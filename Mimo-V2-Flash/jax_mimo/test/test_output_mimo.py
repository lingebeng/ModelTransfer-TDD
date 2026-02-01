import importlib.util
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_LOG_LEVEL", "ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


def _load_package():
    pkg_name = "mimo_v2_flash"
    if pkg_name in sys.modules:
        return
    pkg_path = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        pkg_path / "__init__.py",
        submodule_search_locations=[str(pkg_path)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)


_load_package()

from mimo_v2_flash.jax_mimo import modeling, params
from mimo_v2_flash.pytorch.configuration_mimo_v2_flash import MiMoV2FlashConfig
from mimo_v2_flash.pytorch.modeling_mimo_v2_flash import MiMoV2FlashForCausalLM
import mimo_v2_flash.pytorch.modeling_mimo_v2_flash as mimo_flash


class TestTinyForward(absltest.TestCase):
    def setUp(self):
        super().setUp()
        jax.config.update("jax_default_matmul_precision", "float32")
        torch.manual_seed(0)

        if not getattr(mimo_flash.eager_attention_forward, "_accepts_kwargs", False):
            _orig_eager = mimo_flash.eager_attention_forward

            def _eager_wrapper(*args, **kwargs):
                kwargs.pop("position_ids", None)
                return _orig_eager(*args, **kwargs)

            _eager_wrapper._accepts_kwargs = True
            mimo_flash.eager_attention_forward = _eager_wrapper

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            torch_cfg = MiMoV2FlashConfig.tiny_config()
        torch_cfg.rope_scaling = {"rope_type": "linear", "factor": 1.0}
        torch_cfg._attn_implementation = "eager"
        if getattr(torch_cfg, "sliding_window", None) is None:
            torch_cfg.sliding_window = 8

        torch_model = MiMoV2FlashForCausalLM(torch_cfg).eval()
        for module in torch_model.modules():
            if isinstance(module, mimo_flash.MiMoV2MoEGate):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.e_score_correction_bias is not None:
                    torch.nn.init.zeros_(module.e_score_correction_bias)

        jax_cfg = modeling.ModelConfig.from_torch_config(torch_cfg)
        jax_model = params.create_model_from_torch_state_dict(torch_model.state_dict(), jax_cfg)

        self.torch_model = torch_model
        self.jax_model = jax_model
        self.torch_cfg = torch_cfg
        self.jax_cfg = jax_cfg
        self.batch_size = 2
        self.seq_len = 5

    def _setup_torch_attn(self, input_embeds: torch.Tensor, attention_mask: torch.Tensor):
        past_key_values = DynamicCache(config=self.torch_model.config)
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen, past_seen + input_embeds.shape[1], device=input_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)
        mask_kwargs = {
            "config": self.torch_model.config,
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.torch_model.model.has_sliding_layers:
            causal_mask_mapping["sliding_window_attention"] = create_sliding_window_causal_mask(
                **mask_kwargs
            )
        position_embeddings = self.torch_model.model.rotary_emb(input_embeds, position_ids)
        swa_position_embeddings = self.torch_model.model.swa_rotary_emb(input_embeds, position_ids)
        return {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "attention_mask_mapping": causal_mask_mapping,
            "position_embeddings": position_embeddings,
            "swa_position_embeddings": swa_position_embeddings,
        }

    def test_embedder(self):
        tx = torch.randint(0, self.torch_cfg.vocab_size, size=(self.batch_size, self.seq_len))
        ty = self.torch_model.model.embed_tokens(tx)
        jx = jnp.asarray(tx.numpy())
        jy = self.jax_model.model.embed_tokens(jx)
        np.testing.assert_allclose(
            np.array(jy, dtype=np.float32),
            ty.detach().cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_rms_norm(self):
        tx = torch.randn(
            self.batch_size,
            self.seq_len,
            self.torch_cfg.hidden_size,
            dtype=torch.float32,
        )
        ty = self.torch_model.model.layers[0].input_layernorm(tx)
        jx = jnp.asarray(tx.numpy())
        jy = self.jax_model.model.layers[0].input_layernorm(jx)
        np.testing.assert_allclose(
            np.array(jy, dtype=np.float32),
            ty.detach().cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_self_attention_layer0(self):
        layer = self.torch_model.model.layers[0]
        is_swa = layer.attention_type == "sliding_window_attention"
        attention_mask = torch.ones((self.batch_size, self.seq_len), dtype=torch.int64)
        input_embeds = torch.randn(
            self.batch_size, self.seq_len, self.torch_cfg.hidden_size, dtype=torch.float32
        )
        torch_inputs = self._setup_torch_attn(input_embeds, attention_mask)
        attn_mask = torch_inputs["attention_mask_mapping"][layer.attention_type]
        pos_emb = (
            torch_inputs["swa_position_embeddings"]
            if is_swa
            else torch_inputs["position_embeddings"]
        )
        ty, _ = layer.self_attn(
            input_embeds,
            position_embeddings=pos_emb,
            attention_mask=attn_mask,
            past_key_values=None,
            cache_position=torch_inputs["cache_position"],
            position_ids=torch_inputs["position_ids"],
        )

        jx = jnp.asarray(input_embeds.numpy())
        j_mask = modeling.make_attention_mask(
            jnp.asarray(attention_mask.numpy()),
            self.seq_len,
            sliding_window=self.jax_cfg.sliding_window if is_swa else None,
            dtype=jx.dtype,
        )
        jy = self.jax_model.model.layers[0].self_attn(
            jx,
            attention_mask=j_mask,
            position_ids=None,
            deterministic=True,
        )
        np.testing.assert_allclose(
            np.array(jy, dtype=np.float32),
            ty.detach().cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_tiny_forward_logits(self):
        tx = torch.randint(1, self.torch_cfg.vocab_size, size=(self.batch_size, self.seq_len))
        with torch.no_grad():
            t_logits = self.torch_model(
                input_ids=tx,
                attention_mask=torch.ones_like(tx),
                use_cache=False,
                logits_to_keep=1,
            ).logits

        input_ids = jnp.asarray(tx.numpy())
        attention_mask = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
        j_logits = self.jax_model(
            input_ids,
            attention_mask=attention_mask,
            logits_to_keep=1,
            deterministic=True,
        )

        j_logits_np = np.array(j_logits, dtype=np.float32)
        t_logits_np = t_logits.float().detach().numpy()
        self.assertFalse(np.isnan(j_logits_np).any(), "JAX logits contain NaNs")
        self.assertFalse(np.isnan(t_logits_np).any(), "Torch logits contain NaNs")

        np.testing.assert_allclose(
            j_logits_np,
            t_logits_np,
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == "__main__":
    absltest.main()
