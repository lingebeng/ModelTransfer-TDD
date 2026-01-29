import importlib.util
import os
import sys
import tempfile
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
from flax import nnx
from safetensors.torch import save_file


def _load_package():
    pkg_name = "mimo_v2_flash"
    if pkg_name in sys.modules:
        return
    pkg_path = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        pkg_path / "__init__.py",
        submodule_search_locations=[str(pkg_path)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)


_load_package()

from mimo_v2_flash.config import ModelConfig
from mimo_v2_flash.pytorch.configuration_mimo_v2_flash import MiMoV2FlashConfig
from mimo_v2_flash.pytorch.modeling_mimo_v2_flash import MiMoV2FlashForCausalLM
from mimo_v2_flash import params


class TestTinyForward(absltest.TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def test_tiny_forward_logits(self):
        import mimo_v2_flash.pytorch.modeling_mimo_v2_flash as mimo_flash

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
        if hasattr(torch_cfg, "sliding_window_size"):
            torch_cfg.sliding_window_size = torch_cfg.sliding_window
        torch_model = MiMoV2FlashForCausalLM(torch_cfg).eval()
        # MiMoV2MoEGate uses torch.empty; ensure deterministic finite init for tests.
        for module in torch_model.modules():
            if isinstance(module, mimo_flash.MiMoV2MoEGate):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.e_score_correction_bias is not None:
                    torch.nn.init.zeros_(module.e_score_correction_bias)

        # Build JAX config from torch config
        jax_cfg = ModelConfig(
            num_layers=torch_cfg.num_hidden_layers,
            vocab_size=torch_cfg.vocab_size,
            emb_dim=torch_cfg.hidden_size,
            mlp_dim=torch_cfg.intermediate_size,
            num_heads=torch_cfg.num_attention_heads,
            head_dim=torch_cfg.head_dim,
            num_kv_heads=torch_cfg.num_key_value_heads,
            v_head_dim=getattr(torch_cfg, "v_head_dim", torch_cfg.head_dim),
            rope_theta=torch_cfg.rope_theta,
            max_position_embeddings=torch_cfg.max_position_embeddings,
            norm_eps=torch_cfg.layernorm_epsilon,
            tie_word_embeddings=torch_cfg.tie_word_embeddings,
            attention_bias=getattr(torch_cfg, "attention_bias", False),
            attention_dropout=torch_cfg.attention_dropout,
            partial_rotary_factor=torch_cfg.partial_rotary_factor,
            hybrid_layer_pattern=getattr(torch_cfg, "hybrid_layer_pattern", None),
            sliding_window=getattr(torch_cfg, "sliding_window", None),
            swa_num_heads=getattr(torch_cfg, "swa_num_attention_heads", None),
            swa_num_kv_heads=getattr(torch_cfg, "swa_num_key_value_heads", None),
            swa_head_dim=getattr(torch_cfg, "swa_head_dim", None),
            swa_v_head_dim=getattr(torch_cfg, "swa_v_head_dim", None),
            swa_rope_theta=getattr(torch_cfg, "swa_rope_theta", None),
            add_full_attention_sink_bias=getattr(
                torch_cfg, "add_full_attention_sink_bias", False
            ),
            add_swa_attention_sink_bias=getattr(
                torch_cfg, "add_swa_attention_sink_bias", False
            ),
            n_routed_experts=getattr(torch_cfg, "n_routed_experts", None),
            num_experts_per_tok=getattr(torch_cfg, "num_experts_per_tok", None),
            moe_intermediate_size=getattr(torch_cfg, "moe_intermediate_size", None),
            moe_layer_freq=getattr(torch_cfg, "moe_layer_freq", None),
            routed_scaling_factor=getattr(torch_cfg, "routed_scaling_factor", None),
            scoring_func=getattr(torch_cfg, "scoring_func", "sigmoid"),
            topk_method=getattr(torch_cfg, "topk_method", "noaux_tc"),
            n_group=getattr(torch_cfg, "n_group", None),
            topk_group=getattr(torch_cfg, "topk_group", None),
            norm_topk_prob=getattr(torch_cfg, "norm_topk_prob", True),
        )

        # Dump torch weights to safetensors and load into JAX
        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = Path(tmpdir) / "model.safetensors"
            save_file(torch_model.state_dict(), str(weight_path))
            jax_model = params.create_model_from_safe_tensors(
                tmpdir, jax_cfg, mesh=None
            )
            graph_def, state = nnx.split(jax_model)
            state = jax.tree.map(
                lambda x: x.astype(jnp.float32) if isinstance(x, jax.Array) else x,
                state,
            )
            jax_model = nnx.merge(graph_def, state)
            # Force float32 embedding to reduce numeric issues.
            emb_w = jax_model.embedder.embedding[...]
            jax_model.embedder = nnx.Embed(
                num_embeddings=jax_cfg.vocab_size,
                features=jax_cfg.emb_dim,
                dtype=jnp.float32,
                rngs=nnx.Rngs(params=0),
            )
            jax_model.embedder.embedding[...] = emb_w.astype(jnp.float32)

        batch_size = 2
        seq_len = 5
        tx = torch.randint(1, torch_cfg.vocab_size, size=(batch_size, seq_len))
        with torch.no_grad():
            t_logits = torch_model(input_ids=tx, use_cache=False).logits

        jx = jnp.asarray(tx.numpy())
        cache = jax_model.init_cache(
            jax_cfg,
            batch_size=batch_size,
            token_len=seq_len,
            generate_steps=0,
            dtype=jnp.float32,
        )
        segment_ids = (jx != 0).astype(jnp.int32)
        j_logits = jax_model(jx, segment_ids, cache, num_right_pads=0)

        j_logits_np = np.array(j_logits, dtype=np.float32)
        t_logits_np = t_logits.float().detach().numpy()
        self.assertFalse(np.isnan(j_logits_np).any(), "JAX logits contain NaNs")
        self.assertFalse(np.isnan(t_logits_np).any(), "Torch logits contain NaNs")
        print("JAX logits:", j_logits_np)
        print("Torch logits:", t_logits_np)
        torch.testing.assert_close(
            torch.tensor(j_logits_np),
            torch.tensor(t_logits_np),
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
        )


if __name__ == "__main__":
    absltest.main()
