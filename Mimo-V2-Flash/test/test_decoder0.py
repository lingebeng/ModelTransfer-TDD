import importlib.util
import json
import os
import warnings
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_LOG_LEVEL", "ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

warnings.simplefilter("ignore", FutureWarning)

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask


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
from mimo_v2_flash.modeling import MiMoV2Flash, DecoderLayer
from mimo_v2_flash.pytorch.configuration_mimo_v2_flash import MiMoV2FlashConfig
from mimo_v2_flash.pytorch.modeling_mimo_v2_flash import (
    MiMoV2DecoderLayer,
    MiMoV2FlashRotaryEmbedding,
    MiMoV2FlashForCausalLM,
)


def _load_weight_map(weight_dir: Path):
    index_path = weight_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def _load_tensor(weight_dir: Path, weight_map: dict, key: str):
    import safetensors

    filename = weight_map.get(key)
    if filename is None:
        raise KeyError(f"Missing weight key: {key}")
    with safetensors.safe_open(str(weight_dir / filename), framework="pt") as sf:
        return sf.get_tensor(key)


def _dequant_fp8(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    # Dequantize blockwise FP8 weights: weight_fp8 * scale_inv (block repeated).
    w = weight.float()
    s = scale_inv.float()
    s = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    s = s[: w.shape[0], : w.shape[1]]
    return w * s


def _transform_q(weight, cfg: ModelConfig):
    w = weight.reshape(cfg.num_heads, cfg.head_dim, cfg.emb_dim)
    return w.transpose(2, 0, 1)


def _transform_k(weight, cfg: ModelConfig):
    w = weight.reshape(cfg.num_kv_heads, cfg.head_dim, cfg.emb_dim)
    return w.transpose(2, 0, 1)


def _transform_v(weight, cfg: ModelConfig):
    w = weight.reshape(cfg.num_kv_heads, cfg.v_head_dim, cfg.emb_dim)
    return w.transpose(2, 0, 1)


def _transform_o(weight, cfg: ModelConfig):
    w = weight.T
    return w.reshape(cfg.num_heads, cfg.v_head_dim, cfg.emb_dim)


class TestDecoder0(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        warnings.simplefilter("ignore", FutureWarning)
        cls.weight_dir = Path(__file__).resolve().parents[1] / "weight"
        cls.config_path = cls.weight_dir / "config.json"
        if not cls.config_path.exists():
            return

        with cls.config_path.open("r", encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Ensure eager_attention_forward accepts kwargs (e.g., position_ids).
        import mimo_v2_flash.pytorch.modeling_mimo_v2_flash as mimo_flash

        if not getattr(mimo_flash.eager_attention_forward, "_accepts_kwargs", False):
            _orig_eager = mimo_flash.eager_attention_forward

            def _eager_wrapper(*args, **kwargs):
                kwargs.pop("position_ids", None)
                return _orig_eager(*args, **kwargs)

            _eager_wrapper._accepts_kwargs = True
            mimo_flash.eager_attention_forward = _eager_wrapper

        cls.cfg = ModelConfig(
            num_layers=1,
            vocab_size=cfg_json["vocab_size"],
            emb_dim=cfg_json["hidden_size"],
            mlp_dim=cfg_json["intermediate_size"],
            num_heads=cfg_json["num_attention_heads"],
            head_dim=cfg_json["head_dim"],
            num_kv_heads=cfg_json.get("num_key_value_heads"),
            v_head_dim=cfg_json.get("v_head_dim"),
            rope_theta=cfg_json["rope_theta"],
            max_position_embeddings=cfg_json["max_position_embeddings"],
            norm_eps=cfg_json["layernorm_epsilon"],
            tie_word_embeddings=cfg_json["tie_word_embeddings"],
            attention_bias=cfg_json.get("attention_bias", False),
            attention_dropout=cfg_json.get("attention_dropout", 0.0),
            partial_rotary_factor=cfg_json.get("partial_rotary_factor", 1.0),
            hybrid_layer_pattern=[0],
            sliding_window=cfg_json.get("sliding_window")
            or cfg_json.get("sliding_window_size"),
            swa_num_heads=cfg_json.get("swa_num_attention_heads"),
            swa_num_kv_heads=cfg_json.get("swa_num_key_value_heads"),
            swa_head_dim=cfg_json.get("swa_head_dim"),
            swa_v_head_dim=cfg_json.get("swa_v_head_dim"),
            swa_rope_theta=cfg_json.get("swa_rope_theta"),
            add_full_attention_sink_bias=cfg_json.get(
                "add_full_attention_sink_bias", False
            ),
            add_swa_attention_sink_bias=cfg_json.get(
                "add_swa_attention_sink_bias", False
            ),
            n_routed_experts=None,
            num_experts_per_tok=None,
            moe_intermediate_size=None,
            moe_layer_freq=[False],
            routed_scaling_factor=None,
            scoring_func=cfg_json.get("scoring_func", "sigmoid"),
            topk_method=cfg_json.get("topk_method", "noaux_tc"),
            n_group=None,
            topk_group=None,
            norm_topk_prob=cfg_json.get("norm_topk_prob", True),
        )

        weight_map = _load_weight_map(cls.weight_dir)

        # JAX model skeleton + real layer0
        cls.nnx_model = nnx.eval_shape(
            lambda: MiMoV2Flash(cls.cfg, rngs=nnx.Rngs(params=0))
        )
        cls.nnx_model.layers = nnx.List(
            [DecoderLayer(cfg=cls.cfg, layer_idx=0, rngs=nnx.Rngs(params=0))]
        )
        layer0 = cls.nnx_model.layers[0]

        # Assign JAX layer0 weights
        def _w(key):
            w = _load_tensor(cls.weight_dir, weight_map, key)
            scale_key = f"{key}_scale_inv"
            if scale_key in weight_map:
                scale = _load_tensor(cls.weight_dir, weight_map, scale_key)
                w = _dequant_fp8(w, scale)
            return w.float().cpu().numpy()

        layer0.input_layernorm.scale[...] = jnp.asarray(
            _w("model.layers.0.input_layernorm.weight")
        )
        layer0.post_attention_layernorm.scale[...] = jnp.asarray(
            _w("model.layers.0.post_attention_layernorm.weight")
        )
        layer0.attn.q_proj.w[...] = jnp.asarray(
            _transform_q(_w("model.layers.0.self_attn.q_proj.weight"), cls.cfg)
        )
        layer0.attn.k_proj.w[...] = jnp.asarray(
            _transform_k(_w("model.layers.0.self_attn.k_proj.weight"), cls.cfg)
        )
        layer0.attn.v_proj.w[...] = jnp.asarray(
            _transform_v(_w("model.layers.0.self_attn.v_proj.weight"), cls.cfg)
        )
        layer0.attn.o_proj.w[...] = jnp.asarray(
            _transform_o(_w("model.layers.0.self_attn.o_proj.weight"), cls.cfg)
        )
        layer0.mlp.gate_proj.kernel[...] = jnp.asarray(
            _w("model.layers.0.mlp.gate_proj.weight").T
        )
        layer0.mlp.up_proj.kernel[...] = jnp.asarray(
            _w("model.layers.0.mlp.up_proj.weight").T
        )
        layer0.mlp.down_proj.kernel[...] = jnp.asarray(
            _w("model.layers.0.mlp.down_proj.weight").T
        )

        # Torch model (meta), replace layer0 with real weights
        torch_cfg = MiMoV2FlashConfig(
            **{
                **cfg_json,
                "num_hidden_layers": 1,
                "hybrid_layer_pattern": [0],
                "moe_layer_freq": [0],
                "n_routed_experts": None,
                "num_experts_per_tok": None,
                "moe_intermediate_size": None,
                "add_full_attention_sink_bias": cfg_json.get(
                    "add_full_attention_sink_bias", False
                ),
                "add_swa_attention_sink_bias": cfg_json.get(
                    "add_swa_attention_sink_bias", False
                ),
                "rope_scaling": {"rope_type": "linear", "factor": 1.0},
            }
        )
        prev_device = None
        if hasattr(torch, "get_default_device"):
            prev_device = torch.get_default_device()
            torch.set_default_device("meta")
        try:
            cls.torch_model = MiMoV2FlashForCausalLM(torch_cfg).eval()
        finally:
            if prev_device is not None:
                torch.set_default_device(prev_device)

        torch_layer0 = MiMoV2DecoderLayer(torch_cfg, 0).to(torch.float32)
        with torch.no_grad():
            torch_layer0.input_layernorm.weight.copy_(
                torch.from_numpy(_w("model.layers.0.input_layernorm.weight"))
            )
            torch_layer0.post_attention_layernorm.weight.copy_(
                torch.from_numpy(_w("model.layers.0.post_attention_layernorm.weight"))
            )
            torch_layer0.self_attn.q_proj.weight.copy_(
                torch.from_numpy(_w("model.layers.0.self_attn.q_proj.weight"))
            )
            torch_layer0.self_attn.k_proj.weight.copy_(
                torch.from_numpy(_w("model.layers.0.self_attn.k_proj.weight"))
            )
            torch_layer0.self_attn.v_proj.weight.copy_(
                torch.from_numpy(_w("model.layers.0.self_attn.v_proj.weight"))
            )
            torch_layer0.self_attn.o_proj.weight.copy_(
                torch.from_numpy(_w("model.layers.0.self_attn.o_proj.weight"))
            )
            torch_layer0.mlp.gate_proj.weight.copy_(
                torch.from_numpy(_w("model.layers.0.mlp.gate_proj.weight"))
            )
            torch_layer0.mlp.up_proj.weight.copy_(
                torch.from_numpy(_w("model.layers.0.mlp.up_proj.weight"))
            )
            torch_layer0.mlp.down_proj.weight.copy_(
                torch.from_numpy(_w("model.layers.0.mlp.down_proj.weight"))
            )

        cls.torch_model.model.layers[0] = torch_layer0
        cls.torch_model._test_rotary = MiMoV2FlashRotaryEmbedding(
            torch_cfg, is_swa=False
        )
        cls.relaxed_tol = 1e-2
        cls.batch_size = 1
        cls.num_input_tokens = 4

    @staticmethod
    def _setup_torch_attn(model, input_embeddings: torch.Tensor):
        past_key_values = DynamicCache(config=model.config)
        cache_position = torch.arange(
            0, input_embeddings.shape[1], device=input_embeddings.device
        )
        position_ids = cache_position.unsqueeze(0)
        mask_kwargs = {
            "config": model.config,
            "input_embeds": input_embeddings,
            "attention_mask": None,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        attention_mask = create_causal_mask(**mask_kwargs)
        if hasattr(model, "_test_rotary") and model._test_rotary is not None:
            position_embeddings = model._test_rotary(input_embeddings, position_ids)
        else:
            position_embeddings = model.model.rotary_emb(input_embeddings, position_ids)
        return dict(
            hidden_states=input_embeddings,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            cache_position=cache_position,
        )

    def _init_nnx_cache(self, batch_size: int, token_len: int):
        return self.nnx_model.init_cache(
            self.cfg,
            batch_size=batch_size,
            token_len=token_len,
            generate_steps=0,
            dtype=jnp.float32,
        )

    def test_decoder0(self):
        if not hasattr(self, "nnx_model") or self.nnx_model is None:
            self.skipTest("weights not available")

        nm = self.nnx_model.layers[0]
        tm = self.torch_model.model.layers[0].to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.cfg.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape, dtype=jnp.float32)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        nnx_cache = self._init_nnx_cache(self.batch_size, self.num_input_tokens)
        torch_inputs = self._setup_torch_attn(self.torch_model, tx)

        jy, ty = (
            nm(jx, nnx_cache[0], jnp.ones((self.batch_size, self.num_input_tokens))),
            tm(**torch_inputs),
        )

        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )


if __name__ == "__main__":
    absltest.main()
