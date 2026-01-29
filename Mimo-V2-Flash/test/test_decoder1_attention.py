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
from transformers.masking_utils import create_sliding_window_causal_mask


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
from mimo_v2_flash.modeling import Attention, LayerCache, RMSNorm
from mimo_v2_flash.pytorch.configuration_mimo_v2_flash import MiMoV2FlashConfig
from mimo_v2_flash.pytorch.modeling_mimo_v2_flash import (
    MiMoV2Attention,
    MiMoV2RMSNorm,
    MiMoV2FlashRotaryEmbedding,
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
    w = weight.float()
    s = scale_inv.float()
    s = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    s = s[: w.shape[0], : w.shape[1]]
    return w * s


def _w(weight_dir: Path, weight_map: dict, key: str) -> np.ndarray:
    w = _load_tensor(weight_dir, weight_map, key)
    scale_key = f"{key}_scale_inv"
    if scale_key in weight_map:
        scale = _load_tensor(weight_dir, weight_map, scale_key)
        w = _dequant_fp8(w, scale)
    return w.float().cpu().numpy()


def _transform_q(weight, cfg: ModelConfig, layer_idx: int):
    w = weight.reshape(
        cfg.num_heads_for_layer(layer_idx),
        cfg.head_dim_for_layer(layer_idx),
        cfg.emb_dim,
    )
    return w.transpose(2, 0, 1)


def _transform_k(weight, cfg: ModelConfig, layer_idx: int):
    w = weight.reshape(
        cfg.num_kv_heads_for_layer(layer_idx),
        cfg.head_dim_for_layer(layer_idx),
        cfg.emb_dim,
    )
    return w.transpose(2, 0, 1)


def _transform_v(weight, cfg: ModelConfig, layer_idx: int):
    w = weight.reshape(
        cfg.num_kv_heads_for_layer(layer_idx),
        cfg.v_head_dim_for_layer(layer_idx),
        cfg.emb_dim,
    )
    return w.transpose(2, 0, 1)


def _transform_o(weight, cfg: ModelConfig, layer_idx: int):
    w = weight.T
    return w.reshape(
        cfg.num_heads_for_layer(layer_idx),
        cfg.v_head_dim_for_layer(layer_idx),
        cfg.emb_dim,
    )


class TestDecoder1Attention(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.weight_dir = Path(__file__).resolve().parents[1] / "weight"
        cls.config_path = cls.weight_dir / "config.json"
        if not cls.config_path.exists():
            cls.skipTest(cls, "weight files not found")

        with cls.config_path.open("r", encoding="utf-8") as f:
            cfg_json = json.load(f)

        cls.layer_idx = 1
        cls.batch_size = 1
        cls.num_input_tokens = 4
        cls.relaxed_tol = 1e-2

        cls.cfg = ModelConfig(
            num_layers=2,
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
            hybrid_layer_pattern=[0, 1],
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
            moe_layer_freq=[False, False],
            routed_scaling_factor=None,
            scoring_func=cfg_json.get("scoring_func", "sigmoid"),
            topk_method=cfg_json.get("topk_method", "noaux_tc"),
            n_group=None,
            topk_group=None,
            norm_topk_prob=cfg_json.get("norm_topk_prob", True),
        )

        weight_map = _load_weight_map(cls.weight_dir)

        # JAX modules
        cls.jax_attn = Attention(
            cfg=cls.cfg, layer_idx=cls.layer_idx, rngs=nnx.Rngs(params=0)
        )
        cls.jax_in_norm = RMSNorm(cls.cfg.emb_dim, cls.cfg, rngs=nnx.Rngs(params=0))
        cls.jax_post_norm = RMSNorm(cls.cfg.emb_dim, cls.cfg, rngs=nnx.Rngs(params=0))

        cls.jax_in_norm.scale[...] = jnp.asarray(
            _w(
                cls.weight_dir,
                weight_map,
                f"model.layers.{cls.layer_idx}.input_layernorm.weight",
            )
        )
        cls.jax_post_norm.scale[...] = jnp.asarray(
            _w(
                cls.weight_dir,
                weight_map,
                f"model.layers.{cls.layer_idx}.post_attention_layernorm.weight",
            )
        )
        cls.jax_attn.q_proj.w[...] = jnp.asarray(
            _transform_q(
                _w(
                    cls.weight_dir,
                    weight_map,
                    f"model.layers.{cls.layer_idx}.self_attn.q_proj.weight",
                ),
                cls.cfg,
                cls.layer_idx,
            )
        )
        cls.jax_attn.k_proj.w[...] = jnp.asarray(
            _transform_k(
                _w(
                    cls.weight_dir,
                    weight_map,
                    f"model.layers.{cls.layer_idx}.self_attn.k_proj.weight",
                ),
                cls.cfg,
                cls.layer_idx,
            )
        )
        cls.jax_attn.v_proj.w[...] = jnp.asarray(
            _transform_v(
                _w(
                    cls.weight_dir,
                    weight_map,
                    f"model.layers.{cls.layer_idx}.self_attn.v_proj.weight",
                ),
                cls.cfg,
                cls.layer_idx,
            )
        )
        cls.jax_attn.o_proj.w[...] = jnp.asarray(
            _transform_o(
                _w(
                    cls.weight_dir,
                    weight_map,
                    f"model.layers.{cls.layer_idx}.self_attn.o_proj.weight",
                ),
                cls.cfg,
                cls.layer_idx,
            )
        )

        # Torch modules (attention + norms only)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            torch_cfg = MiMoV2FlashConfig(
                **{
                    **cfg_json,
                    "num_hidden_layers": 2,
                    "hybrid_layer_pattern": [0, 1],
                    "moe_layer_freq": [0, 0],
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
                    "_attn_implementation": "eager",
                }
            )
        import mimo_v2_flash.pytorch.modeling_mimo_v2_flash as mimo_flash

        if not getattr(mimo_flash.eager_attention_forward, "_accepts_kwargs", False):
            _orig_eager = mimo_flash.eager_attention_forward

            def _eager_wrapper(*args, **kwargs):
                kwargs.pop("position_ids", None)
                return _orig_eager(*args, **kwargs)

            _eager_wrapper._accepts_kwargs = True
            mimo_flash.eager_attention_forward = _eager_wrapper
        cls.torch_attn = MiMoV2Attention(torch_cfg, True, cls.layer_idx).to(
            torch.float32
        )
        cls.torch_in_norm = MiMoV2RMSNorm(
            torch_cfg.hidden_size, eps=torch_cfg.layernorm_epsilon
        ).to(torch.float32)
        cls.torch_post_norm = MiMoV2RMSNorm(
            torch_cfg.hidden_size, eps=torch_cfg.layernorm_epsilon
        ).to(torch.float32)
        cls.torch_rotary = MiMoV2FlashRotaryEmbedding(torch_cfg, is_swa=True)

        with torch.no_grad():
            cls.torch_in_norm.weight.copy_(
                torch.from_numpy(
                    _w(
                        cls.weight_dir,
                        weight_map,
                        f"model.layers.{cls.layer_idx}.input_layernorm.weight",
                    )
                )
            )
            cls.torch_post_norm.weight.copy_(
                torch.from_numpy(
                    _w(
                        cls.weight_dir,
                        weight_map,
                        f"model.layers.{cls.layer_idx}.post_attention_layernorm.weight",
                    )
                )
            )
            cls.torch_attn.q_proj.weight.copy_(
                torch.from_numpy(
                    _w(
                        cls.weight_dir,
                        weight_map,
                        f"model.layers.{cls.layer_idx}.self_attn.q_proj.weight",
                    )
                )
            )
            cls.torch_attn.k_proj.weight.copy_(
                torch.from_numpy(
                    _w(
                        cls.weight_dir,
                        weight_map,
                        f"model.layers.{cls.layer_idx}.self_attn.k_proj.weight",
                    )
                )
            )
            cls.torch_attn.v_proj.weight.copy_(
                torch.from_numpy(
                    _w(
                        cls.weight_dir,
                        weight_map,
                        f"model.layers.{cls.layer_idx}.self_attn.v_proj.weight",
                    )
                )
            )
            cls.torch_attn.o_proj.weight.copy_(
                torch.from_numpy(
                    _w(
                        cls.weight_dir,
                        weight_map,
                        f"model.layers.{cls.layer_idx}.self_attn.o_proj.weight",
                    )
                )
            )

    def test_decoder1_attention_only(self):
        key = jax.random.key(0)
        shape = (self.batch_size, self.num_input_tokens, self.cfg.emb_dim)
        jx = jax.random.normal(key, shape=shape, dtype=jnp.float32)
        tx = torch.tensor(np.array(jx, dtype=np.float32))

        # Torch forward: norm -> attn -> residual -> post_norm (no MLP)
        t_in = self.torch_in_norm(tx)
        past_key_values = DynamicCache(config=self.torch_attn.config)
        cache_position = torch.arange(0, tx.shape[1], device=tx.device)
        position_ids = cache_position.unsqueeze(0)
        mask_kwargs = {
            "config": self.torch_attn.config,
            "input_embeds": tx,
            "attention_mask": None,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        attention_mask = create_sliding_window_causal_mask(**mask_kwargs)
        position_embeddings = self.torch_rotary(tx, position_ids)
        t_attn, _ = self.torch_attn(
            hidden_states=t_in,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            cache_position=cache_position,
        )
        t_out = self.torch_post_norm(tx + t_attn)

        # JAX forward: norm -> attn -> residual -> post_norm (no MLP)
        j_in = self.jax_in_norm(jx)
        cache_size = 2 ** int(np.ceil(np.log2(max(self.num_input_tokens, 1))))
        j_cache = LayerCache(
            self.cfg.num_kv_heads_for_layer(self.layer_idx),
            self.cfg.head_dim_for_layer(self.layer_idx),
            self.cfg.v_head_dim_for_layer(self.layer_idx),
            self.cfg.shd_cfg,
            self.batch_size,
            cache_size,
            jnp.float32,
        )
        j_attn = self.jax_attn(
            j_in, j_cache, jnp.ones((self.batch_size, self.num_input_tokens))
        )
        j_out = self.jax_post_norm(jx + j_attn)

        torch.testing.assert_close(
            torch.tensor(np.array(j_out, dtype=np.float32)),
            t_out,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )


if __name__ == "__main__":
    absltest.main()
