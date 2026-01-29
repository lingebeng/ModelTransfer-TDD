import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_LOG_LEVEL", "ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.simplefilter("ignore", FutureWarning)

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx


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
from mimo_v2_flash.modeling import MiMoV2Flash
from mimo_v2_flash.pytorch.configuration_mimo_v2_flash import MiMoV2FlashConfig
from mimo_v2_flash.pytorch.modeling_mimo_v2_flash import MiMoV2FlashForCausalLM


class _Identity(nnx.Module):
    def __call__(self, x):
        return x


def _load_weight_map(weight_dir: Path):
    index_path = weight_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def _load_tensor(weight_dir: Path, weight_map: dict, key: str, framework: str = "pt"):
    import safetensors

    filename = weight_map.get(key)
    if filename is None:
        raise KeyError(f"Missing weight key: {key}")
    with safetensors.safe_open(str(weight_dir / filename), framework=framework) as sf:
        return sf.get_tensor(key)


def _dequant_fp8(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    w = weight.float()
    s = scale_inv.float()
    s = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    s = s[: w.shape[0], : w.shape[1]]
    return w * s


def _w_pt(
    weight_dir: Path, weight_map: dict, key: str, dtype: torch.dtype
) -> torch.Tensor:
    w = _load_tensor(weight_dir, weight_map, key, framework="pt")
    scale_key = f"{key}_scale_inv"
    if scale_key in weight_map:
        scale = _load_tensor(weight_dir, weight_map, scale_key, framework="pt")
        w = _dequant_fp8(w, scale)
    return w.to(dtype)


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


class TestFullForwardTruncated(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.weight_dir = Path(__file__).resolve().parents[1] / "weight"
        cls.config_path = cls.weight_dir / "config.json"
        if not cls.config_path.exists():
            return

        with cls.config_path.open("r", encoding="utf-8") as f:
            cfg_json = json.load(f)

        cls.batch_size = 1
        cls.num_tokens = 1
        cls.relaxed_tol = 2e-2
        cls.vocab_subset = 8192

        vocab_size = min(cls.vocab_subset, cfg_json["vocab_size"])
        cls.cfg = ModelConfig(
            num_layers=2,
            vocab_size=vocab_size,
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
            add_full_attention_sink_bias=False,
            add_swa_attention_sink_bias=False,
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

        # Build JAX model (1 layer) and load weights
        cls.jax_model = MiMoV2Flash(cls.cfg, rngs=nnx.Rngs(params=0))

        # Use safetensors slicing to avoid loading full vocab into memory.
        import safetensors

        embed_file = weight_map["model.embed_tokens.weight"]
        with safetensors.safe_open(
            str(cls.weight_dir / embed_file), framework="numpy"
        ) as sf:
            embed_w_np = sf.get_slice("model.embed_tokens.weight")[:vocab_size, :]
        cls.jax_model.embedder.embedding[...] = jnp.asarray(embed_w_np)

        # Layer0 weights
        q_w = (
            _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.0.self_attn.q_proj.weight",
                torch.float32,
            )
            .cpu()
            .numpy()
        )
        k_w = (
            _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.0.self_attn.k_proj.weight",
                torch.float32,
            )
            .cpu()
            .numpy()
        )
        v_w = (
            _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.0.self_attn.v_proj.weight",
                torch.float32,
            )
            .cpu()
            .numpy()
        )
        o_w = (
            _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.0.self_attn.o_proj.weight",
                torch.float32,
            )
            .cpu()
            .numpy()
        )
        cls.jax_model.layers[0].attn.q_proj.w[...] = jnp.asarray(
            _transform_q(q_w, cls.cfg, 0)
        )
        cls.jax_model.layers[0].attn.k_proj.w[...] = jnp.asarray(
            _transform_k(k_w, cls.cfg, 0)
        )
        cls.jax_model.layers[0].attn.v_proj.w[...] = jnp.asarray(
            _transform_v(v_w, cls.cfg, 0)
        )
        cls.jax_model.layers[0].attn.o_proj.w[...] = jnp.asarray(
            _transform_o(o_w, cls.cfg, 0)
        )

        gate_w = (
            _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.0.mlp.gate_proj.weight",
                torch.float32,
            )
            .cpu()
            .numpy()
        )
        up_w = (
            _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.0.mlp.up_proj.weight",
                torch.float32,
            )
            .cpu()
            .numpy()
        )
        down_w = (
            _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.0.mlp.down_proj.weight",
                torch.float32,
            )
            .cpu()
            .numpy()
        )
        cls.jax_model.layers[0].mlp.gate_proj.kernel[...] = jnp.asarray(gate_w.T)
        cls.jax_model.layers[0].mlp.up_proj.kernel[...] = jnp.asarray(up_w.T)
        cls.jax_model.layers[0].mlp.down_proj.kernel[...] = jnp.asarray(down_w.T)

        in_norm_np = _load_tensor(
            cls.weight_dir,
            weight_map,
            "model.layers.0.input_layernorm.weight",
            framework="numpy",
        )
        post_norm_np = _load_tensor(
            cls.weight_dir,
            weight_map,
            "model.layers.0.post_attention_layernorm.weight",
            framework="numpy",
        )
        cls.jax_model.layers[0].input_layernorm.scale[...] = jnp.asarray(in_norm_np)
        cls.jax_model.layers[0].post_attention_layernorm.scale[...] = jnp.asarray(
            post_norm_np
        )

        # Layer1 weights (attention + dense MLP)
        q1_w = (
            _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.1.self_attn.q_proj.weight",
                torch.float32,
            )
            .cpu()
            .numpy()
        )
        k1_w = (
            _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.1.self_attn.k_proj.weight",
                torch.float32,
            )
            .cpu()
            .numpy()
        )
        v1_w = (
            _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.1.self_attn.v_proj.weight",
                torch.float32,
            )
            .cpu()
            .numpy()
        )
        o1_w = (
            _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.1.self_attn.o_proj.weight",
                torch.float32,
            )
            .cpu()
            .numpy()
        )
        cls.jax_model.layers[1].attn.q_proj.w[...] = jnp.asarray(
            _transform_q(q1_w, cls.cfg, 1)
        )
        cls.jax_model.layers[1].attn.k_proj.w[...] = jnp.asarray(
            _transform_k(k1_w, cls.cfg, 1)
        )
        cls.jax_model.layers[1].attn.v_proj.w[...] = jnp.asarray(
            _transform_v(v1_w, cls.cfg, 1)
        )
        cls.jax_model.layers[1].attn.o_proj.w[...] = jnp.asarray(
            _transform_o(o1_w, cls.cfg, 1)
        )

        # Skip layer1 MLP (MoE weights are too large); align both sides by using identity.
        cls.jax_model.layers[1].mlp = _Identity()

        in1_np = _load_tensor(
            cls.weight_dir,
            weight_map,
            "model.layers.1.input_layernorm.weight",
            framework="numpy",
        )
        post1_np = _load_tensor(
            cls.weight_dir,
            weight_map,
            "model.layers.1.post_attention_layernorm.weight",
            framework="numpy",
        )
        cls.jax_model.layers[1].input_layernorm.scale[...] = jnp.asarray(in1_np)
        cls.jax_model.layers[1].post_attention_layernorm.scale[...] = jnp.asarray(
            post1_np
        )

        final_norm_np = _load_tensor(
            cls.weight_dir, weight_map, "model.norm.weight", framework="numpy"
        )
        cls.jax_model.final_norm.scale[...] = jnp.asarray(final_norm_np)

        lm_file = weight_map["lm_head.weight"]
        with safetensors.safe_open(
            str(cls.weight_dir / lm_file), framework="numpy"
        ) as sf:
            lm_head_np = sf.get_slice("lm_head.weight")[:vocab_size, :]
        cls.jax_model.lm_head.w[...] = jnp.asarray(lm_head_np.T)

        # Build Torch model (1 layer) and load weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            torch_cfg = MiMoV2FlashConfig(
                **{
                    **cfg_json,
                    "num_hidden_layers": 2,
                    "vocab_size": vocab_size,
                    "hybrid_layer_pattern": [0, 1],
                    "moe_layer_freq": [0, 0],
                    "n_routed_experts": None,
                    "num_experts_per_tok": None,
                    "moe_intermediate_size": None,
                    "add_full_attention_sink_bias": False,
                    "add_swa_attention_sink_bias": False,
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

        cls.torch_model = MiMoV2FlashForCausalLM(torch_cfg).eval()
        with torch.no_grad():
            embed_w = _w_pt(
                cls.weight_dir, weight_map, "model.embed_tokens.weight", torch.bfloat16
            )[:vocab_size]
            cls.torch_model.model.embed_tokens.weight.copy_(embed_w)
            layer0 = cls.torch_model.model.layers[0]
            in_norm = _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.0.input_layernorm.weight",
                torch.bfloat16,
            )
            post_norm = _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.0.post_attention_layernorm.weight",
                torch.bfloat16,
            )
            layer0.input_layernorm.weight.copy_(in_norm)
            layer0.post_attention_layernorm.weight.copy_(post_norm)
            layer0.self_attn.q_proj.weight.copy_(torch.from_numpy(q_w))
            layer0.self_attn.k_proj.weight.copy_(torch.from_numpy(k_w))
            layer0.self_attn.v_proj.weight.copy_(torch.from_numpy(v_w))
            layer0.self_attn.o_proj.weight.copy_(torch.from_numpy(o_w))
            layer0.mlp.gate_proj.weight.copy_(torch.from_numpy(gate_w))
            layer0.mlp.up_proj.weight.copy_(torch.from_numpy(up_w))
            layer0.mlp.down_proj.weight.copy_(torch.from_numpy(down_w))

            # Layer1 weights (dense MLP in truncated config)
            q1_w = (
                _w_pt(
                    cls.weight_dir,
                    weight_map,
                    "model.layers.1.self_attn.q_proj.weight",
                    torch.float32,
                )
                .cpu()
                .numpy()
            )
            k1_w = (
                _w_pt(
                    cls.weight_dir,
                    weight_map,
                    "model.layers.1.self_attn.k_proj.weight",
                    torch.float32,
                )
                .cpu()
                .numpy()
            )
            v1_w = (
                _w_pt(
                    cls.weight_dir,
                    weight_map,
                    "model.layers.1.self_attn.v_proj.weight",
                    torch.float32,
                )
                .cpu()
                .numpy()
            )
            o1_w = (
                _w_pt(
                    cls.weight_dir,
                    weight_map,
                    "model.layers.1.self_attn.o_proj.weight",
                    torch.float32,
                )
                .cpu()
                .numpy()
            )
            in1 = _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.1.input_layernorm.weight",
                torch.bfloat16,
            )
            post1 = _w_pt(
                cls.weight_dir,
                weight_map,
                "model.layers.1.post_attention_layernorm.weight",
                torch.bfloat16,
            )
            layer1 = cls.torch_model.model.layers[1]
            layer1.input_layernorm.weight.copy_(in1)
            layer1.post_attention_layernorm.weight.copy_(post1)
            layer1.self_attn.q_proj.weight.copy_(torch.from_numpy(q1_w))
            layer1.self_attn.k_proj.weight.copy_(torch.from_numpy(k1_w))
            layer1.self_attn.v_proj.weight.copy_(torch.from_numpy(v1_w))
            layer1.self_attn.o_proj.weight.copy_(torch.from_numpy(o1_w))
            layer1.mlp = torch.nn.Identity()
            final_norm = _w_pt(
                cls.weight_dir, weight_map, "model.norm.weight", torch.bfloat16
            )
            lm_head = _w_pt(
                cls.weight_dir, weight_map, "lm_head.weight", torch.bfloat16
            )[:vocab_size]
            cls.torch_model.model.norm.weight.copy_(final_norm)
            cls.torch_model.lm_head.weight.copy_(lm_head)

    def test_full_forward_truncated(self):
        if self.jax_model is None or self.torch_model is None:
            self.skipTest("weights not available")

        torch.manual_seed(0)
        tx = torch.randint(
            1, self.cfg.vocab_size, size=(self.batch_size, self.num_tokens)
        )
        jx = jnp.asarray(tx.numpy())

        # Torch forward
        with torch.no_grad():
            t_logits = self.torch_model(input_ids=tx).logits

        # JAX forward
        cache = self.jax_model.init_cache(
            self.cfg, self.batch_size, self.num_tokens, 0, dtype=jnp.float32
        )
        segment_ids = (jx != 0).astype(jnp.int32)
        j_logits = self.jax_model(jx, segment_ids, cache, num_right_pads=0)

        torch.testing.assert_close(
            torch.tensor(np.array(j_logits, dtype=np.float32)),
            t_logits.float(),
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )


if __name__ == "__main__":
    absltest.main()
