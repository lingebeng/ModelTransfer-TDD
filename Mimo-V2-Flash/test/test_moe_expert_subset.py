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
from mimo_v2_flash.modeling import MoEGate, MLP
from mimo_v2_flash.pytorch.configuration_mimo_v2_flash import MiMoV2FlashConfig
from mimo_v2_flash.pytorch.modeling_mimo_v2_flash import MiMoV2MoEGate, MiMoV2MLP


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


class TestMoEExpertSubset(absltest.TestCase):
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

        cls.layer_idx = 1
        cls.batch_size = 2
        cls.num_tokens = 3
        cls.relaxed_tol = 1e-3

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
            n_routed_experts=cfg_json.get("n_routed_experts"),
            num_experts_per_tok=cfg_json.get("num_experts_per_tok"),
            moe_intermediate_size=cfg_json.get("moe_intermediate_size"),
            moe_layer_freq=[False, True],
            routed_scaling_factor=cfg_json.get("routed_scaling_factor"),
            scoring_func=cfg_json.get("scoring_func", "sigmoid"),
            topk_method=cfg_json.get("topk_method", "noaux_tc"),
            n_group=cfg_json.get("n_group"),
            topk_group=cfg_json.get("topk_group"),
            norm_topk_prob=cfg_json.get("norm_topk_prob", True),
        )

        cls.weight_map = _load_weight_map(cls.weight_dir)

        # Gates for selecting experts
        gate_w = _w(
            cls.weight_dir,
            cls.weight_map,
            f"model.layers.{cls.layer_idx}.mlp.gate.weight",
        )
        gate_b = _w(
            cls.weight_dir,
            cls.weight_map,
            f"model.layers.{cls.layer_idx}.mlp.gate.e_score_correction_bias",
        )

        cls.jax_gate = MoEGate(cls.cfg, rngs=nnx.Rngs(params=0))
        cls.jax_gate.w[...] = jnp.asarray(gate_w)
        cls.jax_gate.e_score_correction_bias[...] = jnp.asarray(gate_b)

        torch_cfg = MiMoV2FlashConfig(**cfg_json)
        cls.torch_cfg = torch_cfg
        cls.torch_gate = MiMoV2MoEGate(torch_cfg).eval()
        with torch.no_grad():
            cls.torch_gate.weight.copy_(torch.from_numpy(gate_w))
            cls.torch_gate.e_score_correction_bias.copy_(torch.from_numpy(gate_b))

    def _build_expert_pair(self, expert_idx: int):
        # JAX expert
        jax_expert = MLP(
            self.cfg, rngs=nnx.Rngs(params=0), mlp_dim=self.cfg.moe_intermediate_size
        )
        # Torch expert
        torch_expert = MiMoV2MLP(
            self.torch_cfg, intermediate_size=self.cfg.moe_intermediate_size
        ).to(torch.float32)

        # Load weights
        g_key = (
            f"model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
        )
        u_key = f"model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
        d_key = (
            f"model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
        )
        g_w = _w(self.weight_dir, self.weight_map, g_key)
        u_w = _w(self.weight_dir, self.weight_map, u_key)
        d_w = _w(self.weight_dir, self.weight_map, d_key)

        jax_expert.gate_proj.kernel[...] = jnp.asarray(g_w.T)
        jax_expert.up_proj.kernel[...] = jnp.asarray(u_w.T)
        jax_expert.down_proj.kernel[...] = jnp.asarray(d_w.T)

        with torch.no_grad():
            torch_expert.gate_proj.weight.copy_(torch.from_numpy(g_w))
            torch_expert.up_proj.weight.copy_(torch.from_numpy(u_w))
            torch_expert.down_proj.weight.copy_(torch.from_numpy(d_w))

        return jax_expert, torch_expert

    def test_expert_subset_outputs(self):
        if self.jax_gate is None or self.torch_gate is None:
            self.skipTest("weights not available")

        tx = torch.randn(
            self.batch_size, self.num_tokens, self.cfg.emb_dim, dtype=torch.float32
        )
        jx = jnp.asarray(tx.numpy())

        # Use gate to pick top-k experts
        j_idx, _ = self.jax_gate(jx)
        expert_ids = np.unique(np.array(j_idx))[:2]  # limit to 2 experts for speed

        for e in expert_ids:
            j_expert, t_expert = self._build_expert_pair(int(e))
            j_out = j_expert(jx)
            t_out = t_expert(tx)

            torch.testing.assert_close(
                torch.tensor(np.array(j_out, dtype=np.float32)),
                t_out.detach(),
                rtol=self.relaxed_tol,
                atol=self.relaxed_tol,
                check_dtype=False,
            )


if __name__ == "__main__":
    absltest.main()
