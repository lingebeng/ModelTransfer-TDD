import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_LOG_LEVEL", "ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from absl.testing import absltest
import jax.numpy as jnp
from flax import nnx
import numpy as np


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


class TestEmbedding(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _load_package()
        cls.weight_dir = Path(__file__).resolve().parents[1] / "weight"
        cls.config_path = cls.weight_dir / "config.json"
        cls.emb_path = cls.weight_dir / "model_embedding.safetensors"
        cls.weight_np_f32 = None
        cls.nnx_model = None
        cls.torch_model = None
        cls.vocab_size = None
        cls.emb_dim = None
        cls.relaxed_tol = 1e-3
        cls.batch_size = 2
        cls.num_input_tokens = 3

        if not (cls.config_path.exists() and cls.emb_path.exists()):
            return

        try:
            import safetensors
            import torch
        except Exception:
            return

        with cls.config_path.open("r", encoding="utf-8") as f:
            cfg_json = json.load(f)

        cls.vocab_size = cfg_json["vocab_size"]
        cls.emb_dim = cfg_json["hidden_size"]

        with safetensors.safe_open(str(cls.emb_path), framework="numpy") as sf:
            weight_np = sf.get_tensor("model.embed_tokens.weight")
        cls.weight_np_f32 = weight_np.astype("float32", copy=False)

        from mimo_v2_flash.config import ModelConfig
        from mimo_v2_flash.modeling import MiMoV2Flash
        from mimo_v2_flash.pytorch.configuration_mimo_v2_flash import MiMoV2FlashConfig
        from mimo_v2_flash.pytorch.modeling_mimo_v2_flash import MiMoV2FlashForCausalLM

        # Use minimal layer counts to avoid heavy model construction; embedder still uses real weights.
        cfg = ModelConfig(
            num_layers=0,
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
            hybrid_layer_pattern=[],
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
            moe_layer_freq=[],
            routed_scaling_factor=None,
            scoring_func=cfg_json.get("scoring_func", "sigmoid"),
            topk_method=cfg_json.get("topk_method", "noaux_tc"),
            n_group=None,
            topk_group=None,
            norm_topk_prob=cfg_json.get("norm_topk_prob", True),
        )

        # Build JAX model with abstract params to avoid allocating full weights.
        cls.nnx_model = nnx.eval_shape(
            lambda: MiMoV2Flash(cfg, rngs=nnx.Rngs(params=0))
        )
        cls.nnx_model.embedder = nnx.Embed(
            num_embeddings=cls.vocab_size,
            features=cls.emb_dim,
            dtype=jnp.float32,
            rngs=nnx.Rngs(params=0),
        )
        cls.nnx_model.embedder.embedding[...] = jnp.asarray(cls.weight_np_f32)

        # Build Torch model on meta device, then attach real embedding weights only.
        torch_cfg = MiMoV2FlashConfig(
            **{
                **cfg_json,
                "num_hidden_layers": 0,
                "hybrid_layer_pattern": [],
                "moe_layer_freq": [],
                "n_routed_experts": None,
                "num_experts_per_tok": None,
                "moe_intermediate_size": None,
                "add_full_attention_sink_bias": False,
                "add_swa_attention_sink_bias": False,
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

        weight_torch = torch.from_numpy(cls.weight_np_f32)
        cls.torch_model.model.embed_tokens = torch.nn.Embedding(
            cls.vocab_size, cls.emb_dim, _weight=weight_torch
        )

    def setUp(self):
        super().setUp()
        _load_package()

    def test_embedder(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch not installed")

        if self.nnx_model is None or self.torch_model is None:
            self.skipTest("weights or dependencies not available")

        nm = self.nnx_model.embedder
        tm = self.torch_model.model.embed_tokens

        tx = torch.randint(
            0, self.vocab_size, size=(self.batch_size, self.num_input_tokens)
        )
        jx = jnp.array(tx.cpu().detach().numpy())

        jy = nm.embedding[...].at[(jx,)].get()
        ty = tm(tx)

        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )


if __name__ == "__main__":
    absltest.main()
