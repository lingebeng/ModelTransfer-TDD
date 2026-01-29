import importlib.util
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_LOG_LEVEL", "ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
import torch


MODEL_ID = "XiaomiMiMo/MiMo-V2-Flash"


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


def _resolve_weight_dir():
    env_dir = os.environ.get("MIMO_WEIGHT_DIR")
    if env_dir:
        return Path(env_dir)
    if os.environ.get("ALLOW_HF_DOWNLOAD") != "1":
        return None
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return None
    return Path(
        snapshot_download(
            MODEL_ID,
            allow_patterns=[
                "*.safetensors",
                "*.safetensors.index.json",
                "config.json",
                "tokenizer*",
                "*.model",
                "*.txt",
            ],
        )
    )


class TestTrueWeight(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _load_package()
        cls.weight_dir = _resolve_weight_dir()
        cls.torch_model = None
        cls.jax_model = None
        cls.jax_cfg = None
        cls.torch_cfg = None

        if cls.weight_dir is None:
            return

        from mimo_v2_flash.config import ModelConfig
        from mimo_v2_flash import params
        from mimo_v2_flash.pytorch.configuration_mimo_v2_flash import MiMoV2FlashConfig
        from mimo_v2_flash.pytorch.modeling_mimo_v2_flash import MiMoV2FlashForCausalLM

        # Ensure eager_attention_forward accepts kwargs (e.g., position_ids).
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
            cls.torch_cfg = MiMoV2FlashConfig.from_pretrained(cls.weight_dir)

        if getattr(cls.torch_cfg, "rope_scaling", None) is None:
            cls.torch_cfg.rope_scaling = {"rope_type": "linear", "factor": 1.0}
        cls.torch_cfg._attn_implementation = "eager"
        if getattr(cls.torch_cfg, "sliding_window", None) is None:
            cls.torch_cfg.sliding_window = getattr(
                cls.torch_cfg, "sliding_window_size", None
            )
        if hasattr(cls.torch_cfg, "sliding_window_size"):
            cls.torch_cfg.sliding_window_size = cls.torch_cfg.sliding_window

        cls.torch_model = MiMoV2FlashForCausalLM.from_pretrained(
            cls.weight_dir,
            config=cls.torch_cfg,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
        ).eval()

        cls.jax_cfg = ModelConfig(
            num_layers=cls.torch_cfg.num_hidden_layers,
            vocab_size=cls.torch_cfg.vocab_size,
            emb_dim=cls.torch_cfg.hidden_size,
            mlp_dim=cls.torch_cfg.intermediate_size,
            num_heads=cls.torch_cfg.num_attention_heads,
            head_dim=cls.torch_cfg.head_dim,
            num_kv_heads=cls.torch_cfg.num_key_value_heads,
            v_head_dim=getattr(cls.torch_cfg, "v_head_dim", cls.torch_cfg.head_dim),
            rope_theta=cls.torch_cfg.rope_theta,
            max_position_embeddings=cls.torch_cfg.max_position_embeddings,
            norm_eps=cls.torch_cfg.layernorm_epsilon,
            tie_word_embeddings=cls.torch_cfg.tie_word_embeddings,
            attention_bias=getattr(cls.torch_cfg, "attention_bias", False),
            attention_dropout=cls.torch_cfg.attention_dropout,
            partial_rotary_factor=cls.torch_cfg.partial_rotary_factor,
            hybrid_block_size=getattr(cls.torch_cfg, "hybrid_block_size", None),
            hybrid_layer_pattern=getattr(cls.torch_cfg, "hybrid_layer_pattern", None),
            sliding_window=getattr(cls.torch_cfg, "sliding_window", None),
            swa_num_heads=getattr(cls.torch_cfg, "swa_num_attention_heads", None),
            swa_num_kv_heads=getattr(cls.torch_cfg, "swa_num_key_value_heads", None),
            swa_head_dim=getattr(cls.torch_cfg, "swa_head_dim", None),
            swa_v_head_dim=getattr(cls.torch_cfg, "swa_v_head_dim", None),
            swa_rope_theta=getattr(cls.torch_cfg, "swa_rope_theta", None),
            add_full_attention_sink_bias=getattr(
                cls.torch_cfg, "add_full_attention_sink_bias", False
            ),
            add_swa_attention_sink_bias=getattr(
                cls.torch_cfg, "add_swa_attention_sink_bias", False
            ),
            n_routed_experts=getattr(cls.torch_cfg, "n_routed_experts", None),
            num_experts_per_tok=getattr(cls.torch_cfg, "num_experts_per_tok", None),
            moe_intermediate_size=getattr(cls.torch_cfg, "moe_intermediate_size", None),
            moe_layer_freq=getattr(cls.torch_cfg, "moe_layer_freq", None),
            routed_scaling_factor=getattr(cls.torch_cfg, "routed_scaling_factor", None),
            scoring_func=getattr(cls.torch_cfg, "scoring_func", "sigmoid"),
            topk_method=getattr(cls.torch_cfg, "topk_method", "noaux_tc"),
            n_group=getattr(cls.torch_cfg, "n_group", None),
            topk_group=getattr(cls.torch_cfg, "topk_group", None),
            norm_topk_prob=getattr(cls.torch_cfg, "norm_topk_prob", True),
            rope_scaling=getattr(cls.torch_cfg, "rope_scaling", None),
        )

        try:
            cls.jax_model = params.create_model_from_safe_tensors(
                str(cls.weight_dir), cls.jax_cfg, mesh=None
            )
        except AttributeError as exc:
            if "float8" in str(exc):
                cls.jax_model = None
            else:
                raise

    def setUp(self):
        super().setUp()
        _load_package()

    def test_true_weight_logits(self):
        if os.environ.get("RUN_TRUE_WEIGHT_TEST") != "1":
            self.skipTest("Set RUN_TRUE_WEIGHT_TEST=1 to run this test")

        if self.weight_dir is None:
            self.skipTest(
                "Weights not available; set MIMO_WEIGHT_DIR or ALLOW_HF_DOWNLOAD=1"
            )

        if self.jax_model is None:
            self.skipTest(
                "JAX weight load failed (possible float8); update loader to dequant"
            )

        batch_size = int(os.environ.get("MIMO_TEST_BATCH", "1"))
        seq_len = int(os.environ.get("MIMO_TEST_SEQ", "4"))
        torch.manual_seed(0)

        input_ids = torch.randint(
            1, self.torch_cfg.vocab_size, size=(batch_size, seq_len)
        )
        with torch.no_grad():
            t_logits = self.torch_model(
                input_ids=input_ids,
                use_cache=False,
            ).logits

        jx = jnp.asarray(input_ids.cpu().numpy())
        cache = self.jax_model.init_cache(
            self.jax_cfg,
            batch_size=batch_size,
            token_len=seq_len,
            generate_steps=0,
            dtype=jnp.float32,
        )
        segment_ids = (jx != 0).astype(jnp.int32)
        j_logits = self.jax_model(jx, segment_ids, cache, num_right_pads=0)

        j_logits_np = np.array(j_logits, dtype=np.float32)
        t_logits_np = t_logits.float().cpu().numpy()
        self.assertFalse(np.isnan(j_logits_np).any(), "JAX logits contain NaNs")
        self.assertFalse(np.isnan(t_logits_np).any(), "Torch logits contain NaNs")

        torch.testing.assert_close(
            torch.tensor(j_logits_np),
            torch.tensor(t_logits_np),
            rtol=1e-2,
            atol=1e-2,
            check_dtype=False,
        )


if __name__ == "__main__":
    absltest.main()
