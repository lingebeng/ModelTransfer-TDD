import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

# Import as a package so relative imports inside the model code resolve.
from transformer.configuration_mimo_v2_flash import MiMoV2FlashConfig
import transformer.modeling_mimo_v2_flash as mimo_flash
from transformer.modeling_mimo_v2_flash import MiMoV2FlashForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def _select_keys(weight_map: dict[str, str]) -> dict[str, str]:
    keep_prefixes = (
        "model.embed_tokens.",
        "model.norm.",
        "lm_head.",
        "model.layers.0.",
        "model.layers.1.",
    )
    selected = {}
    for key, filename in weight_map.items():
        if key.endswith("weight_scale_inv"):
            continue
        if key.startswith(keep_prefixes):
            selected[key] = filename
    return selected


def load_partial_weights(model: MiMoV2FlashForCausalLM, model_dir: Path) -> None:
    # Prefer using the index to load only needed keys.
    index_path = model_dir / "model.safetensors.index.json"
    state_dict = {}
    if index_path.exists():
        index = json.load(open(index_path, "r", encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        selected = _select_keys(weight_map)
        if not selected:
            raise RuntimeError("No matching keys found in model.safetensors.index.json")
        files: dict[str, list[str]] = {}
        for key, filename in selected.items():
            files.setdefault(filename, []).append(key)
        for filename, keys in files.items():
            path = model_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing shard: {path}")
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for key in keys:
                    state_dict[key] = f.get_tensor(key)
    else:
        # Fallback: load known shards if index is missing.
        shard_names = [
            "model_embedding.safetensors",
            "model_0.safetensors",
            "model_1.safetensors",
            "model_1_linear_fc1.safetensors",
            "model_1_linear_fc2.safetensors",
            "model_final.safetensors",
        ]
        loaded_any = False
        for name in shard_names:
            path = model_dir / name
            if not path.exists():
                continue
            loaded_any = True
            for k, v in load_safetensors(str(path)).items():
                if not k.endswith("weight_scale_inv"):
                    state_dict[k] = v
        if not loaded_any:
            raise FileNotFoundError(
                f"No expected safetensors shards found under {model_dir}"
            )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"warning: missing {len(missing)} keys (example: {missing[:5]})")
    if unexpected:
        print(f"warning: unexpected {len(unexpected)} keys (example: {unexpected[:5]})")


def run_smoke_test(model_dir: Path) -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="`rope_config_validation` is deprecated.*",
    )
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    config = MiMoV2FlashConfig.from_pretrained(model_dir)
    # Some configs ship without a proper model_type/architectures and trigger a warning.
    if not getattr(config, "model_type", None):
        config.model_type = "mimo_v2_flash"
    if not getattr(config, "architectures", None):
        config.architectures = ["MiMoV2FlashForCausalLM"]
    # Limit layers for a faster smoke test.
    config.num_hidden_layers = 2
    if hasattr(config, "hybrid_layer_pattern"):
        config.hybrid_layer_pattern = list(config.hybrid_layer_pattern)[
            : config.num_hidden_layers
        ]
    if hasattr(config, "moe_layer_freq"):
        config.moe_layer_freq = list(config.moe_layer_freq)[: config.num_hidden_layers]
    # Ensure runtime-required config fields are set for masking + attention.
    config.sliding_window = getattr(config, "sliding_window", 8) or 8
    config._attn_implementation = "eager"

    # Add default RoPE initializer for older Transformers builds.
    if "default" not in ROPE_INIT_FUNCTIONS:

        def _compute_default_rope_parameters(
            config, device=None, seq_len=None, layer_type=None
        ):
            head_dim = (
                getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads
            )
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0) or 1.0
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                config.rope_theta
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(
                        device=device, dtype=torch.float
                    )
                    / dim
                )
            )
            attention_scaling = 1.0
            return inv_freq, attention_scaling

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    if not hasattr(
        mimo_flash.MiMoV2FlashRotaryEmbedding, "compute_default_rope_parameters"
    ):
        mimo_flash.MiMoV2FlashRotaryEmbedding.compute_default_rope_parameters = (
            staticmethod(ROPE_INIT_FUNCTIONS["default"])
        )

    # Transformers eager attention path forwards extra kwargs; wrap to ignore them.
    if not getattr(mimo_flash.eager_attention_forward, "_accepts_kwargs", False):
        _orig_eager = mimo_flash.eager_attention_forward

        def _eager_wrapper(*args, **kwargs):
            kwargs.pop("position_ids", None)
            return _orig_eager(*args, **kwargs)

        _eager_wrapper._accepts_kwargs = True
        mimo_flash.eager_attention_forward = _eager_wrapper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiMoV2FlashForCausalLM(config).to(device)
    load_partial_weights(model, model_dir)
    model.eval()

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )
    with torch.no_grad():
        # out = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        outputs = model(input_ids=input_ids, use_cache=False)
    # print(input_ids)
    # print(out)
    logits = outputs.logits
    print(input_ids)
    print(logits)
    # assert logits.shape == (batch_size, seq_len, config.vocab_size), (
    #     f"unexpected logits shape: {logits.shape}"
    # )

    # print("smoke test passed: logits shape =", tuple(logits.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("Mimo-V2-Flash"),
        help="Path to the downloaded HF model weights folder.",
    )
    args = parser.parse_args()
    run_smoke_test(args.model_dir)
