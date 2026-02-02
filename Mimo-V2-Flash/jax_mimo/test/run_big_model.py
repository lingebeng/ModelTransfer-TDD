import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import jax.numpy as jnp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from jax_mimo import modeling, params


def _load_model_config(path: str) -> modeling.ModelConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    field_names = {f.name for f in dataclasses.fields(modeling.ModelConfig)}
    cfg_kwargs = {k: raw[k] for k in raw.keys() if k in field_names}

    if "sliding_window" not in cfg_kwargs and "sliding_window_size" in raw:
        cfg_kwargs["sliding_window"] = raw["sliding_window_size"]

    if "moe_layer_freq" in cfg_kwargs and cfg_kwargs["moe_layer_freq"] is not None:
        cfg_kwargs["moe_layer_freq"] = [bool(x) for x in cfg_kwargs["moe_layer_freq"]]

    if (
        "hybrid_layer_pattern" in cfg_kwargs
        and cfg_kwargs["hybrid_layer_pattern"] is not None
    ):
        cfg_kwargs["hybrid_layer_pattern"] = [
            int(x) for x in cfg_kwargs["hybrid_layer_pattern"]
        ]

    if cfg_kwargs.get("routed_scaling_factor") is None:
        cfg_kwargs["routed_scaling_factor"] = 1.0

    return modeling.ModelConfig(**cfg_kwargs)


def _resolve_ckpt_dir(args: argparse.Namespace) -> str:
    if args.ckpt_dir:
        return args.ckpt_dir

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - runtime dependency on server
        raise RuntimeError(
            "huggingface_hub is required when --ckpt-dir is not provided. "
            "Install it or pass --ckpt-dir."
        ) from exc

    return snapshot_download(
        repo_id=args.model_id,
        revision=args.hf_revision,
        cache_dir=args.hf_cache_dir,
        token=args.hf_token,
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "config.json",
        ],
    )


def _resolve_config_json(args: argparse.Namespace, ckpt_dir: str) -> str:
    if args.config_json:
        return args.config_json
    hf_config = Path(ckpt_dir) / "config.json"
    if hf_config.exists():
        return str(hf_config)
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "config.json"
    )


def run_big_model() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="Local directory with sharded safetensors. If omitted, download from HF.",
    )
    parser.add_argument(
        "--model-id",
        default="XiaomiMiMo/MiMo-V2-Flash",
        help="Hugging Face model id used when --ckpt-dir is omitted.",
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Optional Hugging Face revision/branch/tag/commit.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to $HF_TOKEN.",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Path to config.json. Default: <ckpt-dir>/config.json if present.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the index mapping, do not load weights.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    args = parser.parse_args()
    ckpt_dir = _resolve_ckpt_dir(args)
    config_json = _resolve_config_json(args, ckpt_dir)
    print(f"Using checkpoint dir: {ckpt_dir}")
    print(f"Using config json: {config_json}")

    if args.validate_only:
        index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
        missing = params.validate_index(index_path)
        print(f"missing keys: {len(missing)}")
        if missing:
            print("\n".join(missing))
        return

    cfg = _load_model_config(config_json)
    model = params.create_model_from_safe_tensors(ckpt_dir, cfg)

    input_ids = jnp.zeros((args.batch_size, args.seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    logits = model(
        input_ids,
        attention_mask=attention_mask,
        logits_to_keep=1,
        deterministic=True,
    )
    print("logits shape:", logits.shape)


if __name__ == "__main__":
    run_big_model()


__all__ = ["run_big_model"]
