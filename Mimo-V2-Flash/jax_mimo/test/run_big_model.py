import argparse
import dataclasses
import json
import os
import sys

import jax
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

    if "hybrid_layer_pattern" in cfg_kwargs and cfg_kwargs["hybrid_layer_pattern"] is not None:
        cfg_kwargs["hybrid_layer_pattern"] = [
            int(x) for x in cfg_kwargs["hybrid_layer_pattern"]
        ]

    if cfg_kwargs.get("routed_scaling_factor") is None:
        cfg_kwargs["routed_scaling_factor"] = 1.0

    return modeling.ModelConfig(**cfg_kwargs)


def run_big_model() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True, help="Directory with sharded safetensors.")
    parser.add_argument(
        "--config-json",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "config.json"
        ),
        help="Path to config.json.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the index mapping, do not load weights.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    args = parser.parse_args()

    if args.validate_only:
        index_path = os.path.join(args.ckpt_dir, "model.safetensors.index.json")
        missing = params.validate_index(index_path)
        print(f"missing keys: {len(missing)}")
        if missing:
            print("\n".join(missing))
        return

    cfg = _load_model_config(args.config_json)
    model = params.create_model_from_safe_tensors(args.ckpt_dir, cfg)

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
