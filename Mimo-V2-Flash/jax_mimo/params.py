import gc
import json
import re
import sys
from enum import Enum
from typing import Dict

import jax
import safetensors
from etils import epath
from flax import nnx

from . import modeling


def _get_key_and_transform_mapping(cfg: modeling.ModelConfig):
    class Transform(Enum):
        BIAS = None
        LINEAR = ((1, 0), None, False)
        EMBED = None
        SCALE = None

    return {
        r"model\.embed_tokens\.weight": ("model.embed_tokens.embedding", Transform.EMBED),
        # attention
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"model.layers.\1.self_attn.q_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"model.layers.\1.self_attn.k_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"model.layers.\1.self_attn.v_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
            r"model.layers.\1.self_attn.o_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
            r"model.layers.\1.self_attn.q_proj.bias",
            Transform.BIAS,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (
            r"model.layers.\1.self_attn.k_proj.bias",
            Transform.BIAS,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
            r"model.layers.\1.self_attn.v_proj.bias",
            Transform.BIAS,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.attention_sink_bias": (
            r"model.layers.\1.self_attn.attention_sink_bias",
            Transform.BIAS,
        ),
        # mlp (dense)
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
            r"model.layers.\1.mlp.gate_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
            r"model.layers.\1.mlp.up_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
            r"model.layers.\1.mlp.down_proj.kernel",
            Transform.LINEAR,
        ),
        # mlp (moe)
        r"model\.layers\.([0-9]+)\.mlp\.gate\.weight": (
            r"model.layers.\1.mlp.gate.weight",
            Transform.BIAS,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.gate\.e_score_correction_bias": (
            r"model.layers.\1.mlp.gate.e_score_correction_bias",
            Transform.BIAS,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.gate_proj\.weight": (
            r"model.layers.\1.mlp.experts.\2.gate_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.up_proj\.weight": (
            r"model.layers.\1.mlp.experts.\2.up_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.down_proj\.weight": (
            r"model.layers.\1.mlp.experts.\2.down_proj.kernel",
            Transform.LINEAR,
        ),
        # norms + lm head
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
            r"model.layers.\1.input_layernorm.weight",
            Transform.SCALE,
        ),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"model.layers.\1.post_attention_layernorm.weight",
            Transform.SCALE,
        ),
        r"model\.norm\.weight": ("model.norm.weight", Transform.SCALE),
        r"lm_head\.weight": ("lm_head.kernel", Transform.LINEAR),
    }


def _should_skip_key(source_key: str) -> bool:
    if source_key.startswith("model.mtp."):
        return True
    return source_key.endswith("weight_scale_inv")


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if not subs:
        return None
    if len(subs) > 1:
        raise ValueError(f"Only one key should be found: {source_key} -> {subs}")
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict=None):
    key, *rest = keys
    if not rest:
        if transform is not None:
            permute, reshape, reshape_first = transform
            if reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
            if permute:
                tensor = tensor.transpose(permute)
            if not reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
        if tensor.shape != state_dict[key].shape:
            raise ValueError(
                f"Shape mismatch for {st_key}: {tensor.shape} vs {state_dict[key].shape}"
            )
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_torch_state_dict(
    torch_state: Dict[str, "torch.Tensor"],
    cfg: modeling.ModelConfig,
) -> nnx.Module:
    import torch

    model = nnx.eval_shape(
        lambda: modeling.MiMoV2FlashForCausalLM(cfg, rngs=nnx.Rngs(params=0, dropout=1))
    )
    graph_def, abs_state = nnx.split(model)
    state_dict = nnx.to_pure_dict(abs_state)

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []
    for torch_key in torch_state.keys():
        if _should_skip_key(torch_key):
            continue
        mapped = _torch_key_to_jax_key(key_mapping, torch_key)
        if mapped is None:
            continue
        jax_key, transform = mapped
        tensor = torch_state[torch_key].detach().cpu().numpy()
        keys = [_stoi(k) for k in jax_key.split(".")]
        try:
            _assign_weights(keys, tensor, state_dict, torch_key, transform.value)
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(
                f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
            )

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(
            f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}"
        )

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["model"]["embed_tokens"][
            "embedding"
        ].T

    gc.collect()
    return nnx.merge(graph_def, state_dict)


def create_model_from_safe_tensors(
    file_dir: str,
    cfg: modeling.ModelConfig,
) -> nnx.Module:
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    model = nnx.eval_shape(
        lambda: modeling.MiMoV2FlashForCausalLM(cfg, rngs=nnx.Rngs(params=0, dropout=1))
    )
    graph_def, abs_state = nnx.split(model)
    state_dict = nnx.to_pure_dict(abs_state)

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []
    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                if _should_skip_key(torch_key):
                    continue
                mapped = _torch_key_to_jax_key(key_mapping, torch_key)
                if mapped is None:
                    continue
                jax_key, transform = mapped
                tensor = sf.get_tensor(torch_key)
                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform.value)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(
            f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}"
        )

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["model"]["embed_tokens"][
            "embedding"
        ].T

    gc.collect()
    return nnx.merge(graph_def, state_dict)


def validate_index(index_path: str, cfg: modeling.ModelConfig | None = None) -> list[str]:
    if cfg is None:
        cfg = modeling.ModelConfig()
    index_path = epath.Path(index_path).expanduser()
    with index_path.open() as f:
        data = json.load(f)
    weight_map = data.get("weight_map", {})
    key_mapping = _get_key_and_transform_mapping(cfg)
    missing = []
    for torch_key in weight_map.keys():
        if _should_skip_key(torch_key):
            continue
        mapped = _torch_key_to_jax_key(key_mapping, torch_key)
        if mapped is None:
            missing.append(torch_key)
    return missing


__all__ = [
    "create_model_from_torch_state_dict",
    "create_model_from_safe_tensors",
    "validate_index",
]


if __name__ == "__main__":
    default_index = (
        epath.Path(__file__).resolve().parent / "config" / "model.safetensors.index.json"
    )
    index_path = epath.Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_index
    missing_keys = validate_index(str(index_path))
    print(f"missing keys: {len(missing_keys)}")
    if missing_keys:
        print("\n".join(missing_keys))
