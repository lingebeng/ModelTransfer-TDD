import gc
import re
from enum import Enum

import jax
import safetensors
from etils import epath
from flax import nnx

from .config import ModelConfig
from .modeling import MiMoV2Flash


def _layer_idx_from_jax_key(jax_key: str) -> int:
    parts = jax_key.split(".")
    if len(parts) >= 2 and parts[0] == "layers":
        return int(parts[1])
    raise ValueError(f"Cannot infer layer index from key: {jax_key}")


def _attn_q_transform(jax_key: str, cfg: ModelConfig):
    idx = _layer_idx_from_jax_key(jax_key)
    return (
        (2, 0, 1),
        (cfg.num_heads_for_layer(idx), cfg.head_dim_for_layer(idx), cfg.emb_dim),
        True,
    )


def _attn_k_transform(jax_key: str, cfg: ModelConfig):
    idx = _layer_idx_from_jax_key(jax_key)
    return (
        (2, 0, 1),
        (cfg.num_kv_heads_for_layer(idx), cfg.head_dim_for_layer(idx), cfg.emb_dim),
        True,
    )


def _attn_v_transform(jax_key: str, cfg: ModelConfig):
    idx = _layer_idx_from_jax_key(jax_key)
    return (
        (2, 0, 1),
        (cfg.num_kv_heads_for_layer(idx), cfg.v_head_dim_for_layer(idx), cfg.emb_dim),
        True,
    )


def _attn_out_transform(jax_key: str, cfg: ModelConfig):
    idx = _layer_idx_from_jax_key(jax_key)
    return (
        (1, 0),
        (cfg.num_heads_for_layer(idx), cfg.v_head_dim_for_layer(idx), cfg.emb_dim),
        False,
    )


def _attn_q_bias_transform(jax_key: str, cfg: ModelConfig):
    idx = _layer_idx_from_jax_key(jax_key)
    return (None, (cfg.num_heads_for_layer(idx), cfg.head_dim_for_layer(idx)), True)


def _attn_k_bias_transform(jax_key: str, cfg: ModelConfig):
    idx = _layer_idx_from_jax_key(jax_key)
    return (None, (cfg.num_kv_heads_for_layer(idx), cfg.head_dim_for_layer(idx)), True)


def _attn_v_bias_transform(jax_key: str, cfg: ModelConfig):
    idx = _layer_idx_from_jax_key(jax_key)
    return (
        None,
        (cfg.num_kv_heads_for_layer(idx), cfg.v_head_dim_for_layer(idx)),
        True,
    )


class Transform(Enum):
    """Transformations for model parameters."""

    BIAS = None
    LINEAR = ((1, 0), None, False)
    EMBED = None
    SCALE = None
    ATTN_Q = _attn_q_transform
    ATTN_K = _attn_k_transform
    ATTN_V = _attn_v_transform
    ATTN_OUT = _attn_out_transform
    ATTN_Q_BIAS = _attn_q_bias_transform
    ATTN_K_BIAS = _attn_k_bias_transform
    ATTN_V_BIAS = _attn_v_bias_transform
    GATE = None


def _get_key_and_transform_mapping(cfg: ModelConfig):
    return {
        r"model\.embed_tokens\.weight": ("embedder.embedding", Transform.EMBED),
        # attention
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"layers.\1.attn.q_proj.w",
            Transform.ATTN_Q,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
            r"layers.\1.attn.q_proj.b",
            Transform.ATTN_Q_BIAS,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"layers.\1.attn.k_proj.w",
            Transform.ATTN_K,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (
            r"layers.\1.attn.k_proj.b",
            Transform.ATTN_K_BIAS,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"layers.\1.attn.v_proj.w",
            Transform.ATTN_V,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
            r"layers.\1.attn.v_proj.b",
            Transform.ATTN_V_BIAS,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
            r"layers.\1.attn.o_proj.w",
            Transform.ATTN_OUT,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.attention_sink_bias": (
            r"layers.\1.attn.attention_sink_bias",
            Transform.SCALE,
        ),
        # norms
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
            r"layers.\1.input_layernorm.scale",
            Transform.SCALE,
        ),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.scale",
            Transform.SCALE,
        ),
        r"model\.norm\.weight": ("final_norm.scale", Transform.SCALE),
        # mlp (dense)
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
            r"layers.\1.mlp.gate_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
            r"layers.\1.mlp.up_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
            r"layers.\1.mlp.down_proj.kernel",
            Transform.LINEAR,
        ),
        # moe gate
        r"model\.layers\.([0-9]+)\.mlp\.gate\.weight": (
            r"layers.\1.mlp.gate.w",
            Transform.GATE,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.gate\.e_score_correction_bias": (
            r"layers.\1.mlp.gate.e_score_correction_bias",
            Transform.GATE,
        ),
        # moe experts
        r"model\.layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.gate_proj\.weight": (
            r"layers.\1.mlp.experts.\2.gate_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.up_proj\.weight": (
            r"layers.\1.mlp.experts.\2.up_proj.kernel",
            Transform.LINEAR,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.down_proj\.weight": (
            r"layers.\1.mlp.experts.\2.down_proj.kernel",
            Transform.LINEAR,
        ),
        # lm head
        r"lm_head\.weight": ("lm_head.w", Transform.LINEAR),
    }


def _resolve_transform(transform, jax_key, cfg: ModelConfig):
    if transform is None:
        return None
    if callable(transform):
        return transform(jax_key, cfg)
    return transform


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), reshape)
        for pat, (repl, reshape) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) == 0:
        return None, None
    if len(subs) != 1:
        raise ValueError(f"Multiple key matches for {source_key}: {subs}")
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict):
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


def create_model_from_safe_tensors(
    file_dir: str, cfg: ModelConfig, mesh: jax.sharding.Mesh | None = None
) -> MiMoV2Flash:
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    mimo = nnx.eval_shape(lambda: MiMoV2Flash(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(mimo)
    state_dict = nnx.to_pure_dict(abs_state)
    sharding = (
        nnx.to_pure_dict(nnx.get_named_sharding(abs_state, mesh))
        if mesh is not None
        else None
    )

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []
    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    continue
                if transform is not None:
                    transform_val = (
                        transform.value if hasattr(transform, "value") else transform
                    )
                    transform = _resolve_transform(transform_val, jax_key, cfg)
                else:
                    transform = None
                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(
                        keys, tensor, state_dict, torch_key, transform, sharding
                    )
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
        state_dict["lm_head"]["w"] = state_dict["embedder"]["embedding"].T
    gc.collect()
    return nnx.merge(graph_def, state_dict)
