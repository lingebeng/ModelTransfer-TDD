"""Tiny-config smoke test for MiMo-V2-Flash.

Runs a minimal forward pass to validate wiring and tensor shapes.
"""

import os
import sys

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from pytorch.configuration_mimo_v2_flash import MiMoV2FlashConfig
from pytorch import modeling_mimo_v2_flash as modeling
from pytorch.modeling_mimo_v2_flash import MiMoV2FlashForCausalLM


_orig_eager_attention_forward = modeling.eager_attention_forward
_DEBUG_FLOW = os.getenv("MIMO_V2_FLASH_DEBUG", "").lower() in {"1", "true", "yes"}
_DEBUG_VALUES = os.getenv("MIMO_V2_FLASH_DEBUG_VALUES", "").lower() in {
    "1",
    "true",
    "yes",
}


def _debug(msg: str) -> None:
    if _DEBUG_FLOW:
        print(msg)


def _debug_tensor(name: str, tensor: torch.Tensor | None) -> None:
    if not _DEBUG_FLOW:
        return
    if tensor is None:
        print(f"{name}: None")
        return
    print(
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
    )
    t = tensor.detach()
    if t.numel() == 0:
        return
    if t.is_floating_point():
        stats = (float(t.min().item()), float(t.max().item()), float(t.mean().item()))
        print(f"{name}.stats: min={stats[0]:.4g} max={stats[1]:.4g} mean={stats[2]:.4g}")
    else:
        stats = (int(t.min().item()), int(t.max().item()))
        print(f"{name}.stats: min={stats[0]} max={stats[1]}")
    if _DEBUG_VALUES:
        flat = t.flatten()
        n = min(flat.numel(), 8)
        vals = ", ".join([f"{v:.4g}" for v in flat[:n].tolist()])
        print(f"{name}.values[:{n}]: {vals}")


def _eager_attention_forward_compat(*args, **kwargs):
    # The local eager attention helper doesn't accept position_ids in this repo.
    kwargs.pop("position_ids", None)
    return _orig_eager_attention_forward(*args, **kwargs)


def _init_moe_gates(model: torch.nn.Module, init_std: float) -> None:
    # Initialize MoE gate weights for deterministic smoke tests without touching model code.
    for module in model.modules():
        if isinstance(module, modeling.MiMoV2MoEGate):
            with torch.no_grad():
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if hasattr(module, "e_score_correction_bias"):
                    module.e_score_correction_bias.zero_()


def _enable_debug_tracing(model: MiMoV2FlashForCausalLM) -> None:
    if not _DEBUG_FLOW:
        return

    def wrap_method(module, wrapper):
        orig = module.forward

        def wrapped(*args, **kwargs):
            return wrapper(orig, module, *args, **kwargs)

        module.forward = wrapped

    def get_arg(args, kwargs, name, index=0, default=None):
        if name in kwargs:
            return kwargs[name]
        if len(args) > index:
            return args[index]
        return default

    def debug_causal_lm_forward(orig, module, *args, **kwargs):
        _debug("=== MiMoV2FlashForCausalLM.forward ===")
        _debug_tensor("input_ids", kwargs.get("input_ids"))
        _debug_tensor("attention_mask", kwargs.get("attention_mask"))
        _debug_tensor("position_ids", kwargs.get("position_ids"))
        _debug(f"logits_to_keep: {kwargs.get('logits_to_keep')}")
        outputs = orig(*args, **kwargs)
        _debug_tensor("logits", outputs.logits)
        return outputs

    def debug_model_forward(orig, module, *args, **kwargs):
        _debug("=== MiMoV2Model.forward ===")
        input_ids = get_arg(args, kwargs, "input_ids", index=0)
        attention_mask = get_arg(args, kwargs, "attention_mask", index=1)
        position_ids = get_arg(args, kwargs, "position_ids", index=2)
        past_key_values = get_arg(args, kwargs, "past_key_values", index=3)
        inputs_embeds = get_arg(args, kwargs, "inputs_embeds", index=4)
        use_cache = get_arg(args, kwargs, "use_cache", index=5)
        cache_position = get_arg(args, kwargs, "cache_position", index=6)

        _debug_tensor("input_ids", input_ids)
        _debug_tensor("inputs_embeds", inputs_embeds)
        _debug_tensor("attention_mask", attention_mask)
        _debug_tensor("position_ids", position_ids)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = module.embed_tokens(input_ids)
        _debug_tensor("inputs_embeds(after embed)", inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = modeling.DynamicCache(config=module.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        _debug_tensor("cache_position", cache_position)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        _debug_tensor("position_ids(final)", position_ids)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": module.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": modeling.create_causal_mask(**mask_kwargs),
            }
            if module.has_sliding_layers:
                causal_mask_mapping["sliding_window_attention"] = (
                    modeling.create_sliding_window_causal_mask(**mask_kwargs)
                )
        if isinstance(causal_mask_mapping, dict):
            _debug(f"causal_mask_mapping keys: {list(causal_mask_mapping.keys())}")
            for key, mask in causal_mask_mapping.items():
                _debug_tensor(f"mask[{key}]", mask)

        hidden_states = inputs_embeds
        position_embeddings = module.rotary_emb(hidden_states, position_ids)
        swa_position_embeddings = module.swa_rotary_emb(hidden_states, position_ids)
        _debug_tensor("position_embeddings.cos", position_embeddings[0])
        _debug_tensor("position_embeddings.sin", position_embeddings[1])
        _debug_tensor("swa_position_embeddings.cos", swa_position_embeddings[0])
        _debug_tensor("swa_position_embeddings.sin", swa_position_embeddings[1])

        extra_kwargs = dict(kwargs)
        for key in (
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "use_cache",
            "cache_position",
        ):
            extra_kwargs.pop(key, None)

        for layer in module.layers[: module.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer.attention_type],
                position_embeddings=(
                    position_embeddings
                    if layer.attention_type == "full_attention"
                    else swa_position_embeddings
                ),
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **extra_kwargs,
            )

        hidden_states = module.norm(hidden_states)
        _debug_tensor("hidden_states(after norm)", hidden_states)
        return modeling.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def debug_decoder_forward(orig, module, *args, **kwargs):
        hidden_states = get_arg(args, kwargs, "hidden_states", index=0)
        attention_mask = get_arg(args, kwargs, "attention_mask", index=1)
        position_ids = get_arg(args, kwargs, "position_ids", index=2)
        past_key_values = get_arg(args, kwargs, "past_key_values", index=3)
        use_cache = get_arg(args, kwargs, "use_cache", index=4, default=False)
        cache_position = get_arg(args, kwargs, "cache_position", index=5)
        position_embeddings = get_arg(args, kwargs, "position_embeddings", index=6)

        layer_idx = getattr(module, "_debug_idx", "?")
        _debug(f"MiMoV2DecoderLayer.forward layer_idx={layer_idx}")
        _debug_tensor("layer.hidden_states(in)", hidden_states)

        residual = hidden_states
        hidden_states = module.input_layernorm(hidden_states)
        _debug_tensor("layer.hidden_states(after input_layernorm)", hidden_states)

        extra_kwargs = dict(kwargs)
        for key in (
            "hidden_states",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "use_cache",
            "cache_position",
            "position_embeddings",
        ):
            extra_kwargs.pop(key, None)

        hidden_states, _ = module.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **extra_kwargs,
        )
        _debug_tensor("layer.attn_output", hidden_states)
        hidden_states = residual + hidden_states
        _debug_tensor("layer.hidden_states(after attn residual)", hidden_states)

        residual = hidden_states
        hidden_states = module.post_attention_layernorm(hidden_states)
        _debug_tensor("layer.hidden_states(after post_attention_layernorm)", hidden_states)
        hidden_states = module.mlp(hidden_states)
        _debug_tensor("layer.mlp_or_moe_output", hidden_states)
        hidden_states = residual + hidden_states
        _debug_tensor("layer.hidden_states(after mlp residual)", hidden_states)
        return hidden_states

    def debug_attention_forward(orig, module, *args, **kwargs):
        hidden_states = get_arg(args, kwargs, "hidden_states", index=0)
        attention_mask = get_arg(args, kwargs, "attention_mask", index=2)
        _debug(f"MiMoV2Attention.forward layer_idx={getattr(module, 'layer_idx', '?')}")
        _debug_tensor("attn.hidden_states", hidden_states)
        _debug_tensor("attn.attention_mask", attention_mask)

        input_shape = hidden_states.shape[:-1]
        qk_hidden_shape = (*input_shape, -1, module.head_dim)
        v_hidden_shape = (*input_shape, -1, module.v_head_dim)
        query_states = module.q_proj(hidden_states).view(qk_hidden_shape).transpose(1, 2)
        key_states = module.k_proj(hidden_states).view(qk_hidden_shape).transpose(1, 2)
        value_states = module.v_proj(hidden_states).view(v_hidden_shape).transpose(1, 2)
        _debug_tensor("attn.query_states", query_states)
        _debug_tensor("attn.key_states", key_states)
        _debug_tensor("attn.value_states", value_states)

        attn_output, attn_weights = orig(*args, **kwargs)
        _debug_tensor("attn.attn_output", attn_output)
        return attn_output, attn_weights

    def debug_moe_forward(orig, module, *args, **kwargs):
        hidden_states = get_arg(args, kwargs, "hidden_states", index=0)
        _debug("MiMoV2MoE.forward")
        _debug_tensor("moe.hidden_states(in)", hidden_states)
        out = orig(*args, **kwargs)
        _debug_tensor("moe.hidden_states(out)", out)
        return out

    def debug_gate_forward(orig, module, *args, **kwargs):
        hidden_states = get_arg(args, kwargs, "hidden_states", index=0)
        _debug("MiMoV2MoEGate.forward")
        _debug_tensor("moe.gate.hidden_states(in)", hidden_states)
        topk_idx, topk_weight = orig(*args, **kwargs)
        _debug_tensor("moe.gate.topk_idx", topk_idx)
        _debug_tensor("moe.gate.topk_weight", topk_weight)
        return topk_idx, topk_weight

    wrap_method(model, debug_causal_lm_forward)
    wrap_method(model.model, debug_model_forward)

    for idx, layer in enumerate(model.model.layers):
        layer._debug_idx = idx
        wrap_method(layer, debug_decoder_forward)
        wrap_method(layer.self_attn, debug_attention_forward)
        if isinstance(layer.mlp, modeling.MiMoV2MoE):
            wrap_method(layer.mlp, debug_moe_forward)
            wrap_method(layer.mlp.gate, debug_gate_forward)


def run_smoke_test() -> None:
    torch.manual_seed(0)

    config = MiMoV2FlashConfig.tiny_config()
    # Ensure rope_type is compatible with the local transformers version.
    config.rope_scaling = {"rope_type": "linear", "factor": 1.0}
    config.standardize_rope_params()
    config.validate_rope()
    # Required by sliding-window attention mask creation.
    config.sliding_window = 4

    # Patch eager attention to ignore unsupported kwargs.
    modeling.eager_attention_forward = _eager_attention_forward_compat

    # Override tiny_config for a 6-layer GA/SWA pattern.
    config.num_hidden_layers = 6
    config.hybrid_layer_pattern = [0, 1, 1, 1, 1, 0]  # GA, SWA, SWA, SWA, SWA, GA
    config.moe_layer_freq = [True] * config.num_hidden_layers

    model = MiMoV2FlashForCausalLM(config)
    model.eval()
    _init_moe_gates(model, init_std=config.initializer_range)
    _enable_debug_tracing(model)

    batch_size = 4
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )

    logits = outputs.logits
    assert logits.shape == (batch_size, 1, config.vocab_size)


def test_tiny_config_smoke() -> None:
    run_smoke_test()


if __name__ == "__main__":
    run_smoke_test()
    print("tiny_config smoke test passed")
