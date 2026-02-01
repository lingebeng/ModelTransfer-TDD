from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import MiMoV2FlashConfig
from .layers import RMSNorm, MLP
from .attention import MiMoV2Attention
from .masking import make_attention_mask
from .moe import MiMoV2MoE


class MiMoV2DecoderLayer(nn.Module):
    config: MiMoV2FlashConfig
    layer_idx: int

    def setup(self) -> None:
        is_swa_layer = self.config.hybrid_layer_pattern[self.layer_idx] == 1
        self.attention_type = "sliding_window_attention" if is_swa_layer else "full_attention"
        self.self_attn = MiMoV2Attention(self.config, is_swa_layer)
        if self.config.n_routed_experts is not None and self.config.moe_layer_freq[self.layer_idx]:
            self.mlp = MiMoV2MoE(self.config)
        else:
            self.mlp = MLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
            )
        self.input_layernorm = RMSNorm(self.config.hidden_size, eps=self.config.layernorm_epsilon)
        self.post_attention_layernorm = RMSNorm(self.config.hidden_size, eps=self.config.layernorm_epsilon)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        position_ids: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MiMoV2Model(nn.Module):
    config: MiMoV2FlashConfig

    def setup(self) -> None:
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size)
        self.layers = [
            MiMoV2DecoderLayer(self.config, layer_idx=i, name=f"layer_{i}")
            for i in range(self.config.num_hidden_layers)
        ]
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.layernorm_epsilon)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (1, seq_len))

        full_mask = make_attention_mask(attention_mask, seq_len, sliding_window=None, dtype=hidden_states.dtype)
        if self.config.sliding_window is None:
            swa_mask = full_mask
        else:
            swa_mask = make_attention_mask(
                attention_mask, seq_len, sliding_window=self.config.sliding_window, dtype=hidden_states.dtype
            )

        for layer in self.layers:
            mask = full_mask if layer.attention_type == "full_attention" else swa_mask
            hidden_states = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=position_ids,
                deterministic=deterministic,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class MiMoV2FlashForCausalLM(nn.Module):
    config: MiMoV2FlashConfig

    def setup(self) -> None:
        self.model = MiMoV2Model(self.config)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        logits_to_keep: int = 0,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        hidden_states = self.model(input_ids, attention_mask=attention_mask, deterministic=deterministic)
        if logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        logits = self.lm_head(hidden_states)
        return logits
