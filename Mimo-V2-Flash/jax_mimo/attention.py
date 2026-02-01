from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import MiMoV2FlashConfig
from .rope import build_rope_cache, apply_rotary_pos_emb


def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    if n_rep == 1:
        return x
    return jnp.repeat(x, repeats=n_rep, axis=1)


class MiMoV2Attention(nn.Module):
    config: MiMoV2FlashConfig
    is_swa: bool

    def setup(self) -> None:
        if self.is_swa:
            self.head_dim = self.config.swa_head_dim
            self.v_head_dim = self.config.swa_v_head_dim
            self.num_attention_heads = self.config.swa_num_attention_heads
            self.num_key_value_heads = self.config.swa_num_key_value_heads
            self.rope_theta = self.config.swa_rope_theta
        else:
            self.head_dim = self.config.head_dim
            self.v_head_dim = self.config.v_head_dim
            self.num_attention_heads = self.config.num_attention_heads
            self.num_key_value_heads = self.config.num_key_value_heads
            self.rope_theta = self.config.rope_theta

        self.rope_dim = int(self.head_dim * self.config.partial_rotary_factor)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        q_hidden_size = self.num_attention_heads * self.head_dim
        k_hidden_size = self.num_key_value_heads * self.head_dim
        v_hidden_size = self.num_key_value_heads * self.v_head_dim
        o_hidden_size = self.num_attention_heads * self.v_head_dim

        self.q_proj = nn.Dense(q_hidden_size, use_bias=self.config.attention_bias, name="q_proj")
        self.k_proj = nn.Dense(k_hidden_size, use_bias=self.config.attention_bias, name="k_proj")
        self.v_proj = nn.Dense(v_hidden_size, use_bias=self.config.attention_bias, name="v_proj")
        self.o_proj = nn.Dense(o_hidden_size, use_bias=False, name="o_proj")
        self.dropout = nn.Dropout(self.config.attention_dropout)

    def _apply_rope(
        self, q: jnp.ndarray, k: jnp.ndarray, position_ids: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        seq_len = q.shape[-2]
        cos, sin = build_rope_cache(seq_len, self.rope_dim, theta=self.rope_theta, dtype=q.dtype)

        q_rope, q_nope = jnp.split(q, [self.rope_dim], axis=-1)
        k_rope, k_nope = jnp.split(k, [self.rope_dim], axis=-1)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        q = jnp.concatenate([q_rope, q_nope], axis=-1)
        k = jnp.concatenate([k_rope, k_nope], axis=-1)
        return q, k

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        position_ids: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, self.num_key_value_heads, self.v_head_dim).transpose(0, 2, 1, 3)

        q, k = self._apply_rope(q, k, position_ids)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights - jnp.max(attn_weights, axis=-1, keepdims=True)
        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout(attn_probs, deterministic=deterministic)

        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output)
