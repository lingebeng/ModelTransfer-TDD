from typing import Tuple

import jax.numpy as jnp


def build_rope_cache(
    seq_len: int, dim: int, theta: float = 10000.0, dtype=jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    positions = jnp.arange(seq_len, dtype=dtype)
    freqs = jnp.einsum("i,j->ij", positions, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb)[None, None, :, :]
    sin = jnp.sin(emb)[None, None, :, :]
    return cos, sin


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
