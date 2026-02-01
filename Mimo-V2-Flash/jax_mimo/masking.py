from typing import Optional

import jax.numpy as jnp


def _causal_mask(seq_len: int) -> jnp.ndarray:
    idx = jnp.arange(seq_len)
    return idx[None, :] <= idx[:, None]


def _sliding_window_mask(seq_len: int, window: int) -> jnp.ndarray:
    idx = jnp.arange(seq_len)
    q = idx[:, None]
    k = idx[None, :]
    return (k <= q) & ((q - k) < window)


def make_attention_mask(
    attention_mask: Optional[jnp.ndarray],
    seq_len: int,
    sliding_window: Optional[int] = None,
    dtype=jnp.float32,
) -> jnp.ndarray:
    if sliding_window is None:
        allow = _causal_mask(seq_len)
    else:
        allow = _sliding_window_mask(seq_len, sliding_window)

    if attention_mask is not None:
        key_mask = attention_mask[:, None, None, :].astype(bool)
        allow = allow[None, None, :, :] & key_mask
    else:
        allow = allow[None, None, :, :]

    zero = jnp.array(0.0, dtype=dtype)
    neg_inf = jnp.array(-1.0e30, dtype=dtype)
    return jnp.where(allow, zero, neg_inf)
