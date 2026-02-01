from typing import Callable

import flax.linen as nn
import jax.numpy as jnp


def get_act_fn(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if name == "silu":
        return nn.silu
    raise ValueError(f"unsupported activation: {name}")


class RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param("weight", nn.initializers.ones, (self.hidden_size,))
        x_float = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
        x_norm = x_float * jnp.reciprocal(jnp.sqrt(variance + self.eps))
        return (x_norm * weight).astype(x.dtype)


class MLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    hidden_act: str

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate_proj = nn.Dense(self.intermediate_size, use_bias=False, name="gate_proj")
        up_proj = nn.Dense(self.intermediate_size, use_bias=False, name="up_proj")
        down_proj = nn.Dense(self.hidden_size, use_bias=False, name="down_proj")
        act_fn = get_act_fn(self.hidden_act)
        return down_proj(act_fn(gate_proj(x)) * up_proj(x))
