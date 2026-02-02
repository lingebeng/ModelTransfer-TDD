import jax.numpy as jnp


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
rotated_x = rotate_half(x)
print("Original x:", x)
print("Rotated x:", rotated_x)
