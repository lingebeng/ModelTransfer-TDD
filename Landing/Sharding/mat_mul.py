import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec as P

devices = jax.devices()

mesh = Mesh(np.array(devices).reshape((2, 2)), ("x", "y"))


@shard_map(
    mesh=mesh,
    in_specs=(P("x", None), P(None, "y")),
    out_specs=P("x", "y"),
)
def verify_case1(x, y):
    """
    [M // 2, K] @ [K, N // 2] -> [M // 2, N // 2]
    :param x: [M, K]
    :param y: [K, N]
    """
    return jnp.matmul(x, y)


@shard_map(
    mesh=mesh,
    in_specs=(P(None, "y"), P(None, None)),
    out_specs=P(None, None),
    check_vma=False,
)
def verify_case2(x, y):
    """
    [M, K // 2] @ [K, N] -> [M, N]
    :param x: [M, K]
    :param y: [K, N]
    """

    x_gather = lax.all_gather(x, axis_name="y", axis=1)  # [M, K // 2] -> [M, 2, K // 2]
    x_full = x_gather.reshape(x.shape[0], -1)  # [M, K]
    return jnp.matmul(x_full, y)


@shard_map(
    mesh=mesh,
    in_specs=(P(None, "y"), P("y", None)),
    out_specs=P(None, None),
)
def verify_case3(x, y):
    """
    [M, K // 2] @ [K // 2, N] -> [M, N]
    :param x: [M, K]
    :param y: [K, N]
    """

    out_partial = jnp.matmul(x, y)  # [M, N]
    out = lax.psum(out_partial, axis_name="y")  # [M, N]
    return out


@shard_map(
    mesh=mesh,
    in_specs=(P("y", None), P(None, "y")),
    out_specs=P("y", None),
)
def verify_case4(x, y):
    """
    [M // 2, K] @ [K, N // 2] -> [M // 2, N // 2]
    :param x: [M, K]
    :param y: [K, N]
    """
    y_gather = lax.all_gather(y, axis_name="y", axis=1)  # [K, N // 2] -> [K, 2, N // 2]
    y_full = y_gather.reshape(y.shape[0], -1)  # [K, N]
    return jnp.matmul(x, y_full)  # [M // 2, N]


@shard_map(
    mesh=mesh,
    in_specs=(P("y", None), P(None, "y")),
    out_specs=P(None, "y"),
)
def verify_case4_other(x, y):
    """
    [M // 2, K] @ [K, N // 2] -> [M // 2, N // 2]
    :param x: [M, K]
    :param y: [K, N]
    """
    x_gather = lax.all_gather(x, axis_name="y", axis=0)  # [M // 2, K] -> [2, M // 2, K]
    x_full = x_gather.reshape(-1, x.shape[1])  # [M, K]
    return jnp.matmul(x_full, y)  # [M, N // 2]


if __name__ == "__main__":
    M, K, N = 8, 16, 8
    x = jnp.arange(M * K).reshape(M, K)
    y = jnp.arange(K * N).reshape(K, N)

    with mesh:
        out1 = verify_case1(x, y)
        out2 = verify_case2(x, y)
        out3 = verify_case3(x, y)
        out4 = verify_case4(x, y)
        out4_other = verify_case4_other(x, y)
        assert jnp.allclose(out1, out2)
        assert jnp.allclose(out1, out3)
        assert jnp.allclose(out1, out4)
        assert jnp.allclose(out1, out4_other)
