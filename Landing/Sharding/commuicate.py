import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
from jax import shard_map, lax
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ("i",))

global_data = jnp.arange(16).reshape(4, 4)


@shard_map(mesh=mesh, in_specs=P("i", None), out_specs=P("i", None))
def run_all_to_all(x):
    return lax.all_to_all(x, axis_name="i", split_axis=1, concat_axis=1, tiled=True)


print("--- 输入数据 (Global) ---")
print(global_data)

output = run_all_to_all(global_data)

print("\n--- 输出数据 (Global) ---")
print(output)
