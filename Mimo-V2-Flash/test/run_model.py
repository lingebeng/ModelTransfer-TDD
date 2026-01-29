import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_LOG_LEVEL", "ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
from flax import nnx

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config import ModelConfig
    from modeling import MiMoV2Flash
else:
    from ..config import ModelConfig
    from ..modeling import MiMoV2Flash


def main():
    cfg = ModelConfig.tiny_config()
    model = MiMoV2Flash(cfg, rngs=nnx.Rngs(params=0))

    batch_size = 2
    seq_len = 8
    key = jax.random.key(0)
    tokens = jax.random.randint(
        key, (batch_size, seq_len), 0, cfg.vocab_size, dtype=jnp.int32
    )

    cache = model.init_cache(
        cfg,
        batch_size=batch_size,
        token_len=seq_len,
        generate_steps=0,
        dtype=jnp.float32,
    )
    segment_ids = (tokens != 0).astype(jnp.int32)
    logits = model(tokens, segment_ids, cache, num_right_pads=0)

    print("logits shape:", logits.shape)
    print("logits dtype:", logits.dtype)


if __name__ == "__main__":
    main()
