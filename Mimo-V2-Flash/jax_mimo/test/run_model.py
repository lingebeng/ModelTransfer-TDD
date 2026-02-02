import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp
from flax import nnx

from jax_mimo import modeling


def run_model() -> None:
    cfg = modeling.ModelConfig.tiny_config()
    cfg.sliding_window = 4
    cfg.num_hidden_layers = 6
    cfg.hybrid_layer_pattern = [0, 1, 1, 1, 1, 0]
    cfg.moe_layer_freq = [True] * cfg.num_hidden_layers

    model = modeling.MiMoV2FlashForCausalLM(cfg, rngs=nnx.Rngs(params=0, dropout=1))

    batch_size = 4
    seq_len = 8
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, cfg.vocab_size)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    logits = model(
        input_ids,
        attention_mask=attention_mask,
        logits_to_keep=1,
        deterministic=True,
    )
    print("logits shape:", logits.shape)

    # KV cache demo (prefill + single-step decode)
    cache = model.init_cache(batch_size=batch_size, max_seq_len=seq_len + 2)
    _ = model(input_ids, attention_mask=attention_mask, cache=cache, deterministic=True)
    next_ids = jax.random.randint(key, (batch_size, 1), 0, cfg.vocab_size)
    next_mask = jnp.ones((batch_size, seq_len + 1), dtype=jnp.int32)
    step_logits = model(next_ids, attention_mask=next_mask, cache=cache, deterministic=True)
    print("kv logits shape:", step_logits.shape)


if __name__ == "__main__":
    run_model()
