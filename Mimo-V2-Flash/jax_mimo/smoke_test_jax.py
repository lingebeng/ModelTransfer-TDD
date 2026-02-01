import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp

from jax_mimo.config import MiMoV2FlashConfig
from jax_mimo.model import MiMoV2FlashForCausalLM


def run_smoke_test() -> None:
    config = MiMoV2FlashConfig.tiny_config()
    config.sliding_window = 4
    config.num_hidden_layers = 6
    config.hybrid_layer_pattern = [0, 1, 1, 1, 1, 0]
    config.moe_layer_freq = [True] * config.num_hidden_layers

    model = MiMoV2FlashForCausalLM(config)
    batch_size = 4
    seq_len = 8
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    params = model.init(jax.random.PRNGKey(1), input_ids, attention_mask=attention_mask, logits_to_keep=1)
    logits = model.apply(params, input_ids, attention_mask=attention_mask, logits_to_keep=1, deterministic=True)
    assert logits.shape == (batch_size, 1, config.vocab_size)


if __name__ == "__main__":
    run_smoke_test()
    print("jax tiny_config smoke test passed")
