from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import MiMoV2FlashConfig
from .layers import MLP


class MiMoV2MoEGate(nn.Module):
    config: MiMoV2FlashConfig

    def setup(self) -> None:
        self.top_k = self.config.num_experts_per_tok
        self.n_routed_experts = self.config.n_routed_experts
        self.routed_scaling_factor = (
            self.config.routed_scaling_factor
            if self.config.routed_scaling_factor is not None
            else 1.0
        )
        self.scoring_func = self.config.scoring_func
        self.topk_method = self.config.topk_method
        self.n_group = self.config.n_group
        self.topk_group = self.config.topk_group
        self.norm_topk_prob = self.config.norm_topk_prob
        self.gating_dim = self.config.hidden_size
        self.weight = self.param(
            "weight", nn.initializers.normal(self.config.initializer_range), (self.n_routed_experts, self.gating_dim)
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = self.param(
                "e_score_correction_bias", nn.initializers.zeros, (self.n_routed_experts,)
            )

    def __call__(self, hidden_states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        bsz, seq_len, h = hidden_states.shape
        x = hidden_states.reshape(-1, h)
        logits = jnp.dot(x.astype(jnp.float32), self.weight.T.astype(jnp.float32))
        if self.scoring_func == "sigmoid":
            scores = jax.nn.sigmoid(logits)
        else:
            raise ValueError(f"unsupported scoring function: {self.scoring_func}")

        if self.topk_method == "noaux_tc":
            scores_for_choice = scores + self.e_score_correction_bias[None, :]
            group_scores = scores_for_choice.reshape(bsz * seq_len, self.n_group, -1)
            group_top2 = jax.lax.top_k(group_scores, k=2)[0]
            group_scores = group_top2.sum(axis=-1)
            group_idx = jax.lax.top_k(group_scores, k=self.topk_group)[1]

            group_mask = jnp.zeros_like(group_scores)
            group_mask = group_mask.at[jnp.arange(bsz * seq_len)[:, None], group_idx].set(1.0)
            score_mask = jnp.repeat(group_mask[:, :, None], self.n_routed_experts // self.n_group, axis=-1)
            score_mask = score_mask.reshape(bsz * seq_len, -1)
            tmp_scores = jnp.where(score_mask > 0, scores_for_choice, -jnp.inf)
            _, topk_idx = jax.lax.top_k(tmp_scores, k=self.top_k)
            topk_weight = jnp.take_along_axis(scores, topk_idx, axis=-1)
        else:
            raise ValueError(f"unsupported TopK method: {self.topk_method}")

        if self.top_k > 1 and self.norm_topk_prob:
            denom = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
            topk_weight = topk_weight / denom
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight


class MiMoV2MoE(nn.Module):
    config: MiMoV2FlashConfig

    def setup(self) -> None:
        self.experts = [
            MLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.moe_intermediate_size,
                hidden_act=self.config.hidden_act,
                name=f"expert_{i}",
            )
            for i in range(self.config.n_routed_experts)
        ]
        self.gate = MiMoV2MoEGate(self.config)

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        final = jnp.zeros_like(x)

        for expert_idx, expert in enumerate(self.experts):
            mask = jnp.any(topk_idx == expert_idx, axis=-1)
            weights = jnp.where(topk_idx == expert_idx, topk_weight, 0.0).sum(axis=-1)
            expert_out = expert(x)
            weighted = expert_out * weights[:, None]
            final = final + jnp.where(mask[:, None], weighted, 0.0)

        return final.reshape(orig_shape)
