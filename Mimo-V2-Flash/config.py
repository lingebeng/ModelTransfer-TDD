# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
from typing import TypeAlias

from jax import numpy as jnp
from jax import P
from jax.sharding import PartitionSpec

ShardingSpec: TypeAlias = PartitionSpec


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingCfg:
    emb_vd: ShardingSpec
    emb_dv: ShardingSpec
    q_weight_ndh: ShardingSpec
    k_weight_kdh: ShardingSpec
    v_weight_kdv: ShardingSpec
    o_weight_nvd: ShardingSpec
    ffw_weight_df: ShardingSpec
    ffw_weight_fd: ShardingSpec
    rms_norm: ShardingSpec
    act_btd: ShardingSpec
    act_btf: ShardingSpec
    act_btnh: ShardingSpec
    act_btnv: ShardingSpec

    @staticmethod
    def no_sharding() -> "ShardingCfg":
        return ShardingCfg(
            emb_vd=P(None, None),
            emb_dv=P(None, None),
            q_weight_ndh=P(None, None, None),
            k_weight_kdh=P(None, None, None),
            v_weight_kdv=P(None, None, None),
            o_weight_nvd=P(None, None, None),
            ffw_weight_df=P(None, None),
            ffw_weight_fd=P(None, None),
            rms_norm=P(None),
            act_btd=P(None, None, None),
            act_btf=P(None, None, None),
            act_btnh=P(None, None, None, None),
            act_btnv=P(None, None, None, None),
        )

    @staticmethod
    def default() -> "ShardingCfg":
        return ShardingCfg(
            emb_vd=P("tp", "fsdp"),
            emb_dv=P("fsdp", "tp"),
            q_weight_ndh=P("tp", "fsdp", None),
            k_weight_kdh=P("tp", "fsdp", None),
            v_weight_kdv=P("tp", "fsdp", None),
            o_weight_nvd=P("tp", None, "fsdp"),
            ffw_weight_df=P("fsdp", "tp"),
            ffw_weight_fd=P("tp", "fsdp"),
            rms_norm=P("tp"),
            act_btd=P("fsdp", None, "tp"),
            act_btf=P("fsdp", None, "tp"),
            act_btnh=P("fsdp", None, "tp", None),
            act_btnv=P("fsdp", None, "tp", None),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ModelConfig:
    # Core model dims (JAX names)
    num_layers: int
    vocab_size: int
    emb_dim: int
    mlp_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int | None
    v_head_dim: int | None
    rope_theta: float
    max_position_embeddings: int
    norm_eps: float
    tie_word_embeddings: bool
    attention_bias: bool = False
    attention_dropout: float = 0.0
    partial_rotary_factor: float = 1.0
    hybrid_block_size: int | None = None
    hybrid_layer_pattern: list[int] | None = None
    sliding_window: int | None = None
    swa_num_heads: int | None = None
    swa_num_kv_heads: int | None = None
    swa_head_dim: int | None = None
    swa_v_head_dim: int | None = None
    swa_rope_theta: float | None = None
    add_full_attention_sink_bias: bool = False
    add_swa_attention_sink_bias: bool = False
    # MoE settings (optional)
    n_routed_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    moe_layer_freq: list[bool] | None = None
    routed_scaling_factor: float | None = None
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    n_group: int | None = None
    topk_group: int | None = None
    norm_topk_prob: bool = True
    # Optional HF-compatible fields (kept for parity; unused by JAX model)
    rope_scaling: dict | None = None
    hidden_act: str = "silu"
    # Sharding
    shd_cfg: ShardingCfg = ShardingCfg.no_sharding()

    def __post_init__(self) -> None:
        num_kv_heads = (
            self.num_kv_heads if self.num_kv_heads is not None else self.num_heads
        )
        v_head_dim = self.v_head_dim if self.v_head_dim is not None else self.head_dim

        swa_num_heads = (
            self.swa_num_heads if self.swa_num_heads is not None else self.num_heads
        )
        swa_num_kv_heads = (
            self.swa_num_kv_heads if self.swa_num_kv_heads is not None else num_kv_heads
        )
        swa_head_dim = (
            self.swa_head_dim if self.swa_head_dim is not None else self.head_dim
        )
        swa_v_head_dim = (
            self.swa_v_head_dim if self.swa_v_head_dim is not None else v_head_dim
        )
        swa_rope_theta = (
            self.swa_rope_theta if self.swa_rope_theta is not None else self.rope_theta
        )

        if self.hybrid_layer_pattern is None:
            if self.hybrid_block_size is not None:
                pattern = [
                    0 if ((i + 1) % self.hybrid_block_size == 0) else 1
                    for i in range(self.num_layers)
                ]
            else:
                pattern = [0 for _ in range(self.num_layers)]
        else:
            pattern = self.hybrid_layer_pattern

        if self.moe_layer_freq is None:
            moe_layer_freq = [False for _ in range(self.num_layers)]
        else:
            moe_layer_freq = self.moe_layer_freq

        moe_intermediate_size = (
            self.moe_intermediate_size
            if self.moe_intermediate_size is not None
            else self.mlp_dim
        )
        num_experts_per_tok = (
            self.num_experts_per_tok if self.num_experts_per_tok is not None else 1
        )
        routed_scaling_factor = (
            self.routed_scaling_factor
            if self.routed_scaling_factor is not None
            else 1.0
        )
        n_group = self.n_group if self.n_group is not None else 1
        topk_group = self.topk_group if self.topk_group is not None else 1

        object.__setattr__(self, "num_kv_heads", num_kv_heads)
        object.__setattr__(self, "v_head_dim", v_head_dim)
        object.__setattr__(self, "swa_num_heads", swa_num_heads)
        object.__setattr__(self, "swa_num_kv_heads", swa_num_kv_heads)
        object.__setattr__(self, "swa_head_dim", swa_head_dim)
        object.__setattr__(self, "swa_v_head_dim", swa_v_head_dim)
        object.__setattr__(self, "swa_rope_theta", swa_rope_theta)
        object.__setattr__(self, "hybrid_layer_pattern", pattern)
        object.__setattr__(self, "moe_layer_freq", moe_layer_freq)
        object.__setattr__(self, "moe_intermediate_size", moe_intermediate_size)
        object.__setattr__(self, "num_experts_per_tok", num_experts_per_tok)
        object.__setattr__(self, "routed_scaling_factor", routed_scaling_factor)
        object.__setattr__(self, "n_group", n_group)
        object.__setattr__(self, "topk_group", topk_group)

    # PyTorch-style aliases for parity/debugging
    @property
    def hidden_size(self) -> int:
        return self.emb_dim

    @property
    def intermediate_size(self) -> int:
        return self.mlp_dim

    @property
    def num_hidden_layers(self) -> int:
        return self.num_layers

    @property
    def num_attention_heads(self) -> int:
        return self.num_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.num_kv_heads

    @property
    def layernorm_epsilon(self) -> float:
        return self.norm_eps

    def is_swa_layer(self, layer_idx: int) -> bool:
        return self.hybrid_layer_pattern[layer_idx] == 1

    def is_moe_layer(self, layer_idx: int) -> bool:
        return bool(self.n_routed_experts) and self.moe_layer_freq[layer_idx]

    def num_heads_for_layer(self, layer_idx: int) -> int:
        return self.swa_num_heads if self.is_swa_layer(layer_idx) else self.num_heads

    def num_kv_heads_for_layer(self, layer_idx: int) -> int:
        return (
            self.swa_num_kv_heads if self.is_swa_layer(layer_idx) else self.num_kv_heads
        )

    def head_dim_for_layer(self, layer_idx: int) -> int:
        return self.swa_head_dim if self.is_swa_layer(layer_idx) else self.head_dim

    def v_head_dim_for_layer(self, layer_idx: int) -> int:
        return self.swa_v_head_dim if self.is_swa_layer(layer_idx) else self.v_head_dim

    def rope_theta_for_layer(self, layer_idx: int) -> float:
        return self.swa_rope_theta if self.is_swa_layer(layer_idx) else self.rope_theta

    def rope_dim_for_layer(self, layer_idx: int) -> int:
        return int(self.head_dim_for_layer(layer_idx) * self.partial_rotary_factor)

    @classmethod
    def from_pytorch_config(
        cls,
        *,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int | None,
        max_position_embeddings: int,
        layernorm_epsilon: float,
        tie_word_embeddings: bool,
        rope_theta: float,
        attention_dropout: float = 0.0,
        partial_rotary_factor: float = 1.0,
        hybrid_block_size: int | None = None,
        hybrid_layer_pattern: list[int] | None = None,
        **kwargs,
    ) -> "ModelConfig":
        head_dim = kwargs.pop("head_dim", hidden_size // num_attention_heads)
        v_head_dim = kwargs.pop("v_head_dim", head_dim)
        return cls(
            num_layers=num_hidden_layers,
            vocab_size=vocab_size,
            emb_dim=hidden_size,
            mlp_dim=intermediate_size,
            num_heads=num_attention_heads,
            head_dim=head_dim,
            num_kv_heads=num_key_value_heads,
            v_head_dim=v_head_dim,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            norm_eps=layernorm_epsilon,
            tie_word_embeddings=tie_word_embeddings,
            attention_dropout=attention_dropout,
            partial_rotary_factor=partial_rotary_factor,
            hybrid_block_size=hybrid_block_size,
            hybrid_layer_pattern=hybrid_layer_pattern,
            **kwargs,
        )

    @classmethod
    def tiny_config(cls) -> "ModelConfig":
        emb_dim = 256
        num_heads = 4
        head_dim = emb_dim // num_heads
        num_layers = 4
        hybrid_block_size = 2  # pattern: [SWA, GA, SWA, GA]

        n_routed_experts = 4
        num_experts_per_tok = 2
        moe_intermediate_size = 512
        moe_layer_freq = [True] * num_layers

        return cls(
            num_layers=num_layers,
            vocab_size=8192,
            emb_dim=emb_dim,
            mlp_dim=768,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=2,
            v_head_dim=head_dim,
            rope_theta=10000.0,
            max_position_embeddings=2048,
            norm_eps=1e-6,
            tie_word_embeddings=False,
            attention_bias=False,
            attention_dropout=0.0,
            partial_rotary_factor=1.0,
            hybrid_block_size=hybrid_block_size,
            sliding_window=8,
            swa_num_heads=num_heads,
            swa_num_kv_heads=2,
            swa_head_dim=head_dim,
            swa_v_head_dim=head_dim,
            swa_rope_theta=10000.0,
            add_full_attention_sink_bias=False,
            add_swa_attention_sink_bias=False,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            moe_layer_freq=moe_layer_freq,
            routed_scaling_factor=1.0,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=2,
            topk_group=1,
            norm_topk_prob=True,
            shd_cfg=ShardingCfg.no_sharding(),
        )


__all__ = ["ModelConfig", "ShardingCfg", "ShardingSpec", "PartitionSpec", "jnp"]
