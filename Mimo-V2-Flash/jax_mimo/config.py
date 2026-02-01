from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class MiMoV2FlashConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    layernorm_epsilon: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    attention_dropout: float = 0.0
    hybrid_block_size: Optional[int] = None
    hybrid_layer_pattern: Optional[Sequence[int]] = None
    partial_rotary_factor: float = 1.0
    # Extra attributes used by the model
    head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    swa_head_dim: Optional[int] = None
    swa_v_head_dim: Optional[int] = None
    swa_num_attention_heads: Optional[int] = None
    swa_num_key_value_heads: Optional[int] = None
    swa_rope_theta: float = 10000.0
    attention_bias: bool = False
    add_full_attention_sink_bias: bool = False
    add_swa_attention_sink_bias: bool = False
    n_routed_experts: Optional[int] = None
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 512
    moe_layer_freq: Optional[Sequence[bool]] = None
    routed_scaling_factor: float = 1.0
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    sliding_window: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.v_head_dim is None:
            self.v_head_dim = self.head_dim
        if self.swa_head_dim is None:
            self.swa_head_dim = self.head_dim
        if self.swa_v_head_dim is None:
            self.swa_v_head_dim = self.v_head_dim
        if self.swa_num_attention_heads is None:
            self.swa_num_attention_heads = self.num_attention_heads
        if self.swa_num_key_value_heads is None:
            self.swa_num_key_value_heads = self.num_key_value_heads
        if self.hybrid_block_size is not None and self.hybrid_layer_pattern is None:
            self.hybrid_layer_pattern = [
                0 if ((i + 1) % self.hybrid_block_size == 0) else 1
                for i in range(self.num_hidden_layers)
            ]
        if self.hybrid_layer_pattern is None:
            self.hybrid_layer_pattern = [0] * self.num_hidden_layers
        if self.moe_layer_freq is None:
            self.moe_layer_freq = [False] * self.num_hidden_layers

    @classmethod
    def tiny_config(cls) -> "MiMoV2FlashConfig":
        hidden_size = 256
        num_attention_heads = 4
        head_dim = hidden_size // num_attention_heads
        num_hidden_layers = 4
        hybrid_block_size = 2
        n_routed_experts = 4
        num_experts_per_tok = 2
        moe_intermediate_size = 512
        moe_layer_freq = [True] * num_hidden_layers

        return cls(
            vocab_size=8192,
            hidden_size=hidden_size,
            intermediate_size=768,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            hybrid_block_size=hybrid_block_size,
            rope_theta=10000.0,
            attention_dropout=0.0,
            partial_rotary_factor=1.0,
            head_dim=head_dim,
            v_head_dim=head_dim,
            swa_head_dim=head_dim,
            swa_v_head_dim=head_dim,
            swa_num_attention_heads=num_attention_heads,
            swa_num_key_value_heads=2,
            swa_rope_theta=10000.0,
            attention_bias=False,
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
        )
