import dataclasses
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx


@dataclasses.dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
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
    # Extra attention attributes
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
    # MoE
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
    # Sliding window
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False

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
    def tiny_config(cls) -> "ModelConfig":
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

    @classmethod
    def from_torch_config(cls, torch_cfg) -> "ModelConfig":
        return cls(
            vocab_size=torch_cfg.vocab_size,
            hidden_size=torch_cfg.hidden_size,
            intermediate_size=torch_cfg.intermediate_size,
            num_hidden_layers=torch_cfg.num_hidden_layers,
            num_attention_heads=torch_cfg.num_attention_heads,
            num_key_value_heads=getattr(torch_cfg, "num_key_value_heads", None),
            hidden_act=torch_cfg.hidden_act,
            max_position_embeddings=torch_cfg.max_position_embeddings,
            initializer_range=torch_cfg.initializer_range,
            layernorm_epsilon=torch_cfg.layernorm_epsilon,
            rope_theta=torch_cfg.rope_theta,
            rope_scaling=getattr(torch_cfg, "rope_scaling", None),
            attention_dropout=torch_cfg.attention_dropout,
            hybrid_block_size=getattr(torch_cfg, "hybrid_block_size", None),
            hybrid_layer_pattern=getattr(torch_cfg, "hybrid_layer_pattern", None),
            partial_rotary_factor=getattr(torch_cfg, "partial_rotary_factor", 1.0),
            head_dim=getattr(torch_cfg, "head_dim", None),
            v_head_dim=getattr(torch_cfg, "v_head_dim", None),
            swa_head_dim=getattr(torch_cfg, "swa_head_dim", None),
            swa_v_head_dim=getattr(torch_cfg, "swa_v_head_dim", None),
            swa_num_attention_heads=getattr(torch_cfg, "swa_num_attention_heads", None),
            swa_num_key_value_heads=getattr(torch_cfg, "swa_num_key_value_heads", None),
            swa_rope_theta=getattr(torch_cfg, "swa_rope_theta", 10000.0),
            attention_bias=getattr(torch_cfg, "attention_bias", False),
            add_full_attention_sink_bias=getattr(torch_cfg, "add_full_attention_sink_bias", False),
            add_swa_attention_sink_bias=getattr(torch_cfg, "add_swa_attention_sink_bias", False),
            n_routed_experts=getattr(torch_cfg, "n_routed_experts", None),
            num_experts_per_tok=getattr(torch_cfg, "num_experts_per_tok", 2),
            moe_intermediate_size=getattr(torch_cfg, "moe_intermediate_size", 512),
            moe_layer_freq=getattr(torch_cfg, "moe_layer_freq", None),
            routed_scaling_factor=getattr(torch_cfg, "routed_scaling_factor", 1.0),
            scoring_func=getattr(torch_cfg, "scoring_func", "sigmoid"),
            topk_method=getattr(torch_cfg, "topk_method", "noaux_tc"),
            n_group=getattr(torch_cfg, "n_group", 1),
            topk_group=getattr(torch_cfg, "topk_group", 1),
            norm_topk_prob=getattr(torch_cfg, "norm_topk_prob", True),
            sliding_window=getattr(torch_cfg, "sliding_window", None),
            tie_word_embeddings=getattr(torch_cfg, "tie_word_embeddings", False),
        )


def _get_act_fn(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if name == "silu":
        return jax.nn.silu
    raise ValueError(f"unsupported activation: {name}")


def _causal_mask(seq_len: int) -> jnp.ndarray:
    idx = jnp.arange(seq_len)
    return idx[None, :] <= idx[:, None]


def _sliding_window_mask(seq_len: int, window: int) -> jnp.ndarray:
    idx = jnp.arange(seq_len)
    q = idx[:, None]
    k = idx[None, :]
    return (k <= q) & ((q - k) < window)


def make_attention_mask(
    attention_mask: Optional[jnp.ndarray],
    seq_len: int,
    sliding_window: Optional[int],
    dtype: jnp.dtype,
) -> jnp.ndarray:
    if sliding_window is None:
        allow = _causal_mask(seq_len)
    else:
        allow = _sliding_window_mask(seq_len, sliding_window)

    if attention_mask is not None:
        key_mask = attention_mask[:, None, None, :].astype(bool)
        allow = allow[None, None, :, :] & key_mask
    else:
        allow = allow[None, None, :, :]

    zero = jnp.array(0.0, dtype=dtype)
    neg_inf = jnp.array(jnp.finfo(dtype).min, dtype=dtype)
    return jnp.where(allow, zero, neg_inf)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def build_rope_cache(seq_len: int, dim: int, theta: float, dtype: jnp.dtype):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    positions = jnp.arange(seq_len, dtype=dtype)
    freqs = jnp.einsum("i,j->ij", positions, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb)[None, None, :, :]
    sin = jnp.sin(emb)[None, None, :, :]
    return cos, sin


def apply_rotary_pos_emb(
    q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nnx.Module):
    def __init__(self, hidden_size: int, eps: float, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(nnx.initializers.ones_init()(rngs.params(), (hidden_size,)))
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = x.dtype
        x_float = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
        x_norm = x_float * jnp.reciprocal(jnp.sqrt(variance + self.eps))
        return (x_norm * self.weight[...]).astype(dtype)


class MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        *,
        rngs: nnx.Rngs,
    ):
        self.gate_proj = nnx.Linear(hidden_size, intermediate_size, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(hidden_size, intermediate_size, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(intermediate_size, hidden_size, use_bias=False, rngs=rngs)
        self.act_fn = _get_act_fn(hidden_act)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiMoV2MoEGate(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = (
            config.routed_scaling_factor if config.routed_scaling_factor is not None else 1.0
        )
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nnx.Param(
            nnx.initializers.normal(stddev=config.initializer_range)(
                rngs.params(), (self.n_routed_experts, self.gating_dim)
            )
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nnx.Param(
                nnx.initializers.zeros_init()(rngs.params(), (self.n_routed_experts,))
            )

    def __call__(self, hidden_states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        bsz, seq_len, h = hidden_states.shape
        x = hidden_states.reshape(-1, h)
        logits = jnp.dot(x.astype(jnp.float32), self.weight[...].T.astype(jnp.float32))
        if self.scoring_func == "sigmoid":
            scores = jax.nn.sigmoid(logits)
        else:
            raise ValueError(f"unsupported scoring function: {self.scoring_func}")

        if self.topk_method == "noaux_tc":
            scores_for_choice = scores + self.e_score_correction_bias[None, :]
            grouped = scores_for_choice.reshape(bsz * seq_len, self.n_group, -1)
            group_top2 = jax.lax.top_k(grouped, k=2)[0]
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


class MiMoV2MoE(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.experts = nnx.List(
            [
                MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                    rngs=rngs,
                )
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = MiMoV2MoEGate(config, rngs=rngs)

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


def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    if n_rep == 1:
        return x
    return jnp.repeat(x, repeats=n_rep, axis=1)


class MiMoV2Attention(nnx.Module):
    def __init__(self, config: ModelConfig, is_swa: bool, layer_idx: int, *, rngs: nnx.Rngs):
        self.config = config
        self.is_swa = is_swa
        self.layer_idx = layer_idx

        if is_swa:
            self.head_dim = config.swa_head_dim
            self.v_head_dim = config.swa_v_head_dim
            self.num_attention_heads = config.swa_num_attention_heads
            self.num_key_value_heads = config.swa_num_key_value_heads
            self.rope_theta = config.swa_rope_theta
        else:
            self.head_dim = config.head_dim
            self.v_head_dim = config.v_head_dim
            self.num_attention_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.rope_theta = config.rope_theta

        self.rope_dim = int(self.head_dim * config.partial_rotary_factor)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        q_hidden_size = self.num_attention_heads * self.head_dim
        k_hidden_size = self.num_key_value_heads * self.head_dim
        v_hidden_size = self.num_key_value_heads * self.v_head_dim
        o_hidden_size = self.num_attention_heads * self.v_head_dim

        self.q_proj = nnx.Linear(config.hidden_size, q_hidden_size, use_bias=config.attention_bias, rngs=rngs)
        self.k_proj = nnx.Linear(config.hidden_size, k_hidden_size, use_bias=config.attention_bias, rngs=rngs)
        self.v_proj = nnx.Linear(config.hidden_size, v_hidden_size, use_bias=config.attention_bias, rngs=rngs)
        self.o_proj = nnx.Linear(o_hidden_size, config.hidden_size, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(config.attention_dropout, rngs=rngs)
        self.use_sink = (config.add_swa_attention_sink_bias and is_swa) or (
            config.add_full_attention_sink_bias and not is_swa
        )
        if self.use_sink:
            self.attention_sink_bias = nnx.Param(
                nnx.initializers.zeros_init()(rngs.params(), (self.num_attention_heads,))
            )
        else:
            self.attention_sink_bias = None

    def _apply_rope(self, q: jnp.ndarray, k: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        seq_len = q.shape[-2]
        cos, sin = build_rope_cache(seq_len, self.rope_dim, theta=self.rope_theta, dtype=q.dtype)
        q_rope, q_nope = jnp.split(q, [self.rope_dim], axis=-1)
        k_rope, k_nope = jnp.split(k, [self.rope_dim], axis=-1)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        q = jnp.concatenate([q_rope, q_nope], axis=-1)
        k = jnp.concatenate([k_rope, k_nope], axis=-1)
        return q, k

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        position_ids: Optional[jnp.ndarray],
        deterministic: bool = True,
    ) -> jnp.ndarray:
        del position_ids
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, self.num_key_value_heads, self.v_head_dim).transpose(0, 2, 1, 3)

        q, k = self._apply_rope(q, k)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        if self.use_sink:
            sinks = self.attention_sink_bias[None, :, None, None]
            sinks = jnp.broadcast_to(sinks, (bsz, self.num_attention_heads, seq_len, 1))
            attn_weights = jnp.concatenate([attn_weights, sinks], axis=-1)

        attn_weights = attn_weights - jnp.max(attn_weights, axis=-1, keepdims=True)
        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        if self.use_sink:
            attn_probs = attn_probs[..., :-1]
        attn_probs = self.dropout(attn_probs, deterministic=deterministic)

        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class MiMoV2DecoderLayer(nnx.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.config = config
        self.layer_idx = layer_idx
        is_swa_layer = config.hybrid_layer_pattern[layer_idx] == 1
        self.attention_type = "sliding_window_attention" if is_swa_layer else "full_attention"
        self.self_attn = MiMoV2Attention(config, is_swa_layer, layer_idx, rngs=rngs)
        if config.n_routed_experts is not None and config.moe_layer_freq[layer_idx]:
            self.mlp = MiMoV2MoE(config, rngs=rngs)
        else:
            self.mlp = MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                rngs=rngs,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, rngs=rngs)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        position_ids: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MiMoV2Model(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.embed_tokens = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)
        self.layers = nnx.List(
            [MiMoV2DecoderLayer(config, layer_idx=i, rngs=rngs) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, rngs=rngs)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        _, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (1, seq_len))

        full_mask = make_attention_mask(attention_mask, seq_len, sliding_window=None, dtype=hidden_states.dtype)
        if self.config.sliding_window is None:
            swa_mask = full_mask
        else:
            swa_mask = make_attention_mask(
                attention_mask, seq_len, sliding_window=self.config.sliding_window, dtype=hidden_states.dtype
            )

        for layer in self.layers:
            mask = full_mask if layer.attention_type == "full_attention" else swa_mask
            hidden_states = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=position_ids,
                deterministic=deterministic,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class MiMoV2FlashForCausalLM(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.model = MiMoV2Model(config, rngs=rngs)
        self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        logits_to_keep: int = 0,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        hidden_states = self.model(input_ids, attention_mask=attention_mask, deterministic=deterministic)
        if logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        logits = self.lm_head(hidden_states)
        return logits


__all__ = [
    "ModelConfig",
    "MiMoV2FlashForCausalLM",
    "MiMoV2Model",
    "MiMoV2DecoderLayer",
    "MiMoV2Attention",
    "MiMoV2MoE",
    "RMSNorm",
    "make_attention_mask",
]
