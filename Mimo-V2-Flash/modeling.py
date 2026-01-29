import dataclasses
import math
from functools import partial
from typing import TypeAlias

import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, ArrayLike

from .config import ModelConfig, ShardingCfg

_K_MASK = jnp.finfo(jnp.bfloat16).min
ShardingSpec: TypeAlias = P


def shard(x, spec: ShardingSpec):
    if get_abstract_mesh().empty or spec is None:
        return x
    return reshard(x, spec)


@dataclasses.dataclass(slots=True)
class LayerCache:
    k_cache: nnx.Cache
    v_cache: nnx.Cache
    size: int
    start_ind: nnx.Variable
    cur_ind: nnx.Variable

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        shd_cfg: ShardingCfg,
        batch_size: int,
        cache_size: int,
        dtype: jnp.dtype,
    ):
        k_shape = (batch_size, cache_size, num_kv_heads, head_dim)
        v_shape = (batch_size, cache_size, num_kv_heads, v_head_dim)
        self.k_cache = shard(
            nnx.Cache(jnp.zeros(k_shape, dtype=dtype)), shd_cfg.act_btnh
        )
        self.v_cache = shard(
            nnx.Cache(jnp.zeros(v_shape, dtype=dtype)), shd_cfg.act_btnv
        )
        self.size = self.k_cache.shape[1]
        batch_sharding = P(shd_cfg.act_btnh[0]) if shd_cfg.act_btnh else P(None)
        self.start_ind = shard(
            nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32)), batch_sharding
        )
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))


Cache: TypeAlias = list[LayerCache]


class Einsum(nnx.Module):
    def __init__(
        self,
        einsum_str: str,
        shape: tuple[int, ...],
        *,
        shd: ShardingSpec,
        rngs: nnx.Rngs,
        use_bias: bool = False,
        bias_shape: tuple[int, ...] | None = None,
    ):
        self.einsum_str = einsum_str
        self.shape = shape
        self.w = shard(nnx.Param(nnx.initializers.normal()(rngs.params(), shape)), shd)
        self.b = None
        if use_bias:
            if bias_shape is None:
                raise ValueError("bias_shape must be provided when use_bias=True")
            self.b = shard(
                nnx.Param(nnx.initializers.zeros_init()(rngs.params(), bias_shape)),
                P(None),
            )

    @jax.named_scope("einsum")
    def __call__(self, x: ArrayLike) -> Array:
        y = jnp.einsum(self.einsum_str, x, self.w[...])
        if self.b is not None:
            y = y + self.b[...]
        return y


def _generate_pos_embeddings(
    positions: jax.Array, head_dim: int, rope_theta: float
) -> tuple[jax.Array, jax.Array]:
    if head_dim == 0:
        zeros = jnp.zeros((*positions.shape, 0), dtype=jnp.float32)
        return zeros, zeros
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum(
        "BT,k->BTk",
        positions,
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    sin = jnp.concatenate([sin, sin], axis=-1)
    cos = jnp.concatenate([cos, cos], axis=-1)
    return sin, cos


def rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope_partial(
    x: jax.Array, sin: jax.Array, cos: jax.Array, rope_dim: int
) -> jax.Array:
    if rope_dim == 0:
        return x
    x_rope, x_nope = x[..., :rope_dim], x[..., rope_dim:]
    sin = sin[:, :, None, :]
    cos = cos[:, :, None, :]
    x_rope = (x_rope * cos) + (rotate_half(x_rope) * sin)
    return jnp.concatenate([x_rope, x_nope], axis=-1).astype(x.dtype)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.scale = shard(
            nnx.Param(nnx.initializers.ones_init()(rngs.params(), dim)),
            cfg.shd_cfg.rms_norm,
        )
        self.norm_eps = cfg.norm_eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        rms = jnp.sqrt(
            jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True)
            + self.norm_eps
        )
        return jnp.astype(self.scale[...] * x / rms, dtype)


def count_left_pads(x: jax.Array) -> int:
    return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)


def count_right_pads(x: jax.Array, pad_id) -> int:
    result = jnp.where(
        jnp.all(x == pad_id, axis=1),
        x.shape[1],
        jnp.argmin(jnp.flip(x == pad_id, axis=1).astype(jnp.int32), axis=1),
    )
    return jnp.max(result)


def compute_positions_from_segment_ids(seg_ids):
    return jax.vmap(
        lambda row: jnp.where(
            row != 0, jnp.arange(seg_ids.shape[1]) - jnp.argmax(row), 2**30
        )
    )(seg_ids)


def repeat_kv(hidden_states: jax.Array, n_rep: int) -> jax.Array:
    batch, slen, num_kv_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.repeat(hidden_states[:, :, :, None, :], n_rep, axis=3)
    return hidden_states.reshape(batch, slen, num_kv_heads * n_rep, head_dim)


class MoEGate(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.top_k = cfg.num_experts_per_tok
        self.n_routed_experts = cfg.n_routed_experts
        self.routed_scaling_factor = cfg.routed_scaling_factor
        self.scoring_func = cfg.scoring_func
        self.topk_method = cfg.topk_method
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.norm_topk_prob = cfg.norm_topk_prob
        self.gating_dim = cfg.emb_dim
        self.w = nnx.Param(
            nnx.initializers.normal()(
                rngs.params(), (self.n_routed_experts, self.gating_dim)
            )
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nnx.Param(
                nnx.initializers.zeros_init()(rngs.params(), (self.n_routed_experts,))
            )
        else:
            # Keep as a Param to avoid static-attribute assignment issues in newer nnx.
            self.e_score_correction_bias = nnx.Param(
                nnx.initializers.zeros_init()(rngs.params(), (self.n_routed_experts,))
            )

    def __call__(self, hidden_states: Array) -> tuple[Array, Array]:
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, h)
        logits = jnp.matmul(
            hidden_states.astype(jnp.float32), self.w[...].T.astype(jnp.float32)
        )
        if self.scoring_func == "sigmoid":
            scores = jax.nn.sigmoid(logits)
        else:
            raise NotImplementedError(
                f"Unsupported scoring function: {self.scoring_func}"
            )

        if self.topk_method != "noaux_tc":
            raise NotImplementedError(f"Unsupported TopK method: {self.topk_method}")

        if self.e_score_correction_bias is None:
            raise ValueError("e_score_correction_bias must be set for noaux_tc")

        scores_for_choice = scores + self.e_score_correction_bias[...][None, :]
        scores_grouped = scores_for_choice.reshape(bsz * seq_len, self.n_group, -1)
        top2 = jax.lax.top_k(scores_grouped, k=2)[0]
        group_scores = jnp.sum(top2, axis=-1)
        group_idx = jax.lax.top_k(group_scores, k=self.topk_group)[1]
        group_mask = jnp.sum(jax.nn.one_hot(group_idx, self.n_group), axis=1)
        score_mask = jnp.repeat(
            group_mask[:, :, None], scores_grouped.shape[-1], axis=-1
        ).reshape(bsz * seq_len, -1)
        tmp_scores = jnp.where(score_mask.astype(bool), scores_for_choice, -jnp.inf)
        topk_idx = jax.lax.top_k(tmp_scores, k=self.top_k)[1]
        topk_weight = jnp.take_along_axis(scores, topk_idx, axis=-1)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class MLP(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs, mlp_dim: int | None = None):
        self.shd_cfg = cfg.shd_cfg
        self.hidden_dim = cfg.emb_dim
        self.mlp_dim = mlp_dim if mlp_dim is not None else cfg.mlp_dim
        linear = partial(nnx.Linear, use_bias=False, rngs=rngs)
        self.gate_proj = shard(
            linear(self.hidden_dim, self.mlp_dim), self.shd_cfg.ffw_weight_df
        )
        self.up_proj = shard(
            linear(self.hidden_dim, self.mlp_dim), self.shd_cfg.ffw_weight_df
        )
        self.down_proj = shard(
            linear(self.mlp_dim, self.hidden_dim), self.shd_cfg.ffw_weight_fd
        )

    @jax.named_scope("feed_forward")
    def __call__(self, x: ArrayLike) -> Array:
        activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
        activations = shard(activations, self.shd_cfg.act_btf)
        outputs = self.down_proj(activations)
        return outputs


class MoE(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.experts = nnx.List(
            [
                MLP(cfg, rngs=rngs, mlp_dim=cfg.moe_intermediate_size)
                for _ in range(cfg.n_routed_experts)
            ]
        )
        self.gate = MoEGate(cfg, rngs=rngs)
        self.n_routed_experts = cfg.n_routed_experts

    def __call__(self, hidden_states: Array) -> Array:
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        tokens = hidden_states.reshape(-1, hidden_states.shape[-1])

        expert_outputs = jnp.stack([expert(tokens) for expert in self.experts], axis=0)
        weights = jnp.sum(
            jax.nn.one_hot(topk_idx, self.n_routed_experts) * topk_weight[..., None],
            axis=1,
        )
        outputs = jnp.einsum("END,NE->ND", expert_outputs, weights)
        return outputs.reshape(orig_shape)


class Attention(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, layer_idx: int, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        self.is_swa = cfg.is_swa_layer(layer_idx)
        self.num_heads = cfg.num_heads_for_layer(layer_idx)
        self.num_kv_heads = cfg.num_kv_heads_for_layer(layer_idx)
        self.head_dim = cfg.head_dim_for_layer(layer_idx)
        self.v_head_dim = cfg.v_head_dim_for_layer(layer_idx)
        self.n_rep = self.num_heads // self.num_kv_heads
        self.rope_dim = cfg.rope_dim_for_layer(layer_idx)
        self.rope_theta = cfg.rope_theta_for_layer(layer_idx)
        self.scaling = self.head_dim**-0.5
        self.attention_bias = cfg.attention_bias
        self.attention_dropout = cfg.attention_dropout
        self.sliding_window = cfg.sliding_window if self.is_swa else None

        self.q_proj = Einsum(
            "BTD,DNH->BTNH",
            (cfg.emb_dim, self.num_heads, self.head_dim),
            shd=self.shd_cfg.q_weight_ndh,
            rngs=rngs,
            use_bias=self.attention_bias,
            bias_shape=(self.num_heads, self.head_dim),
        )
        self.k_proj = Einsum(
            "BSD,DKH->BSKH",
            (cfg.emb_dim, self.num_kv_heads, self.head_dim),
            shd=self.shd_cfg.k_weight_kdh,
            rngs=rngs,
            use_bias=self.attention_bias,
            bias_shape=(self.num_kv_heads, self.head_dim),
        )
        self.v_proj = Einsum(
            "BSD,DKV->BSKV",
            (cfg.emb_dim, self.num_kv_heads, self.v_head_dim),
            shd=self.shd_cfg.v_weight_kdv,
            rngs=rngs,
            use_bias=self.attention_bias,
            bias_shape=(self.num_kv_heads, self.v_head_dim),
        )
        self.o_proj = Einsum(
            "BTNV,NVD->BTD",
            (self.num_heads, self.v_head_dim, cfg.emb_dim),
            shd=self.shd_cfg.o_weight_nvd,
            rngs=rngs,
        )

        use_sink_bias = (cfg.add_full_attention_sink_bias and not self.is_swa) or (
            cfg.add_swa_attention_sink_bias and self.is_swa
        )
        self.use_sink_bias = use_sink_bias
        self.attention_sink_bias = nnx.Param(
            nnx.initializers.zeros_init()(rngs.params(), (cfg.num_heads,))
        )

    @jax.named_scope("attention")
    def __call__(self, x: Array, cache: LayerCache, segment_ids: Array) -> Array:
        query_proj = shard(self.q_proj(x), self.shd_cfg.act_btnh)  # [B, T, N, H]
        key_proj = shard(self.k_proj(x), self.shd_cfg.act_btnh)  # [B, T, K, H]
        value_proj = shard(self.v_proj(x), self.shd_cfg.act_btnv)  # [B, T, K, V]

        left_pads = count_left_pads(segment_ids)
        left_pads = shard(left_pads, P(self.shd_cfg.act_btnh[0]))
        cache.start_ind[...] = jnp.where(
            cache.start_ind[...] < 0, left_pads, cache.start_ind[...]
        )
        position_ids = (
            compute_positions_from_segment_ids(segment_ids) + cache.cur_ind[...]
        )
        sin, cos = _generate_pos_embeddings(
            position_ids, self.rope_dim, self.rope_theta
        )
        query_proj = apply_rope_partial(query_proj, sin, cos, self.rope_dim)
        key_proj = apply_rope_partial(key_proj, sin, cos, self.rope_dim)

        slice_indices = (0, cache.cur_ind[...], 0, 0)
        cache.v_cache[...] = jax.lax.dynamic_update_slice(
            cache.v_cache[...], value_proj, slice_indices
        )
        cache.k_cache[...] = jax.lax.dynamic_update_slice(
            cache.k_cache[...], key_proj, slice_indices
        )

        b, t, n, _ = query_proj.shape
        key_cache = repeat_kv(cache.k_cache[...], self.n_rep)
        value_cache = repeat_kv(cache.v_cache[...], self.n_rep)

        attn_logits = (
            jnp.einsum("BTNH,BSNH->BTNS", query_proj, key_cache) * self.scaling
        )

        q_pos = (
            cache.cur_ind[...]
            + jnp.arange(t, dtype=jnp.int32)[None, :]
            - cache.start_ind[...][:, None]
        )
        ts = jnp.arange(cache.size, dtype=jnp.int32)
        kv_segment_ids = (ts[None, :] >= cache.start_ind[...][:, None]) & (
            ts[None, :] < cache.cur_ind[...] + t
        )
        k_pos = ts[None, :] - cache.start_ind[...][:, None]
        causal_mask = k_pos[:, None, :] <= q_pos[:, :, None]
        segment_mask = kv_segment_ids[:, None, :] == segment_ids[:, :, None]
        if self.sliding_window is None:
            window_mask = True
        else:
            window_mask = k_pos[:, None, :] >= (
                q_pos[:, :, None] - (self.sliding_window - 1)
            )
        final_mask = causal_mask & segment_mask & window_mask
        attn_logits = jnp.where(final_mask[:, :, None, :], attn_logits, _K_MASK)

        if self.use_sink_bias:
            sink_logits = self.attention_sink_bias[...][None, None, :, None]
            sink_logits = jnp.broadcast_to(sink_logits, (b, t, n, 1))
            attn_logits = jnp.concatenate([attn_logits, sink_logits], axis=-1)

        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(
            attn_logits.dtype
        )
        if self.use_sink_bias:
            attn_weights = attn_weights[..., :-1]

        attn_output = jnp.einsum("BTNS,BSNV->BTNV", attn_weights, value_cache)

        cache.cur_ind[...] = cache.cur_ind[...] + t
        return shard(self.o_proj(attn_output), self.shd_cfg.act_btd)


class DecoderLayer(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, layer_idx: int, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.attn = Attention(cfg=cfg, layer_idx=layer_idx, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.mlp = (
            MoE(cfg, rngs=rngs) if cfg.is_moe_layer(layer_idx) else MLP(cfg, rngs=rngs)
        )

    def __call__(self, x: Array, cache: LayerCache, segment_ids: Array) -> Array:
        inputs_normalized = self.input_layernorm(x)
        attn_output = x + self.attn(inputs_normalized, cache, segment_ids)
        outputs = attn_output + self.mlp(self.post_attention_layernorm(attn_output))
        return outputs


class MiMoV2Flash(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.embedder = shard(
            nnx.Embed(
                num_embeddings=cfg.vocab_size,
                features=cfg.emb_dim,
                dtype=jnp.bfloat16,
                rngs=rngs,
            ),
            cfg.shd_cfg.emb_vd,
        )
        self.out_emb_shd = None if get_abstract_mesh().empty else cfg.shd_cfg.act_btd
        self.layers = nnx.List(
            [
                DecoderLayer(cfg=cfg, layer_idx=i, rngs=rngs)
                for i in range(cfg.num_layers)
            ]
        )
        self.final_norm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.lm_head = Einsum(
            einsum_str="BTD,DV->BTV",
            shape=(cfg.emb_dim, cfg.vocab_size),
            shd=cfg.shd_cfg.emb_dv,
            rngs=rngs,
        )

    def init_cache(
        self,
        cfg: ModelConfig,
        batch_size: int,
        token_len: int,
        generate_steps: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> Cache:
        cache_size = 2 ** math.ceil(math.log2(max(token_len + generate_steps, 1)))
        caches = []
        for i in range(cfg.num_layers):
            caches.append(
                LayerCache(
                    cfg.num_kv_heads_for_layer(i),
                    cfg.head_dim_for_layer(i),
                    cfg.v_head_dim_for_layer(i),
                    cfg.shd_cfg,
                    batch_size,
                    cache_size,
                    dtype,
                )
            )
        return caches

    def __call__(self, tokens, segment_ids, cache, num_right_pads):
        x = (
            self.embedder.embedding[...]
            .at[(tokens,)]
            .get(out_sharding=self.out_emb_shd)
        )
        for i, layer in enumerate(self.layers):
            x = layer(x, cache[i], segment_ids)
        logits = self.lm_head(self.final_norm(x))
        return logits


@jax.jit
def forward(
    model: nnx.Module, cache: Cache, tokens: Array, pad_id: int
) -> tuple[Array, nnx.Cache]:
    segment_ids = 1 * (tokens != pad_id)
    num_right_pads = count_right_pads(tokens, pad_id)
    logits = model(tokens, segment_ids, cache, num_right_pads)
    target_ind = tokens.shape[-1] - num_right_pads - 1
    return logits[:, target_ind], cache


__all__ = ["ModelConfig", "MiMoV2Flash", "forward"]
