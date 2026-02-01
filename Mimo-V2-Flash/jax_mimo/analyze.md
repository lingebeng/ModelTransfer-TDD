# MiMo-V2-Flash JAX (nnx) 版本逐模块详解

> 面向 **JAX 小白 + LLM 小白** 的超详细解释。目标：看懂 `Mimo-V2-Flash/jax_mimo/modeling.py` 里每一个模块的语法、设计逻辑、意图、以及关键张量形状。

---

## 0. 总览：这份代码做了什么？

这份 `modeling.py` 是 **MiMo-V2-Flash** 的 JAX 实现，采用 **Flax 的 nnx API**。

模型整体结构：

```
input_ids  ->  Embedding
          ->  (Layer 0..N-1): [RMSNorm -> Attention -> Residual] + [RMSNorm -> MLP/MoE -> Residual]
          ->  Final RMSNorm
          ->  LM Head (Linear)
          ->  logits
```

它支持：
- **Full Attention** 和 **Sliding Window Attention (SWA)** 混合层
- **RoPE** 位置编码
- **MoE (Mixture-of-Experts)** 前馈层

---

## 1. JAX / Flax nnx 基础概念（必须先理解）

### 1.1 JAX 基础
- `jax.numpy`（缩写 `jnp`）是 NumPy 风格的 API，但 **底层是可编译的函数图**。
- JAX 张量的核心就是 **不可变**：操作会产生新张量。
- 常见操作：
  - `jnp.arange`, `jnp.reshape`, `jnp.transpose`, `jnp.einsum`, `jnp.where` 等。

### 1.2 Flax nnx API 是什么？
- `nnx` 是 Flax 的 **新式模块系统**。
- **与旧的 `flax.linen` 不同**：
  - nnx 是 **对象式 (stateful)**：模型实例里直接带参数。
  - 参数不是一个“外部 dict”，而是直接挂在模块上。

关键概念：
- `nnx.Module`: nnx 模块基类。
- `nnx.Param`: 参数对象（比如权重）。
- `nnx.Linear`: 线性层（类似 PyTorch 的 nn.Linear）。
- `nnx.Embed`: embedding 层。
- `nnx.Dropout`: dropout 层（需要 RNG）。
- `nnx.List`: nnx 对应的模块列表容器。

### 1.3 nnx 的 RNGs
- nnx 里 **需要显式传入 RNG streams**：例如 `nnx.Rngs(params=0, dropout=1)`
- 如果用了 `Dropout` 但没传 `dropout` rng，会报错。

---

## 2. ModelConfig：模型超参数的“说明书”

```python
@dataclasses.dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    ...
```

### 2.1 作用
- 用一个 dataclass 把所有超参数集中管理。
- 相当于 PyTorch 里的 `Config`。

### 2.2 重要字段解释
- `vocab_size`: 词表大小
- `hidden_size`: token embedding/隐藏向量的维度 (D)
- `num_hidden_layers`: transformer 层数
- `num_attention_heads`: 头数 (Hq)
- `num_key_value_heads`: KV 头数 (Hkv)
- `head_dim`: 每个 head 的维度 (通常 = hidden_size / num_heads)
- `v_head_dim`: value head 维度（MiMo 里可与 head_dim 不同）
- `hybrid_layer_pattern`: 每层 attention 类型，0=full, 1=SWA
- `moe_layer_freq`: 每层是否使用 MoE
- `rope_theta`: RoPE 频率常数

### 2.3 `__post_init__` 自动补全逻辑
- 如果 `num_key_value_heads` 未指定，则默认等于 `num_attention_heads`。
- 如果 `head_dim` 未指定，自动 = `hidden_size / num_attention_heads`。
- 如果 `hybrid_layer_pattern` 未提供，就自动生成（根据 `hybrid_block_size`）。
- 如果 `moe_layer_freq` 未提供，就默认全部 False。

> **直觉**：`__post_init__` 就是保证所有派生字段都完整可用。

### 2.4 `tiny_config`
- 这是专门用于“冒烟测试”的小配置：
  - hidden_size=256
  - heads=4
  - layers=4
  - MoE 开启
- 这样做的目的是 **跑得快**、验证结构是否正确。

### 2.5 `from_torch_config`
- 直接把 PyTorch config 转成 JAX config，方便对齐测试。

---

## 3. 工具函数：小但关键

### 3.1 `_get_act_fn`
- 将 `hidden_act` 字符串映射到 JAX 函数。
- 目前只支持 `silu`。

### 3.2 Mask 相关

#### `_causal_mask(seq_len)`
- 生成一个 **下三角** mask
- `allow[i, j] = True` 表示 query i 可以看 key j
- 也就是保证 **自回归**：不能看未来

#### `_sliding_window_mask(seq_len, window)`
- 只允许看过去 `window` 长度以内

#### `make_attention_mask(...)`
- 将 causal/swa mask + attention_mask 合成
- 输出形状 `(B, 1, T, T)`
- allowed 的位置是 `0`，禁止的位置是 `-inf`（方便 softmax）

> 这是标准做法：用 `-inf` 把不允许的 attention 权重消掉。

### 3.3 RoPE

#### `build_rope_cache`
- 生成 cos/sin 旋转矩阵
- shape: `(1, 1, T, rope_dim)`

#### `apply_rotary_pos_emb`
- 对 Q/K 的前 `rope_dim` 维度应用旋转

---

## 4. RMSNorm

```python
class RMSNorm(nnx.Module):
```

### 4.1 公式
- RMSNorm 的核心公式：

```
RMS(x) = sqrt(mean(x^2) + eps)
output = x / RMS(x) * weight
```

### 4.2 代码细节
- `self.weight`: 形状 `(hidden_size,)`
- `x_float = x.astype(jnp.float32)`：用 float32 防止数值误差
- `variance = mean(x^2)`
- `x_norm = x / sqrt(variance + eps)`

### 4.3 输入输出形状
- 输入：`(B, T, D)`
- 输出：`(B, T, D)`

---

## 5. MLP

```python
class MLP(nnx.Module):
```

### 5.1 结构
- gate_proj: `D -> F`
- up_proj:   `D -> F`
- down_proj: `F -> D`

### 5.2 公式
```
MLP(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
```

> 这是常见的 **Gated MLP** 形式（类似 SwiGLU）。

### 5.3 输入输出形状
- 输入 `(B, T, D)`
- 输出 `(B, T, D)`

---

## 6. MiMoV2MoEGate（MoE 路由器）

```python
class MiMoV2MoEGate(nnx.Module):
```

### 6.1 作用
- 给每个 token 选择 **Top-K 专家**
- 输出两个东西：
  - `topk_idx`: (B*T, K) 每个 token 选中的专家索引
  - `topk_weight`: (B*T, K) 每个专家的权重

### 6.2 关键张量
- `hidden_states`: (B, T, D)
- flatten 后 `x`: (B*T, D)
- `weight`: (E, D)，E = 专家数
- `logits = x @ weight.T`: (B*T, E)

### 6.3 TopK 选择逻辑（noaux_tc）
- 首先计算 `scores` (sigmoid(logits))
- 再把专家分成 `n_group` 组
- 每组选 top2，然后在组之间再选 topk_group
- 形成 mask，再在 mask 内做 topK

> 这属于 **DeepSeek/MiMo 风格的 group-topk 选择**，减少专家冲突。

---

## 7. MiMoV2MoE（真正的专家混合）

```python
class MiMoV2MoE(nnx.Module):
```

### 7.1 结构
- `experts`: 一个专家列表，每个专家是一个 MLP
- `gate`: 上面的 MoE Gate

### 7.2 前向过程
1. `topk_idx, topk_weight = gate(hidden_states)`
2. flatten `(B*T, D)`
3. 对每个专家：
   - 找到它负责的 token
   - 计算专家输出
   - 按权重加回 `final`

### 7.3 输出
- 输出 shape = 输入 shape `(B, T, D)`

---

## 8. Attention 模块（MiMoV2Attention）

```python
class MiMoV2Attention(nnx.Module):
```

### 8.1 作用
- 对输入序列做自注意力
- 既支持 **full attention** 又支持 **sliding window attention**

### 8.2 Q/K/V 形状
假设：
- B=batch
- T=seq_len
- H=attention heads
- Hkv=kv heads
- Dh=head_dim
- Dv=v_head_dim

```
q_proj: (B, T, H*Dh)
reshape -> (B, H, T, Dh)

k_proj: (B, T, Hkv*Dh)
reshape -> (B, Hkv, T, Dh)

v_proj: (B, T, Hkv*Dv)
reshape -> (B, Hkv, T, Dv)
```

### 8.3 RoPE
- 对 Q/K 的前 `rope_dim` 维度做旋转

### 8.4 repeat_kv
- 如果 H > Hkv，则需要把 K/V 复制到多头
- repeat_kv 会把 (B, Hkv, T, Dh) 变成 (B, H, T, Dh)

### 8.5 attention weights
```
attn = Q @ K^T / sqrt(Dh)
```
shape: `(B, H, T, T)`

### 8.6 attention mask
- mask 形状 `(B, 1, T, T)`
- 加到 logits 上，禁止的位置是 -inf

### 8.7 attention sink bias
- 如果启用，会在 key 维度拼接一个额外位置（用作 sink）

### 8.8 softmax + dropout
- `softmax` 得到概率
- `dropout` 用于训练

### 8.9 输出
- 计算 `attn_probs @ V` -> `(B, H, T, Dv)`
- reshape -> `(B, T, H*Dv)`
- 线性投影 `o_proj` -> `(B, T, D)`

---

## 9. Decoder Layer

```python
class MiMoV2DecoderLayer(nnx.Module):
```

结构是标准 Transformer Decoder：

1. **输入层归一化**
2. **Attention** + 残差
3. **后归一化**
4. **MLP/MoE** + 残差

关键点：
- `hybrid_layer_pattern` 决定当前层是 full attention 还是 sliding window attention
- `moe_layer_freq` 决定当前层是 MoE 还是普通 MLP

---

## 10. MiMoV2Model（backbone）

```python
class MiMoV2Model(nnx.Module):
```

### 10.1 前向流程
1. **Embedding**
2. **生成 position_ids**
3. **构造两套 attention mask**
   - full mask
   - sliding window mask
4. 遍历每一层：
   - 按 layer.attention_type 选择 mask
   - forward
5. 最后做一次 RMSNorm

### 10.2 关键输出
- 返回 `hidden_states`，shape `(B, T, D)`

---

## 11. MiMoV2FlashForCausalLM（带 LM Head）

```python
class MiMoV2FlashForCausalLM(nnx.Module):
```

- 包含 backbone (`MiMoV2Model`)
- 加一个 `lm_head`，即线性投影到 vocab

### 11.1 logits_to_keep
- 如果 `logits_to_keep = 1`，只保留最后一个 token 的 logits
- 常用于生成时只预测下一个 token

### 11.2 输出
- logits 形状 `(B, T', vocab_size)`
- T' = `logits_to_keep` 或 `T`

---

## 12. 小白常见问题 FAQ

### Q1: 为什么要用 RoPE？
- RoPE 是一种位置编码方式，不需要额外的 position embedding 表。
- 直接通过旋转 Q/K 让注意力知道“相对位置”。

### Q2: 为什么 attention mask 是 (B,1,T,T)？
- B：batch
- 1：broadcast 到所有 heads
- T,T：query 和 key 的维度

### Q3: MoE 的意义？
- 每个 token 只走少数专家 => 计算更省
- 但总参数量很大 => 模型表达能力强

### Q4: 为什么用 nnx 而不是 linen？
- nnx 更接近 PyTorch 的“对象式参数”，方便权重映射和 debug。
- 并且 bonsai 项目整体也是 nnx 风格。

---

## 13. 建议的阅读顺序

1. `ModelConfig`（理解各种超参数含义）
2. `make_attention_mask`（理解 mask 维度）
3. `RMSNorm` / `MLP`
4. `MiMoV2Attention`
5. `MiMoV2MoEGate` / `MiMoV2MoE`
6. `MiMoV2DecoderLayer`
7. `MiMoV2Model`
8. `MiMoV2FlashForCausalLM`

---

## 14. 配图式数据流（ASCII 图）

### 14.1 整体数据流（从 input_ids 到 logits）

```
input_ids (B, T)
    |
    v
Embed (vocab_size -> hidden_size)
    |
    v
hidden_states (B, T, D)
    |
    v
for layer in 0..L-1:
    +---------------------------+
    | RMSNorm                   |
    |   (B, T, D) -> (B, T, D)  |
    +---------------------------+
                |
                v
    +---------------------------+
    | Attention (Full or SWA)   |
    |   (B, T, D) -> (B, T, D)  |
    +---------------------------+
                |
                v
    +---------------------------+
    | Residual Add              |
    |   x = x + attn(x)         |
    +---------------------------+
                |
                v
    +---------------------------+
    | RMSNorm                   |
    |   (B, T, D) -> (B, T, D)  |
    +---------------------------+
                |
                v
    +---------------------------+
    | MLP or MoE                |
    |   (B, T, D) -> (B, T, D)  |
    +---------------------------+
                |
                v
    +---------------------------+
    | Residual Add              |
    |   x = x + ffn(x)          |
    +---------------------------+

    |
    v
Final RMSNorm
    |
    v
LM Head (Linear D -> vocab_size)
    |
    v
logits (B, T or 1, vocab_size)
```

---

### 14.2 注意力（Attention）内部数据流

```
hidden_states (B, T, D)
   |
   |  q_proj / k_proj / v_proj
   v
Q: (B, T, H*Dh) -> reshape -> (B, H, T, Dh)
K: (B, T, Hkv*Dh) -> reshape -> (B, Hkv, T, Dh)
V: (B, T, Hkv*Dv) -> reshape -> (B, Hkv, T, Dv)
   |
   |  RoPE on first rope_dim of Q/K
   v
Q', K'
   |
   |  repeat_kv if H > Hkv
   v
K', V' -> (B, H, T, Dh/Dv)
   |
   |  attn_weights = Q @ K^T / sqrt(Dh)
   v
attn_weights: (B, H, T, T)
   |
   |  + attention_mask  (causal or sliding)
   v
masked_weights
   |
   |  softmax
   v
attn_probs: (B, H, T, T)
   |
   |  attn_output = attn_probs @ V
   v
attn_output: (B, H, T, Dv)
   |
   |  transpose + reshape
   v
(B, T, H*Dv)
   |
   |  o_proj (Linear)
   v
output: (B, T, D)
```

---

### 14.3 RoPE 位置编码数据流（简化）

```
positions: 0..T-1
   |
   v
cos/sin tables: (1, 1, T, rope_dim)
   |
   v
Q_rope, K_rope: (B, H, T, rope_dim)
   |
   |  rotate_half + cos/sin mix
   v
Q_rot, K_rot
   |
   v
Q = concat(Q_rot, Q_nope)  (last dim = Dh)
K = concat(K_rot, K_nope)
```

---

### 14.4 Attention Mask 生成（Full vs SWA）

```
seq_len = T

Full causal mask (T x T):
allow[i, j] = (j <= i)

Sliding window mask (T x T):
allow[i, j] = (j <= i) AND (i - j < window)

attention_mask (B, T):
1 = real token, 0 = padding

Final mask:
allow = allow[None, None, :, :] & attention_mask[:, None, None, :]

Return mask values:
allowed -> 0
blocked -> -inf
```

---

### 14.5 MoE（Mixture-of-Experts）数据流

```
hidden_states (B, T, D)
   |
   v
Gate:
  logits = x @ W_gate^T  -> (B*T, E)
  scores = sigmoid(logits)
  topk_idx, topk_weight = TopK(scores)

Experts:
  for each expert e:
    select tokens where e in topk_idx
    expert_out = MLP_e(x_tokens)
    weighted_out = expert_out * topk_weight
    add into final

final_hidden_states (B, T, D)
```

---

### 14.6 形状速查表（常用符号）

```
B  = batch size
T  = sequence length
D  = hidden_size
H  = num_attention_heads
Hkv= num_key_value_heads
Dh = head_dim
Dv = v_head_dim
E  = n_routed_experts
K  = num_experts_per_tok
```

---

如果你希望，我可以继续加：
- **逐行注释版**（直接把代码拷贝并逐行解释）
- **图解版数据流**（每层 shape 图）
- **配套练习题**（帮助你检查理解）
