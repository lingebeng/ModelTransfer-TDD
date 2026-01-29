## 嵌入/解码器测试中遇到的问题

### 1) JAX TPU 后端日志噪音
- 现象：即使在 CPU 上运行，也会打印 `Unable to initialize backend 'tpu': Failed to open libtpu.so`。
- 处理：在测试里设置环境变量：
  - `JAX_PLATFORMS=cpu`
  - `JAX_LOG_LEVEL=ERROR`
  - `TF_CPP_MIN_LOG_LEVEL=2`
- 备注：`JAX_PLATFORM_NAME=cpu` 仍会触发 TPU 探测日志，必须用 `JAX_PLATFORMS=cpu`。

### 2) NNX `.value` 弃用警告
- 现象：访问/赋值 `nnx.Param` 时出现 `'.value' access is now deprecated...`。
- 位置：`modeling.py` 与测试中有多处 `.value` 使用。
- 当前影响：仅警告，不影响正确性。
- 潜在修复：改用 `param[...]` 与 `param.get_value()` / `param.set_value(...)`。

### 3) safetensors 中 FP8 权重加载失败
- 现象：`AttributeError: module 'numpy' has no attribute 'float8_e4m3fn'`（`framework="numpy"` 时）。
- 根因：权重是 `torch.float8_e4m3fn`，numpy 不支持该 dtype。
- 处理：使用 `framework="pt"` 加载，再用 `weight_scale_inv` 做解量化。
- 解量化逻辑：`weight_fp8.float() * scale_inv.float()`，按 128x128 block 扩展。

### 4) Torch eager_attention_forward 参数不兼容
- 现象：`TypeError: eager_attention_forward() got an unexpected keyword argument 'position_ids'`。
- 根因：本地 PyTorch 的 `eager_attention_forward` 不接收 `position_ids`，但调用处传了。
- 处理：在测试中封装/替换函数，丢弃 `position_ids`。

### 5) Torch meta device 上的 rotary embedding 报错
- 现象：`NotImplementedError: Cannot copy out of meta tensor; no data!`
- 根因：为了省内存用 meta device 构建模型，但 `rotary_emb` 需要真实数据。
- 处理：单独创建 `MiMoV2FlashRotaryEmbedding`（真实设备），用于测试。

### 6) Rope 配置校验错误
- 现象：`KeyError: Missing required keys in rope_parameters ...`
- 根因：HF 5.x 对 `rope_scaling` 要求必须含 `rope_type` 及其必需字段。
- 处理：在测试 config 中传入 `rope_scaling={"rope_type": "linear", "factor": 1.0}`。
