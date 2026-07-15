# Adapter Registry 能力边界审计

状态：研究记录 / 阶段快照
适用版本：以文中日期、提交和运行环境为准；不作为当前 main 操作说明

日期：2026-07-04

范围：只读审计 `networks/registry.py`、LoRA family factory/save、插件注册、推理加载、merge 和 Web 续训服务。未修改代码，未运行真实推理。

## 主结论

当前 registry 已经能表达“如何创建、嗅探、预处理和保存某些 adapter”，但还不能统一表达“是否可 merge、推理应该静态合并还是动态挂载、Web 续训是否支持、复杂互斥策略”。这些能力仍散落在 `merge_to_dit.py`、推理模型加载、LoRA factory 和 Web 服务里。

最需要优先处理的风险是：插件权重如果进入普通静态 `load_safetensors_with_lora()` 路径，可能被 warning 后静默忽略，而不是自动走动态 hook 或明确失败。

## Registry 当前能表达的内容

- `NetworkSpec`
  - `name`
  - `module_class`
  - `save_variant`
  - `kwarg_flags`
  - `selector` / `validate`
  - `module_kwargs`
  - `preprocess_weights`
  - 权重 header 嗅探
  - continue-lora 权重类型识别
- core specs 已覆盖 `lora`、`dora`、`ortho`、`hydra`、`ortho_hydra`、`chimera_hydra`、`stacked_experts_global_fei`。
- LoHa / GLoRA / VeRA / StepExpert 等插件已接入 registry，并能表达部分互斥和保存 handler。
- fresh training 能通过 `resolve_network_spec()` 做 selector 优先级和部分互斥校验。
- from-weights 加载只部分 registry 化：插件检测走 `detect_network_spec_from_weights()`，但 Hydra / Chimera / DoRA / Ortho / ReFT / register 等仍在 factory 里硬编码扫描。

## 仍散落的能力

### Merge 能力

- `LoRANetwork.is_mergeable()` 当前主要看 register tokens。
- `scripts/merge_to_dit.py` 另用 key marker 拒绝 Hydra / ReFT / StepExpert / postfix / register 等非 bakeable 权重。
- registry 没有统一字段表达 `static_mergeable` 或 `merge_policy`。

### 推理加载能力

- `library/inference/models.py` 用 header key / metadata 判断 Hydra、Chimera、StepExpert 是否需要动态挂载。
- 普通静态路径 `load_safetensors_with_lora()` 只识别 `.lora_down.weight` / `.lora_up.weight`。
- LoHa / GLoRA / VeRA 等插件 key 进入静态路径时，有静默无效风险。

### 保存和 metadata

- 插件 save handler 处理 LoHa / GLoRA / VeRA。
- core `lora_save.py` 继续按 `save_variant` 字符串分派 Hydra / Chimera / standard。
- `LoRANetwork.save_weights()` 手写大量 spec、路由、Chimera、VeRA、DoRA metadata。
- `metadata=None` 时，`ss_network_spec` 是否总会写入尚未被测试固定。

### Web 续训

- Web 续训服务会先调用插件识别，但服务层白名单只收 LoRA / DoRA / LoHa / LoKr / GLoRA。
- VeRA 插件可返回 `VeRA`，但当前服务层仍可能拒绝。

## 低冲突清账建议

1. 新增 header-only 分类函数，例如：

```python
classify_adapter_capabilities(keys, metadata) -> AdapterCapabilities
```

输出字段建议：

- `spec_name`
- `adapter_kind`
- `static_mergeable`
- `requires_dynamic_hook`
- `continue_supported`
- `save_variant`
- `reason`

2. 先让这些入口消费分类结果，并保留旧逻辑 fallback：

- `library/inference/models.py`
- `scripts/merge_to_dit.py`
- `web/services/continue_lora_service.py`

3. 给 registry 补低风险字段或外部映射：

- `merge_policy`
- `inference_policy`
- `continue_kind`
- `non_bakeable_markers`

不要一次性把 Hydra / Chimera 的复杂 loader 全搬进 registry。

4. 对插件推理先做安全处理：

- 非 standard LoRA key 进入静态 `load_safetensors_with_lora()` 时，自动转 dynamic hook，或明确报错。
- 不要只 warning 后继续生成看似成功但 adapter 未生效的结果。

## 可测试风险清单

- 静态推理加载 LoHa / GLoRA / VeRA：构造最小 safetensors，走 `load_dit_model()` 非 pgraft 路径，断言不会静默忽略插件 key。
- StepExpert 直走 `create_network_from_weights()`：带 `.lora_ups.N.weight` 和 turbo metadata 的文件应走专门路径或明确报错，避免被 Hydra stack 预处理误收。
- registry/save 覆盖：遍历 `NETWORK_REGISTRY`，断言每个 `save_variant` 要么有 `SAVE_HANDLERS`，要么属于 core save 白名单。
- merge policy 一致性：同一 synthetic 权重在 classifier、`is_mergeable()`、`scan_non_bakeable_keys()` 上结论一致。
- metadata drift：`LoRANetwork.save_weights(..., metadata=None)` 对 lora / 插件 / Hydra 是否写入必要 spec 与 routing metadata。
- 混合插件 key：同一文件混入 LoHa / GLoRA / VeRA 结构时应拒绝，而不是由最后一个 detection 覆盖结论。

## 建议推进顺序

1. 先加 `AdapterCapabilities` 纯分类和测试，不改实际加载行为。
2. 接入 merge 拒绝路径，减少“能不能 bake”的重复判断。
3. 接入 Web 续训服务，补齐 VeRA 和插件类型提示。
4. 最后改推理加载路径，处理插件静态路径的静默无效风险。
