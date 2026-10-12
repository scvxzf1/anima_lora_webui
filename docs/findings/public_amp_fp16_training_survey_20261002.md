# 公网 AMP FP16 训练实现调研

日期：2026-10-02。状态：公开源码与文档调研完成，未修改训练实现或运行 GPU。
范围：Diffusers、OneTrainer、sd-scripts、Musubi-Tuner、SimpleTuner；重点是
Flux、Wan、Z-Image 和确切的 Qwen-Image 2.1。源码固定至下列 commit，
不把配置入口、训练可达保护或文档声明当作质量及 Turing 硬件认证。

## 结论与工程取舍

已有可复用的原生 AMP FP16 训练设计。最直接的先例是 Diffusers 官方
Flux LoRA 脚本：冻结基座 FP16、可训练 LoRA FP32、Accelerate 管理 AMP；
加载训练状态后再次确保 LoRA FP32。Wan 的 FP32 norm/modulation/residual
代码则提供了模型级保护模板。它们支持当前 Anima 的设计方向，但没有证明
不同模型可以共享同一组敏感层或直接转为稳定的 FP16 训练。

| 项目 / 模型 | 核实的机制 | 证据及限制 |
| --- | --- | --- |
| Diffusers / Flux LoRA | FP16 冻结模型 + FP32 可训练参数；加载 hook 再次 upcast trainables | 官方训练代码，未复现外部训练、质量或旧卡性能 |
| OneTrainer / Flux | 可训练参数全为 FP32 时启用 FP16 scaler；T5 单独避免 FP16 autocast | 训练实现；组件 fallback 的实际 dtype 另由配置决定 |
| Diffusers / Wan | FP32 norm、调制、部分残差及 final norm/modulation，随后转回模型 dtype | 训练 forward 可达；不保护所有加法、矩阵乘及 FP16 转回边界 |
| Diffusers / Flux、Qwen-Image 2.1 | FP16 block 输出 clip 到 ±65504 | 训练 forward 可达；会改变饱和区梯度，无法修复已有 NaN |
| Musubi-Tuner / Wan | 文档及入口要求权重 dtype 与 AMP 训练 dtype 匹配 | 可选 FP16 模型训练能力；不等于任意 BF16 权重转换已通过 |
| sd-scripts / Flux、SD3 | Accelerate 的普通 AMP forward | 未在所查文件中找到专门的 scaled-FP16 稳定化配方 |
| SimpleTuner / 多模型 | 精度总表以 BF16 为主，Z-Image 单独文档提到 FP16 | 文档存在不一致，未核实独立 FP16 保护或训练质量证据 |

研究范围内未找到经过多模型及 Turing 验收的通用自动 FP16 校准轮子，
也未找到可直接套用所有 BF16 底模的训练 weight/activation rescale 配方。
这只是本轮检索结果，不是对整个生态不存在此类方案的证明。

## 1. Diffusers 官方 Flux LoRA 训练

固定版本：`578c9b2c6636ab2424a0e56186268b83623656b2`。

[训练脚本的模型 dtype 设置](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/examples/dreambooth/train_dreambooth_lora_flux.py#L1229-L1250)
根据 `accelerator.mixed_precision` 选择 `weight_dtype`，FP16 时将 VAE、
transformer 和文本编码器转换为 FP16。
[训练参数转换](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/examples/dreambooth/train_dreambooth_lora_flux.py#L1378-L1384)
随后只将可训练参数恢复 FP32：

```python
if args.mixed_precision == "fp16":
    # only upcast trainable parameters (LoRA) into fp32
    cast_training_params(models, dtype=torch.float32)
```

[加载 hook](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/examples/dreambooth/train_dreambooth_lora_flux.py#L1355-L1363)
在加载 LoRA 后重复此保证。这是训练和恢复路径的实际代码，不只是 CLI 的
`fp16` 选项；本轮没有运行它。Flux 与本项目 Krea-2-Raw 的实现和参数语义不同，
不能继承其保护或训练结论。

## 2. OneTrainer 的组件精度与 scaler

固定版本：`23df3832e8f213d64c337e39f9365ba46e99fed0`。

[enable_grad_scaling](https://github.com/Nerogar/OneTrainer/blob/23df3832e8f213d64c337e39f9365ba46e99fed0/modules/util/dtype_util.py#L9-L16)
仅在训练 dtype 为 FP16、全部可训练参数为 FP32 时启用 scaler，使用项目自定义
`CustomGradScaler`。
[训练 loop](https://github.com/Nerogar/OneTrainer/blob/23df3832e8f213d64c337e39f9365ba46e99fed0/modules/trainer/GenericTrainer.py#L757-L788)
执行 loss scaling、clip 前 unscale、scaler step/update。

[BaseFluxSetup.setup_optimizations](https://github.com/Nerogar/OneTrainer/blob/23df3832e8f213d64c337e39f9365ba46e99fed0/modules/modelSetup/BaseFluxSetup.py#L49-L53)
为 T5 设置 `disable_fp16_autocast=True`。
[组件 fallback helper](https://github.com/Nerogar/OneTrainer/blob/23df3832e8f213d64c337e39f9365ba46e99fed0/modules/util/dtype_util.py#L55-L76)
在外层 FP16 autocast 中选择独立 fallback dtype，说明文本编码器可以有不同于
DiT 的精度策略。

可借鉴组件隔离及可训练参数的精度约束；不要直接复制 helper 的所有实现假设。
该源码称 CUDA FP32 autocast upcast 属于未文档化行为。本项目若需要严格的
FP32 island，应显式禁用 autocast，并核验输入、权重和运算的真实 dtype。
此外，autocast 对 norm 的常见处理不等于自定义 norm/residual 全部获得 FP32 保护。

## 3. Wan 的模型级 FP32 保护

[WanTransformerBlock.forward](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/models/transformers/transformer_wan.py#L472-L502)
显式使用 `temb.float()`、FP32 norm/modulation、自注意力及 FFN 残差加法。
典型代码为：

```python
norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
```

[FP32LayerNorm.forward](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/models/normalization.py#L84-L93)
将 input/weight/bias 明确转为 FP32，结果再转回原 dtype。
[final norm/modulation](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/models/transformers/transformer_wan.py#L719-L725)
也使用 FP32 计算；它不是 final Linear 全部 FP32 的证明。

该模式可作为各模型 family 的 FP32 guard 模板。cross-attention residual 仍是
普通加法，回转 FP16 仍可能溢出，因此不能直接宣称所有前向风险已消除。

## 4. Flux / Qwen-Image 2.1 的 FP16 clip

[Flux single-stream block](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/models/transformers/transformer_flux.py#L394-L411)
在 residual 相加后对 FP16 输出 `clip(-65504, 65504)`；
[double-stream block](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/models/transformers/transformer_flux.py#L486-L495)
对应分支只裁剪 text/context stream。

确切的 [QwenImage21TransformerBlock.forward](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/models/transformers/transformer_qwenimage21.py#L638-L647)
使用 `gate.tanh()` 并在 block 末尾裁剪 FP16 输出。
[grad-enabled checkpoint 分支](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/models/transformers/transformer_qwenimage21.py#L982-L987)
调用同一 block，因此它是训练 forward 可达的代码。
[QwenImage21ZeroCenterRMSNorm](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/models/transformers/transformer_qwenimage21.py#L170-L185)
还在 FP32 中计算平方均值以及有效 scale `weight + 1`，最后转回输入 dtype。
不能用普通 RMSNorm 替换这个零中心参数语义。

clip 不是 FP32 residual 的等价方案。它将超范围输出饱和，会改变饱和区梯度；
已有 NaN 也不会被正常 clip 修复。可列为需要质量对照的实验，不作为默认补丁。

## 5. Z-Image 已有的保护与空缺

[ZImageTransformerBlock](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/models/transformers/transformer_z_image.py#L252-L283)
使用 tanh gate 和 attention/FFN branch 输出归一化。
其 [RMSNorm](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/models/normalization.py#L551-L568)
普通实现已在 FP32 计算 variance；不需要再次添加同一项“优化”。
本轮所查 block 未找到额外的 FP16 residual clamp，final LayerNorm/modulation/Linear
也不是完整显式 FP32 island。

## 6. Musubi、sd-scripts 与 SimpleTuner

Musubi 固定版本：`f8a1b03794a49239a3539015075f5123d6c07d66`。
[Wan 文档的支持矩阵](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/docs/wan.md#L46-L72)
声明 FP16/BF16 模型可用，矩阵要求模型与训练精度匹配；
[Wan 入口](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/wan_train_network.py#L60-L76)
核验 dtype 配对。共享 trainer 在
[full precision 设置](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/training/trainer_base.py#L1815-L1837)
主动关闭 `full_fp16/full_bf16`；残留 full-FP16 unscale patch 不能视为当前开放能力。
`fp8_scaled` 属于另一个权重量化路径，不是 AMP FP16 稳定化证据。

sd-scripts 固定版本：`690ea7f96c23182352ec63def76d431c6120bd2f`。
[Flux forward](https://github.com/kohya-ss/sd-scripts/blob/690ea7f96c23182352ec63def76d431c6120bd2f/flux_train_network.py#L377-L392)
和 [SD3 forward](https://github.com/kohya-ss/sd-scripts/blob/690ea7f96c23182352ec63def76d431c6120bd2f/sd3_train_network.py#L356-L369)
采用 `accelerator.autocast()`，说明普通 AMP 在其他模型训练实现中存在。
在此次核对的训练文件中未找到额外模型专用 FP16 rescale 配方。

SimpleTuner 固定版本：`d6b5939104d7d09a08e6fe8a63a0f34407004d5c`。
[总精度表](https://github.com/bghira/SimpleTuner/blob/d6b5939104d7d09a08e6fe8a63a0f34407004d5c/documentation/QUICKSTART.md#L107-L150)
将 Flux、Krea2、Z-Image、Qwen Image 2.1、Wan 等列为 BF16；
[Z-Image quickstart](https://github.com/bghira/SimpleTuner/blob/d6b5939104d7d09a08e6fe8a63a0f34407004d5c/documentation/quickstart/ZIMAGE.md#L135-L139)
则写 `bf16 on modern GPUs; fp16 otherwise`。这两处文案不足以建立 FP16
训练配方已验收的结论，本轮未找到与该句配套的专用保护或训练验收。

## 对本项目的后续建议

1. 通用部分采用 FP16 冻结基座、FP32 trainables、原生 autocast/scaler、
   保存/恢复后实参校验和轻量精度报告。各模型声明独立的保护及 attention 合同。
2. 首选 Krea-2 非量化路径，检查实际 residual/modulation/final 的 dtype；
   将 Flux 与 Wan 的代码作为风险定位参考，不复制它们的模型数学。NF4 保留 BF16。
3. 先补权重转换前后有限性及前向输出有限性，再记录有效更新、scaler skip、
   保存/续训和固定 FP16/BF16 训练质量对照。clip/rescale 实验另列，不混入默认保护。
4. 复用启动 OOM 的新进程 swap 调度，但只有 family 精度合同已建立才能开放；
   不通过改变 dtype、batch 或模型语义掩盖容量问题。

PyTorch 的 [AMP 文档](https://github.com/pytorch/pytorch/blob/75d3bcc52b24422d8da9d1700a4feab46329aa2e/docs/source/amp.md#L125-L145)
明确指出 BF16 预训练模型可能超出 FP16 的 65504 范围。GradScaler 在 loss/backward
阶段调节梯度尺度，无法修复前向已产生的 Inf/NaN 或权重转换时已经发生的溢出。
FP32 LoRA 参数、FP32 norm、FP32 residual 和 FP32 final 是不同保护边界。

本报告只提供公开实现依据。没有运行外部脚本、下载模型或核实第三方训练质量，
也没有将任何源码路径外推为图灵兼容或速度/显存收益承诺。
