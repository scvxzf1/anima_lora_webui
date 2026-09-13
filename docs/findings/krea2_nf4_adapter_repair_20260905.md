# Krea-2 NF4：DoRA / OrthoLoRA / ReFT 修复验证

日期：2026-09-05

状态：实验修复；不是生产能力扩展

适用版本：本地未提交工作树，不能仅凭 HEAD 复现

前序记录：[170HX adapter 变体测试](krea2_adapter_variants_170hx_300step_20260904.md)

## 结论与范围

本轮修复了三个变体对 NF4 逻辑权重、数值精度和运行生命周期的处理，
并补充实际 adapter 文件保存/重载及原生训练状态恢复测试。
三者均在 CMP 170HX 上通过 3-step Krea-2 全模型训练及 checkpoint 保存，
随后从各自 step3 原生 state 成功续训到 step5。六个进程全部正常退出，未遗留训练任务。

这不等于 300-step 长训、预览质量或通用独立推理通过。生产 Krea-2 family registry
仍限制为 plain LoRA；实验 runner 仅在本进程放宽 `supported_network_specs` 和
`plain_lora_only`，其余训练入口与兼容性检查保留。未修改 Z-Image 的 NF4 支持边界。

## 修复内容

### 共享权重读取

`networks/lora_modules/weight_access.py` 区分逻辑矩阵与 bitsandbytes 打包存储，
使用 `quant_state` 反量化并校验模块声明的逻辑维度。识别 Params4bit 子类和恢复后的
quant state，不依赖单一类名相等或 `bnb_quantized` 标记。缺失必要量化状态时拒绝读取；
dense 返回值也不与基座共享可写存储。dense 路径不强制导入 bitsandbytes。

### DoRA

- 初始化按逻辑 dense 权重计算 magnitude 和基础行范数，释放临时大矩阵。
- 保留 FP32 基础行范数缓存的原始值，避免 `.bfloat16()` 先舍入再 `.float()` 的伪 FP32。
- 用无 bias 的 rank-sized NF4 matmul 计算交叉项；低秩 Gram 和范数代数关闭 autocast、保持 FP32。
- 使用残差形式，零初始化在 BF16 含 bias 情况下也精确返回原始输出。
- `multiplier` 缩放整个 DoRA edit；0 为精确基座，1 为完整 DoRA，中间/负值不再只缩放方向分支。
- 原有 detached-norm 训练规则保留，不引入另一种 DoRA 梯度定义。
- NF4 merge/fuse 明确拒绝，防止写坏打包权重。这里的 fuse 不是 fused AdamW 或 FlashAttention。

### OrthoLoRA

- 从逻辑反量化矩阵初始化 SVD bases，避免把 packed storage 当成 `[out, in]`。
- OrthoLoRA / OrthoHydra 在初始化前检查 rank；SVD 使用基座所在设备，不偷偷选择默认 GPU。
- Cayley solve 强制 FP32，BF16 adapter cast 不再使求解输入降成 BF16。
- 尊重 enabled 和零 multiplier；原始 NF4 基座不被改写。
- 导出 `.safetensors` 是蒸馏后的标准 LoRA。续训必须使用保留 bases、`S_p/S_q/lambda_layer`
  及 optimizer/scheduler/RNG 的原生 state 目录。BF16 导出只作容差等价，不声称 bitwise exact。

### ReFT

- 维度从 block 的 `x_dim/features` 获取，缺省才回退 model config；校验 rank 不超过 residual width。
- Krea `SingleStreamBlock` 明确将 `_forward` 暴露为 ReFT 目标，干预位于 checkpoint 和 resident
  compile 内部，不改变公共 block 对象及 swap backward hook 的归属。Anima 保留原来的 forward 目标。
- apply 幂等；重复安装第二个 ReFT、compile 后才安装 ReFT 均明确拒绝。
- 网络 `set_enabled()` 同时控制 ReFT，零 multiplier 精确关闭；QR 初始化不选择默认 CUDA 设备。
- 真实文件 round-trip 发现原实现恢复 `alpha` buffer 却未恢复 Python `scale`，导致输出显著改变。
  已在递归状态加载时同步恢复每块 scale；alpha buffer 固定为浮点类型，避免整数构造时截断小数 checkpoint。

## CMP 170HX 三步验证

实验根目录：`output/runs/krea2-nf4-repair-20260905/125447/`。
启动脚本：`output/runs/krea2-nf4-repair-20260905/run_smoke.py`。

| 项目 | 设置 |
| --- | --- |
| GPU | NVIDIA CMP 170HX 64GB；PyTorch 可见 63.39 GiB；按 UUID 绑定，未使用 RTX 3080 |
| PyTorch / CUDA | `2.12.0+cu130` / `13.0` |
| 权重 | 本地 Krea-2 self-contained NF4；base compute / mixed precision 均 BF16 |
| attention / optimizer | `attn_mode="flash"` / AdamW `fused=True` |
| checkpoint / swap | full checkpoint / swap8，slab restore，BF16 transfer |
| compile | fixed-seq resident scope，Inductor default mode |
| 数据 / 优化 | 沿用前序单图 1024 bucket cache；batch1，seed114，lr2e-4，constant |
| adapter | LoRA rank16/alpha8；ReFT rank16/alpha16 |
| ReFT blocks | `0,20,21,22,23,24,25,26,27`，同时覆盖 resident compiled block 与 swapped tail |
| 保存 | step3 adapter + 原生 state；最终 adapter + 原生 state |

| 变体 | 完成步数 | loss 平均值 | step3 recent s/step | peak allocated / reserved GiB | 进程总耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| DoRA | 3/3 PASS | 0.09349446 | 3.764 | 9.632 / 11.350 | 340.174 s |
| OrthoLoRA | 3/3 PASS | 0.09358871 | 3.674 | 8.250 / 9.506 | 314.893 s |
| LoRA + ReFT | 3/3 PASS | 0.09401993 | 3.599 | 8.931 / 10.521 | 113.179 s |

`recent_s_per_step` 来自 progress JSONL，不含初始化和保存，三步不足以建立稳态性能结论。
显存为 PyTorch CUDA peak，不是系统内存或 NVML 全进程峰值。CPU 权重读取/SVD 初始化仍有明显开销，
本轮没有宣称减少初始化耗时。三个 `run_end` 均为 `status=ok, final_step=3`。
逐步 gradient-flow 记录中全部 28 个 block 分组均无缺失梯度，分组 gradient/update norm 非零；
这只是分组级证据，不意味着每个参数在零初始化的第一步都必须有非零梯度。

## 原生状态续训与文件核验

续训目录：`output/runs/krea2-nf4-repair-20260905/resume-130835/`。
脚本：`run_resume.py`、`inspect_checkpoints.py`，均在本轮实验根目录。
三个新进程分别加载 step3 的模型、optimizer、scheduler、sampler 和 RNG state，
训练入口确认跳过前三步后，只执行 step4/5，保持原来的 NF4/Flash/checkpoint/swap/compile/fused 配置。
ReFT 的最终 alpha/scale 加载修复由这轮全模型 resume 及非默认 alpha 单测共同覆盖。

| 变体 | 恢复并完成 | step5 当前 loss | step5 recent s/step | peak allocated / reserved GiB | 进程总耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| DoRA | 3 -> 5 PASS | 0.10540245 | 3.827 | 9.632 / 11.406 | 301.591 s |
| OrthoLoRA | 3 -> 5 PASS | 0.10575544 | 3.706 | 8.221 / 9.584 | 302.949 s |
| LoRA + ReFT | 3 -> 5 PASS | 0.10458524 | 3.468 | 8.931 / 10.639 | 77.724 s |

所有 optimizer 参数 state 的 `step` 都是 5，`train_state.current_step` 也是 5；
step4/5 的 28 个 block 分组仍无缺失梯度。源 step3 与新 step5 的原生模型文件对比：

| 变体 | 原生 tensor keys | 导出 tensor keys | 续训后变化的 tensor 数 | 变体证据 |
| --- | ---: | ---: | ---: | --- |
| DoRA | 784 | 784 | 588 | 原生 196 个 magnitude；导出 196 个 `dora_scale` 均非零 |
| OrthoLoRA | 1176 | 588 | 588 | 原生各 196 个 `S_p/S_q/lambda_layer` 均非零；导出为 standard LoRA |
| LoRA + ReFT | 624 | 624 | 419 | 9 个 block 的 learned-source weight/bias 均非零；27 个 ReFT 参数 tensor 均继续更新 |

所有原生和导出 tensor 均有限，无 NaN/Inf。详细核验保存在续训目录的 `checkpoint_audit.json`。
Ortho 导出仍可带 `ss_network_spec=ortho` 来源标记，但张量布局已经蒸馏，不能据此认定保留了 Cayley 续训状态。

小模型测试验证了保存前后输出及下一次 optimizer update 的精确一致；全模型本轮没有额外执行
不中断的 5-step 对照，因此只宣称真实恢复/更新/保存链路通过，不宣称全模型逐位续训一致。
不同 step 的输入和噪声不同，不能把此处 loss 数字或三步平均值当作质量排名。

## 测试覆盖

以下四个修复测试文件连同 construction、custom autograd、registry、Krea compile/selective checkpoint
和 self-contained NF4 既有回归，共 **195 passed**（CPU 隔离 GPU，12.76 s）。

adapter 源文件、application/builders 和新增测试的 Ruff check 通过。额外检查 Krea `dit.py` 时，
发现既有 `TextFusionTransformer.forward` 第 313 行变量 `l` 触发 E741，位于本次四行改动之外，未顺手重命名。
文档完整性检查为 7 passed / 1 failed；唯一失败是
既有 `docs/findings/anima_dual_gpu_parallel_probe_20260904.md` 前 25 行缺少状态标签，未改动该无关文档。
本次文档的文件链接、索引可达性检查通过。

- `tests/test_nf4_adapter_contracts.py`：权重读取、DoRA 精度/梯度/强度、dense merge parity、
  NF4 merge 拒绝、Ortho 初始化/启停、原生 adapter+AdamW 恢复。
- `tests/test_krea2_reft_contracts.py`：真实小 Krea block 的 NF4 checkpoint 输出/梯度/dropout 等价，
  saved-tensor 边界、compile 目标及重编译、非默认 alpha 加载、原生优化器续训。
- `tests/test_krea2_nf4_adapter_persistence.py`：真实 writer -> safetensors -> factory -> strict load；
  DoRA 和 ReFT 输出精确一致，Ortho BF16 蒸馏按 `rtol=2e-2, atol=1e-2` 比较。
- `tests/test_krea2_nf4_adapter_variants.py`：首次修复的 NF4 smoke 回归。

CPU AOT 测试仅将 CUDA-only attention 探测替换为 CPU SDPA，以验证真实 block 的 compile/checkpoint
契约；全模型硬件测试使用真实 Flash 路径，不能将 CPU stub 称为 Flash 验证。

## 已遇问题与保留边界

1. 前序 DoRA packed shape、Ortho packed SVD、ReFT 缺少 `x_dim` 的失败保留在旧报告，不覆写历史。
2. 前序另一次 ReFT 修复探针在 step0 遇到 `/tmp` 下 PTX 的 `OSError: [Errno 5] Input/output error`。
   这不是已证明的 NF4 算法错误，也不能算通过。本轮 TMPDIR/Inductor cache 放在实验所在数据盘，未清理用户文件。
3. 本轮 ReFT round-trip 曾出现最大绝对输出差 `0.34765625`；定位为 alpha/scale 不同步，修复后精确通过。
   新增每块不同 alpha 测试又捕获整数 buffer 截断 `0.5 -> 0`，已一并修复。
4. 未重跑 300 步、未采集新预览，不给三种变体做质量排名；旧 300-step 报告中的其他变体样图不能代替本轮证据。
5. 未开放 Krea-2 WebUI/独立推理能力，未声称 NF4 merge/fuse 可用。ComfyUI vendor 副本尚未同步，
   发布节点前需运行 `python tasks.py vendor-sync` 并复测节点加载。
