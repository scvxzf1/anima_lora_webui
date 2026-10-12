# Qwen 编辑采样文本/参考图 DiT K/V 缓存 A/B/C（2026-10-02）

状态：实现完成；C 未通过预登记数值门槛，未取得实质速度收益；训练预览默认保留 B。
范围：CMP 90HX、BF16、plain LoRA、torch/native SDPA、swap24/slab、32-block Inductor compile。

## 结论

C 接入官方文本和参考图前缀 K/V 缓存，首步 extract、后续 cached。
同一批两张 28 步样张的平均耗时 A=486.562 秒、B=263.642 秒、C=263.207 秒。
C 相对 B 仅少 0.165%，不能作为可靠加速；C 相对 A 快 45.90%，该收益主要来自已经验收的 B。
C 峰值 allocated 比 B 多约 0.504 GiB，reserved 多约 0.736 GiB。

A/B latent 和解码 PNG 像素完全相同；C 两次复跑也完全相同，但 C 与 A/B 不同。
28 步两张图的 latent relative L2 为 0.4401% / 0.3901%，超过预登记 0.1% 门槛。
两步 CFG=3 probe 的最终 latent relative L2 为 1.2409%，同样未通过。
没有为了出具通过结论放宽门槛。缓存路径和训练恢复验收通过不等于数值质量验收通过。

因此训练预览只在 JSON/TOML 样张条目显式设置 `use_kv_cache=true` 时使用 C；
省略或设为 `false` 保留 B。文本 prompt 行和当前 UI 未新增缓存控件。
用户确认方案 B 为默认后，直接调用 `generate_preview` 的 API 也默认关闭缓存，
只有显式传 `use_kv_cache=True` 才使用 C。
本机用户 prompt、训练配置和原训练输出未改，未重启服务或恢复原长训。

## 性能对照

同一进程、同一 DiT/第400步 LoRA、同一冻结条件及种子，运行 A→B→C→C→B→A。
A/B 源码分别冻结归档，B 不是已经修改的实时 sampler。

| 顺序 | 路径 | 竖图 / 秒 | 横图 / 秒 | 批次 / 秒 | allocated / GiB | reserved / GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | A 原逐预测 prepare | 239.451 | 239.145 | 486.558 | 3.955 | 4.195 |
| 2 | B 单次布局初始化 | 127.883 | 127.824 | 263.655 | 3.953 | 4.195 |
| 3 | C B+K/V | 127.642 | 127.611 | 263.207 | 4.457 | 4.932 |
| 4 | C 复跑 | 127.651 | 127.585 | 263.206 | 4.457 | 4.932 |
| 5 | B 复跑 | 127.869 | 127.807 | 263.630 | 3.953 | 4.195 |
| 6 | A 复跑 | 239.454 | 239.138 | 486.566 | 3.955 | 4.195 |

各批次包含进入 inference、两张样图生成、恢复 training、latent 保存和事件落盘。
forward 前后 CUDA 同步；额外保存 target prediction 的 CPU trace，三路使用同样插桩。
forward 计时不含 CPU trace 复制，批次计时包含该开销；这些是统一验收口径，非纯生产延迟。
模型加载、TE/VAE 条件编码、PNG 解码和独立预热不在表内。
与此前 A/B 报告的 480.568/262.233 秒口径不同，不混用两份数据计算 C 收益。

C 两张图合计每轮 extract=2、cached=54，所有32层 K/V 均在 cuda:0；
A/B 均为56次 no_cache。A 去噪内 prepare56次，B/C为0，三路进入/恢复各1次。
第一张图 prefix812 tokens，32层 BF16 K/V 共406 MiB；第二张748 tokens，共374 MiB。
CFG 正负分支缓存各自独立，约812 MiB合计。前一张图完成后即清空，不能跨图累积复用。

C 首次两步预热第一张38.629秒、第二张9.133秒，包含新增编译。
C 预热增加3个 Dynamo unique graphs；所有正式 A/B/C 和 CFG probe 的 compile delta 均为空。
B 正式 forward 平均约4.564秒，C cached约4.556秒，未取得可靠下降。
官方 cached 路径仍执行模型入口 img_in/txt_in、布局/metadata，并且 target attention 仍读取全部前缀 K/V；
24块权重交换也继续存在。本次没有 transfer/wait profile，不能将无提速根因确定为 PCIe 等待。

## 数值与图像

预先登记的 provisional gate 同时要求 finite、shape相同、max_abs≤0.0625、relative L2≤1e-3。
CPU小模型 FP32 容差测试通过不替代真实 BF16 GPU 验收。

| 对照 | 竖图 max_abs | 竖图 relative L2 | 横图 max_abs | 横图 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| A vs B 两轮 | 0 | 0 | 0 | 0 |
| A vs C 两轮 | 0.0625 | 0.00440058 | 0.0546875 | 0.00390088 |
| C vs C复跑 | 0 | 0 | 0 | 0 |

CFG3两步：A/B完全相同；C最终latent max_abs=0.078125、relative L2=0.0124092。
完整 trace 保存逐预测误差，这些是各自演化的 sampler 轨迹，不能称为相同 latent 下的单步误差。

8bit RGB解码后，C对A的竖图 mean abs=0.5255、RMSE=1.0738、最大单通道差68，
横图 mean abs=0.3481、RMSE=0.7091、最大差17；约65.56%/55.30%像素存在至少一个通道差异。
并排图整体内容接近，但没有进行盲评或宽泛质量验收，不能因视觉接近忽略数值门槛失败。

## 实现与恢复

- `preview_kv.py`：复用 family 的首步输入验证，调用官方 extract；后续 cached 独立校验 target-only 输出。
  缓存只属于当前采样调用的一个条件分支，完成或异常显式清空所有层 K/V。
- `sampling.py`：保留 B 连续换块环，新增 API 开关和 CFG 分支缓存；训练 forward 未修改。
- `training_preview.py`：仅从样张条目显式读取实验开关，默认 B。
- `preview_conditions.py`：预处理前拒绝非布尔开关，避免字符串 `"false"` 被当作启用。
- 官方 causal/eval 模型、静态 LoRA/LoHa/LoKr和已知 processor 才缓存；
  动态 hook、未知 processor、自定义 forward、训练/非causal模型回退无缓存。
  helper 固定条件的 tensor identity/version，条件替换或普通 tensor 原地修改需要新缓存。
  inference tensor 没有 version counter，原地修改检测仅限普通 tensor；生产调用本身不修改条件。
- cache在下一张样图/下一轮重新建立，不跨 optimizer 更新，不缓存训练反向图。

真实异常探针在第二张图的第二次 cached预测失败：两个prompt缓存全部清空，
待传输任务=0，DiT/network恢复 training，进入/恢复各prepare1次。
之后同一模型执行full-checkpoint forward/backward/AdamW：loss=0.74990249，
梯度有限且16768985元素非零，adapter update L2=0.03684527，基座保持冻结。
目标latent为[1,64,8,8]（128×128），保留完整真实参考条件；这不是原1024数据集长训或optimizer/global-step续训验收。

CPU独立官方API诊断13组、78次分支forward中，helper/direct cached输出逐位一致；
FP32官方cached/uncached最大prediction误差6.56e-7，BF16 CPU小模型逐位一致。
CPU支持双reference、不同text/targetshape、CFG、eager-backendcompile及静态adapter非零delta。
小模型CPU不能说明真实CUDA大模型的漂移根因。

### 真实 GPU 同输入官方对照

另在独立进程加载同一权重/条件，固定相同latent/sigma，比较新helper、独立构造官方API
和无缓存forward。先运行32-block Inductor，再显式解包为同一模型eager；没有改权重。
第二步输入统一来自compiled无缓存首步CFG3与同一官方scheduler，避免不同轨迹输入混淆。
四个隔离缓存（helper/native各正负两支）共1624 MiB，仅用于诊断，不代表生产CFG占用。

在compiled和eager的8次分支forward中，helper/native输出与全部32层K/V均逐元素相同；
改变target值和sigma后重新extract，prefixK/V仍逐元素不变。两阶段缓存清空、training恢复均通过。
这验证了新接线复现官方缓存，而不是由错误的prefix/slot/CFG分支引入偏差。

| 路径 | 首步正/负 prediction relative L2 | 次步正/负 relative L2 | 对照 |
| --- | --- | --- | --- |
| Inductor | 0.006267 / 0.005294 | 0.004097 / 0.004170 | 官方缓存 vs 同输入无缓存 |
| eager | 0 / 0 | 0.003995 / 0.004197 | 官方缓存 vs 同输入无缓存 |

eager首步extract与无缓存完全一致，后续cached仍有约0.4%差异，因此偏差不只来自编译。
Inductor首步也因cache模式产生计算路径差异；此独立诊断包含首次trace，不是正式热态性能轮次。
没有进一步定位到具体CUDA kernel，不能声称已证明全部误差只来自某个GEMM或attention实现。
官方缓存减少query/projection行数，会改变BF16计算路径；本轮证据支持保留实验选项，
不支持将官方API一致性当作A/B数值门槛通过。

对照目录：`output/runs/qwen-kv-gpu-oracle-20261002-r1`，exit0、summary状态`ok`，
这里的ok仅表示helper/native exact与生命周期通过；无缓存偏差仍完整记录，未放宽门槛。
复现命令与ABC相同config/checkpoint，入口换成`kv_gpu_oracle.py`，使用新的不存在输出目录。

## 产物与复现

完整 A/B/C：`output/runs/qwen-sampling-kv-abc-20261002-r1`。
进程exit0、summary状态`numerical_review_required`，表示测试执行完成且数值门槛未通过。
该目录保留snapshot源码/输入哈希、每步predictions、latent/PNG、compile counters、events.jsonl，
以及`abc_comparison_image0.png`/`abc_comparison_image1.png`。
冻结条件复用此前r4的`conditions.pt`；记录了本次hash，但r4旧summary没有该文件历史hash，
因此只能证明本次ABC共享同一冻结文件，不能单凭manifest证明它从r4至今从未改变。

```bash
CUDA_VISIBLE_DEVICES=GPU-806debd9-d254-da57-d515-ccae15b4ac6a .venv/bin/python -u \
  scripts/qwen_image_2_1/sampling_kv_abc.py \
  --config output/runs/9-27-50x-qwen21-edit-20261002-073957/config.runtime.toml \
  --checkpoint output/runs/9-27-50x-qwen21-edit-20261002-073957/training_output/9-27-50x-qwen21-edit-hot1000-20261002-step00000400.safetensors \
  --output output/runs/qwen-sampling-kv-abc-recheck --order A B C C B A
```

只能在目标GPU没有其他compute任务时运行，输出目录必须不存在，不覆盖已有结果。
完整过程通过UUID验证目标CMP90HX，只暴露cuda:0，不使用显示GPU1050。
训练CLI、UI、权重、采样步数、尺寸、原配置及参考图输入均未为C降级。
参考图来自a-2原图；b-2编辑后图是训练目标，不作为采样输入。
正式样张384×512/seed42、512×352/seed43，Euler28、CFG1；同第400步plain LoRA。

最终定向测试：以下统一命令76项通过（22.79秒），另有独立审查46项通过。
两组覆盖相互重叠，不应相加声称总测试数。

```bash
CUDA_VISIBLE_DEVICES='' timeout 60 .venv/bin/python -m pytest -q \
  tests/test_qwen_image_2_1_preview_kv.py tests/test_qwen_preview_kv_opt_in.py \
  tests/test_qwen_sampling_kv_abc_harness.py tests/test_qwen_image_2_1_preview.py \
  tests/test_qwen_image_2_1_family.py
```

ABC快照之后sampler新增非布尔开关的拒绝检查，并在用户确认后将API默认值改为false；
正式运行及GPU官方对照显式启用缓存，缓存算法与`preview_kv.py`未改变。
最终caller与API都默认关闭，是验收后的生产选择。
仅已测plain LoRA完成本次真实GPU矩阵；不外推全部adapter、Flash、其他GPU、INT8 swap或所有多参考图组合。
