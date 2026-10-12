# Anima 旧 FP16 集中验收与质量协议

状态：实验；严格12格已完成、0/12通过，质量标准已登记、数据未绑定、质量尚未运行。
日期：2026-10-01。范围：当前 Anima plain LoRA、CMP 90HX SM86。
相关证据：[五步旧FP16对照](../findings/anima_legacy_fp16_comparison_90hx_20261001.md)。
最新结果：[12格集中验收](../findings/anima_native_fp16_acceptance_90hx_20261001.md)，证据自洽通过、严格数值失败。

2026-10-02另行完成[单seed FP16/BF16各200步配对热训](../findings/anima_fp16_bf16_ab200_90hx_20261002.md)，
100/200步采样共16张、固定验证240次；这是独立实验身份，未执行下述三seed50步质量协议，
不改写严格矩阵失败，也不把单seed接近结果当作完整质量认证。

本轮结束05/04的默认扩张，只验收旧 native FP16 一个固定方案：原生 AMP，
基座 FP16，LoRA 参数 FP32，28 block 及 final FP32 保护，初始 GradScaler 1024。
生产默认与原严格门槛保持不变；旧默认 scale、Turing 和长期质量尚未认证。

## 严格12格与来源

复用批次2两张真实缓存输入、4032/4200 native token，交叉 sigma .2/.5/.8 与
initial/nonzero adapter、AdamW 状态，共12格。两个 worker 各6格，逐格独立恢复
参数、buffers、optimizer moments/step及 CPU/CUDA/Python/NumPy RNG，随后执行真实更新。
完整预算内即使某格误差失败也继续记录其余预定格；非finite、运行或来源异常则停止并列未完成格。
不追加候选，不调层、scale、seed或严格阈值。

旧 reference、replay、StateBundle 及其原数值源码哈希原样保留。
新增入口只在私有 worker 中临时替换构建、参数检查与 forward，并恢复原入口；
新执行源码和实际策略独立绑定到冻结计划，不把新实现伪装成旧 source identity。
manifest中的 canonical signature 描述原来源/replay比较合同；实际执行另记 native execution。

固定 FP32 latent/noise/sigma 先混合，调用模型时 noisy/timestep/prompt/padding 转 FP16，
target 保持 `(noise-latents).float()`。仍以转换前 FP32 noisy leaf 的梯度作为比较坐标，
标记 `canonical_fp32_pre_boundary_cast`；实际模型输入的 dtype/digest 另存。
这复现固定随机样本下的生产精度边界，不声称复现生产随机采样全过程。

严格门槛独立保持：prediction relL2≤.01，input/adapter grad relL2≤.02，
两种 grad cosine≥.999，全部 finite。实际 AdamW 更新误差额外记录，没有新接受阈值。
`evidence_integrity`、`strict_diagnostic`、`chain`、`quality` 分别输出；
证据自洽可以和严格数值失败同时成立。六格独立更新不冒充连续六步训练。

## 质量预登记

冻结标准位于 `output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2-legacy-fp16-protocol/quality-protocol.json`。
本轮只登记，`quality_status=not_run`、`future_run_authorized=false`。
未来运行前须绑定完整数据、caption/cache、prompt、代码/配置、初始状态和固定验证 noise 的哈希；
尚未绑定的数据不能写成已冻结或已测试。任何协议修订都创建新身份并保留旧记录。

候选与生产 BF16 对照使用三组配对 seed：20261003/04/05，各50次有效更新、零跳步，
共6次运行、300次更新。未来 GPU worker 总墙钟上限3600s，总墙钟5400s、产物检查预算4GiB，
不自动延长。该窗口只能评价有限训练行为，不能认证长期收敛；实际执行另行制定并授权。
两路径逐 seed 使用相同初始 adapter 张量、optimizer状态、数据顺序及非精度配置，
rank/alpha4、AdamW、lr1e-4、constant scheduler、clip1、batch/accum1、swap20、
torch attention、full checkpoint、关闭 compile，保存 BF16。

训练集至少16张，独立 validation 至少8张，每个4032/4200 token family至少2张。
validation不得包含既往矩阵、训练或调参样本；原两张矩阵样本不再称为未见质量集。
检查点0/25/50使用同一个 BF16 evaluator，固定sigma .2/.5/.8与两个固定noise seed，
保留 max text padding，对样本/sigma/noise等权平均 flow prediction MSE；validation不做更新。

对每个配对 seed 定义终点比值 `R=L_fp16(50)/L_bf16(50)`，改善
`I=(L(0)-L(50))/L(0)`。预设工程非劣化标准为：

- 三 seed 的 `R` 中位数≤1.01，任一 seed 的 `R`≤1.03，所有评估值finite。
- BF16 的改善中位数至少0.005，至少2个seed改善；参考信号不足则 `inconclusive`。
- FP16改善中位数至少保留BF16改善中位数的80%；BF16改善的seed上FP16也须改善。

这些是本协议的工程门槛，不是数学上的通用安全界限；三个seed不支持统计置信度声明。
零或无效的loss分母、评估缺失、未完成预算、跳步、来源漂移或不可判定参考均不能报质量通过。
完整有效运行违反门槛则fail；可复现的数值非finite单独记失败与链路异常。

另外冻结4条目标明确的prompt、推理seed20261021及训练seed20261003，共8张配对图，
同用BF16 evaluator、Euler20步、CFG3。盲评保存分数、顺序键和原图，评价prompt遵循、
结构瑕疵与预先声明的画风/主体目标。每项0为严重失败、1明显更差、2相当、3更好，
候选中位数至少2且无严重失败；参考图不可用或缺少可复核评审则 `inconclusive`。
不要求生成图复现训练caption。质量通过要求数值标准与图像评审同时通过，结论仍限定该窗口。
性能的步时/allocated/reserved/整卡显存/RSS/跳步另表记录，不参与质量阈值。

严格误差失败不会被质量或链路结论改写，质量成功也不会将原严格失败改判为通过。
不得事后改阈值、挑seed、删坏样本、换prompt或选择性补跑。

## 实验入口

新增入口为 [`anima_native_acceptance.py`](../../bench/adaptive_runtime/anima_native_acceptance.py)，
私有执行与窄比较器为 [`anima_native_validation.py`](../../bench/adaptive_runtime/anima_native_validation.py)。
旧通用 probe/replay/comparator/state 和生产训练代码保持原样。
CPU prepare 需要新的输出目录，`--run`才启动两个串行监控worker；拒绝覆盖或续接结果。

```bash
.venv/bin/python -m bench.adaptive_runtime.anima_native_acceptance \
  --source-dir output/runs/<batch2-source> \
  --selection output/runs/<batch2-inputs>/selection.json \
  --weights /path/to/anima-base-v1.0.safetensors \
  --quality-protocol output/runs/<protocol>/quality-protocol.json \
  --output-dir output/runs/<fresh-native-acceptance> \
  --gpu-uuid <verified-90hx-uuid>
```

参数不提供swap/scale/候选搜索开关。预检核对完整旧来源、实际张量、当前执行源码及质量协议SHA。
单worker240s、batch480s、外层540s，4GiB为运行前后检查预算，不是操作系统硬配额。
运行结果不会作为跨模型、跨adapter或Turing自动选择证书。
