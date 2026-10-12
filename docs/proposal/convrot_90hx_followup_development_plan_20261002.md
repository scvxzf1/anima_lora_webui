# ConvRot：90HX 后续开发与验证计划

状态：用户已授权P0两处生命周期修复与90HX短测；P0实现、9格生产8步采样复测、9格确定性插入预览对照与定向回归已完成，复测见[修复报告](../findings/convrot_lifecycle_fix_90hx_20261003.md)。其余阶段保留为后续计划；用户明确暂不进行500步长测。依据：[已完成90HX验证](../findings/convrot_90hx_validation_20261002.md)。原S0–S5筛选流程已收尾，500步长测因无合格候选跳过。本计划保持原数值门槛，不将失败结果改写为通过。

优先顺序是 **采样内存生命周期 → 确定性数值参考 → 反向候选 → 同预算实用性 → 条件长测**。当前Anima1024/rank32在90HX上BF16无需换层，1.572秒/步，W8A16为1.739秒/步；没有理由先把W8A8接成生产默认。

## P0：修复训练→采样→训练的内存边界

拆成两个独立改动，先单独验证，再组合；不取消managed payload的placement保护。

| 改动 | 责任与位置 | 验收证据 |
|---|---|---|
| 已完成step的graph释放 | 训练loop/采样调度边界；将采样所需诊断量detach后，释放不再使用的output/loss图引用；业务逻辑放小模块，热点文件只编排 | swap0量化组在加载VAE前不再被旧训练图占住约1.57GiB；下一步gradient/update与未采样控制一致 |
| offloader-owned CPU park/restore | `library/runtime/offloading.py`及payload/reconfigure模块；新增显式生命周期API，等待pending futures/streams后迁移managed payload，可靠恢复resident/swapped状态 | swap12采样decode前managed GPU payload不再滞留约0.94GiB；恢复后placement合法且下一step正常 |

设计前亲自读取实际修改的源码；保持CPU master、storage复用、pinned memory和异型权重布局契约。park/restore必须覆盖异常退出与VAE OOM，不能留下半恢复状态；多次park/restore行为明确。不要在autograd backward中无条件清空ctx常量；需要retain_graph/create_graph的调用仍须有效。`empty_cache()`只释放allocator缓存，不能代替清理活引用。

定向CPU测试覆盖offloader状态、重复调用、异常恢复和managed payload所有权；CUDA小测试覆盖graph保留语义及checkpoint。随后在目标90HX做BF16/W8A16/W8A8 × swap0/12/26，共9个新进程用例，先各2步smoke再8有效更新，第4/8步1024采样。保持原输入、adapter、optimizer与FP32/TF32规则，锁定UUID，不改风扇/功耗。

必须记录step完成、graph释放、park完成、VAE加载、decode结束、restore完成、下一step结束的allocated/reserved/phase peak，以及managed payload bytes和offloader状态。以完整train+sample的峰值衡量收益，不能只比较结束时allocated；对照原证据BF16 swap0约5.57GiB、W8A16约6.25GiB。释放的两个局部占用不能简单相加当作最终峰值改善。

采样前后验证有限值、adapter/optimizer step、保存与恢复，并用deterministic replay对比含采样与无采样的共同训练输入。采样RNG必须隔离或还原。生产dataloader游标是否精确恢复另列任务，不能用受控replay替代生产游标验收。

## P1：建立完整确定性数值参考

现有单样本deterministic Flash控制差值为0，下一步扩展到完整18样本。先跑B0/B2自身两个独立进程重复；逐层梯度、单步AdamW更新、参数名称/shape/dtype、零范数和finite均检查。任何缺样本/缺gradient都fail closed。

若确定性栈自身仍超门槛，先排查RNG、初始化、checkpoint、matmul设置和offloader，不评价候选质量。通过自校准后，依次重测B1/B2对B0、E3对B2、N1对其固定FP32-backward父方案和B0；N2当前拒绝，不优先耗费资源重跑。冻结原output/loss/gradient/update及逐层门槛；如另立训练质量协议，先说明目的和准则，不能悄悄放宽数值等价标准。

确定性数值栈与默认Flash吞吐栈分别记录manifest；不能将deterministic诊断耗时当默认生产速度，也不能把一次output完全相同称为完整训练等价。完整矩阵通过后才执行3 seeds ×20连续轨迹，比较loss/gradient/update/参数漂移；trajectory模块目前只是准备代码，未经该长轨迹GPU验收。

## P2：反向精度和算子优化的有限开发

N1（BF16 dX）保留为待核验候选。当前梯度偏离与B2自身重复量级相近，但AdamW更新不能认证；先完成P1再优化。N2（INT8 dX）已有额外逐层梯度误差，按原门槛拒绝，除非后续提出独立且预登记的质量研究问题。

E3可继续优化真实热点，但前向微测加速不能替代整步收益。保留code/scale/output精确测试，尤其scale舍入、crop/padding、reverse layout和CUDA device guard。以CUDA Event作CMP局部诊断，注明CUPTI不可用，不宣称已取得完整CUDA trace。

按M/K/N矩阵类别考虑混合W8A16/W8A8，仅在对应类别有整步贡献证据时设计路由；先固定scope与Hadamard，不同时叠加regular RHT、compile和新精度，以免不可归因。任何新增policy默认关闭，不能写进生产checkpoint metadata而没有恢复兼容设计。

## P3：重新验证实用收益

P0稳定后，重测固定swap12与统一9.1458GiB预算两个比较。后者逐模式寻找实际最小可用swap，再在各自最优合法swap上比较；不能强迫BF16多换层制造量化优势。每模式三个独立进程轮次，每轮10warm+30measure，交替顺序；记录完整train+sample峰值、分配/驻留、温度/时钟与数值栈身份。

默认基准仍为BF16 swap0。候选须满足原计划实用性门槛，至少有可解释且可重复的速度/预算收益；若只是省训练驻留却采样峰值更高，不能推荐。若90HX本场景没有收益，明确结论并停止扩大矩阵；更大模型或更严显存预算是新的实验范围，不能由本轮外推。

候选真正通过后再做adapter apply之后的compile兼容、采样、保存/续训和policy身份校验；compile与swap组合单独验收，不改变生产默认。

## P4：条件500步训练验收

仅在数值、生命周期、保存/续训与同预算实用性全部合格时运行。BF16与一个胜出候选分别500有效optimizer updates，250/500步采样，1024×1024、相同60张数据、同初始LoRA/optimizer和样本计划；提示词沿用数据集caption衍生的4caption×2seed manifest。记录完整峰值、吞吐、finite/跳步、样本图、checkpoint及optimizer状态。

新增held-out质量集需先定义与训练集隔离及打分协议；当前60张训练集只能作训练诊断。500步完成也不自动代表更广模型、分辨率或硬件质量等价。没有候选通过时状态仍为`SKIPPED_NO_ACCEPTED_CANDIDATE`。

## 交付与执行边界

每阶段独立结果目录、机器可读状态、source/input/config hash、失败原文和结论；保留本轮原始数据与失败日志。实际修复者自测后由不同代理独立验收；主代理审阅关键diff和证据。现有工作区含大量他人改动，实现时隔离工作树，明确文件归属，不回退、不清理运行数据。

本文件是可审阅的后续计划，不表示已授权启动新增长训练或执行全部修复。开始下一轮时先根据用户指定范围推进；不下载大模型、不更改GPU功耗、不自动启动500步任务。磁盘空间有限，新完整梯度/轨迹矩阵先核容量，避免重复保存可重用的相同tensor。
