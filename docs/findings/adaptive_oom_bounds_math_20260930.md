# 动态 OOM 上限与递增重试的静态数学论证

状态：条件证明、有限域枚举和 CPU 小夹具，2026-09-30。
范围：正式训练 supervisor 的启动 OOM 策略；未执行 CUDA、真实 OOM、底模加载或训练。
本文补充并替代[旧静态契约报告](adaptive_precision_oom_static_proof_20260924.md)
中固定步长、尚无 host 上限的描述；不修改生产策略，也不扩大中途恢复能力。

## 1. 证明对象与前提

主要源码：

- [`capacity.py`](../../library/training/auto_block_swap/capacity.py)：加载前结构信息与加载后核验。
- [`policy.py`](../../library/training/auto_block_swap/policy.py)：CPU 增量预算模型。
- [`supervisor.py`](../../library/training/adaptive_runtime/supervisor.py)：冻结配置和初始上限。
- [`retry.py`](../../library/training/adaptive_runtime/retry.py)：上限收紧、递推和停止条件。
- [`training_runtime.py`](../../library/training/adaptive_runtime/training_runtime.py)：测量与更新边界。

训练入口校验给出整数初值 `s_0>=0`、整数起始增量 `d>=1`、整数尝试预算
`1<=A<=16`；本轮证明要求 worker 调用能返回、结构化结果如实反映生命周期，且
读写/进程管理未被无限期阻塞。正式入口固定 full checkpoint，不改变 batch/accumulation，
不开放其它内存计划变化。通用 `next_plan()` 的显式 batch/checkpoint 扩展不纳入该递推。

证明结论是：候选有界、有限、无重复，已观测资源上限不被后续候选突破，失败不会
自动改变精度或越过更新起点。不是 GPU 可运行性、峰值内存、吞吐或数值质量证明。

## 2. 模型能力与用户预算

令真实可交换容器长度为 `N`，当前三个适配器的合法上限为：

```text
L = N - 2, N >= 3
```

这是当前交换 API 的结构约束，不是所有模型的普遍定理；更换交换实现时必须重新
确认。它也不表示最后两块永远驻 GPU：共享 offloader 初始化前 `N-s` 个 GPU 槽，
尾部 `s` 个冻结权重在 CPU，前后向逐步交换参与块。不能按固定尾块身份解释 `N-2`。

加载前 Anima 从 safetensors header 检查 28/40 块布局，Krea-2 从生产加载配置读取
28 块，Z-Image 从本地配置读取 `n_layers`，单文件路径复用官方 30 层 header 校验。
加载后按实际 `blocks`/`layers` 长度复核，结构不一致停止。元数据一致不证明权重
内容身份相同；结构上限允许也不等于该变体的全部训练能力已被放行。

设用户预算为非负整数 `q`，`q=0` 是自动哨兵，而非禁止交换：

```text
Q = L, if q = 0
Q = q, if 1 <= q <= L
C_0 = min(L, Q)
require 0 <= s_0 <= C_0
```

`q>L` 被拒绝，不能把它静默解释为授权任意合法值。由于初值满足上限且每次推进
都取 `min`，在未知 host 预算的启动阶段，所有候选都满足结构及用户预算约束。

## 3. 时变主机上限的归纳不变量

令 `C_k` 为第 `k` 次尝试启动前已知有效上限，`h_k` 为本次加载后的整数预算测量，
或未知值 `None`。通过字段一致性和范围校验后，更新为：

```text
D_k = C_k,                 if h_k is unknown
D_k = min(C_k, h_k),       if h_k is known
```

`D_k` 是观测后的上限；若 `s_k>D_k`，立即停止 `host_limit`，没有下一个 worker。
只有满足重试资格时才令 `C_(k+1)=D_k` 并生成下一个候选。

**基例：** supervisor 要求 `s_0<=C_0<=L` 且不超过用户预算。
**归纳步：** 已知 `s_k<=C_k`。新观测可能令 `D_k<s_k`，此时停止；否则
生成 `s_(k+1)<=D_k=C_(k+1)`。因此所有实际启动的下一候选都满足之前已观测的上限，
而上限序列只会下降、不因随后更宽松的测量而回升。

注意：不能证明 `s_k<=D_k` 对每次观测都成立，更不能证明历史计划小于未来预算。
反例为 `s_0=3,C_0=8,h_0=2`：计划按当时信息合法启动，观测后停止，不能继续尝试。
这也是快照预算无法保证未来系统压力的直接例子。

加载 OOM 可能发生在测量前。正 swap 的 `model_load` OOM 允许 `h_k=None`，
仅有结构/用户上限及进程级主机保护，不称为 host 安全认证。正 swap 的成功或
非加载阶段 OOM 若仍缺测量，则停止 `missing_host_budget`。初始零 swap 的 GPU
常驻模型不套用 CPU 加载后的增量公式，可保留未知；这不是所有未知预算都可放行。

## 4. 指数递推、尝试界与最后机会

设第 `k` 次结果允许推进，剩余可启动次数 `r_k=A-k-1>0`，观测后的上限为 `D_k`。
仅当 `s_k<D_k` 时，正式 supervisor 的规则为：

```text
s_(k+1) = D_k,                         if r_k = 1
s_(k+1) = min(D_k, s_k + d*2^k),      if r_k > 1
```

因 `d*2^k>=1` 且 `s_k<D_k`，两种分支均给出 `s_(k+1)>s_k`，同时
`s_(k+1)<=D_k`。full checkpoint、batch 和 accumulation 不变，所以完整计划也不重复。
结合有限 `A` 与整数计数，实际启动次数 `K` 满足：

```text
1 <= K <= min(A, C_0 - s_0 + 1)
```

失败、预算变小、取消、未知状态或精度校验拒绝只能更早停止。通用 runner 的
`seen` 是额外保护；正式入口在上述前提下严格递增已经足以排除循环。

### 固定上限的闭式

若上限始终为 `C`，前序均为允许阶段的 CUDA OOM，且没有其它停止条件，则在
最后机会跃迁之前，由几何级数得：

```text
s_j = min(C, s_0 + d*(2^j - 1))
```

归纳使用 `sum(k=0..j-1,2^k)=2^j-1`；一旦到达 `C`，下次 OOM 没有授权的
swap 增量，立即停止。令 `r` 为满足 `d*(2^r-1)>=C-s_0` 的最小非负整数，则
`K<=min(A,r+1)`。当 `s_0<C`，可写成
`r=ceil(log2(1+(C-s_0)/d))`；测试用整数不等式求 `r`，不依赖浮点 `log2` 舍入。

如果能推进到最后机会，末次候选为 `C`，不再服从未修改的几何级数。例如
`A=3,s_0=0,d=2,C=10` 时是 `0,2,10`，不是 `0,2,6`；`A=4,C=38` 时是
`0,2,6,38`。`A=1` 只运行初始值，不应宣称最后总会尝试上限。

以上证明的是 worker 调用次数，不是严格端到端墙钟上界。单 worker 超时不包含
所有磁盘 IO、进程创建与清理成本；没有对这些操作的有界假设，不能直接声称总时间
必定小于 `A*timeout`。

## 5. CPU 增量预算的整数模型

在底模已加载到 CPU、尚未建立 offloader masters 的测量点，令 `b_i>0` 为
inventory 统计的第 `i` 块冻结基础 `module.weight` 字节，`V` 为当时可用物理内存，
`R` 为保留量，定义：

```text
M = sum(i=0..N-1, b_i)
T(s) = sum(i=N-s..N-1, b_i)
F(s) = R + 2*max_i(b_i) + M + T(s), 1 <= s <= N-2
```

当前 dense master 策略以 `M` 建模全部块的额外 masters，以 `T(s)` 建模尾部副本，
以两倍最大块计暂存预算；已有 CPU 模型占用已经反映在 `V` 中，不能再把 `V`
当作加载前空闲内存。`host_swap_limit()` 返回 `F(s)<=V` 的最大合法正整数 `s`，
若无正数满足则返回 0。

因为 `F(s+1)-F(s)=b_(N-s-1)>0`，可行正计数构成前缀，一次失败后停止扫描是
该模型下正确的；相等端点 `F(s)=V` 可行。固定 `b,R` 时提高 `V` 不会降低上限；
固定 `V,R` 时逐块增大 `b` 会使每个 `F(s)` 非降，因此不会提高上限。

返回 0 只是没有正 swap 通过此公式，既不认证 zero-swap 可行，也不说明 zero-swap
不可行。公式可能已经无法容纳 `M`，仍返回 0；zero-swap 路径不建立同一 masters 布局。

普通 FP16/BF16 weight 字节为 `2*numel`，FP32 为 `4*numel`；混合块必须逐 weight
统计实际 `element_size()`，不能因策略名 `bf16` 就把 FP32 master 算成两字节。
本轮 CPU 小夹具仅验证 inventory 表示；不加载 NF4，不外推量化策略的表示和费用。

**不包含的保证：** `b_i` 不是全部 parameter/buffer 或模型 RSS，公式不精确覆盖
adapter、optimizer、activation、allocator 碎片、并发进程变化和全部传输缓冲峰值。
预算满足只证明上述模型不等式，运行中主机保护仍必须保留；不构成“不 OOM”证明。

共享 helper 还有显式 sparse 分支：`P_s=[0,s) union [N-s,N)`，用参与集字节替代
`M`。由于 `P_s` 随 `s` 嵌套且尾部成本递增，该分支模型也单调；正式 adaptive
入口未开启 sparse，本轮枚举不把 sparse 当作已验收的恢复能力。

## 6. 精度与更新起点不能被预算证明替代

worker 从冻结配置生成，仅改授权资源字段，并重算精度请求摘要；已安装模型还需
验证 dtype manifest。OOM 递推和 cap 收紧没有改变 FP32 名单、初始 scale 或 backend。
摘要依赖低碰撞概率来防意外漂移，不是对抗性证明，也不认证梯度/质量等价。

精度摘要故意不含 batch/checkpoint 等全部训练语义字段；“摘要相同”本身不能证明
这些字段不变。正式 supervisor 固定相应计划，必须同时检查策略分支，而非只看 hash。

`optimizer_started` 在调用更新前 sticky 置位，正式 worker 固定
`committed_checkpoint=None`。因此更新起点后的 CUDA OOM 缺少可恢复提交点，必然停止，
即使新的内存计划完全满足预算也不能自动重启。预算不等式没有证明数据游标或部分更新可恢复。

## 7. 不完备搜索与性能反例

对任意实际成功谓词 `S(s)`，本实现没有证明它随 swap 单调。fake runner 可设
`S(4)=true` 而其它点失败，指数候选 `0,2,6,8` 全部失败却存在未测成功点。
因此预算耗尽应解释为所试候选失败，不是所有配置不可行。

旧 AUTO 也只是在有限候选中选择：若最大值 8 失败就停止，同样不能排除 4 成功。
没有单调性前提，二分或最大点失败不能证明全域。最后机会跳到上限提高结构范围覆盖，
不构成完备搜索或保证启动的算法。

性能也没有单调定理：纯表格 `t(2)=12,t(4)=9,t(6)=11` 就同时否定“更多 swap
一定更快”和“一定更慢”的普遍命题；这不是实际 GPU 步时测量。

## 8. 可执行证据

新增 [`test_adaptive_oom_math.py`](../../tests/test_adaptive_oom_math.py)，仅纯策略和 CPU：

| 命题 | 有限域证据 |
| --- | --- |
| 固定 cap 闭式、最后机会、尝试界、不重复、计划守恒 | 900 组 |
| 动态 host cap 只降、历史计划与新观测区分、未来不越界 | 11,252 组 |
| dense helper 与独立全域可行性整数 oracle 一致 | 9,288 组 |
| 阈值端点、资源增加与逐块费用增加的单调性 | 351 个 inventory |
| mixed dtype inventory、精度 hash、更新起点与未知预算边界 | CPU/纯策略夹具 |
| 跳过成功点和非单调步时 | fake runner/纯表格反例 |

这些枚举检验实现与命题在所列有限域的一致性，不替代任意规模的归纳证明。
构造 OOM 状态仅是结果字典，没有执行 CUDA 分配或真实异常注入。

```bash
CUDA_VISIBLE_DEVICES='' timeout 60 .venv/bin/python -m pytest -q \
  tests/test_adaptive_oom_math.py
```

新增集合为 `13 passed`；与 `test_adaptive_retry_policy.py`、`test_adaptive_swap_capacity.py`、
`test_auto_block_swap_policy.py`、`test_adaptive_precision_contract.py` 和
`test_adaptive_training_supervisor.py` 同跑为 `114 passed in 6.63s`，退出码 0。
独立只读验收通过，Ruff 与定向 `git diff --check` 通过；文档本地链接可达。

证据只支持本报告列出的条件结论；真实模型峰值、预算估计误差、恢复成功率、数值质量
和吞吐仍需各自独立验收。本轮不生成推荐配置、不修改默认策略、不开放中途恢复。
