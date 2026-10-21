# MOSS-Image A/B 六轴最终对比（E7 定稿）

状态：实验（本轮热测 + 审计后的最终对比结论；硬件与配置边界见 §10）
适用版本：A = `krea2-moss-ab @ 214dbac5`；B = `KohakuMossTrainer @ c73be89`
硬件：NVIDIA CMP 50HX（sm75 / TU102，10240 MiB 标称、9788 MiB torch 可见、225 W、PCIe Gen1 ×8）
相关代码：A `library/training/precision_policy.py:19-53`；B `src/kohakumoss/training/callbacks/finite.py:9-60`、`src/kohakumoss/training/optim/bundle/fused.py:57-98`
相关证据：`output/analysis/moss-ab/six-axis/`（E4/E5/E6a/E6b 与 `E6b_B_fit/scope.json`）、`docs/findings/moss_equal_vram_comparison_20261020.md`、`docs/findings/moss_image_512_ab_remeasure_20261020.md`

> **本文是什么**：t29（A 高 swap 档）/ t30（B `balanced`）/ t34（A 3 seed 112 步）/ t35（B 2 seed 守卫中止）四轮热测，加 t28/t31/t32/t33 四轮审计之后的**最终汇总与判决**。数字全部可追到落盘；推断标 `[算术推导]`，未测标 `[未采集]`。
> **本文不是什么**：不是实现效率排名，不是收敛速度排名，也不是「B 结构性不行」的判决。跨侧 loss 绝对值、跨 scope 成本倍数、跨 scope 内存差**都不在本文件的主张范围内**（§1.4 列了禁止表述）。

## 0. 一句话判决

**在 CMP 50HX / sm75 / 512×512 / batch=1 / fp16 这一格子里，训练线路选 A（krea2-webui）：同显存预算下 A 步时少 8.98%（B/A = 1.0986×），三计数可用、512 档无 NaN；B 的价值在机制与门禁设计（逐步非有限明细、参数级定位、安装前守卫），不在这一格的性能。**

判据链（每个数都能在本文或下面指出的落盘/推导处找到出处）：① 同预算 2.45 GiB 档 A 10.9530 s 对 B 12.0330 s（§8）；② 同预算 6.1 GiB 档 A 6.2087 s `[算术推导·插值]`（swap≈21.31，按字节在 swap=8 与 swap=24 两点间线性插值，推导**不在**本文 §8：算式见 `docs/findings/moss_equal_vram_comparison_20261020.md:263`、表行见 `docs/findings/moss_equal_vram_comparison_20261020.md:269`；本文 §8 是显存轴，只有实测三口径，没有 6.2087 的推导）对 B 8.0024 s（§8）；③ 若问「一次完全有限更新候选要多久」，按**各侧自有判据**折算 A 2.8408 s / B 28.115 s（seed_wise）或 28.20 s（pooled），但这**只是成本口径，不是效率判决**（§7）；④ B 在本轮两轮热测里都被自家 `NonFiniteGuard` 中止，未达 100 候选（§5）。

## 1. 口径与术语（先读，否则后面的数会读错）

### 1.1 「候选」不是「完整权重更新证明」

- **B 侧的候选**：`loss` 有限 ∧ `grad_norm` 有限 ∧ `nonfinite_params == 0`（`E6b_B_fit/result.json` `protocol_notes.effective_criteria`；判据实现见 `E6b_B_fit/scope_recompute.py`）。它是**软件日志判据**。
- **A 侧的候选**：`loss` 有限 + 8/336 个被抽样参数的哈希确有变化 + 无 loss_scale 回退（`E6a_A_fit/result.json` `effective_updates.scope`），并且并列报告 trainer 自己的 `attempted/effective/skipped` 计数。
- 两侧判据**不同源**（A 抽样 8/336，B 全量 504 非有限检测）。所以「候选成本」只能叫 **per fully-finite update candidate（按各侧判据计的候选成本）**，不能叫「真实完整权重更新成本」，也不能当学习质量指标；即便窗口/端点对齐，也**不能**说两侧的「更新证据」完全一致。
- 512 档 B3/B4 的末态 per-param `step = 7` 只属于那两次 512 运行（`output/km-runs/b512_fp16_fp32res_{bal,low}_r1/checkpoints/step_00000023/optim.pt`）；E6b 两轮**没有**收尾 checkpoint（`optimizer_state.available = false`，守卫先抛错），因此 E6b 的候选数**不能**借 512 的末态计数补强。

### 1.2 A 侧四个不同的计数（112 / 109 / 109 / 111），别混用

| 名字 | 值 | 含义 | 出处 |
| --- | --- | --- | --- |
| `steps_run` | **112** | 实际跑的 step 数（global_step 1..112，0-based 字段 0..111） | `E6a_A_fit/result.json`；`progress.seed4*.jsonl` |
| `n_step_sec` | **109** | 稳态窗口内的 update 行数：0-based 3..111 = 1-based **4..112** | 同上 |
| 区间数 | **109** | 109 个 `ts` 差分（含边界 `up[2] → up[3]`，即 1-based 3→4 那一段） | `output/analysis/moss-ab/six-axis/E6b_B_fit/scope_recompute.py:43-45`；`scope.json` `A.42.n_intervals = 109` |
| 候选 | **111 + 1 unknown** | 整轮候选数；`unknown` 是 1-based step 1（窗口外） | `E6a_A_fit/result.json` `class_counts` |

> **注意**：只对 `progress.seed*.jsonl` 里 step 4..112 的相邻两点求差，得到 **108** 个 Δ；那是**另一事件源/边界**（不含 3→4 那一段），与 `scope.json` 的 109 不是同一个样本集。本文统一用 `scope.json` 的 **109 行 / 109 个 Δ**，并明确 offset 与时间端点。

### 1.3 成本聚合：scope × 聚合规则，缺一不可

- **scope**：`S1` = 稳态（丢前 3 步；A 用 `ts` 差分 109 个间隔，B 用原生 `step_time` 83 / 61 个值，**不做差分**）；`S2` = 整轮（墙钟 ÷ 候选数，**含加载、预热与中止尾部**）。
- **聚合规则**：`seed_wise` = 各种子成本的平均（种子等权）；`pooled` = Σ耗时 ÷ Σ候选（候选等权）。两者只在候选数不等时分叉。
- 本文**同时给两栏**，引用必须带 scope + 聚合标签；两侧用**同一条规则**。
- 本轮的聚合陷阱（已改正）：B 的 S2 曾写成 `634.5 s ÷ 22.5 = 28.115 s`，该算式实际等于 **28.20**（pooled），`28.115` 是 `mean(722/25, 547/20)`（seed_wise）。`scope.json` `protocol_notes.aggregation_pitfall` 已登记。
- A 侧 S2 的 seed_wise 与 pooled 恰好同值（2.84080 / 2.84084），原因是三个种子的候选数都是 111，**不是抄错**。

### 1.4 允许 / 禁止的表述

| 允许 | 禁止 |
| --- | --- |
| 「per fully-finite update candidate 成本」+ scope + 聚合标签 | 「真实完整权重更新成本」「训练效率」「收敛速度」 |
| 「同显存预算下的步时比」 | 「实现效率差」（工作量未对齐，§4） |
| 「B 在这两轮被自家守卫中止、未达 100 候选」 | 「B 结构性做不到」「第三个 seed 也会中止」「参数永久冻结」「online+probe 是唯一修复路线」 |
| 同侧自比（A 的 `loss@100/loss@20`、B 的 `loss@20`）与「按各侧判据」的对齐 | 跨侧 loss 绝对值比、跨 scope 内存相减 |

## 2. 表1 六轴总表

| 轴 | A（krea2-moss-ab） | B（KohakuMossTrainer） | 比值 | 证据分级 | 是否可比 |
| --- | --- | --- | --- | --- | --- |
| ① CPU 内存 | 15504.6 MiB（≈15.14 GiB）单进程 VmRSS，swap=8 与 swap=20 两档同值 | `subtree_rss` 同 tick 峰值 20487.4 / 20488.2 MiB（含加载期）；稳态 16.40 / 16.24 GiB | **不给**（scope 不同） | `[本项目实测]` + `[未采集]`（PSS） | ✘ 不可相减 |
| ② 实际拟合速度 | 3 seed × 112 步；`loss@100/loss@20` 均值 0.927（0.889–0.954），但替代估计（block 中位比 1.003、w5 比 0.979、OLS 斜率 −0.030…+0.027）方向不稳 ⇒ **预注册降幅落在噪声内** | 2 seed，候选 25 / 20，`loss@20` 1.3094（w5 0.9017）/ 0.8007；`loss@100 = null` | **不给**（不可跨侧比 loss） | `[本项目实测]` + `[算术推导]`（噪声判定） | ✘ loss 绝对值不可比 |
| ③ 实际有效步数 | attempted 112 / effective 112 / skipped 0；候选 111 + 1 unknown | attempted 86 / 64；候选 25 / 20；全量非有限 61 / 44；局部 0；未知 0 | 候选率 1.0000 对 0.2907 / 0.3125 | `[本项目实测]` | ⚠️ 判据不同源（§1.1） |
| ④ NaN 处理 | 整步丢弃；计数说谎（48/48/0）；loss 全程有限；评级 **中** | 逐参数掩码 + `_restore()`；`nonfinite` 正确、`skipped` 漏记；评级 **中** | 无 | `[本项目实测]` + `[代码事实]` | ✔ 机制可对照，等级同为中 |
| ⑤ 步时 | swap=8 2.4181 s（正式 20 步）/ 2.4132 s（稳态）；swap=24/32/38/40 = 6.9750 / 9.2505 / 10.9530 / 11.5245 s；拟合 `t = 0.1383 + 0.28468·swap`（R² = 0.9999992） | `balanced` 8.0024 s；`low` 12.0330 s；E6b 单 attempt 7.9996 s | 同预算 1.0986×（A 步时少 8.98%）；单 attempt 3.295× | `[本项目实测]` | ✔ 同 scope 内可比 |
| ⑥ 显存 | alloc 8488.5 MiB / reserved 9138–9140 MiB（E6a，swap=8）；512 档 8992.2 / 9138 / 9499 | alloc 6146.6 MiB / reserved 6448.0（E6b `balanced`）；512 档 `low` 2499.0 / `balanced` 6116.4 | 同 2.45 GiB 档 1.0986×；各自已测最快档 3.309×（显存不对等） | `[本项目实测]` | ✔ 显存三口径可比，跨口径不可混 |

> 表内「比值」列**不带 scope/聚合标签的只出现在同 scope 的三项**（同预算步时、单 attempt、显存三口径）；候选成本比 10.76/10.78/9.90/9.93 放在 §7，不放进总表以免被抽掉标签。

## 3. 轴① CPU 内存

**定义**：训练进程占用的主机内存。两侧口径不同：A 是**单进程 VmRSS 峰值**（`process_rss_max_bytes`），B 是**进程组同 tick 的 VmRSS 求和**（`subtree_rss`）。`sum(RSS)` 会重复计共享页（共享库、CoW、只读映射），**既不是 PSS 也不是独占 RAM**。

**两侧实测值**

| 量 | A | B |
| --- | --- | --- |
| 峰值 | **15504.6 / 15504.5 / 15504.6 MiB**（3 seed，≈15.14 GiB，swap=8） | **20487.4 / 20488.2 MiB**（`subtree_rss` 同 tick 峰值，seed42/43） |
| 峰值出现在 | 采样阶段未在落盘声明 `[未采集]` | **加载期**（开跑后约 10 s，04:01:44 / 04:19:42），不是 step 稳态 |
| 稳态 | — | 末 20% tick 16.40 / 16.24 GiB |
| 非同时上界 | — | `sum_of_peaks` 22460.8 / 22459.9 MiB（**不当峰值用**） |
| 与 swap 的关系 | **无关**：swap=8（E6a）与 swap=20（早期生产轮 `resources.process_rss_max_bytes`）同为 15.13–15.14 GiB | `low` 档未采集同口径 RSS `[未采集]` |

**证据路径**：`output/analysis/moss-ab/six-axis/E6a_A_fit/result.json`（`rss_max_mib` / `rss_mean_mib`）；`output/analysis/moss-ab/six-axis/E6b_B_fit/result.json`（`rss`、`rss_peak_bytes = 21482577920`、`rss_peak_scope`）；`output/analysis/moss-ab/six-axis/SUSPENDED_20261020.md:41`（A swap=20 的 15.13 GiB 读数与出处）；B 的共享/私有拆解见 `output/analysis/moss-ab/512-ab/run-B2/interrupted/*/ranks/rank0.jsonl`（`host_rss_gib 16.09` / `host_private_gib 1.14` / `host_shared_gib 14.95`）。

**结论行**：① A 的主机占用**与 swap 深度无关**（两档同值），此前「swap 越大 host 越吃紧」的猜测被实测推翻；② B 的 20.0 GiB 是**含加载期的 sum(RSS)**，其中共享页占绝大部分（同一批日志里 private 仅 1.14 GiB），因此**不能**与 A 的 15.14 GiB 单进程 RSS 相减或相除；③ PSS、A 侧采样阶段、B `low` 档 RSS 均为 `[未采集]`，本文不给内存差值与倍数。

## 4. 轴② 实际拟合速度

**定义**：固定数据、幅面、batch 下 loss 的下降能力。**同源才可比**：E4 判定 batch=1 时 A 的 `Σd²/(128·H·W)` 与 B 的 `Σ_tokens mean_c(d²)/n_img` **代数等价**，但两侧 σ 采样分布（A 均匀 mean 0.5005 / B 截断 logit-normal mean 0.7442）、LoRA 注入面、参数数、图 token 数都不同（§1.4、等显存文档 §1.5），因此**只能同侧自比**。

**两侧实测值**

| 量 | A（3 seed × 112 步） | B（2 seed，候选 25 / 20） |
| --- | --- | --- |
| `loss@20` | 0.8665 / 0.8283 / 0.8833 | **1.3094**（±2 有效步中位 0.9017）/ **0.8007** |
| `loss@100` | 0.8131 / 0.7901 / 0.7852 | **null**（未达 100 候选，不补造） |
| 主估计 `loss@100 / loss@20` | 均值 **0.927**（0.8889–0.9538，`covers_one = false`） | null |
| 替代估计 1：block 中位比 | 1.0183 / 1.1032 / 0.8886，均值 **1.0034**（跨 1.0） | null |
| 替代估计 2：±2 有效步中位比（w5） | 0.9880 / 0.9473 / 1.0006，均值 **0.9787** | null |
| 替代估计 3：OLS 斜率 | −0.000305 / +0.000265 / −0.000135 per step（每 100 次更新 −0.030 / +0.027 / −0.014） | null |
| loss 值域 | 0.4365–1.0745 / 0.4378–1.0760 / 0.4491–1.0856 | 0.7395–1.5841 / 0.7378–1.5941 |
| 预注册降幅 vs 噪声 | 预注册 0.0535 / 0.0383 / 0.0981；实测 block 均值差 −0.0099 ± 0.0451（0.22σ）、+0.0397 ± 0.0515（0.77σ）、−0.045 ± 0.0466（0.97σ） | 不适用 |

**证据路径**：`output/analysis/moss-ab/six-axis/E6a_A_fit/result.json`（`loss_at_20`、`loss_at_100`、`aggregate`、`robustness`、`noise`）；`output/analysis/moss-ab/six-axis/E6b_B_fit/result.json`（`loss_at_20`、`loss_at_100 = null`、`aggregate`）；σ 与归约等价性见 `output/analysis/moss-ab/six-axis/E4_loss_comparability.md` §1.2–1.4、§1.6。

**结论行**：① **跨侧 loss 绝对值不可比**，本文不做、也不建议引用「A 比 B loss 低多少」；② A 侧自身的**预注册降幅落在噪声内**（block 均值差 0.22–0.97σ），替代估计方向不稳（block 中位比均值 1.003 而 w5 比均值 0.979、OLS 有正有负）⇒ **不给 A 自己的收敛结论**；③ B 未达 100 候选 ⇒ `loss@100` 与降幅如实记 **null**，不用插值或外推补位。

## 5. 轴③ 实际有效步数

**定义**：一次 attempt（跑了一步）里，有多少步真的产生了**按该侧判据计的完全有限更新候选**。两侧判据**不同源**（A：8/336 抽样哈希 + 无 scale 回退；B：`loss`/`grad_norm`/`nonfinite_params` 三判据全量标记），故给出的是「各侧候选率」，不是同一把尺子。

**两侧实测值**

| 量 | A（E6a，swap=8，3 seed） | B（E6b，`balanced`，2 seed） |
| --- | --- | --- |
| attempted | 112 / 112 / 112 | **86** / **64** |
| 候选 | **111 + 1 unknown**（unknown = 1-based step 1） | **25** / **20**（稳态窗口 4..86 / 4..64 内 **25 / 19**） |
| 候选率 | **1.0000**（窗口 109/109） | **0.2907 / 0.3125**（稳态 0.3012 / 0.3115；seed_wise 均值 0.3063、pooled 44/144 = 0.3056） |
| 全量非有限步 | 0 | **61 / 44** |
| 局部掩码 / 未知 | — | 0 / 0 |
| trainer 计数 | `attempted 112 / effective 112 / skipped 0`（E5 已证该计数器不可单信） | `raw_skipped_steps_last` 61 / 44（plain bundle 下与实际非有限步数一致）；**fused 档 `skipped` 恒 0，属计数缺口**（§6） |
| 终止原因 | 跑完 112 步 | **`NonFiniteGuard` 抛错**：scale 触底 1.0 后连续 8 步非有限（step 86 / 64） |

**证据路径**：`E6a_A_fit/result.json`（`class_counts`、`effective_updates.scope`）；`E6b_B_fit/result.json`（`class_counts`、`effective_updates`、`optimizer_state.available = false`）；`E6b_B_fit/notes.md` §2/§5；守卫定义 `src/kohakumoss/training/callbacks/finite.py:9-60`（`NONFINITE_PATIENCE` 默认 8，0 = off）；512 档对照 `output/analysis/moss-ab/512-ab/run-B{3,4}/effective_updates.json`（21/60，末态 per-param `step = 7` **仅属那两次运行**）。

**结论行**：① A 侧候选率 1.0000（窗口 109/109），B 侧 0.2907 / 0.3125——**这是两把不同的尺子**，只能当「各侧自报的候选率」；② B 在本轮两轮热测里**都被自家守卫中止、未达 100 候选**；这一句只对「当前配置 + 这两个 seed」成立，**不**推出「B 结构性做不到」「第三个 seed 也会中止」「参数永久冻结」；③ seed43 的两个分母必须并列写：**全程 20/64**、**稳态 19/61**，差的这 1 个候选落在**预热期 step 1**（`cost-aggregation-check.md` §5 逐行核过）；④ 512 档的 21/60（35%）与末态 `step = 7` 只属于 B3/B4，**不能**移作 E6b 的证据。

## 6. 轴④ NaN 处理

**定义**：非有限值（NaN/Inf）的**检测 → 隔离 → 计数 → 留证**四件事。判断标准不是「有没有 NaN」，而是「权重有没有被写坏、计数有没有说谎、事后能不能定位」。

**两侧实测值**（t33 注入热测 + 512/E6b 自然 NaN 观测）

| 环节 | A（krea2-webui） | B（KohakuMossTrainer） |
| --- | --- | --- |
| 检测 | accelerate `GradScaler` + fail-fast（默认关闭） | 每步检测 `found_inf`；`NonFiniteGuard` 默认 8 步连续非有限即抛错（`NONFINITE_PATIENCE`，0 = off） |
| 隔离 | **整步丢弃**：命中即不调用 `optimizer.step()`（实测 1/336 参数范数与 16 元素 hash 不变；「整步 336 个一起丢」是代码推论） | **plain 档整步 skip**（`skipped` 如实 +1，512 档 = 16）；**fused 档逐参数掩码 + `_restore()`**，单参数 inf 只回滚该参数（注入轮 `nonfinite_params = 1`） |
| 计数 | `precision_report` 48/48/**0 skipped**，把被跳过的步算成 effective（**计数说谎**） | `nonfinite` +1 正确；**fused 档 `skipped` 恒 0**（漏增，`src/kohakumoss/training/optim/bundle/fused.py:303`）；plain 档 skipped 正确 |
| 留证 | 无逐参数名、无非有限步明细的专表 | `InstabilityDump` + `nonfinite_names`（定位到参数名）+ 逐步 `loss`/`grad_norm`/`loss_scale`/`nonfinite_params` 落盘 |
| 512 档自然 NaN | 0（16–32 步窗口） | `loss_scale` 4096→1 后仍有 13 步非有限、未自愈；E6b 两轮 32768/65536→1 后连续 8 步非有限被守卫中止 |
| 注入实验评级 | **中**（跳过与降 scale 符合预期；`avr_loss` 被污染的预测被证伪） | **中**（`_restore`/掩码生效、`nonfinite` 自增、`skipped` 漏记为 0） |

**证据路径**：`output/analysis/moss-ab/six-axis/E5_nan_injection.md` §0（结论表）、§0.1（观测粒度）、§7.1（机制链）、§12（512 末态补证）、§14（R1–R13 对照）；复核 `output/analysis/moss-ab/six-axis/verify_E5_r2.md`；B 侧代码 `src/kohakumoss/training/optim/bundle/{base.py:190-219,fused.py:57-98}`、`src/kohakumoss/training/callbacks/finite.py:9-60`；A 侧精度策略 `library/training/precision_policy.py:19-53`。

**结论行**：① 两侧都是「**权重安全、计数说谎**」，差别在粒度：A **整步**丢弃，B 在 fused 档**按参数**掩码；② B 的留证能力（逐步明细 + 参数名）明显更强，A 的 `attempted/effective/skipped` 在**正常路径**可用但在跳过步失真；③ A 侧 `avr_loss` 被永久污染的预测**被实测证伪**（E5 §0），不得再作为 A 的缺陷引用；④ 两侧评级同为**中**，不存在「一方已解决 NaN」的结论；NaN 根因**未定位**（§10）。

## 7. 轴⑤ 步时

**定义**：单步墙钟（不含周期落盘）。三种口径必须分开：**裸步时**、**per fully-finite update candidate 成本**、**同预算推进量**。

**两侧实测值**

| 口径 | A | B | 比值 | scope/标签 |
| --- | --- | --- | --- | --- |
| 裸步时（各自已测档） | swap=8 **2.4181 s**（正式 20 步）/ 2.4132 s（稳态）；swap=24/32/38/40 = 6.9750 / 9.2505 / 10.9530 / 11.5245 s；拟合 `t = 0.1383 + 0.28468·swap`（R² = 0.9999992） | `balanced` **8.0024 s**；`low` **12.0330 s**；E6b 单 attempt 7.9996 s | 3.309×（显存不对等：8992.2 对 6116.4 MiB） | 实测 |
| 同 2.45 GiB 预算 | swap=38 **10.9530 s** | `low` **12.0330 s** | **1.0986×**（A 步时少 8.98%） | 实测/实测 |
| 同 26 块字节 | 7.5435 s `[算术推导·插值]` | `balanced` 8.0024 s | 1.0608× | 含插值 |
| per-candidate **S1**（稳态，丢前 3 步） | **2.4277 s**（109 行 / 109 Δ；seed_wise = pooled） | **26.1208 s**（seed_wise）/ **26.1803 s**（pooled，Σ1151.93 / Σ44） | **10.76×** / **10.78×** | 候选判据两侧不同（§1.1） |
| per-candidate **S2**（全程，含加载+预热+尾部） | **2.8408 s**（seed_wise = pooled；两规则同值） | **28.1150 s**（seed_wise）/ **28.2000 s**（pooled，Σ1269.0 / Σ45） | **9.90×** / **9.93×** | A 尾部 = 正常收尾；B 尾部 = 守卫中止 |
| 单 attempt（不折算候选率） | 2.4277 s | 7.9996 s | **3.295×** | 实测 |
| 同预算推进量（60 min；**本行全部为 `[算术推导]`，无实测值**） | swap=38：328.7 步；swap≈21.31：579.8 步 `[算术推导]` | `low`：299.2 步（其中候选 ≈ 104.7）；`balanced`：449.9 步（候选 ≈ 157.5），两档候选数同标 `[算术推导]` | — | 步数 = 3600 ÷ 该档**实测**步时 `[算术推导]`；候选数 = 步数 × 512 档 21/60 `[算术推导]`；四档步数与两个候选数都只是把实测步时折算到 60 min 尺度，**不得**读成实测推进量 |

**证据路径**：A 步时 `output/analysis/moss-ab/512-ab/run-A5/result.swap*.json`、`run-A3/result.json`、`run-A5/fit.txt`；B 步时 `run-B4/compare.vs-A.json`、`run-B3/result.json`；derived 成本 `output/analysis/moss-ab/six-axis/E6b_B_fit/scope.json`（`aggregates` / `decomposition`）、`E6b_B_fit/result.json`（`per_candidate_steady_state_sec` / `sec_per_fully_finite_candidate_whole_run_sec`）；独立复核 `output/analysis/moss-ab/cost-aggregation-check.md` §1–§2（109Δ、四组倍率、3.295× 全部复现）。

**结论行**：① 同一显存预算下 A 的步时少 **8.98%**（1.0986×），同一字节量下少 5.73%（1.0608×，A 侧含插值）；② 若改问「一次完全有限更新候选要多久」，按各侧判据计为 **10.76× / 10.78×**（S1）与 **9.90× / 9.93×**（S2）——**必须同时带 scope 与聚合标签**，且这不是实现效率；③ 差距的一半以上来自 B 的候选率（1/0.3063 ≈ 3.265×），**这部分是 fp16 NaN 浪费**，不是 kernel 或搬运性能；④ B 的 S1 是**右截尾**样本（守卫中止前的步），不能与 A 的完整 112 步当同一个总体。

> **适用边界（脚本能力）**：`scope_recompute.py` 用 `x == x` 判 NaN（对 Inf 无效），且 `nonfinite_params` 缺失时默认 0。本批数据（138 行）字段齐全、只出现 finite/NaN，因此上述结果可用；但**不得**把该 helper 当作能处理任意 Inf / 缺字段的一般分类器。该局限为说明性边界，不触发脚本改动。

## 8. 轴⑥ 显存消耗

**定义**：三口径（torch `peak alloc` / `peak reserved` / 整卡 `nvidia-smi` 峰值）+ 两个等预算口径（各自已测最快档 / 近等显存档）。

**两侧实测值**

| 口径 | A | B | 说明 |
| --- | --- | --- | --- |
| E6a（swap=8） | alloc **8488.5 MiB** / reserved 9138–9140 MiB | — | 112 步正式跑，探针开启 |
| 512 档（swap=8 / `balanced`） | alloc **8992.2** / reserved 9138 / smi 9499 | alloc **6116.4** / reserved 6408 / smi 6771 | 不对等预算：A 多占 **2875.8 MiB**（+47.0%） |
| 512 档（swap=38 / `low`） | alloc **2511.8** / reserved 2658 / smi 3019 | alloc **2499.0** / reserved 2630 / smi 3133 | 三口径差 +12.8 / +28.0 / −114 MiB ⇒ **同一显存预算**（≈2.45 GiB） |
| E6b（`balanced`） | — | alloc **6146.6** / reserved 6448.0（seed42）、6428.0（seed43） | 比 512 轮高 30 MiB 量级（数据/进程差异） |
| B 自述档位 vs 实测 | — | 预测 8.2 GiB / 实测 5.97 GiB（`balanced`，**高估 24.1%**）；预测 3.9 / 实测 3.06（`low`，高估 27.5%） | 档位估计系统性偏高 |

**证据路径**：`run-A3/result.json`、`run-A5/result.swap38.json`、`run-A5/result.swap8.json`；`run-B4/compare.vs-A.json`、`run-B4/notes.md`、`run-B3/result.json`；E6a/E6b 各自 `result.json`（`vram_alloc_max_mib` / `vram_reserved_max_mib`）；等预算推导与三口径对照见 `docs/findings/moss_equal_vram_comparison_20261020.md` §4。

**结论行**：① 显存**可以对齐**：2.45 GiB 档三口径差 ≤4%，可判同一预算；② 但两侧**工作量未对齐**（LoRA 168 / 252 模块、33,030,144 / 74,317,824 参数、σ、LR 调度、精度治理都不同），因此 1.0986× 只是**吞吐对照**，不是实现效率差；③ B 的档位间隔很大（2.50 GiB ↔ 5.97 GiB 之间无中间档），A 用 swap 扫描可以连续取点；④ 本文**不做**内存（host）差值与倍数（§3）。

## 9. 表2 结论修正表（本轮被推翻或修正的旧结论）

| # | 原说法 | 原出处 | 新证据 | 修正后说法 | 分级 |
| --- | --- | --- | --- | --- | --- |
| (a) | 「A 比 B 快 4.98×」 | `docs/findings/moss_image_512_ab_remeasure_20261020.md:23`（表行 4.976×）、`:29`（可比性限定） | 等预算三口径实测：1.0986×（2.45 GiB）/ 1.2889×（6.1 GiB，A 插值）/ 1.0608×（同 26 块字节） | 4.98× 是「A 多用 6493 MiB 显存」买来的局部比值；**同预算下 A 快 ≈1.10×**，且三口径里两个 A 侧含插值 | `[本项目实测]` + `[算术推导]` |
| (b) | 「A 的 host 占用随 swap 上升」 | `output/analysis/moss-ab/six-axis/PLAN.md:56`（预注册自述，作者已自纠） | swap=8（E6a）与 swap=20（早期生产轮）同为 **15.13–15.14 GiB** | A 的主机占用**与 swap 深度无关**（整模型 fp16 权重常驻 host，再按块搬上 GPU） | `[本项目实测]` |
| (c) | 「B loss ≈ 0.047」 | `docs/findings/moss_historical_measurements_audit_20261020.md:114`（**A 侧** krea2-webui 的 bf16 历史事件 d：逐步 `avr_loss` = 0.0472 / 0.597 / 0.0483 / 0.0538 …，run 目录 `output/runs/moss-image-bf16-baseline10-20261017-085341/`，在 A 仓）；**B 侧当前没有对应落盘**，故原说法属把 A 侧读数挂到 B 的**侧别误植** | 512 档 B 实测 finite loss **0.7547–1.5840**（`km-runs/b512_fp16_fp32res_{bal,low}_r1/metrics.jsonl`，各 7 步）；A E6a **0.4365–1.0856** | 原说法属**侧别误植**：0.0472 是 A 侧 bf16 旧轮次的 `avr_loss`，**不是** B 的出处；`0.047` 作为「B 的 loss」**作废**，该历史 A 值也**不得**当当前 B 的对照；两侧 loss 同量级，但因 σ / 工作量不同**不可跨侧比绝对值** | `[本项目实测]`（A 侧 bf16 事件 d；侧别如上） |
| (d) | 「B `skipped_steps = 0` 说明没跳过」 | `output/analysis/moss-ab/history-review-contracts.md:451`（旧读法：只说「该字段不是这条路径的计数器」）、`output/analysis/moss-ab/six-axis/PLAN.md:36`（登记漏增） | `src/kohakumoss/training/optim/bundle/fused.py:303` 只 `nonfinite += 1`；E5 注入轮 `nonfinite_params = 1` 且被注入参数未变；512 `balanced`（plain）档 `skipped = 16` 如实 | fused 档 `skipped` **恒 0 是计数缺口**；回滚由 `_restore()` / 掩码**真实发生**；该字段不能当分母，也不能读成「零跳过」 | `[代码事实]` + `[本项目实测]` |
| (e) | 「B 每有效更新 11.9× / 11.3×（同口径性能）」 | t35 输出文本（未进 findings 正文；`output/analysis/moss-ab/six-axis/E6b_B_fit/notes.md:130` 已作废该说法） | `scope.json` 与独立复核：S1 seed_wise **10.76×** / pooled **10.78×**；S2 seed_wise **9.90×** / pooled **9.93×** | 四栏并列、带 scope + 聚合标签；**不是**同口径性能倍数 | `[算术推导]` |
| (f) | 「B 结构性做不到 100 次有效更新」 | t35 输出文本（同上） | 仅两轮实测：86 / 64 attempted 被守卫中止 | 只写「**当前配置 + 这两个 seed 的运行均被守卫中止、未达 100 候选**」；不推结构性、不推第三 seed | `[本项目实测]` |
| (g) | 「A 的 `avr_loss` 会被永久污染」（预注册预测） | `output/analysis/moss-ab/six-axis/PLAN.md:86`（预注册预期） | E5 三次注入轮 loss 与 `avr_loss` **全程有限** | 该预测**被证伪**，A 的 `avr_loss` 污染不再作为缺陷 | `[本项目实测]` |
| (h) | 「每真更新成本 22.86 / 34.38 s」「35% 有效更新率」 | `docs/findings/moss_equal_vram_comparison_20261020.md` 早期版本（t31，已在 t44 改为统计量口径） | 21/60 由 `finite_loss_steps` 统计量推得，非逐步完整更新证明 | 改称「**按各侧判据计的候选成本**」；512 档 21/60 仅对 B3/B4 成立 | `[算术推导]` |
| (i) | 「B `low` 的 `optimizers` 状态为空、无 per-param 计数」 | `output/analysis/moss-ab/verify-equal-vram-r2.md:50`（第二轮原判断，已在 §7 撤回） | 实测 `optimizers[0]` 为 504 项映射、每项 `(master, m, v, step)`、第 4 位 504/504 = 7 | 两档都有**末态** per-param 计数，序列化不同、语义相同；均不能替代逐 step 全权重快照 | `[本项目实测]` |
| (j) | 「B 的 kernel 体系在本机收益上界为 0」 | `docs/findings/moss_image_512_ab_remeasure_20261020.md` 早期版本（t27 已改） | H2D 下界 11.764 s 对实测 12.033 s ⇒ 非 H2D 项 ≤0.27 s/步（≈2.3%） | 改为「**条件推算的上界 ≈2.3%，不是 0，也不是实测收益**」 | `[算术推导]` |
| (k) | 「B `low` 每步多出的 1.4314 GB = 激活往返」 | 512 / 等显存早期版本 | 与同档 D2H 计数 1.4324 GB 近似等量，但 buffer 类别未归因 | 改称「**未分类的额外单向流量**」；等量≠同类 | `[本项目实测]` + `[未采集]`（buffer 归因） |

**修正条数：11 条**（其中 (a)(b)(d)(e)(h) 是会影响对外结论的主项；其余为术语/证据层级修正）。

## 10. 表3 未测 / 不可测清单（不得用推断填充）

| # | 项 | 状态 | 原因 / 边界 |
| --- | --- | --- | --- |
| 1 | 多卡 / DDP / TP / PP | `[未采集]` | 本轮全部单卡；两侧多卡路径均未跑 |
| 2 | B 的 1024 档 | `[未采集]` | 时间与显存预算未覆盖；不得由 512 外推 |
| 3 | bf16 路径 | `[未采集]` | sm75 无原生 bf16；A 的 bf16 请求被 `library/training/precision_policy.py:19-53` 就地改写为 fp16；B 的 bf16 未跑 |
| 4 | sm80+ / 其它 PCIe 宽度 | `[未采集]` | 结论只在这张卡、这条 Gen1 ×8 链路上成立 |
| 5 | B `speed` / `minimal` 档 | `[未采集]` | 只测 `balanced` / `low` |
| 6 | A `swap≈21` 实测 | `[未采集]` | 现为插值（`[算术推导]`） |
| 7 | PSS（两侧） | `[未采集]` | 只有 RSS / VmHWM；`sum(RSS)` 重复计共享页 |
| 8 | A 侧 RSS 采样阶段 | `[未采集]` | 落盘只有 `rss_max_mib`，未声明加载期/稳态分离 |
| 9 | B `low` 档 RSS | `[未采集]` | 本轮未采 |
| 10 | 全张量 bitwise 判定 | `[未采集]` | E5 权重已删；现有证据只覆盖 1/336 参数与 16 元素 hash |
| 11 | E6b 末态 optimizer 计数 | `[未采集]` | 两轮均被守卫中止、无收尾 checkpoint（`optimizer_state.available = false`） |
| 12 | B 达 100 候选 / 中止后自愈 | `[未采集]` | 未跑第三 seed、未关守卫、未改配置 |
| 13 | 跨侧 loss 绝对值 / 收敛排名 | **不可比** | σ 分布、工作量、LR 调度、种子数都不同（§4） |
| 14 | NaN 根因 | `[未采集]` | 两侧都只做到检测/隔离/计数，未定位到算子或数值路径 |

**未测项：14 条**（其中 13 为「不可比」，其余为缺测）。

## 11. 最终结论

### 11.1 一句话判决

**在 CMP 50HX / sm75 / 512×512 / batch=1 / fp16 这一格里，训练线路选 A（krea2-webui）：同显存预算下步时少 8.98%（B/A = 1.0986×），三计数可用、512 档 0 NaN、112 步跑完；B（KohakuMossTrainer）在这两轮热测里被自家 `NonFiniteGuard` 中止（86 / 64 attempted，候选 25 / 20）。要取的是 B 的机制与门禁设计，不是它在这一格的性能。**

判决覆盖范围：单卡、单数据集（16 图 / 512 短字幕）、fp16、PCIe Gen1 ×8。**不覆盖**多卡、1024 档、bf16、sm80+（§10）。

### 11.2 分场景建议

| 场景 | 推荐 | 依据（同 scope 才给倍数） | 代价 / 风险 |
| --- | --- | --- | --- |
| 显存紧张（≤2.5 GiB） | **A swap=38** | 10.9530 s 对 B `low` 12.0330 s = **1.0986×**；且 B 512 档只有 21/60 候选（fp16 NaN 浪费） | A 需换出 38/42 块；A 的 D2H 未计量（可能略微高估其优势） |
| 显存充足（可占满 9.8 GiB） | **A swap=8** | 2.4181 s 对 B `balanced` 8.0024 s = **3.309×**（但 A 多占 2875.8 MiB，**不是**同预算） | 若必须给其它进程留 3 GiB 以上，则 B `balanced`（5.97 GiB）是唯一选择 |
| 追求最长无人值守稳定 | **A** | B 两轮在 12 / 9 分钟内被守卫中止（未达 100 候选）；A 本轮 112 步 0 NaN | A 的长训练证据仍是**未测**（本轮最长 112 步、16 图） |
| 需要最丰富的 NaN 诊断证据 | **B** | 逐步明细 + `InstabilityDump` + `nonfinite_names` 定位到参数名 | 这套证据在 B 的 fused 档才完整；plain 档与 A 同粒度 |
| 要求「计数可信」的报表 | **A 的正常路径 + B 的留证** | A 的 `attempted/effective/skipped` 三计数在正常路径正确；B 的 fused 档 `skipped` 漏增 | 任一侧单用都会在异常步上失真（§6） |

### 11.3 合并建议清单

| # | 方向 | 内容 | 风险 | 工作量 | 立即做？ |
| --- | --- | --- | --- | --- | --- |
| M1 | B → A | **逐步非有限明细 + `InstabilityDump` 证据留存**（每步 loss / grad_norm / loss_scale / nonfinite_params + 命中的参数） | 低（纯观测，不改训练路径） | S（一个 callback + 落盘 schema） | **是** |
| M2 | B → A | **`nonfinite_names` 参数级定位**（把 504/336 个候选名落到张量名） | 低（只读梯度） | S | **是** |
| M3 | B → A | **安装前 `NonFiniteGuard`**（默认 8 步连续非有限即停，`NONFINITE_PATIENCE` 可关） | 中（A 是整步丢弃 + GradScaler，阈值需重标定，避免误杀） | S | 试点（不影响默认路径） |
| M4 | A → B | **`attempted / effective / skipped` 三计数 + 逐步 effective 标记**，修掉 fused 档 `skipped` 漏增 | 低 | S | **是** |
| M5 | A → B | **`precision_report.resources` 字段**（RSS / 显存 / 设备信息一次性落盘，含采样阶段标签） | 低 | S | **是** |
| M6 | A → B | **residual-range 守卫**带来的 fp16 稳定性（A 在 512 档 16–32 步与 112 步窗口均 0 NaN） | 中（B 需先证明不改变数值语义） | M（需对照实验） | 试点 |
| M7 | B → A | per-parameter mask/restore（`src/kohakumoss/training/optim/bundle/fused.py:57-98` 那套逐参数回滚） | 高（A 是整步 skip，收益未测，改动面在优化器封装） | L | **不建议立即做** |

合并排序理由：M1/M2/M4/M5 都只影响**证据与报表**，不改数值路径，风险最低；M3/M6 触及失败策略与精度治理，必须先有对照；M7 是结构性改造，在本轮数据里没有任何收益证据支持。

### 11.4 迁移性警告

- **10.76× / 10.78×（S1）与 9.90× / 9.93×（S2）是「按各侧判据计的候选成本比」**，其中 1/0.3063 ≈ 3.265× 来自 B 的 NaN 浪费；这部分在**没有 NaN 的环境里会消失**。单 attempt 差 3.295× 才更接近实现差异，且它仍含工作量差异（§8 结论行）。
- **sm75 没有原生 bf16**：A 的 bf16 请求被 `library/training/precision_policy.py:19-53` 就地改写为 fp16；换到 sm80+ 或真正 bf16 后，两侧的数值行为与此处的 fp16 结论**不可互推**。
- **显存相关的 3.309× / 1.0986× / 1.0608× 只在这条 PCIe Gen1 ×8 链路上成立**：两侧步时都被「每步搬多少字节 ÷ 链路速率」主导，换到 ×16 或别的链路，倍数会重排（字节模型与拟合式见等显存文档 §2/§6）。
- **数据量与步数都很小**（16 图、112 步 / 86 步），本判决不外推到长训练、1024 档与大分辨率。

## 12. 自检

- **行数 / 表数**：本文 **292 行 / 14 张表**（落盘后由脚本复核，命令见下）。
- `git diff --check`：对本文路径干净；未跟踪新文件另用 `git diff --no-index --check /dev/null <doc>` 查空白，无输出。
- **引用与相对路径**：逐条对「本工作树 / A 源树 / B 源树」三根校验，0 坏引用（`path:line` 形式全部命中）。
- **表格列数**：脚本按表块统计，0 错列。
- **数字一致性**：六轴数字与 `scope.json` / `E6a_A_fit/result.json` / `E6b_B_fit/result.json` / `run-A5` / `run-B3` / `run-B4` 逐项对齐；`cost-aggregation-check.md` §1–§2 独立复现了 109Δ、四组倍率与 3.295×。
- **口径声明**：本文所有成本均带 scope + 聚合标签；跨侧 loss、跨 scope 内存、结构性推断均未出现（§1.4）。

## 13. 证据索引

| 数字 / 事实 | 落盘文件 |
| --- | --- |
| A 3 seed × 112 步：`mean_step_sec` 2.4274 / 2.4276 / 2.4281（`n_step_sec = 109`）、候选 111 + 1 unknown、RSS 15504.6 MiB、VRAM 8488.5 / 9138–9140 MiB、loss 统计与噪声块 | `output/analysis/moss-ab/six-axis/E6a_A_fit/result.json` |
| A 侧逐步事件与 `ts`（0-based 0..111） | `output/analysis/moss-ab/six-axis/E6a_A_fit/progress.seed4{2,3,4}.jsonl` |
| A 侧 109Δ / 109 行 / S1·S2 聚合（seed_wise 与 pooled 两栏） | `output/analysis/moss-ab/six-axis/E6b_B_fit/scope.json`、`output/analysis/moss-ab/six-axis/E6b_B_fit/scope_recompute.py:43-45` |
| B 两轮：attempted 86 / 64、候选 25 / 20（稳态 25 / 19）、wall 722 / 547 s、S1 26.5576 / 25.6839、S2 28.88 / 27.35、VRAM 6146.6 / 6448–6428 MiB、RSS 20487.4 / 20488.2 MiB、sum_of_peaks 22460.8 / 22459.9 | `output/analysis/moss-ab/six-axis/E6b_B_fit/result.json`、`notes.md` |
| B 守卫中止（8 步连续非有限、scale 1.0、step 86 / 64） | `output/analysis/moss-ab/six-axis/E6b_B_fit/notes.md` §2；`src/kohakumoss/training/callbacks/finite.py:9-60` |
| 四组倍率与 109Δ 的独立复现 | `output/analysis/moss-ab/cost-aggregation-check.md` §1–§2 |
| A 步时 8/24/32/38/40 与拟合式 | `output/analysis/moss-ab/512-ab/run-A5/result.swap*.json`、`run-A5/fit.txt` |
| B `balanced` / `low` 步时与显存三口径 | `output/analysis/moss-ab/512-ab/run-B4/compare.vs-A.json`、`run-B3/result.json` |
| 512 档 B 候选 21/60 与末态 `step = 7` | `output/analysis/moss-ab/512-ab/run-B{3,4}/effective_updates.json`、`output/km-runs/b512_fp16_fp32res_{bal,low}_r1/checkpoints/step_00000023/optim.pt` |
| 512 档 B 的 finite loss 值（0.7547–1.5840） | `output/km-runs/b512_fp16_fp32res_{bal,low}_r1/metrics.jsonl` |
| E5 注入实验：A 48/48/0、B `nonfinite_params = 1`、`skipped` 漏增、双方评级中 | `output/analysis/moss-ab/six-axis/E5_nan_injection.md` §0/§7.1/§12；`verify_E5_r2.md` |
| σ 分布与 loss 归约等价性 | `output/analysis/moss-ab/six-axis/E4_loss_comparability.md` §1.2–1.4、§1.6 |
| 等显存三口径与三组比值 | `docs/findings/moss_equal_vram_comparison_20261020.md` §0/§4 |
| A/B 工作负载差异（LoRA 168 / 252、33,030,144 / 74,317,824 参数） | `output/analysis/moss-ab/history-review-contracts.md` §3.1–3.3 |

> **与其它文档的关系**：本文是最终汇总，数字以本文 §13 的落盘为准；等显存口径的推导细节仍见 `docs/findings/moss_equal_vram_comparison_20261020.md`，512 档原始复测见 `docs/findings/moss_image_512_ab_remeasure_20261020.md`，功能主线与合并清单见 `docs/findings/moss_vs_kohakumoss_trainer_ab_20261019.md`。本文不修改任何历史文档正文，冲突一律记入 §9。
