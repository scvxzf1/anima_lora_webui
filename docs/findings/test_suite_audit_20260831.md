# 测试集审计与分层治理建议

状态：当前维护基线
适用版本：2026-08-31 工作区
范围：`tests/`、pytest 配置、`tasks.py` 测试入口和 GitHub Actions 测试门禁

## 执行摘要

当前测试集确实过多且过杂，但问题不能用批量删除解决。2026-08-09 的五轮清理把 Git 跟踪的
`tests/test_*.py` 从 225 个降到 191 个；当前又增长到 252 个、77,008 行、pytest 收集
2,804 项。净增 61 个文件（+31.9%）中有相当部分来自 Dragon、Tagging、Z-Image 和多模型
能力，属于真实功能增长；失控点是这些测试没有进入明确的层级和预算。

本轮审计的核心结论：

1. `test-unit` 实际运行整个 `tests/`，混合 unit、Node/jsdom、subprocess、分布式、CUDA、
   benchmark 和 probe，不是单元测试入口。
2. 默认全套在 60 秒保护窗口内只推进到约 15%；收集本身耗时 12.37 秒。它不适合作为日常
   反馈门禁。
3. 前端与少量训练测试大量读取生产源码，再断言函数名、变量名、缩进或代码片段存在。这类
   测试既容易被等价重构打碎，又不能证明行为正确。
4. 真硬件验证、研究 probe 和 benchmark helper 已混入默认 pytest 发现范围，但没有
   `integration`、`hardware`、`benchmark` 或 `probe` marker。
5. 13 个测试文件超过 1,000 行，最大文件 3,691 行。部分文件应拆分；另一些文件虽然大，
   承载的是路径安全、状态恢复和 fail-closed 不变量，不能按体积删除。

建议先分层、再改写、最后删并。第一阶段不以减少断言数为目标，而是让默认门禁重新变得明确、
快速、可重复；第二阶段才删除真实重复和伪覆盖。

## 审计口径

审计基于当前脏工作区。审计期间已有多个前端和前端测试文件由用户修改；本轮没有覆盖、回退或
归因这些改动。

| 指标 | 当前值 |
| --- | ---: |
| Git 跟踪的 `tests/test_*.py` | 252 |
| 总行数 | 77,008 |
| pytest 收集项 | 2,804 |
| 收集耗时 | 12.37 秒 |
| 超过 1,000 行的测试文件 | 13 |
| 少于 100 行的测试文件 | 64 |
| 使用 `read_text()` 的测试文件 | 91 |
| 使用 subprocess API 的测试文件 | 46 |
| 引用 CUDA/device CUDA 的测试文件 | 12 |
| 使用 skip/importorskip 的测试文件 | 32 |

`read_text()` 并不自动代表坏测试：配置、序列化产物和文档完整性测试可以合理读取文件。风险对象
是读取生产源码后断言实现字面量或代码位置的 source-probe 测试。

## 主要发现

### P1：默认测试入口没有语义边界

`pyproject.toml` 只有 `fast` 和 `focused` 两个 marker。`scripts/tasks/utilities.py::cmd_test_unit`
直接执行 `pytest -q tests/`；`tests/conftest.py` 也没有统一的网络、CUDA、耗时或外部运行时隔离。

因此默认入口会收集：

- 46 个含 subprocess 调用的测试文件；
- 12 个含 CUDA 路径的测试文件；
- Node/jsdom 前端运行时测试；
- `test_flash_attn_v100_compile.py` 的 V100 + CUDA + Inductor fullgraph backward；
- `test_compile_checkpoint_block_swap_hot.py` 的组合矩阵；
- 多个直接测试 `bench/`、`scripts/experiments/` 和 `scripts/krea2/probe_*` 的文件。

风险不是环境不满足时会 skip，而是维护者无法回答“默认 pytest 究竟保证什么”。硬件测试在普通
机器上长期 skip 也会形成假门禁。

### P1：源码字面量测试冒充行为测试

高密度区域包括：

- `tests/test_training_frontend_config_ui.py:91`：断言桥接函数的完整 export 文本、handler 名和
  具体缩进；该文件共 3,691 行。
- `tests/test_training_frontend_history.py:141`：用函数体片段断言刷新去重、按钮状态和事件绑定。
- `tests/test_training_frontend_modules.py:15`：硬编码大量模块路径、已删除文件名和 cache token。
- `tests/test_deferred_sample_cleanup.py:16`：读取多个训练源码文件并用 `source.index()` 和字符串
  存在性推断清理顺序，没有执行真实生命周期。
- `tests/test_dragon_next_training_config_acceptance.py:27`：对 TOML 产物做字符串包含断言，而不是
  解析结构化配置。

应保留的源码级护栏只有少数架构契约，例如前端 import graph、禁止 `globalThis` 回流、关键 DOM
ID 和 cache token 一致性。业务状态、序列化和生命周期必须改为调用公共 API 或可执行模块。

### P1：实验、benchmark 和硬件验证混入生产单测

首批迁层候选：

| 文件 | 当前职责 | 建议层级 |
| --- | --- | --- |
| `test_flash_attn_v100_compile.py` | V100 FlashAttention + Inductor 真机 backward | `hardware` |
| `test_int8_blockswap_equivalence_probe.py` | 实验 probe 的 toy 数值和 CLI/profile | `probe` |
| `test_attention_injection_probe.py` | `bench` 私有统计 helper | `probe` |
| `test_convrot_fusion_microbench.py` | experiment microbench 的 parser/决策 | `benchmark` |
| `test_convrot_step_profile.py` | profile probe 的策略输出 | `probe` |
| `test_krea2_nf4_ex_spectrum.py` | Krea-2 研究 helper | `probe` |
| `test_krea2_nf4_correction.py` | NF4 回补实验 helper | `probe` |
| `test_compile_checkpoint_block_swap_hot.py` | compile/checkpoint/swap 组合栈 | `integration` |

迁层不等于删除。生产不变量应保留小型 contract test；研究矩阵、硬件 backward 和性能决策应由
显式任务运行，避免默认机器的 skip 状态被误认为覆盖。

### P1：测试增长没有预算或归属约束

8 月 9 日基线到当前净增 61 个文件。8 月 10 日后可直接定位到 33 个新增文件，其中 8 月 31 日
Dragon/Tagging 提交一次新增 18 个。Dragon 相关出现大量 50--120 行、1--4 项测试的 runtime
碎片，例如：

- `test_dragon_visibility_poller_runtime.py`
- `test_dragon_live_dom_runtime.py`
- `test_dragon_live_log_runtime.py`
- `test_dragon_pointer_frame_runtime.py`
- `test_dragon_route_styles_runtime.py`
- `test_dragon_history_search_runtime.py`
- `test_dragon_dataset_runtime.py`
- `test_dragon_dataset_cleanup_runtime.py`

这些不是无价值测试，但文件粒度跟随每次修复增长。应归并成 Dragon runtime、history、dataset、
config 等领域套件，并复用统一 Node/jsdom harness。

### P2：巨型测试文件混合多个职责

最大的维护热点：

| 文件 | 行数 | 主要问题 | 处理原则 |
| --- | ---: | --- | --- |
| `test_training_frontend_config_ui.py` | 3,691 | 源码字符串、重复 Node runner、多 feature 混合 | 按 feature 拆并改行为测试 |
| `test_block_swapping.py` | 2,608 | CPU 状态机、CUDA、profile、ConvRot 混合 | 分 unit/hardware/schema；不删不变量 |
| `test_web_config_datasets.py` | 2,366 | editor、preflight、preset、legacy 混合 | 按服务边界拆分 |
| `test_training_queue.py` | 1,984 | 全局路径 monkeypatch 和状态场景重复 | fixture/表驱动后拆分 |
| `test_training_frontend_history.py` | 1,866 | source-probe 与 Node 行为混合 | 按 API/render/runtime 拆分 |
| `test_web_config_preflight.py` | 1,636 | legacy facade 与真实 preflight 混合 | 兼容层独立 |
| `test_preprocess_paths.py` | 1,225 | 路径和缓存安全面宽 | 只拆分，不按体积删 |
| `test_preview_service.py` | 1,191 | 路径安全与普通 listing 混排 | 安全/行为分层 |
| `test_network_registry.py` | 1,184 | schema、resolution、plugin、metadata 混合 | 按 registry 契约拆分 |
| `test_training_frontend_modules.py` | 1,152 | import graph、路径清单、token 混合 | 保留架构护栏，删字面量噪声 |
| `test_lokr.py` | 1,134 | 单方法多层契约 | 后续单独审查 |
| `test_web_config_file_groups.py` | 1,130 | 文件组场景集中 | 参数化并按服务边界拆分 |
| `test_training_frontend_live.py` | 1,012 | 大段内嵌 DOM fixture | 抽 Node/jsdom harness |

拆分本身不会减少测试数量，但会阻止热点继续变成新的测试石山。删减应发生在职责拆清之后。

### P2：存在可确认的真实重复和隐式依赖

可以立即收口而不损失行为覆盖的候选：

- `test_plain_lora_speed_runner.py:68`、`test_mfu_bench.py:107`、
  `test_signal_probe_runner.py:45` 中两组 GPU guard/torch mapping 测试逐字重复。改为以 runner 模块
  为参数的共享 contract，可从 6 个复制函数收口为 2 个参数化 contract。
- `test_krea2_attention_backend.py:22` 的 default-to-torch 与 aliases-to-torch 参数集语义重叠，
  可合为一张输入/规范值表。
- 多个前端文件重复 `_read`、repo root 和 Node subprocess boilerplate；已有
  `frontend_test_support.py`，但尚未提供统一 read/Node/jsdom harness。
- `test_training_queue.py:27` 的 internal/external root patch helper 重复修改多个模块全局常量，
  应由隔离 service context fixture 替代。
- `test_training_frontend_config_ui.py` 等通过 `globals().update(vars(support))` 隐式注入符号，
  应改为显式 import；否则 support 新增同名符号可能静默覆盖测试全局。

### P2：CI 没有任何测试门禁

仓库只有 `.github/workflows/release.yml`，并明确注明不设 test gate。CUDA/FlashAttention 无法在免费
runner 上覆盖是合理限制，但纯 CPU unit、配置、路径安全、Web 后端 smoke 和 Node 语法/行为测试
仍可以分层运行。当前本地 2,804 项与发布流程之间没有自动连接。

## 不应删除的覆盖

以下内容即使文件很大也必须保留，只能重组或使用更小 fixture：

- 输出根、路径穿越、symlink 和跨域删除边界；
- queue/history 的崩溃恢复、重试、archive/manual override 和外置配置根；
- lazy model loading 顺序与 compile-after-apply；
- constant token bucket、native flatten 和 DiT 5D latent shape；
- model family registry 未知 family/缺 handler/未注册 mode 的 fail-closed；
- Z-Image plain LoRA、attention、checkpoint 和 block-swap 范围限制；
- Krea-2 attention、compile、full checkpoint、block swap 和 NF4 BF16 compute 强制边界；
- LoRA 三轴 routing、旧 metadata 拒绝、fused/split projection 保存加载；
- 前端生产入口 import graph、关键 DOM contract、禁止隐式全局状态回流。

这些测试的维护成本高，但保护的是用户数据、模型兼容性或已验证的数值边界。不能为了文件数目标
把它们降成源码字符串断言或直接删除。

## 建议治理顺序

### 阶段 1：先建立套件语义

1. 新增 `integration`、`hardware`、`benchmark`、`probe` marker。
2. 新增显式 `test-core`、`test-integration`、`test-hardware`、`test-experimental`、`test-all`
   任务；第一轮保持现有 `test-unit` 行为兼容，待调用方迁移后再决定是否重命名。
3. 为 Node/jsdom、CUDA 和外部网络能力提供集中 fixture/gate，不再每个文件自行 skip。
4. 将上表首批 8 个文件标入正确层级；混合文件按 test 粒度标记，避免整文件误排除。

阶段验收：CPU `test-core` 在 60 秒内完成；`test-all --collect-only` 仍收集全部预期测试且无错误。

### 阶段 2：删除真实重复和实验伪单测

1. 参数化三套 runner 的共享 GPU contract。
2. 合并同一 normalize/reject 输入矩阵。
3. 将 benchmark/probe 私有 helper 的测试与实验实现一起归入实验层；生产层只留公共 contract。
4. 对只打印、自测手写逻辑、只验证文件存在的测试继续执行零容忍删除。

阶段验收：每个删除项必须说明被哪个保留测试或上层任务替代，禁止只用减少行数作为依据。

### 阶段 3：改写 source-probe 和收口 fixture

1. 优先改写 `test_deferred_sample_cleanup.py`，用可注入的小训练生命周期 fixture 验证调用顺序。
2. 前端只保留 module graph/DOM/cache token 等少量静态护栏；业务状态用 Node/jsdom 或浏览器行为
   测试。
3. TOML/JSON/metadata 一律用结构化 parser 断言，不比较格式字符串。
4. 建立统一的 frontend Node/jsdom harness 和 Web service context fixture。

阶段验收：新增源码字面量断言必须在 review 中说明它保护的架构不变量；普通业务行为不得使用此
模式。

### 阶段 4：拆分热点并加轻量 CI

1. 将所有超过 1,200 行的测试按生产服务/feature 拆分，避免继续追加。
2. Dragon 小文件按 runtime/history/dataset/config 归并；不要重新合成单个 Dragon 巨型文件。
3. GitHub Actions 至少运行 CPU core、backend smoke、Node 语法和结构化配置/文档完整性；硬件和
   probe 保持显式本地或自托管任务。

阶段验收：默认 CPU 门禁稳定、失败能定位到明确领域；发布不再完全依赖人工记得运行测试。

## 建议的规模目标

规模目标用于约束增长，不作为删覆盖的 KPI：

- 第一轮通过 Dragon 归并和实验迁层，将顶层测试文件数控制到约 225--235；
- 默认 CPU core 在 60 秒内完成，完整收集保持无错误；
- 单测试文件原则上不超过 1,200 行，前端行为文件优先控制在 600--800 行；
- 新功能优先加入现有领域套件，只有新的独立运行时、服务边界或依赖层级才新建文件；
- 每个 hardware/probe 测试必须有显式入口、依赖条件和对应生产 contract。

文件数不是最终质量指标。更重要的指标是默认门禁耗时、skip 分布、source-probe 占比、重复 fixture
数量，以及一次生产改动触发的无关测试失败数。

## 本轮验证

- `pytest --collect-only -q tests`：2,804 项，12.37 秒，collection error 为 0。
- `timeout 60 pytest -q tests --durations=25`：超时退出，进度约 15%；因未完成，无法获得可靠
  durations 排名。
- 审计仅新增本文并更新 findings 索引；未修改测试、生产代码、用户配置或运行数据。
