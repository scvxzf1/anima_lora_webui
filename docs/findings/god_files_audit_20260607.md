# Anima LoRA WebUI 上帝文件审计报告

日期：2026-06-07
范围：只做静态分析和维护建议，未修改业务代码。

## 结论摘要

项目里确实存在几类明显的“上帝文件”。最需要优先治理的是 WebUI 前端过渡层和 Web 后端服务层：

| 优先级 | 文件 | 行数 | 主要问题 | 建议策略 |
|---|---:|---:|---|---|
| P0 | `web/static/js/features/legacy-app.js` | 17882 | 单文件承载配置表单、TOML 管理、数据集预设、训练控制、历史、WebSocket、全局设置、拖拽排序等大量状态和 DOM 逻辑 | 按 feature 继续拆出 `createXFeature(ctx)`，它应只剩旧模块胶水 |
| P0 | `web/static/style.css` | 15528 | 所有页面、组件、主题、响应式、动效集中在一个 CSS 文件，选择器范围和回归风险不断扩大 | 按 base/components/features 拆 CSS，并和前端 feature 边界对齐 |
| P1 | `web/services/training_service.py` | 5635 | `TrainingService` 同时管训练进程、预处理、队列、历史、runtime 目录、日志指标、GPU 监控、恢复训练和文件删除安全 | 先拆纯函数工具和存储层，再拆运行时/队列/历史服务 |
| P1 | `web/services/config_service.py` | 4242 | 配置合并、数据集预设、TOML 文件管理、分组/锁、导入导出、预检、sample prompts、输出 run 配置全部混在一个模块 | 按配置域拆模块，保留兼容 facade |
| P2 | `train.py` | 2711 | 仍是训练入口和 `AnimaTrainer` 主生命周期中心，但已有 `library/training/` 承接拆分 | 继续瘦身入口，优先抽离 sample、metadata、router hook glue |
| P2 | `networks/lora_anima/network.py` | 3309 | `LoRANetwork` 聚合 router、buffer wiring、metrics、load/save、optimizer params、merge 等职责 | 按 router state、metrics、state_dict、optimizer 分阶段拆 mixin/helper |

其中 `legacy-app.js` 和 `style.css` 是最典型的上帝文件；`training_service.py` / `config_service.py` 是后端维护风险最高的上帝模块。`train.py` 和 `networks/lora_anima/network.py` 更像“核心编排大文件”，已有部分拆分，不建议和 WebUI 债务一起大动。

## 判定标准

本次按以下维度判断，而不是只看行数：

- 文件是否承载多个业务域。
- 是否持有跨域状态或大量全局可变状态。
- 是否直接绑定大量 DOM / API / 文件系统 / 子进程等外部边界。
- 是否让测试需要大面积 monkeypatch 同一个模块。
- 是否已经有模块化方向，但旧文件仍承担过渡债务。
- 是否修改一个小功能时容易牵动无关区域。

## 关键扫描数据

| 文件 | 行数 | 函数/方法数 | 类数 | 备注 |
|---|---:|---:|---:|---|
| `web/static/js/features/legacy-app.js` | 17882 | 943 | 0 | 196 个唯一 DOM id 查询，291 个事件监听，66 处 API 调用 |
| `web/static/style.css` | 15528 | 0 | 0 | 约 2196 个选择器/规则块，9 个大 section |
| `web/services/training_service.py` | 5635 | 275 | 1 | `TrainingService` 类内 86 个方法，模块级 helper 189 个 |
| `web/services/config_service.py` | 4242 | 205 | 0 | 模块级函数 202 个 |
| `train.py` | 2711 | 66 | 1 | `AnimaTrainer` 主入口，已有 bootstrap/loop/checkpoint 等拆分 |
| `networks/lora_anima/network.py` | 3309 | 69 | 4 | LoRA family 核心模型对象 |

`web/routes/training.py` 有 35 个 handler，`web/routes/config.py` 有 45 个 handler。路由文件虽然偏长，但基本还是 HTTP 层薄封装，问题小于 service 层。

## P0：`legacy-app.js`

### 当前职责

`legacy-app.js` 目前至少混合了这些职责：

- 全局初始化、主题、GPU 选择器、顶层 tab。
- 配置表单渲染、字段草稿、兼容 patch、字段搜索/折叠。
- 数据集编辑器、数据集预设、数据集图片预览。
- TOML 文件编辑器、配置分组、锁定、导入导出、拖拽排序。
- 训练启动/停止、预处理、续训、继续训练来源选择。
- WebSocket、训练状态轮询、指标/日志/图表。
- 历史任务、集合分组、历史拖拽、时间线。
- 全局设置、预览图、队列接入、权重分析接入。
- 大量 DOM 工具、路径格式化、下载、确认弹窗等工具函数。

项目维护说明里已经明确它是“第一阶段拆分的过渡层”，所以这个文件属于已知技术债，不是自然增长出来的合理入口。

### 主要风险

- 任意配置页改动都可能碰到训练、历史或队列状态。
- 状态变量集中在 `createLegacyApp()` 闭包顶部，难以确认某个状态只被哪个功能使用。
- 拖拽、弹窗、列表渲染等交互逻辑重复出现，修一个边界容易漏另一个。
- 前端测试大量直接搜索 `legacy-app.js` 文本，拆分时需要同步调整静态钩子测试。

### 拆分建议

建议按“最少跨域依赖”顺序拆：

1. `web/static/js/features/gpu-picker/`
   - 拆出 GPU 列表加载、本地选择、展示文案。
   - 对外只暴露 `selectedGpuPayload()`、`loadGpuOptions()`、`render()`。

2. `web/static/js/features/config-form/`
   - 拆配置表单状态、字段渲染、搜索/折叠、network args 表单适配。
   - `legacy-app.js` 保留 `loadConfig()` 调用入口即可。

3. `web/static/js/features/dataset-presets/`
   - 拆数据集预设列表、预览、保存/导入/删除、preset group UI。
   - 和后端未来的 dataset service 对齐。

4. `web/static/js/features/toml-manager/`
   - 拆 TOML 编辑器、文件分组、锁定、导出、拖拽排序。
   - 拖拽底层抽成 `js/shared/drag-sort.js`，供历史集合继续复用。

5. `web/static/js/features/training-dashboard/`
   - 拆训练控制、状态轮询、WebSocket、指标面板。
   - 队列和 preview 已经有模块，不要回填新逻辑到 legacy。

6. `web/static/js/features/history/`
   - 当前已有 `history-detail/`，但历史列表/集合/拖拽仍在 legacy。
   - 先拆历史列表和集合管理，详情继续走已有模块。

拆分时不要一次性重写 UI。先做“搬家型重构”：函数移动、状态局部化、公开窄 API，确保行为不变。

### 建议验收

- `timeout 60 python -m pytest tests/test_training_frontend_state.py`
- 浏览器手验：配置切换、TOML 保存、数据集预览、训练启动预检、历史列表、队列启动、预览弹窗。

## P0：`style.css`

### 当前职责

CSS 文件覆盖范围包括：

- 全局 theme token 和浅色主题覆盖。
- Buttons、config split layout、config form。
- 数据集编辑器、TOML 管理器、输出 run 管理。
- training tab、历史集合、队列、训练状态。
- preview workspace、preflight dialog、训练 forge surface。
- responsive 规则、print、权重分析页面。

这已经超过“一个静态样式入口”的合理边界。当前 CSS 的主要问题不是某一段写得差，而是所有视觉状态都共享同一个巨大选择器空间。

### 拆分建议

建议拆成这些文件：

- `web/static/css/base/tokens.css`：`:root`、主题变量、基础字体/页面背景。
- `web/static/css/base/reset.css`：通用元素、focus、基础表单。
- `web/static/css/components/buttons.css`
- `web/static/css/components/dialog.css`
- `web/static/css/components/forms.css`
- `web/static/css/components/drag-sort.css`
- `web/static/css/features/config.css`
- `web/static/css/features/datasets.css`
- `web/static/css/features/toml-manager.css`
- `web/static/css/features/training.css`
- `web/static/css/features/history.css`
- `web/static/css/features/preview.css`
- `web/static/css/features/weight-analysis.css`
- `web/static/css/responsive.css`

落地方式建议先拆文件但保持选择器不改名，通过 `style.css` 用 `@import` 聚合。等功能模块稳定后，再考虑按 feature 在 HTML 中拆入口，避免第一步就改缓存策略和加载顺序。

### 建议验收

- `timeout 60 python -m pytest tests/test_training_frontend_state.py`
- 桌面宽屏、900px、640px 三个视口截图检查：配置页、训练页、历史页、预览页、权重分析页。

## P1：`training_service.py`

### 当前职责

这个模块已经不是单纯的训练服务，实际包含：

- 子进程启动/停止、环境变量、GPU 白名单。
- 预处理和训练串联。
- 训练队列持久化、重试、取消、清理、失败策略。
- 历史任务 meta、日志、artifact、集合设置、批量删除。
- runtime run 目录创建、runtime config/dataset config 克隆。
- nl/tag mix 和 trigger clone 数据集物化。
- stdout/tqdm/progress jsonl 解析、指标去重、WebSocket 广播。
- GPU/system stats 轮询。
- resume checkpoint 扫描和诊断。
- JSON/JSONL 文件 IO、安全路径检查、删除目录边界检查。

`TrainingService` 自身有 86 个方法，模块级 helper 189 个。这里最危险的是职责边界和文件安全逻辑混在一起：未来改队列或历史删除时，很容易误触 runtime/output root 边界。

### 拆分建议

建议先拆纯模块，不改变外部 API，`TrainingService` 暂时作为 facade：

1. `web/services/training/gpu.py`
   - `_get_gpu_stats`、`_list_available_gpus`、`_apply_gpu_whitelist`、白名单 normalize。
   - 对应测试：`tests/test_training_gpu_selection.py`。

2. `web/services/training/queue_store.py`
   - queue.json 读写、backup、状态 normalize、item id。
   - 先不搬调度逻辑，只搬存储纯函数。

3. `web/services/training/history_store.py`
   - history task 读取、摘要、artifact path、collection settings、JSONL 读取。
   - 保留路径安全校验，单测覆盖逃逸路径。

4. `web/services/training/runtime.py`
   - `_prepare_web_runtime_config`、runtime meta、auto probe path、runtime dataset clone。
   - nl/tag mix 和 trigger clone 可以再下沉到 `runtime_dataset.py`。

5. `web/services/training/progress_parser.py`
   - tqdm、progress jsonl、metric line 解析、rate 格式化。
   - 这是最适合先拆的低风险纯逻辑。

6. `web/services/training/process_runner.py`
   - 子进程生命周期、stdout drain、stop family、broadcast 消息组合。
   - 最后再拆，因为它和 `TrainingService` 状态耦合最多。

### 建议验收

- `timeout 60 python -m pytest tests/test_training_queue.py`
- `timeout 60 python -m pytest tests/test_training_resume.py`
- `timeout 60 python -m pytest tests/test_training_gpu_selection.py`
- `timeout 60 python -m pytest tests/test_preview_service.py`

## P1：`config_service.py`

### 当前职责

`config_service.py` 集中了：

- method/variant/preset 列表。
- merged config 加载和 Web 自动数据路径。
- 数据集预设 CRUD、诊断、图片预览、caption 检测。
- 数据集编辑器保存和训练配置 patch。
- 训练配置预检。
- raw TOML 读取、保存、patch、删除、另存。
- sample prompts 文件读写。
- 配置分组、排序、锁定、导出 zip。
- output run config 浏览和另存。
- 路径 normalize、安全 resolve、系统预设还原。
- nl/tag caption 分类。

这类模块的问题是“工具函数看似都和配置有关”，但实际上至少有五个不同领域：训练配置、数据集、文件管理、预检、输出 run。

### 拆分建议

建议目标结构：

- `web/services/config/methods.py`
  - `list_methods`、`list_variants`、variant metadata。
- `web/services/config/merge.py`
  - `load_merged_config`、`apply_auto_data_dirs`、训练配置路径。
- `web/services/config/datasets.py`
  - dataset preset CRUD、dataset editor、dataset summary。
- `web/services/config/dataset_preview.py`
  - 图片扫描、caption meta、preview image resolve。
- `web/services/config/preflight.py`
  - `preflight_training_config` 和各类检查器。
- `web/services/config/files.py`
  - raw TOML save/patch/delete、system restore。
- `web/services/config/groups.py`
  - file group、lock、export zip。
- `web/services/config/sample_prompts.py`
  - sample prompts 路径策略与读写。
- `web/services/config/output_runs.py`
  - output run 列表、读取、另存。
- `web/services/config/paths.py`
  - `_safe_resolve`、normalize、display path 等共享路径工具。

迁移时建议保留 `web/services/config_service.py` 作为兼容 facade，先 re-export 原函数名。这样 `web/routes/config.py` 和现有测试可以分批迁移。

### 建议验收

- `timeout 60 python -m pytest tests/test_web_config_service.py`
- `timeout 60 python -m pytest tests/test_gui_variants.py`
- `timeout 60 python -m pytest tests/test_preprocess_paths.py`
- `timeout 60 python -m pytest tests/test_preview_service.py`

## P2：`train.py`

`train.py` 很长，但当前不属于最差的上帝文件。原因：

- 项目已经把训练 setup 拆到了 `library/training/bootstrap.py`。
- 训练循环已拆到 `library/training/loop.py`。
- checkpoint、metadata、loss、sampler、progress、memory probe 等也已有独立模块。
- `train.py` 仍承担入口、`AnimaTrainer` hook、模型加载、batch 处理、sample、parser 等编排职责。

后续可以继续拆，但不建议优先级高于 WebUI。建议方向：

- 把 sample preview / deferred sample decode 相关逻辑拆到 `library/training/samples.py`。
- 把 `setup_parser()` 继续瘦身，按已有 `library/training/cli_args.py` 聚合。
- 把 router conditioning / method adapter glue 的训练入口逻辑继续收口到 `library/training/method_adapter.py` 或专门模块。
- 保持 `AnimaTrainer` 作为兼容入口，不做一次性改名或大范围继承改造。

建议验收：

- `timeout 60 python -m pytest tests/test_training_bootstrap.py tests/test_deferred_sample_cleanup.py tests/test_progress_sink.py`
- 涉及 batch/adapter 时追加 `tests/test_network_registry.py`、`tests/test_method_network_lifecycle.py`。

## P2：`networks/lora_anima/network.py`

这个文件是 LoRA family 的核心对象，长是事实，但部分复杂度来自真实业务边界：router、timestep mask、FEI、stacked experts、state_dict、optimizer param 分组、merge/save 都和网络对象生命周期强相关。

已有拆分基础：

- `networks/lora_anima/config.py`：冻结配置对象。
- `networks/lora_anima/factory.py`：创建网络和 checkpoint sniff。
- `networks/lora_anima/loading.py`：state_dict 形状转换。
- `networks/lora_modules/`：单个变体模块。

建议不要立刻拆类继承层。更稳的方式：

- `router_state.py`：`set_sigma`、`set_fei`、routing buffers、clear caches 相关 helper。
- `router_metrics.py`：entropy、balance loss、router stats。
- `grad_stats.py`：up grad stats 采集和格式化。
- `optimizer_params.py`：LoRA+ lr ratio 和 optimizer param groups。
- `state_io.py`：`save_weights`、`load_weights` 辅助逻辑继续从 `network.py` 下沉。

这类拆分需要数值和 checkpoint 兼容性验证，不适合和 WebUI 重构混在同一批。

建议验收：

- `timeout 60 python -m pytest tests/test_network_registry.py tests/test_network_cfg.py tests/test_factory_metadata_flow.py`
- `timeout 60 python -m pytest tests/test_lora_custom_autograd.py tests/test_global_router.py tests/test_chimera_router_stats.py`

## 非优先候选

以下文件也偏大，但暂不建议作为“上帝文件治理”优先目标：

- `web/static/index.html`：1277 行，主要是 DOM 锚点和页面骨架。前端 JS/CSS 拆完后再考虑模板化，否则会增加同步成本。
- `web/routes/training.py` / `web/routes/config.py`：handler 多，但职责基本是 HTTP 参数读取和 service 调用。应等 service 层拆完后再瘦路由。
- `inference.py` / `library/inference/generation.py`：推理拆分已有 `docs/separation_plan.md`，按那个计划推进即可，不建议混入 WebUI 上帝文件治理。
- 大测试文件：`tests/test_training_resume.py`、`tests/test_web_config_service.py`、`tests/test_training_frontend_state.py` 体量大，但目前是给大模块兜底的安全网。拆源码前不要先拆测试。

## 推荐执行路线

### 阶段 1：纯搬家，降低风险

- 拆 `training_service.py` 的 GPU、progress parser、queue/history JSON IO 纯函数。
- 拆 `config_service.py` 的 path、sample prompts、output runs、dataset preview。
- 前端先拆 GPU picker、theme、轻量工具模块。
- 每次只移动一组函数，保留原导入路径 facade。

### 阶段 2：按功能模块收口状态

- `legacy-app.js` 中配置表单、数据集预设、TOML manager、历史列表分别变成 feature。
- 对每个 feature 定义自己的 state，不再共享 legacy 顶部大闭包状态。
- CSS 同步按 feature 拆文件，但选择器先不重命名。

### 阶段 3：重新定义边界

- `TrainingService` 只保留高层编排：start/stop/status/queue facade。
- `config_service.py` 只保留 re-export 或删除，路由直接调用新模块。
- `legacy-app.js` 缩到 bootstrap/旧兼容 glue，新增功能不得再写进去。

### 阶段 4：治理核心训练/network 大文件

- 在 WebUI 债务下降后，再处理 `train.py` 和 `network.py`。
- 每次拆分都要求 checkpoint、metadata、router、训练 step 行为可回归。

## 建议增加的防回归规则

- 前端新增功能禁止直接写入 `legacy-app.js`，除非只是接入新 feature。
- `legacy-app.js` 行数设置软上限：每次 PR 不允许净增加，除非同时解释原因。
- `style.css` 拆分后，新增样式必须进入对应 feature/component CSS。
- `web/services/training_service.py` 和 `config_service.py` 拆分期间保留 facade，但新函数不要继续加回旧模块。
- 对危险路径操作保留独立单测：output root、runtime dir、history artifact、queue runtime 删除。

## 最小测试矩阵

WebUI 前端拆分：

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py
```

配置服务拆分：

```bash
timeout 60 python -m pytest tests/test_web_config_service.py tests/test_gui_variants.py tests/test_preprocess_paths.py
```

训练服务拆分：

```bash
timeout 60 python -m pytest tests/test_training_queue.py tests/test_training_resume.py tests/test_training_gpu_selection.py
```

预览、output root、安全路径相关：

```bash
timeout 60 python -m pytest tests/test_preview_service.py tests/test_weight_analysis_service.py
```

训练/network 核心拆分：

```bash
timeout 60 python -m pytest tests/test_training_bootstrap.py tests/test_network_registry.py tests/test_network_cfg.py tests/test_factory_metadata_flow.py
```

## 总体判断

当前项目不是“全盘屎山”，而是 WebUI 快速增长后留下了明显的过渡层债务。`legacy-app.js`、`style.css`、`training_service.py`、`config_service.py` 是优先处理对象；训练和 LoRA network 核心虽然也大，但已有模块化方向，且业务耦合更真实，应该放在 WebUI 债务治理之后。

最稳的维护路线不是大爆破重写，而是持续把旧文件变成 facade：先搬纯逻辑，再局部化状态，最后收紧新功能入口。这样既能降低回归风险，也能让后续功能开发不再继续给上帝文件加砖。
