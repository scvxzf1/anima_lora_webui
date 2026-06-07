# Anima LoRA WebUI 上帝文件复审报告

日期：2026-06-07
范围：基于当前工作区重新静态扫描；只分析和给出维护建议，未修改业务代码。

## 复审结论

这轮修改已经推进了 GOAL-01 的一部分：`training_service.py` 的 GPU 和 progress 解析逻辑、`config_service.py` 的路径工具已经拆出独立模块。这个方向是对的，且新模块目前没有形成新的上帝文件。

但主债务还没有结构性解除：

- `web/static/js/features/legacy-app.js` 仍是 17882 行，还是项目最大上帝文件。
- `web/static/style.css` 从旧报告的 15528 行涨到 15621 行，样式债务继续扩大。
- `web/services/training_service.py` 从 5635 行降到 5400 行，有进展，但仍是后端最大服务上帝文件。
- `web/services/config_service.py` 当前是 4248 行，比旧报告 4242 行略多，说明只拆 path 工具还不足以扭转增长。
- 新增关注：`library/datasets/base.py` 已到 2128 行，`BaseDataset` 同时承担 bucket、图像读取、latents/text cache、IP features、identity pairs、contrastive negatives、inversion/BYG 等职责，已经接近训练数据管线的上帝类。

## 当前优先级

| 优先级 | 文件 | 当前行数 | 旧报告行数 | 当前判断 |
|---|---:|---:|---:|---|
| P0 | `web/static/js/features/legacy-app.js` | 17882 | 17882 | 未变薄；仍是最大风险 |
| P0 | `web/static/style.css` | 15621 | 15528 | 继续增长；CSS 拆分应提前 |
| P1 | `web/services/training_service.py` | 5400 | 5635 | 有效瘦身 235 行，但仍需拆 queue/history/runtime |
| P1 | `web/services/config_service.py` | 4248 | 4242 | path helper 已拆，但主模块仍增长 |
| P1 | `library/datasets/base.py` | 2128 | 1956 | 新增重点；数据管线职责过密 |
| P2 | `train.py` | 2782 | 2711 | 稍有增长；仍是核心编排大文件 |
| P2 | `networks/lora_anima/network.py` | 3317 | 3309 | 基本持平；核心 network 大类 |
| P2 | `library/anima/models.py` | 2541 | 2533 | 大模型定义集合，暂不优先拆 |

## 当前扫描数据

| 文件 | 行数 | 函数/方法数 | 类数 | 备注 |
|---|---:|---:|---:|---|
| `web/static/js/features/legacy-app.js` | 17882 | 943 | 0 | 196 个唯一 DOM id 查询，291 个事件监听，66 处 API 调用 |
| `web/static/style.css` | 15621 | 0 | 0 | 约 2210 个选择器/规则块 |
| `web/services/training_service.py` | 5400 | 275 | 1 | 类内/模块级函数仍大量集中 |
| `web/services/config_service.py` | 4248 | 205 | 0 | 仍承担配置、数据集、文件、分组、预检等领域 |
| `networks/lora_anima/network.py` | 3317 | 70 | 4 | LoRA family 核心模型对象 |
| `train.py` | 2782 | 67 | 1 | 训练入口和 `AnimaTrainer` 生命周期 |
| `library/datasets/base.py` | 2128 | 48 | 1 | `BaseDataset` 单类职责过密 |
| `library/anima/models.py` | 2541 | 多类 | 多类 | 模型结构定义，复杂但边界相对明确 |

本次统计排除了 `.venv/`、`models/`、`output/`、`logs/`、训练历史、队列、`__pycache__` 和 `_vendor` 目录。

## 已完成拆分的评价

### `web/services/training/gpu.py`

当前 151 行，8 个函数。职责清楚：

- GPU 白名单解析。
- `CUDA_VISIBLE_DEVICES` 注入。
- `nvidia-smi` GPU 统计和可用 GPU 列表解析。

判断：健康。这个模块可以继续保持独立，不需要再拆。

### `web/services/training/progress_parser.py`

当前 262 行，19 个函数。职责主要是 progress JSONL 和 metric 文本解析：

- step rate 中位数。
- metric normalize / seen key。
- progress line 解析。
- progress JSONL event 转 metric。
- display step 分配。

判断：目前健康，但要注意边界。后续如果继续加入 history timeline、chart formatting、WebSocket 消息组装，就会变成新的“小上帝工具箱”。建议只保留纯解析和纯数值逻辑。

### `web/services/config/paths.py`

当前 49 行，5 个函数。职责清楚：

- config 相对路径 normalize。
- configs 边界内 safe resolve。
- config 子目录 safe resolve。
- 项目路径 resolve / display path。

判断：健康。后续 config 模块拆分应继续复用这里，不要在新模块里重复写路径解析。

## P0：`legacy-app.js` 仍未实质拆分

当前状态：

- 行数保持 17882。
- DOM id 查询数量仍是 196。
- 事件监听仍是 291。
- API 调用仍是 66。
- 主要 section 仍覆盖配置表单、TOML、训练控制、WebSocket、设置、预览、队列、轮询、事件绑定和工具函数。

本轮前端只做了很小的测试护栏和局部调整，并没有拆 feature。旧结论仍成立：它是当前最典型的上帝文件。

建议下一步：

1. 先拆 `theme`、`gpu-picker`、`standalone-warning` 这种低耦合 feature。
2. 再拆 `config-form`、`dataset-presets`、`toml-manager`。
3. 最后拆 `training-dashboard` 和 `history-list`。

不要直接跳到配置表单大拆。当前 `legacy-app.js` 闭包状态太多，先用轻量 feature 建立模式更稳。

## P0：`style.css` 债务继续扩大

当前状态：

- 行数从 15528 增到 15621。
- 选择器/规则块约 2210。
- 仍然包含全局 token、按钮、配置页、数据集页、训练页、历史、预览、弹窗、响应式、权重分析等所有样式。
- 当前没有 `web/static/css/` 拆分目录。

判断：CSS 已经不只是“长”，而是会阻碍任何前端 feature 拆分。JS feature 拆出来后，如果样式仍挤在一个文件里，后续改 UI 仍会跨域碰撞。

建议提前做 GOAL-06 的第一步：

- 先不改选择器，只按 section 原样搬运。
- `style.css` 暂时作为 `@import` 聚合入口。
- 拆分文件和 feature 边界对齐。

建议拆分顺序：

1. `base/tokens.css`
2. `base/reset.css`
3. `components/buttons.css`
4. `components/forms.css`
5. `features/config.css`
6. `features/training.css`
7. `features/history.css`
8. `features/preview.css`
9. `features/weight-analysis.css`
10. `responsive.css`

## P1：`training_service.py` 有进展，但仍是后端主债务

已改善：

- GPU 相关逻辑已转到 `web/services/training/gpu.py`。
- progress/metric 解析逻辑已转到 `web/services/training/progress_parser.py`。
- 旧函数名保留为兼容 facade。

仍存在的问题：

- `TrainingService` 仍同时管理启动/停止、预处理、队列、历史、runtime、日志、指标、系统监控、续训、删除安全。
- 模块级 helper 仍覆盖 runtime dataset clone、nl/tag mix、trigger clone、history timeline、resume checkpoint、JSON/JSONL IO、queue runtime 删除等多个领域。
- 测试仍大量 monkeypatch `training_service.HISTORY_DIR`、`training_service.resolve_output_root`、`training_service._write_json_atomic` 等模块全局。

建议下一步拆分：

- `web/services/training/json_store.py`
  - `_read_json`、`_write_json_atomic`、`_read_jsonl_limited`、`_count_jsonl` 等。
- `web/services/training/queue_store.py`
  - queue state 读写、backup、normalize、item id、clearable state。
- `web/services/training/history_store.py`
  - history meta 读取、summary、artifact path、collection settings。
- `web/services/training/runtime.py`
  - runtime config 准备、runtime meta、auto probe path。
- `web/services/training/runtime_dataset.py`
  - dataset clone、nl/tag mix、trigger clone。

注意：queue/history/runtime 都涉及用户数据和删除安全，拆分时必须保留原有路径边界检查。

## P1：`config_service.py` 仍需大拆

已改善：

- `web/services/config/paths.py` 已承接路径 normalize 和 safe resolve。
- 测试已开始直接覆盖 `config_paths`。

仍存在的问题：

- 当前行数 4248，比旧报告略多。
- 仍集中 method/variant、merged config、dataset presets、dataset preview、dataset editor、preflight、raw TOML、sample prompts、file groups、locks、output runs、system restore。
- `web/routes/config.py` 仍主要依赖 `config_service.py` 这个大 facade。

建议下一步按低风险顺序拆：

1. `web/services/config/methods.py`
   - `list_methods`、`list_variants`、variant metadata。
2. `web/services/config/sample_prompts.py`
   - sample prompts path 策略、读写、分叉。
3. `web/services/config/output_runs.py`
   - output run list/read/save-as。
4. `web/services/config/raw_files.py`
   - raw TOML load/save/delete/patch/preview patch。
5. `web/services/config/groups.py`
   - file groups、locks、export zip。
6. `web/services/config/datasets.py`
   - dataset preset/editor/doc build/summary。
7. `web/services/config/preflight.py`
   - preflight checks。

建议继续保留 `config_service.py` facade，直到路由和测试稳定迁移。

## P1：新增关注 `library/datasets/base.py`

这个文件在旧报告里没有被列为重点，但当前复审建议提升到 P1。

当前问题：

- `BaseDataset` 单类 2000+ 行。
- 同时负责 caption 处理、tokenize、bucket、图像注册、latents cache、text encoder cache、image load、IP feature sidecar、identity pair、contrastive negatives、inversion runs、BYG tuple、`__getitem__` 样本拼装。
- 训练数据管线的多个实验能力都追加到了同一个类里。

它和 WebUI 上帝文件不同：这里有真实训练数据生命周期耦合，不能粗暴拆。但继续往 `BaseDataset` 追加新训练特性，会让数据行为越来越难验证。

建议拆分方向：

- `library/datasets/cache_latents.py`
  - latents cache 检查、缓存写入、读取策略桥接。
- `library/datasets/cache_text.py`
  - text encoder output cache 检查、缓存写入、读取。
- `library/datasets/ip_features.py`
  - IP-Adapter PE feature sidecar 读取和 identity pair feature 选择。
- `library/datasets/contrastive.py`
  - contrastive negatives sampler 接入和负样本 TE 读取。
- `library/datasets/editing_sources.py`
  - inversion runs、BYG tuple、cond latent 加载。
- `library/datasets/sample_builder.py`
  - `__getitem__` 中样本 dict 组装的纯逻辑。

建议先拆 helper，不要先拆 `BaseDataset` 类继承结构。数据集行为很容易影响训练数值，优先保持 API 不变。

建议验收：

```bash
timeout 60 python -m pytest tests/test_preprocess_dataset.py tests/test_latents_cache_strategy.py tests/test_constant_token_buckets.py
timeout 60 python -m pytest tests/test_identity_pairs.py tests/test_soft_tokens_contrastive.py
```

## P2：`train.py`

当前 2782 行，比旧报告增加 71 行。它仍是训练入口和 `AnimaTrainer` 主生命周期中心，但已有 `library/training/bootstrap.py`、`loop.py`、`checkpoints.py`、`progress.py` 等承接拆分。

当前不建议抢先大拆。可以等 WebUI 和 dataset 债务稳定后处理：

- sample preview / deferred sample decode。
- parser 剩余大块。
- router/method adapter glue。
- constant-token bucket 辅助函数。

## P2：`networks/lora_anima/network.py`

当前 3317 行，基本持平。仍是 LoRA family 核心对象，聚合 router、buffer wiring、metrics、state_dict、optimizer params、merge/save。

它的复杂度相当一部分来自真实 adapter 生命周期，不建议和 WebUI 同期大拆。

建议后续拆：

- `router_state.py`
- `router_metrics.py`
- `grad_stats.py`
- `optimizer_params.py`
- `state_io.py`

验收重点必须放在 checkpoint、metadata、router、load/save 数值兼容。

## 非优先大文件

以下文件大，但当前不建议作为上帝文件治理第一优先级：

- `library/anima/models.py`：2541 行，主要是模型结构定义；可以长期优化，但不是当前最乱的业务聚合点。
- `library/models/qwen_vae.py`：2049 行，VAE 模型实现，接近外部模型定义/适配层，不宜优先拆。
- `tests/test_training_resume.py`：3892 行，测试大是后端上帝文件的副作用。源码拆稳前，不建议先拆测试。
- `tests/test_web_config_service.py`：3194 行，同上，先作为安全网保留。
- `web/routes/config.py` / `web/routes/training.py`：handler 多，但仍基本是 HTTP 层，优先级低于 service。

## 当前风险判断

### 正向变化

- 已出现 `web/services/training/` 和 `web/services/config/` 目录，说明拆分方向落地。
- 新模块职责目前清楚，没有新上帝文件。
- 后端测试已开始直接覆盖新 helper，兼容 facade 策略可行。

### 仍需警惕

- `training_service.py` 保留大量 wrapper 后，函数数没有明显下降；继续拆时要逐步迁移测试直接打新模块，否则旧模块会长期保持“大而全 facade”。
- `config_service.py` 只拆 path 工具收益太小，下一步必须拆真实业务域。
- 前端还没开始实质拆分，`legacy-app.js` 的护栏只能防止增长，不能降低复杂度。
- `style.css` 仍在增长，可能抵消后续 JS feature 拆分收益。
- `library/datasets/base.py` 已经是新的中期风险点，训练实验功能不要继续直接塞进 `BaseDataset`。

## 更新后的执行建议

建议从原 GOAL 顺序稍微调整：

1. 继续 GOAL-02：优先拆 `config_service.py` 的 methods、sample prompts、output runs、raw files。
2. 继续 GOAL-03：拆 `training_service.py` 的 JSON store、queue store、history store、runtime paths。
3. 提前插入 CSS 第一阶段拆分：`style.css` 原样按 section 搬运，避免继续单文件增长。
4. 执行 GOAL-04：拆 `legacy-app.js` 的 theme、GPU picker、standalone warning。
5. 新增 DATASET-GOAL-01：拆 `library/datasets/base.py` 的 cache 和 sidecar helper。

不建议现在做：

- 不建议直接拆 `train.py`。
- 不建议直接拆 `networks/lora_anima/network.py`。
- 不建议重写配置表单 UI。
- 不建议删除 `config_service.py` / `training_service.py` facade。

## 建议测试矩阵

后端 config 拆分：

```bash
timeout 60 python -m pytest tests/test_web_config_service.py tests/test_gui_variants.py tests/test_preview_service.py
```

后端 training 拆分：

```bash
timeout 60 python -m pytest tests/test_training_queue.py tests/test_training_resume.py tests/test_training_gpu_selection.py
```

前端 feature / CSS 拆分：

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py
```

dataset 拆分：

```bash
timeout 60 python -m pytest tests/test_preprocess_dataset.py tests/test_latents_cache_strategy.py tests/test_constant_token_buckets.py tests/test_identity_pairs.py tests/test_soft_tokens_contrastive.py
```

训练/network 核心后续拆分：

```bash
timeout 60 python -m pytest tests/test_training_bootstrap.py tests/test_network_registry.py tests/test_network_cfg.py tests/test_factory_metadata_flow.py
```

## 总体判断

本轮修改是正确方向，但仍处在“拆出第一批 helper”的早期阶段。项目当前的上帝文件排序没有根本改变：前端 `legacy-app.js` / `style.css` 仍是 P0，后端 `training_service.py` / `config_service.py` 仍是 P1。

新的重要变化是：`library/datasets/base.py` 已经值得纳入治理队列。它不像 WebUI 文件那样显眼，但训练数据链路越复杂，越容易成为后续训练行为回归的源头。建议在 WebUI 后端 facade 稳住后，把 dataset helper 拆分作为独立中期目标推进。
