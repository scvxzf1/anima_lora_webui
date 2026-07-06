# 项目清理强制长跑目标书：跨子系统持续推进版

一句话：这份任务书是接在 `bd591b83` 之后的新入口，目标是让下一轮至少连续推进 3 小时，并且必须跨多个子系统做真实、可验证的小阶段。

日期：2026-07-06
状态：完成归档（2026-07-06）
前置已完成：

- `project_cleanup_long_running_goal_20260705.md`：已完成归档，提交 `f74b8255`
- `project_cleanup_next_stage_goal_20260705.md`：已完成归档，提交 `7c5c277c`
- `project_cleanup_sustained_goal_20260705.md`：已完成归档，提交 `bd591b83`

默认发布目标：本地 `main` -> `webui/main`
最低运行窗口：至少 3 小时。
理想运行窗口：4 到 6 小时。

---

## 🎯 1. 总目标

一句话：下一轮不要继续只围着一个局部打转，而是跨 WebUI、runtime、训练启动、LoRA/config 护栏做持续清理。

目标名称：

```text
跨子系统项目清理：WebUI 真实证据 + runtime/path 安全 + training launch 护栏 + LoRA/config 残余保护
```

总目标：

- 继续补 `TASK-04`：WebUI 真 feature / DOM / 前端模块证据，优先补结构测试或本地静态 smoke。
- 继续补 `TASK-02` / `TASK-06`：runtime、路径隔离、launch helper、block-swap 低风险护栏。
- 继续补 `TASK-08`：训练启动和 forward helper 的顺序保护，只做 monkeypatch / 小 fixture 测试。
- 小步延续 `TASK-07` / `TASK-09` / `TASK-10`：只补残余 characterization tests、入口审计和门禁文档，不做大拆。
- 每轮更新 checkpoint，明确哪些只是阶段收口，哪些仍不能夸大。

---

## ⛔ 2. 硬完成条件

一句话：这次的完成条件比上一轮更硬，不能靠 2 小时或十几个阶段就收口。

只有同时满足以下条件，才允许标记目标完成：

1. **最低耗时门槛**
   - `goal.timeUsedSeconds >= 10800`，也就是至少 3 小时。
   - 如果当前环境没有 goal 时间工具，就用启动时的 `date +%s` 和当前 `date +%s` 计算。
   - 不允许靠 `sleep`、空等、反复无意义轮询凑时间。

2. **最低阶段数量**
   - 至少完成 `20` 个可验收小阶段。
   - 每个阶段必须产出至少一种真实证据：
     - 新测试或加宽测试。
     - 小范围源码护栏。
     - 只读审计清单。
     - 文档/checkpoint 更新，且必须绑定具体审计、测试或代码改动，不能单独刷阶段数。
     - 验证命令结果，且必须对应本阶段的具体审计范围、测试范围或改动范围。

3. **最低推进轮数**
   - 至少完成 `5` 个推进轮。
   - 每轮必须包含：
     - 只读审计。
     - 至少 1 个测试、源码小改或明确的文档修正。
     - 至少 1 组验证。
     - checkpoint 更新，且必须说明本轮新增事实，不能只重复上一轮结论。

4. **跨子系统覆盖**
   - 至少覆盖 `4` 个不同子系统。
   - 允许计入的子系统：
     - WebUI 前端 / DOM / 静态模块。
     - WebUI 后端服务 / 队列 / history / preview。
     - runtime / launch / config path / output root。
     - training bootstrap / forward / compile order。
     - LoRA save/load/config 护栏。
     - docs / CLI / type-check 门禁。

5. **验证密度**
   - 至少完成 `6` 组非纯文档验证。
   - 至少有 `3` 组 pytest 或 type-check。
   - 所有后台测试单条命令必须加 `timeout 60`。
   - 如果某个宽命令 60 秒内跑不完，必须拆成窄命令并记录。
   - 拆分后的验证必须通过；如果仍失败，必须修复或记录为阻塞，不能当作通过收口。

6. **最终收口**
   - 最后一轮必须跑总验证。
   - 最终必须显式 stage 实际修改文件。
   - 不允许 `git add -A`。
   - 提交并推送到 `webui/main`。

如果基础轮提前完成但未满足任一门槛：

- 不允许 complete。
- 必须进入第 13 节 EXT 扩展阶段池继续推进。

如果遇到阻塞：

- 只有连续 3 轮同一阻塞且无法安全推进，才允许标记 blocked。
- blocked 前必须写清楚已完成阶段、阻塞原因、恢复条件和未提交改动状态。

---

## 🛡️ 3. 禁止事项

一句话：长跑只能做低风险清理，不能把危险操作包装成“推进”。

禁止：

- 不跑真实训练。
- 不下载模型。
- 不删除、移动或清理用户数据。
- 不删除 `_legacy.py`。
- 不改 LoRA checkpoint key 格式。
- 不改 LoRA public API。
- 不改三轴路由语义：`use_moe_style` / `route_per_layer` / `router_source`。
- 不做 `git reset --hard`。
- 不做 `git checkout -- <path>`。
- 不做 force push。
- 不做 `git clean`。
- 不用 `git add -A`。
- 不做批量移动 / 批量删除 / 批量重命名。
- 不做核心依赖升级、全局安装或卸载。
- 不终止用户正在跑的训练、daemon 或 WebUI 队列。
- 不调用生产 API，不发送敏感数据。
- 不为了凑阶段做无意义格式化。
- 不为了凑时长拆空 helper。
- 不把全量 Web config 测试 60 秒超时说成通过。

禁止默认触碰目录：

- `.venv/`
- `.worktrees/`
- `models/`
- `output/`
- `logs/`
- `post_image_dataset/`
- `configs/imported/`
- `configs/web-training-history/`
- `configs/web-training-queue/`
- `web/configs/`
- `bench/mfu/assets/`
- `tmp/`

---

## 🧭 4. 启动规则

一句话：开跑前先确认仓库、远端、旧目标归档和计时起点。

启动时必须执行：

```bash
git status --short --branch
git log -1 --oneline --decorate
git rev-list --left-right --count HEAD...webui/main
date +%s
```

启动时必须阅读：

- `AGENTS.md`
- `docs/findings/project_cleanup_checkpoint_20260705.md`
- `docs/findings/project_cleanup_sustained_goal_20260706.md`

启动时必须确认：

- `project_cleanup_sustained_goal_20260705.md` 已是完成归档，不再重复执行。
- 当前活跃入口是本文件。
- 本轮不会碰用户数据目录。

每轮开始必须执行：

```bash
git status --short --branch
git diff --name-only
```

每轮结束必须执行：

```bash
git diff --check
```

每 30 分钟左右必须盘点：

- 当前 `goal.timeUsedSeconds` 或手动耗时。
- 已完成阶段数。
- 已覆盖子系统数。
- 已跑验证列表。
- 剩余安全阶段。
- 是否需要进入 EXT。

---

## 🧩 5. 推进轮总览

一句话：本目标至少 5 轮，每轮都要有不同方向的真实产物。

| 轮次 | 主线 | 最低阶段数 | 预计耗时 | 风险 |
|---|---|---:|---:|---|
| R0 | 基线、旧目标归档、候选池确认 | 2 | 15-25 分钟 | Low |
| R1 | WebUI 前端 / DOM / 静态模块证据 | 4 | 45-75 分钟 | Low-Medium |
| R2 | WebUI 后端服务 / 队列 / output root 安全 | 4 | 45-75 分钟 | Low-Medium |
| R3 | runtime / launch / config path 护栏 | 4 | 45-75 分钟 | Low-Medium |
| R4 | training bootstrap / forward / compile order 保护 | 4 | 45-75 分钟 | Low-Medium |
| R5 | LoRA/config/type-check 残余保护 | 3 | 35-60 分钟 | Low-Medium |
| R6 | 总验证、checkpoint、提交推送 | 3 | 30-45 分钟 | Low |
| EXT | 扩展阶段池 | 不限 | 直到满足 3 小时 | Low-Medium |

最低要求：

- 必须完成 R0 和 R6。
- R1 到 R5 至少完成其中 4 轮。
- 总阶段数必须达到 20。
- 子系统覆盖必须达到 4 类。
- 如果任何主轮风险过高，跳过原因必须写进 checkpoint，并用 EXT 补足。

---

## ✅ 6. R0：基线和旧目标归档确认

一句话：先把入口关系理清，避免下一轮误跑 20260705 的旧目标。

### A00 基线确认

目标：

- 确认本地 `main` 与 `webui/main` 状态。
- 记录开始时间。

验收：

- checkpoint 写入启动状态、远端同步状态、开始时间。

### A01 旧目标归档扫描

目标：

- 扫描 20260705 三份目标书是否都不再自称活跃入口。
- 扫描 docs 中是否仍有“请按 20260705 sustained 执行”的可复制 prompt。

建议命令：

```bash
rg -n "状态：活跃|project_cleanup_sustained_goal_20260705.md" docs/findings/project_cleanup_*20260705.md docs/findings/project_cleanup_checkpoint_20260705.md
```

验收：

- 旧目标只作为历史记录。
- 新入口指向 `project_cleanup_sustained_goal_20260706.md`。

---

## 🖥️ 7. R1：WebUI 前端 / DOM / 静态模块证据

一句话：这轮补 WebUI 真实功能的静态和轻量行为证据，不做视觉大改。

### A02 前端模块依赖图审计

目标：

- 只读审计 `web/static/js/features/anima-app/`、`live-training/`、`queue/`、`history-detail/` 的 import 边界。
- 找出最适合补结构测试的薄弱点。

建议命令：

```bash
rg -n "from './|from \"./|globalThis|querySelector|getElementById|addEventListener" web/static/js/features/anima-app web/static/js/features/live-training web/static/js/features/queue web/static/js/features/history-detail
```

验收：

- checkpoint 写入候选清单和不动原因。

### A03 DOM id 契约补测

目标：

- 如果审计发现 index.html 与 JS selector 缺少测试，补一个窄测试。
- 优先写入 `tests/test_training_frontend_state.py`。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "dom or selector or dataset or queue or history"
```

### A04 live-training 状态更新证据

目标：

- 补 `updateStatus()` / `updateProgress()` 周边的静态或小 fixture 行为保护。
- 不启动真实训练。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "live or progress or status"
```

### A05 queue/history 前端入口保护

目标：

- 补 queue/history 前端模块的结构测试或 import smoke。
- 只测前端契约，不改 UI 样式。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "queue or history"
```

### A06 CSS import / responsive 守门

目标：

- 只读确认 `style.css` import 顺序和相关 CSS 文件没有断链。
- 只有发现明确问题才修。

验证：

```bash
git diff --check -- web/static/style.css web/static/css tests/test_training_frontend_state.py
```

---

## 🧱 8. R2：WebUI 后端服务 / 队列 / output root 安全

一句话：这轮补服务层路径边界和队列行为保护，不碰真实队列数据。

### A07 output root 边界审计

目标：

- 只读审计 preview、training history、queue、settings 对 output root 的解析。
- 找到一个最小可测边界。

建议命令：

```bash
rg -n "resolve_output_root|output_root|HISTORY_DIR|QUEUE_DIR|training-history|training-queue" web/services tests
```

### A08 preview / settings 路径测试

目标：

- 给 `preview_service` 或 `settings_service` 补一个路径隔离测试。
- 不写真实 `configs/`，只用 `tmp_path`。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py tests/test_env_config_paths.py
```

### A09 training queue runtime config 保护

目标：

- 补一个 queue runtime config / meta 写入的小测试。
- 必须 monkeypatch launch，不启动训练。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py -k "runtime or metadata or output_root or launch"
```

### A10 training resume / history 窄验证

目标：

- 跑一个和改动相关的 `training_resume` 窄筛选。
- 如果 60 秒不够，拆更窄，不计失败为通过。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_resume.py -k "history or meta or output_root or runtime"
```

### A11 service 层不能越界清单

目标：

- checkpoint 写清楚哪些 service 仍不能做批量删除、真实队列清理或训练启动。

验收：

- checkpoint 有“禁止对外夸大”和“未做真实训练/未碰用户数据”记录。

---

## ⚙️ 9. R3：runtime / launch / config path 护栏

一句话：这轮处理启动命令、路径解析和 runtime helper 的低风险边界。

### A12 launch command builder 审计

目标：

- 只读审计 `scripts/tasks/_common.py`、`library/runtime/launch.py`、`tests/test_launch_config.py`。
- 找一个最小命令构造边界测试。

建议命令：

```bash
rg -n "build_launch_cmd|accelerate_training_command_prefix|ANIMA_ACCELERATE_LAUNCH|PROFILE_STEPS|python_exe" scripts/tasks/_common.py library/runtime/launch.py tests
```

### A13 launch helper 补测

目标：

- 补一个不启动子进程的 launch command 测试。
- 优先写入 `tests/test_launch_config.py` 或 `tests/test_tasks_runner.py`。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py tests/test_tasks_runner.py
```

### A14 config root 外置路径护栏

目标：

- 补 `ANIMA_CONFIGS_ROOT` / `.anima-webui-settings.toml` 的路径边界测试。
- 只用 `tmp_path`，不写真实本机设置文件。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py tests/test_config.py -k "configs_root or env or path"
```

### A15 runtime harness 低风险验证

目标：

- 补或跑 `compile_blocks_for_training` / no-DiT guard / token budget 的模型无关测试。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_runtime_harness_cli.py tests/test_native_flatten.py
```

### A16 block swap 不扩大审计

目标：

- 只读确认 `offloading.py` 剩余复杂逻辑不适合本轮拆。
- 如果发现纯 helper 候选，只写 checkpoint，不急着拆。

验收：

- checkpoint 记录“为什么不拆 CUDA stream / Event / thread pool / hook 调度”。

---

## 🧪 10. R4：training bootstrap / forward / compile order 保护

一句话：这轮只做模型无关的训练启动和 forward 顺序测试。

### A17 lazy loading 顺序审计

目标：

- 只读审计 `train.py` 和 `library/training/bootstrap.py` 里 text encoder / VAE / DiT 的加载顺序。
- 找一个可以用 monkeypatch 固定的顺序不变量。

建议命令：

```bash
rg -n "load_qwen3_text_encoder|load_vae|load_target_model|compile_blocks_for_training|network.apply_to|load_weights" train.py library/training library/runtime/harness.py tests
```

### A18 bootstrap compile-after-apply 保护

目标：

- 补一个不加载模型的测试，确认 compile helper 在 adapter apply/load 后调用，或确认已有测试覆盖。
- 优先写入 `tests/test_training_bootstrap.py` 或 `tests/test_runtime_harness_cli.py`。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_bootstrap.py tests/test_runtime_harness_cli.py -k "compile or adapter or apply"
```

### A19 forward helper 归属审计

目标：

- 只读审计 `library/training/forward/` 和旧 shim。
- 只记录下一步候选，不做 train.py 大拆。

验收：

- checkpoint 写清楚哪些 forward helper 可迁移，哪些不能碰。

### A20 prior / inversion / router conditioning 窄验证

目标：

- 跑现有相关测试或补一个小 fixture。
- 不启动真实训练。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_bootstrap.py tests/test_training_optimizers.py tests/test_training_resume.py -k "prior or inversion or router or compile or bootstrap"
```

---

## 🧬 11. R5：LoRA / config / type-check 残余保护

一句话：这轮只补剩余边界，不继续把上轮已经收口的方向无限扩大。

### A21 LoRA 残余候选审计

目标：

- 只读审计 `networks/lora_anima/loading.py`、`factory.py`、`network.py` 中仍未覆盖的保守拒绝路径。
- 优先找“坏输入不应被静默兼容”的测试。

建议命令：

```bash
rg -n "raise ValueError|return state_dict|metadata|router|split|qkv|legacy|sigma_mlp" networks/lora_anima tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_save_pipeline.py
```

### A22 LoRA 一个窄边界测试

目标：

- 只补一个 characterization test。
- 不改保存格式、不改加载格式、不改三轴语义。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_save_pipeline.py
```

### A23 config facade 只读确认

目标：

- 确认 `_legacy.py` 仍不能删除。
- 如果发现旧入口文案误导，就只修文档或测试。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "legacy or module_imports_without_facade_cycle or metadata"
```

### A24 type-check 候选只读审计

目标：

- 只读找下一个低依赖候选。
- 默认不要继续扩大 Web config 白名单，除非有明确 direct-import 测试。

建议命令：

```bash
timeout 60 .venv/bin/python tasks.py type-check
sed -n '1,260p' scripts/tasks/utilities.py
sed -n '1,220p' tests/test_type_check_targets.py
```

验收：

- checkpoint 写入“扩或不扩”的理由。

---

## 🔚 12. R6：总验证、checkpoint、提交推送

一句话：最后只做收口，不再扩大重构。

### A25 checkpoint 总结

必须写入：

- `docs/findings/project_cleanup_checkpoint_20260705.md`

必须记录：

- 完成阶段编号。
- 覆盖子系统。
- 修改文件列表。
- 验证命令和结果。
- 60 秒超时拆分情况。
- 跳过阶段原因。
- 未做事项：
  - 没真实训练。
  - 没下载模型。
  - 没碰用户数据目录。
  - 没删除 `_legacy.py`。
  - 没改 checkpoint key / public API / 三轴路由语义。
  - 没建立全仓 type-check，除非真的做到了。

### A26 总验证

建议按实际改动选用，不要求无关全量：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "dom or selector or queue or history or live or progress"
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py tests/test_env_config_paths.py tests/test_launch_config.py tests/test_runtime_harness_cli.py
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py -k "runtime or metadata or launch or output_root"
PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_save_pipeline.py
timeout 60 .venv/bin/python tasks.py type-check
git diff --check
```

如果宽筛选 60 秒内跑不完：

- 拆成更窄 `-k`。
- 记录“宽命令未计为通过，拆分验证通过”。

### A27 提交推送

命令：

```bash
git status --short --branch
git diff --name-only
git add <本轮实际修改文件>
git diff --cached --check
git commit -m "refactor: continue cross-system cleanup"
git fetch webui --prune
git log webui/main..HEAD
git push webui main:main
git status --short --branch
git rev-parse HEAD webui/main
```

提交前检查：

- `goal.timeUsedSeconds >= 10800`。
- 已完成至少 20 个阶段。
- 已完成至少 5 个推进轮。
- 已覆盖至少 4 个子系统。
- 已有至少 6 组非纯文档验证。
- 不满足则不能 final，必须继续 EXT。

---

## 🔁 13. EXT：扩展阶段池

一句话：如果基础轮做完还不够 3 小时，就继续从这里拿低风险阶段，不允许提前停。

EXT 规则：

- 每次只做一个 EXT 阶段。
- 同一 EXT 编号不能重复计数；除非输入范围不同、产出新证据，并在 checkpoint 写清楚差异。
- 每做 3 个 EXT 阶段，必须跑一次相关测试或 type-check。
- 不能连续 4 个纯文档 EXT；第 4 个必须是测试、源码护栏或验证。
- 如果候选池剩余少于 6 个，必须先只读审计补充候选池，再继续推进。

| 编号 | 目标 | 子系统 | 风险 |
|---|---|---|---|
| E01 | 补一个 `tests/test_training_frontend_state.py` DOM selector 契约 | WebUI frontend | Low |
| E02 | 补一个 history-detail 静态模块断言 | WebUI frontend | Low |
| E03 | 补一个 queue 前端状态切换断言 | WebUI frontend | Low |
| E04 | 审计 `web/static/js/features/anima-app/chunks/` 超大 chunk，只写候选 | WebUI frontend | Low |
| E05 | 检查 CSS import 顺序和未引用 CSS 文件，只写结果 | WebUI CSS | Low |
| E06 | 补 `preview_service` output root 越界拒绝测试 | WebUI backend | Low |
| E07 | 补 `settings_service` configs root 保存路径测试 | WebUI backend | Low |
| E08 | 补 training queue meta 写入小测试 | WebUI queue | Low |
| E09 | 补 training queue launch lock 窄回归 | WebUI queue | Low-Medium |
| E10 | 拆分一个 `training_resume` 60 秒内能跑完的窄验证组 | WebUI history | Low |
| E11 | 审计 `library/runtime/launch.py` env flag 行为，只写清单 | runtime | Low |
| E12 | 补 `build_launch_cmd` 不启动子进程测试 | runtime | Low |
| E13 | 补 `ANIMA_ACCELERATE_LAUNCH` 参数组合测试 | runtime | Low |
| E14 | 补 configs root 外置路径拒绝 `..` 的测试 | config path | Low |
| E15 | 补 `.anima-webui-settings.toml` 缺失时 fallback 测试 | config path | Low |
| E16 | 审计 `library/runtime/harness.py` compile order 说明是否过期 | runtime | Low |
| E17 | 补 `compile_blocks_for_training` 参数转发测试 | runtime | Low |
| E18 | 补 native flatten / token budget 的轻量回归 | runtime | Low |
| E19 | 只读审计 `library/runtime/offloading.py` 剩余不可拆区域 | runtime | Low |
| E20 | 补 block-swap config helper 小测试 | runtime | Low-Medium |
| E21 | 审计 `train.py` lazy loading 顺序，只写 checkpoint | training | Low |
| E22 | 补 `training_bootstrap` compile-after-apply monkeypatch 测试 | training | Low-Medium |
| E23 | 审计 `library/training/forward/` shim 归属，只写候选 | training | Low |
| E24 | 补 router conditioning 小 fixture 测试 | training | Low-Medium |
| E25 | 补 prior-preservation forward shim 兼容测试 | training | Low-Medium |
| E26 | 审计 LoRA loading 保守拒绝路径，只写候选 | LoRA | Low |
| E27 | 补一个 LoRA malformed state_dict characterization test | LoRA | Low-Medium |
| E28 | 补 LoRA save metadata/hash 边界测试 | LoRA | Low |
| E29 | 审计 `factory.py` key sniff 剩余重叠逻辑，只写清单 | LoRA | Low |
| E30 | 补 config facade direct-import smoke | config | Low |
| E31 | 补 `_legacy.py` 兼容导出测试 | config | Low |
| E32 | 审计 `_legacy.py` 剩余 10 个非转发函数，只写分类 | config | Low |
| E33 | 对一个非 Web config 候选单跑 type-check 并记录 | type-check | Low |
| E34 | 更新 `tests/test_type_check_targets.py` 的“禁止误扩”护栏 | type-check | Low |
| E35 | 审计 README/docs 是否仍引用旧目标书或旧命令 | docs | Low |
| E36 | 修正一个明确过期的 docs 命令说明 | docs | Low |
| E37 | 补 `git diff --check -- docs/findings` 验证记录 | docs | Low |
| E38 | 补当前目标阶段计数和覆盖子系统表 | docs | Low |
| E39 | 做一次中宽验证组合并记录结果 | validation | Low |
| E40 | 做最终 stage 文件清单预审但不提交 | validation | Low |

---

## 🧾 14. 可直接复制给 Codex 的强制长跑 Prompt

一句话：下面这段是下一次真正要跑 3 小时以上的目标 prompt。

```text
请按 docs/findings/project_cleanup_sustained_goal_20260706.md 执行跨子系统强制长跑项目清理目标。

硬性要求：
1. 先读 AGENTS.md、docs/findings/project_cleanup_checkpoint_20260705.md、docs/findings/project_cleanup_sustained_goal_20260706.md。
2. 不允许在 goal.timeUsedSeconds < 10800 时标记 complete；连续 3 轮同一阻塞且无法安全推进时，只能标记 blocked，不能标记 complete。
3. 不允许只完成一个小闭环就停；必须至少完成 20 个可验收小阶段、至少 5 个推进轮。
4. 至少覆盖 4 个子系统：WebUI frontend、WebUI backend、runtime/launch/config path、training bootstrap/forward、LoRA/config/type-check 等。
5. 如果基础轮提前完成但未满 3 小时、未满 20 阶段或覆盖不足，必须进入 EXT 扩展阶段池继续推进。
6. 不允许靠 sleep、空等、无意义轮询凑时间；必须持续做低风险审计、测试、helper 小拆分、文档检查点。
7. 禁止真实训练、模型下载、删除/移动用户数据、删除 _legacy.py、改 checkpoint key、改 LoRA public API、改三轴路由语义。
8. 每轮都要更新 checkpoint，记录阶段编号、覆盖子系统、修改文件、验证命令和不能夸大的边界。
9. 最后显式 stage 实际修改文件，不要 git add -A。
10. 满足硬完成条件后，提交并推送到 webui/main。

最终完成条件：
- goal.timeUsedSeconds >= 10800。
- 已完成至少 20 个小阶段。
- 已完成至少 5 个推进轮。
- 已覆盖至少 4 个子系统。
- 至少 6 组非纯文档验证完成。
- 总验证通过，或 60 秒超时项已拆分验证、拆分命令通过并记录。
- checkpoint 文档已更新。
- 本地 main 已推送到 webui/main。
```

---

## 📌 15. 不能夸大的边界

一句话：长跑完成后也只能按证据说话，不能把阶段收口说成全仓清空。

不能说：

- 不能说全仓技术债清完。
- 不能说 WebUI 已做真实浏览器全链路验证，除非真的启动服务并用浏览器工具验证。
- 不能说训练性能提升，除非跑过 bench 或真实训练证据。
- 不能说 LoRA save/load/builder/router 已彻底拆完，除非对应链路真的完成并有测试。
- 不能说 `_legacy.py` 可以删除，除非外部 import surface 已迁移并验证。
- 不能说建立全仓 type-check，除非默认 type-check 真覆盖全仓并通过。

可以说：

- 完成了哪些阶段。
- 覆盖了哪些子系统。
- 跑了哪些验证。
- 哪些 helper 被小步拆分。
- 哪些文档入口已经归档或替换。
- 哪些风险仍然保留。

---

## 🔚 16. 完成归档记录

一句话：本目标已按硬完成条件完成，并由 checkpoint 与 Git 发布结果作为最终证据。

最终状态：

- 最低耗时门槛已满足：`goal.timeUsedSeconds >= 10800`，收口前最新读取为 `10889` 秒。
- 推进轮数和阶段数已满足：checkpoint 已记录 `R0` 到 `R6`，并通过 EXT 扩展到 78 组以上低风险阶段。
- 子系统覆盖已满足：WebUI frontend、WebUI backend / queue / preview、runtime / launch / config path、training bootstrap、LoRA/config/type-check、docs/archive 均有真实证据。
- 总验证已完成：pytest 分批、`tasks.py type-check`、`git diff --check`、docs 可达性、归档索引和远端同步预检均通过。
- 最终发布目标：本地 `main` 推送到 `webui/main`。

不能夸大的边界：

- 没有跑真实训练。
- 没有下载模型。
- 没有清理或移动用户数据目录。
- 没有删除 `_legacy.py`。
- 没有改 LoRA checkpoint key、public API 或三轴路由语义。
- 没有建立全仓 type-check；当前是明确白名单门禁。
