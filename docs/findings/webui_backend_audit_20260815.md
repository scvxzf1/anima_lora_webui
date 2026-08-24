状态：稳定
适用版本：当前 main（截至 20260815）
入口命令：`.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102`
相关代码：`web/`（`server.py`、`routes/`、`services/`）

# WebUI 后端审计报告（20260815）

本次审计面向 `web/` 后端（aiohttp WebUI），覆盖 116 个 Python 模块、约 28.9k 行。
审计维度：安全性、错误处理、并发与状态机、不可破坏不变量、架构健康度、测试覆盖。
方法：人工精读核心入口/约束层/全部路由/危险调用面 + 后台子代理并行审计并发与测试覆盖。
未执行真实训练或大模型推理，所有结论以静态阅读为准。

## 一、安全性

### 1.1 鉴权（设计正确，但有边界值得确认）

`web/server.py:100-134` 的 `auth_middleware` 设计正确：
- 仅当 bind 超出 loopback 时强制 token，loopback 默认无鉴权（本地工作流）。
- token 来源覆盖 `Authorization: Bearer` / `X-Anima-Token` / cookie / `?token=`，`?token=` 命中后写 `HttpOnly; SameSite=Lax` cookie，避免静态资源重复带 query。
- 非 loopback 无 token 时 `main()` 直接 `sys.exit(2)` 拒绝启动（L214-222），不存在"外部绑定但忘了开鉴权"的窗口。

**低危·信息性**：token 比较是 `provided != expected`（L114），非常量时间比较，理论上可侧信道泄漏。loopback-only 部署无影响；外部部署建议后续换 `hmac.compare_digest`。
**低危**：`?token=` 进 cookie 后，token 仍可能出现在浏览器历史/代理日志；这是该模式的固有代价，文档已隐含说明，可接受。

### 1.2 路径穿越（核心安全面，实现扎实）

四条文件读取/下载路径全部收敛到 `web/services/path_safety.py`：
- `resolve_allowed_file`（L176-206）：normalize → 拒 `..` → 相对锚定 root 并 `relative_to` 兜底 → 强制 `allowed_dirs` 白名单 → 强制 suffix。
- `safe_resolve`（config/paths.py L16-36）：绝对路径也必须 `relative_to(configs_dir)` 否则返回 None。
- preview 图片/权重下载、weight_analysis、continue_lora、image_test weight 全部走这条管线（已逐一核验：`preview/common.py:178-200`、`weight_analysis/paths.py:117-140`、`continue_lora_service.py:79-93`、`image_test_service.py:451-493`）。
- 历史日志/产物下载（`history_meta.py:485-531`）用 `HISTORY_ARTIFACT_FILES` / `HISTORY_RUNTIME_ARTIFACT_FIELDS` 白名单 key + `relative_to(task_dir/run_dir)` + `resolve_output_root` 边界三重校验，`artifact_key` 不能任意拼路径。

`static_handler`（server.py L52-62）：`resolve()` 后断言 `STATIC_DIR in path.parents`，防穿越正确。
`/next/{path:.*}` 同理走 FileResponse 但**未做 STATIC_DIR 边界校验**（L37-49 只判 is_file）——不过它锚定 `STATIC_DIR/"dragon-next"/"index.html"` 固定路径，`{path}` 实际未拼进文件路径，无风险。

### 1.3 命令注入 / subprocess

**整体安全**：所有 subprocess 都用 `create_subprocess_exec`（list 形式），无 `shell=True`、无 `os.system`、无 `eval/exec/pickle`（grep 全空）。
- 训练/preprocess：`launcher_job.py:149` `create_subprocess_exec(*cmd, ...)`。
- nvidia-smi：`gpu.py:21,85` 固定 argv list。
- image_test：`image_test_service.py:120` `create_subprocess_exec(*cmd, ...)`，命令由 `_build_generation_command` 构造，参数全部经 `_normalize_*` 校验（sampler/attn_mode/dtype 走 choices 白名单，宽高步数走 `_normalize_positive_int`，weight 走 allowlist）。
- `file_group_paths.py:145` 的 `subprocess.run` 用于 exiftool 调用，需单独确认（见下）。

**高危发现·`extra_args` 任意参数注入**（`web/routes/training.py` + `launcher_start.py:124-125,254-255`）：
- `/api/training/start`、`/preprocess`、`/queue`、`/queue/batch-start` 接受 `extra_args: list[str]`，路由层只 `isinstance(list)` 校验，**无内容白名单**，直接 `cmd.extend(extra_args)` 进入 train.py / tasks.py preprocess 的 argv。
- 影响：用户（任何能访问 WebUI 的本机用户，或外部绑定下任何持 token 者）可注入任意 train.py 参数绕过 preflight/runtime 路径边界，例如 `--config_file <任意 toml>`、`--network_weights <任意 safetensors>`（绕过 continue_lora 的 kind/路径校验）、`--base_args` 覆盖 DiT 路径、`--output_dir` 写到任意目录。
- 不走 shell 所以不能起任意进程，但可读任意 safetensors、写任意 output_dir、覆盖 runtime config。
- 缓解现状：continue_lora inspect 走独立校验，但 `extra_args` 里的 `--network_weights` **不会**再过 inspect；preflight 只校验 `config_file` 指定的配置，不校验 extra_args 注入的 override。
- 建议：对 `extra_args` 做 train.py 已知参数白名单 + 值域校验，或至少禁用 `--config_file/--network_weights/--base_args/--output_dir/--data_dir` 等路径/覆盖类参数，仅放行纯数值/枚举参数。这是本次审计最重要的可操作发现。

**待确认**：`file_group_paths.py:145` 的 `subprocess.run` 调用面（exiftool？）参数来源，下一轮应核验其 argv 是否含用户输入。本轮未深读。

### 1.4 上传面

- `analysis.py:48-67` `inspect-upload` 流式读 multipart，累计 `len(data) > MAX_UPLOAD_WEIGHT_BYTES` 即拒，空文件拒，安全。
- `client_max_size = 512 MiB`（server.py L16,143）兜底。
- 上传 bytes 走 `read_safetensors_header_bytes`（path_safety.py L114-136）纯内存解析 header，有 header_len 合法性校验，不会无界分配。

### 1.5 CORS / CSRF

无 CORS 中间件（单机 WebUI，前后端同源，可接受）。
改状态的全是 POST/PUT/PATCH/DELETE，aiohttp 默认不跨域，CSRF 面小。`?token=` cookie 模式下若用户访问恶意页面，理论上可构造表单 POST——但 loopback 部署无 token，外部部署建议后续加 CSRF token 或 `SameSite=Strict`。

## 二、错误处理

### 2.1 路由层一致性（良好）

- 256 处 `json_response`，状态码分布：400×105 / 404×28 / 409×13 / 500×3 / 503×5 / 403×3，语义基本一致（400 业务校验/ValueError，404 缺资源，409 状态冲突/RuntimeError，500 OSError 磁盘）。
- `config.py` 21 处 `except Exception` 裸接，但全部 `{"ok": False, "error": str(e)}, status=400`，未吞没，统一转 400——可接受，但会掩盖 5xx 类系统错误为 400。
- `training.py` 4 处 `except Exception` 同样统一转 400/409。
- `handle_raw_get`（config.py L399-404）**未 try/except**，`load_raw_file` 抛异常会冒泡成 aiohttp 默认 500——轻微不一致，建议补 try。

### 2.2 services 层静默吞没（45 处，多数合理，少数需关注）

- 多数为防御性回退：`service_state.py:73,83`（queue backup 恢复/写失败静默回退）、`settings_service.py:401-413`（history/queue root 显示降级）、`history_ops.py:146`（meta.json 解析失败→空 dict 继续清理）——合理工程取舍。
- **中危**：`history_ops.py:356` `_append_history_jsonl` 写 metrics/system 日志失败 `except Exception: pass`，会静默丢失训练历史时间线数据且无任何告警，磁盘满或权限错时用户不可见。建议至少 `_remember_log("error", ...)` 一条。
- **低危**：`gpu.py:37,93` nvidia-smi 失败静默返回空/空列表——可接受（无 GPU 机器应正常工作），但建议在 status 里带一个 `gpu_probe_error` 字段方便排障。
- `web/` 整体 `logger`/`logging` 使用仅 10 处，绝大多数错误靠 `_remember_log` 进 WS 推送 + 返回 error 字段——对 WebUI 够用，但服务端无持久化错误日志，排障依赖 history logs.jsonl。

### 2.3 API 响应体一致性

成功响应有的带 `ok: True` 有的不带（`handle_merged` 直接返回 config dict，`handle_methods` 带 `ok/items`），前端需各自适配。非阻塞性文档负债。

## 三、并发与状态机

锁覆盖清单（grep 确认，12 处锁字段 + 4 处 create_task）：
- `_launch_lock`（asyncio.Lock，1 个实例，8 处 `async with`）：`launcher_start` start/start_preprocess、`launcher_job` stop/shutdown、`queue_control` 2 处、`queue_dispatch` 2 处。统一串行化"启动/停止/队列调度/队列控制"——正确，避免 launch 与 stop 交错。
- `_progress_jsonl_lock`（asyncio.Lock，每 generation 重建）：仅保护 jsonl offset 跟踪，粒度小，不与 output_task 互锁——正确。
- `_run_generation`（int 计数器，非锁）：`stop` 后新 `launch` 自增 generation，旧 generation 的后台 task（output/monitor/tail）完成时不写新状态——正确的"代际隔离"。
- `start_new_session=True`（launcher_job.py:155）：子进程独立进程组，`stop` 用 psutil terminate→wait 3s→kill 整族——正确，避免孤儿训练进程。
- `_queue_dispatch_task`（单例 create_task）：shutdown 时 cancel + gather return_exceptions——干净。

**主审确认的关键点**：
- `atomic_write_text` 是 fsync+os.replace 模式，所有 config/history/queue/settings 写都走它（已 grep 确认）。
- `_save_queue` 在 `enqueue_training`/批量 enqueue 内调用，queue file read-modify-write 依赖 `_launch_lock` 串行化——批量 enqueue 在锁内循环，OK。但 `_save_queue` 本身无独立锁，若未来有非 `_launch_lock` 路径写 queue 会竞态（当前无此路径）。
- `asyncio.gather` 仅 2 处（shutdown + task_lifecycle cancel），均 `return_exceptions=True`——不会因一个 task 抛错中断其他清理。
- `image_test_service` 用单一 `_monitor_task` 且 start 时检查 `status != "running"`——简单单任务模型，无锁需求，正确。

**潜在风险（主审标注，待子代理细化）**：
- `settings_service._load_settings` 读 + `save_global_settings`/`save_preview_settings` 写是 read-modify-write，**全程无锁**，且 `_load_settings` 每次 re-parse TOML + 3 次 `.resolve()`。多 tab 并发改设置时有丢失更新风险（见 5.3 settings 性能热点）。
- `_append_history_jsonl`（history_ops.py:348-357）append 写无锁，但只在训练主 output_task 单线程内调用——实际无竞态，OK。
- queue backup 写（service_state.py:83）在主锁外静默吞没失败——见 2.2。

**并发安全总评**：核心训练生命周期（launch/stop/queue dispatch）的并发模型设计正确，代际隔离 + 单锁串行 + 进程组 kill 是稳健组合。主要风险在 settings/queue 的 read-modify-write 无锁（中危，单用户影响小）和 history jsonl 静默吞没（中危，见 2.2）。

## 四、不可破坏不变量遵守

- **`resolve_output_root` 边界**：历史产物下载（`history_meta.py:526-528`）显式校验 `run_dir` 在 `resolve_output_root()` 或 web runtime dir 内；preview/weight/continue_lora 全部以此为白名单源。遵守良好。
- **Lazy loading 顺序**：WebUI 本身不加载 DiT/TE/VAE，只起 train.py 子进程；`_on_startup` 只 init TrainingService + ImageTestService，不触模型。遵守。
- **外置配置拒绝 `..`**：`_normalize_output_root`/`_normalize_config_path`/`_normalize_image_test_save_root` 全部 `if ".." in path.parts: raise`。`normalize_user_path_value` 不拒 `..`（靠 `resolve_allowed_file`/`safe_resolve` 后续拒），路径安全不假象安全——正确。
- **`image_test_save_root` / `output_root` 外部根**：`resolve_output_root` 允许绝对路径（外置输出目录），但下载面用 `is_under_allowed_dirs` 二次收敛。OK。
- **Text Encoder padding / Constant Token Buckets / LoRA family 三轴**：这些是 train.py/networks 侧不变量，WebUI 后端只读不写，无破坏面。
- **memory_probe_jsonl / block_swap_profile_jsonl = "auto"** 路径解析在 `launcher_job.py:97-126` 锚定 `task_dir`，不写回用户配置——遵守 AGENTS.md。

## 五、架构与健康度

### 5.1 上帝文件

`web/` 无严重上帝文件。最大几个：
- `preview/images.py` 972、`image_test_service.py` 898、`routes/training.py` 818、`training/runtime_datasets.py` 741、`routes/config.py` 677。
- `routes/training.py` 53 个路由 handler 集中一文件，但每个 handler 都是薄分发（10-25 行），职责单一，属可接受的"宽表路由"。
- `image_test_service.py` 898 行混了 normalize/command/env/weight 解析/monitor，**违反反上帝守则**（单文件建议 <400 行类/<100 行函数），应拆 `image_test/{normalize,command,weight,monitor}.py`。`_normalize_image_test_request` 单函数 ~120 行也超 100 行红线。
- `preview/images.py` 972 行同问题，应继续按 detect/meta/filter/delete 拆。

### 5.2 facade / legacy shim 健康度

`config/_legacy.py` + `legacy_shims_*.py`（12 文件）是 config_service 的内部兼容层，外部路由/服务**只通过 `config_service` 单一入口**访问（grep 确认无路由直接 import `_legacy`/`legacy_shims`）。shim 仍在被 facade 内部使用，非死代码，但层数偏多（raw_files 自身又有 `_exported` wrapper + `_sync_from_facade`），属历史机械拆分遗留，后续可继续收敛。

### 5.3 测试覆盖

tests/ 共 220 个测试文件。Web 后端覆盖矩阵（按命名归类 + 子代理核验）：

| 模块 | 测试文件 | 覆盖深度 |
|---|---|---|
| path_safety | test_path_safety.py（140 行，专测 `../` 逃逸/绝对路径越界/allowlist 内通过，并 cross-check continue_lora/preview_common/analysis_paths 三处共用策略） | 高 |
| config raw_files | test_web_config_raw_files.py（891 行）+ test_raw_file_warnings_contract.py（用 _FakeJsonRequest + tmp_path + monkeypatch，断言 400/不写文件不变量/blank output_name/retired 字段删除/SPD 嵌套表/CAME 三 beta 修复） | 高 |
| dataset_presets | test_dataset_preset_stage_import.py + test_web_config_datasets.py | 中-高 |
| preflight | test_web_config_preflight.py + test_web_preflight_compat_matrix.py + test_training_compat_matrix.py | 高（含 compat matrix） |
| queue | test_training_queue.py + test_training_queue_retry_wake.py + test_training_queue_resume.py + test_queue_item_retry_override.py + test_training_retry_integration.py | 高 |
| history | test_training_history_delete/_list/_timeline/_artifacts + test_cross_domain_delete_boundaries.py | 中-高 |
| preview | test_preview_service.py + test_preview_async_offload.py + test_image_listing.py | 中（路径穿越靠 path_safety 兜底，路由级专门用例偏少） |
| weight_analysis | test_weight_analysis_service.py（用极小真实 safetensors 断言 delta/alpha/rank/block parse，无大模型） | 中-高 |
| continue_lora | test_training_resume_actions.py + test_path_safety.py + test_training_start_preprocess.py（41 处 continue_info 断言，含变体/路径错误测试） | 高 |
| image_test | test_image_test_service.py + test_settings_image_test.py | 中 |
| settings | test_global_settings_runtime.py + test_ui_scale_settings.py + test_settings_model_family.py + test_settings_image_test.py | 中-高 |
| environment | test_environment_check_service.py（1 个文件，深度中） | 中 |
| subprocess launch | test_launch_config.py + test_training_task_lifecycle.py + test_training_start_preprocess.py + test_daemon.py（26 处 subprocess monkeypatch） | 中-高（隔离良好） |
| server/auth | test_web_static_server.py + test_web_route_registry.py（REQUIRED_ROUTES release-blocking）+ test_web_http_contracts.py | 高（外部绑定拒否/token 传递/loopback 忽略/路由注册完整性均有断言） |
| atomic_io | test_atomic_io.py | 高 |
| file_group | test_web_config_file_groups.py | 中-高 |
| schema_gate | test_schema_gate_observability.py | 中 |
| runtime_config | 4 个 test_*runtime* | 中-高 |
| config _legacy/shims 金字塔 | test_web_config_legacy_shims.py + raw_files/preflight 中的 **子进程级 import-cycle 护栏**（`test_raw_files_module_imports_without_facade_cycle` / `test_preflight_module_imports_without_facade_cycle`） | 中（结构/契约测试，最值得保留的网） |
| **facade_compat.py（training）** | **无** | **无（201 行惰性注册表 ~60+ 名零直接测试，运行时 KeyError 暴露）** |
| **training_facade/resume_facade 派发层** | **无** | **无（history_meta/runtime_paths/history_timeline 高频 `getattr(_training_facade(),<name>)`，仅被间接覆盖）** |
| **legacy_shims_{preflight,merge,estimation,dataset}** | **无** | **无（仅 common/sample_prompts/output_runs/raw_files/file_groups 五桶有转发断言）** |

**关键缺口（严重→次要）**
1. `facade_compat.py` 零直接测试——~60+ 注册名拼错/目标搬家只在运行时 KeyError。
2. `training_facade`/`resume_facade` 派发层零契约测试。
3. `legacy_shims_{preflight,merge,estimation,dataset}` 四桶转发无直接断言。
4. `settings_service` 无并发写测试，save 路径 read-modify-write 全程无锁。
5. preview/image_test 路由级路径穿越专门用例偏少（靠 path_safety 兜底）。
6. **`extra_args` 注入拒绝` 无断言**——`test_training_start_preprocess.py:61` 用 `["--foo"]` 验证了"透传"，但无任何用例断言危险参数（`--config_file/--network_weights`）应被拒绝，与 P0 发现呼应。

**facade/shim 健康度**
- 循环依赖：低，且有 2 个子进程级 import-cycle 护栏测试（raw_files、preflight）。
- 死代码嫌疑：`_legacy.py` 在 web/ 中仅被 `config_service` 导入 1 次；9 个 `legacy_shims_<domain>.py` 仅被聚合器 `legacy_shims.py` 导入；`facade_compat.py` 仅被 `training_service` 消费——均为单消费者兼容层，未死但 glue-to-logic 比极高。

**settings 性能热点（精确调用清单）**
`_load_settings()` 单次成本：`toml.loads(web-ui-settings.toml)` + `toml.loads(base.toml)`（`_load_base_model_path_defaults`）+ 3 次 fs `.resolve()`（history_root/queue_root/configs_root）。`_load_raw_settings` 与 `_load_base_model_path_defaults` **均无 mtime 缓存、无 memoize**；`SETTINGS_FILE` 是 `DynamicPath(lambda)`，每次 `Path(SETTINGS_FILE)` 重新求值 lambda。

运行时热路径（每次 HTTP 请求触发一次完整重盘 2 个 TOML + 3 次 fs resolve）：
- preview：`preview/common.py:349`（每次图片/权重列表请求）
- output_runs：`config/output_runs.py:99,220`
- preflight：`config/preflight_paths.py:137,159,186`（每个 preflight 检查 3 次 resolve_output_root）
- history：`training/history_store.py:425` + `history_meta.py:526`
- training runtime：`runtime_prepare.py:106` / `runtime_state.py:56` / `runtime_resume.py:64`
- image_test：`image_test_service.py:323,415,603`（status/start 各调 get_global_settings + resolve_output_root）
- environment：`environment_check_service.py:104,239`
- weight_analysis：`weight_analysis/paths.py:220`
- resume：`training/resume.py:182`

**影响判断**：
- 性能：单个 WebUI 页面并发拉取 history + preview + output-runs + environment 时触发 4-8 次 `_load_settings`，`base.toml` 在 resolve_output_root 路径里被反复解析最浪费（近乎静态）。属"单次微秒-低毫秒级、但每请求重复"的累积浪费，非瓶颈但无必要。
- 并发正确性（更重要）：`save_global_settings`/`save_training_policy`/`_save_path_overrides` 都是 load raw → merge → atomic_write 读改写，**全程无锁**。aiohttp 同步 handler 在线程池跑，两个并发保存会 last-write-wins 丢更新。`atomic_write_text` 保证不读到半写文件，但 merge 丢失是真实风险。
- 建议：给 `_load_raw_settings`/`_load_base_model_path_defaults` 加 mtime-keyed memoize；给 save 路径加 `threading.Lock` 包住 read-modify-write。优先级：并发锁 > base.toml 缓存 > settings 文件缓存。

**测试总评**：业务域行为测试扎实，大模型依赖隔离良好；但自身架构胶水层（facade_compat、training_facade 派发、4 个 shim bucket 转发）缺契约测试，是随 split 模块搬家静默腐烂的高风险区。

## 六、可操作建议（按优先级）

| 优先级 | 项 | 位置 | 动作 |
|---|---|---|---|
| P0 | extra_args 任意参数注入 | routes/training.py + launcher_start.py | 加 train.py 参数白名单 + 路径类参数拒否 |
| P1 | _append_history_jsonl 静默吞没 | history_ops.py:356 | 失败时 `_remember_log("error",...)` |
| P1 | handle_raw_get 无 try | config.py:399 | 补 try/except 统一 400 |
| P2 | token 非常量时间比较 | server.py:114 | 换 `hmac.compare_digest` |
| 信息 | file_group_paths git show argv | file_group_paths.py:145 | 已核：list 形式 + cwd=ROOT + git 内部路径引用，不走 shell，风险极低 |
| P2 | image_test_service 上帝文件 | image_test_service.py | 按 normalize/command/weight/monitor 拆 |
| P2 | facade_compat 注册表零测试 | training/facade_compat.py | 补导出集合快照 + 每注册名 lazy-load 成功 + 未知名抛 KeyError 测试 |
| P2 | training_facade/resume_facade 派发层零契约测试 | history_meta/runtime_paths/history_timeline | 补属性解析契约 + 缺失属性行为测试 |
| P2 | 4 个 legacy_shims bucket 转发无断言 | legacy_shims_{preflight,merge,estimation,dataset} | 补转发契约测试（对齐现有 5 桶模式） |
| P2 | settings 并发写无锁 | settings_service save_* | save 路径加 threading.Lock 包住 read-modify-write |
| P2 | settings 反复重盘 TOML | settings_service _load_settings/_load_base_model_path_defaults | 加 mtime-keyed memoize（base.toml 优先） |
| P3 | ?token= cookie SameSite | server.py:127 | 外部部署建议 Strict + CSRF token |
| P3 | preview/images.py 继续拆 | preview/images.py | 按 detect/meta/filter 拆 |
| P3 | extra_args 注入拒绝无测试断言 | tests/test_training_start_preprocess.py | 补危险参数（--config_file/--network_weights）应被拒绝的用例，呼应 P0 |

## 七、总评

后端安全基线扎实：路径穿越四条管线统一收敛到 path_safety，subprocess 无 shell，鉴权设计正确，上传有界，不变量遵守良好。
最显著的可操作问题是 `extra_args` 任意参数注入（P0），其余多为健壮性/可维护性改进。
并发与测试覆盖的详细结论在对应子代理章节补全。
