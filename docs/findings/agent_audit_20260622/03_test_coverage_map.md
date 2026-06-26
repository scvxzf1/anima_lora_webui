# R3 — 测试覆盖地图

## 1. 反向索引（按子系统）

### 配置 / Web config
- `tests/test_config.py`, `test_config_normalize.py`, `test_web_config_service.py` — TOML 合并、Web 配置服务、sample prompts

### WebUI 前端静态
- `tests/test_training_frontend_state.py` — app.js→anima-app、禁 legacy-app、history collections、DOM/CSS 契约
- `tests/test_web_static_server.py` — 静态资源服务

### Web 训练 / 队列 / 预览
- `test_training_queue.py`, `test_training_resume.py`, `test_preview_service.py`, `test_weight_analysis_service.py`

### 训练核心
- `test_training_bootstrap.py`, `test_training_optimizers.py`, `test_training_gpu_selection.py`, `test_launch_config.py`

### 不变量 / buckets / harness
- `test_constant_token_buckets.py`, `test_native_flatten.py`, `test_runtime_harness_cli.py`, `test_ensure_text_strategies.py`

### Networks / LoRA family
- `test_network_registry.py`, `test_network_cfg.py`, `test_method_network_lifecycle.py`, `test_factory_metadata_flow.py`
- `test_lora_custom_autograd.py`, `test_loha.py`, `test_lokr.py`, `test_vera.py`, `test_glora.py`, `test_dora_lora.py`
- `test_global_router.py`, `test_router_compute.py`, `test_hydra_sigma_band.py`, `test_fera_fecl_handler.py`, `test_chimera_router_stats.py`

### 推理 / 编辑
- `test_generation_request.py`, `test_edit_dispatcher.py`, `test_directedit_v_injection.py`, `test_experimental_inference_tasks.py`, `test_inference_hydra_sigma.py`

### 预处理 / 数据
- `test_preprocess_dataset.py`, `test_preprocess_paths.py`, `test_caption_index.py`, `test_read_caption.py`, `test_latents_cache_strategy.py`

### Daemon / GUI / 其他
- `test_daemon.py`, `test_gui_variants.py`, `test_gui_jsonl_progress.py`, `test_block_swapping.py`, `test_smoke.py`

**全量文件数:** 72 个 `tests/test_*.py`（快照 2026-06-22）

## 2. 高优先级缺口（≥20，节选）
| # | 缺口 | 建议测什么 | 模板 |
|---|------|------------|------|
| 1 | launcher auto jsonl 路径 | mock task_dir 断言 CLI | test_training_queue |
| 2 | block_swap_profile auto | 同 launcher | test_launch_config |
| 3 | peak_probe auto | launcher | test_training_queue |
| 4 | Web preflight 路径逃逸 | 越界 output_root | test_preview_service |
| 5 | config merge gui vs methods_subdir | 双轨一致 | test_web_config_service |
| 6 | merge 拒绝 Hydra moe 键 | scan_non_bakeable | 新建 test_merge |
| 7 | DCW calibrator + aspect_id | 顺序变更检测 | test_constant_token_buckets |
| 8 | DirectEdit + generation 组合 | e2e mock | test_directedit_v_injection |
| 9 | EasyControl cond cache | preprocess | test_preprocess_dataset |
| 10 | IP-Adapter PE cache miss | 加载错误信息 | test_preprocess_paths |
| 11 | turbo/spd 非 print-config | CLI 拒绝 | test_config |
| 12 | vendor-sync 漂移检测 | hash 对比可选 | test_chimera_node_loader |
| 13 | FEI 推理每步 set_fei | mock network | test_global_router |
| 14 | attention_dispatch 新 backend | layout | test_lora_custom_autograd |
| 15 | LoKr OOM preset 字段 | catalog 键存在 | test_training_frontend_state |
| 16 | queue batch-start 双路由 | routes 别名 | test_training_queue |
| 17 | history collections API | settings order | test_training_frontend_state |
| 18 | weight_analysis 越界 ckpt | resolve_output_root | test_weight_analysis_service |
| 19 | spd sampler 元数据 | inference task | test_experimental_inference_tasks |
| 20 | BYG tuple cache 缺失 | preprocess | test_preprocess_dataset |
| 21 | colorize dataset_config | merge | test_config |
| 22 | memory_probe resolve auto 无 output_dir | resolve_path None | 新建 |

## 3. Smoke pytest（6 类改动，venv + timeout 60）
1. Web 前端: `.venv/bin/python -m pytest tests/test_training_frontend_state.py -q`
2. Web 配置: `.venv/bin/python -m pytest tests/test_web_config_service.py tests/test_config.py -q`
3. 队列/预览: `.venv/bin/python -m pytest tests/test_training_queue.py tests/test_preview_service.py -q`
4. 不变量: `.venv/bin/python -m pytest tests/test_constant_token_buckets.py tests/test_native_flatten.py tests/test_runtime_harness_cli.py -q`
5. Networks: `.venv/bin/python -m pytest tests/test_network_registry.py tests/test_factory_metadata_flow.py tests/test_global_router.py -q`
6. 训练启动: `.venv/bin/python -m pytest tests/test_training_bootstrap.py tests/test_launch_config.py tests/test_daemon.py -q`

**环境注记:** 系统 `python` 可能缺 `torch`/`toml`；审计机应用 `.venv/bin/python`。

## 立即可做 / 需改代码 / 不做什么
- 立即可做: 按改动类型跑上表 1 条 smoke
- 需改代码: 新行为先补缺口表对应测试
- 不做: 无 timeout 全量 pytest
