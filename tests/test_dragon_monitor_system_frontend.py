from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "web" / "static"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_training_monitor_does_not_repeat_dashboard_entry():
    category_map = _read("js/dragon-ui/category-map.js")
    entry = _read("js/dragon-ui/index.js")
    monitor = category_map.split("id: 'training-monitor'", 1)[1].split("id: 'model-system'", 1)[0]
    assert "id: 'dashboard'" not in monitor
    assert "id: 'live-training'" in monitor
    assert "parts[0] === 'dashboard'" in entry
    assert "navigate({ type: 'page', page: 'dashboard' })" in entry


def test_dashboard_and_history_use_safe_semantic_actions():
    dashboard = _read("js/dragon-ui/pages/dashboard.js")
    history = _read("js/dragon-ui/pages/history-view.js")
    image_test = _read("js/dragon-ui/pages/image-test.js")
    queue_view = _read("js/dragon-ui/pages/queue-view.js")
    controls = _read("js/dragon-ui/pages/training-controls.js")
    assert 'role="progressbar"' in dashboard
    assert "window.confirm('确认停止当前训练吗？" in dashboard
    assert "model.status.status = 'idle'" not in dashboard
    assert 'href="#history/${encodeURIComponent(task.id || \'\')}"' in dashboard
    history_view = _read("js/dragon-ui/pages/history-view.js")
    assert 'href="#history/${encodeURIComponent(task.id || \'\')}"' in history_view
    assert 'role="button" tabindex="0"' not in history
    assert "window.confirm('确认停止当前推理吗？" in image_test
    assert '<main class="dragon-tool-panel dragon-queue-worklist' not in queue_view
    assert 'name="training_config_file" autocomplete="off"' in controls
    assert 'name="training_preset" autocomplete="off"' in controls
    assert 'name="training_gpu"' in controls


def test_history_page_uses_complete_backend_contract_and_archive_workspace():
    controller = _read("js/dragon-ui/pages/history.js")
    view = _read("js/dragon-ui/pages/history-view.js")
    css = _read("css/dragon/03-dragon-dashboard.css")

    assert "/api/training/history?limit=200" in controller
    assert "/api/preview/images?source=training&task_id=" in controller
    assert "/api/preview/weights?task_id=" in controller
    assert "/resume-options" in controller
    assert "'/api/training/resume'" in controller
    assert "'/api/training/queue/resume'" in controller
    assert "window.confirm(`确认${action}" in controller
    assert "task.name" in view
    assert "task.group" in view
    assert "task.started_at_text" in view
    assert "task.config_snapshot" in view
    assert "payload.config_toml" in view
    assert 'type="search" name="history_search" autocomplete="off"' in view
    assert 'name="history_status"' in view
    for state in ("done", "interrupted", "canceled", "compiling"):
        assert state in view
    assert 'width="${escapeAttribute(image.width || 1)}"' in view
    assert 'height="${escapeAttribute(image.height || 1)}"' in view
    assert 'loading="lazy"' in view
    assert "Intl.DateTimeFormat" in view
    assert ".dragon-history-controls" in css
    assert ".dragon-history-detail-grid" in css
    assert ".dragon-history-preview-grid" in css


def test_live_training_uses_real_backend_contract_and_full_workspace():
    page = _read("js/dragon-ui/pages/live-training.js")
    view = _read("js/dragon-ui/pages/live-training-view.js")
    state = _read("js/dragon-ui/pages/live-training-state.js")
    log_tools = _read("js/dragon-ui/pages/live-training-log-tools.js")
    ws = _read("js/dragon-ui/ws.js")
    assert "progress.current" in state
    assert "latestMetric.loss" in state
    assert "vram_used_gb" in state
    assert "message.state" in page
    assert "onUnmount" in page
    assert "/api/training/logs?limit=300" in page
    assert "Promise.allSettled" in page
    assert "model.apiConnected = false" in page
    assert "onOpen(" in page and "onClose(" in page
    assert "实时连接已断开" in state
    assert "训练速度" in view and "预计剩余（ETA）" in view
    assert "峰值显存" in view and "峰值等待采样" in view
    assert "轮数" not in view
    assert "role=\"progressbar\"" in view
    assert "data-live-log" in view
    assert 'type="search" name="live_log_search" autocomplete="off"' in view
    for action in ("copy", "download", "pause", "clear"):
        assert f"logToolButton('{action}'" in view
    assert 'data-live-log-action="${action}"' in view
    assert "model.logClearBeforeId" in log_tools
    assert "后端日志未删除" in log_tools
    assert "Intl.DateTimeFormat" in log_tools
    assert "return () => removeHandler('open', callback)" in ws
    assert "return () => removeHandler('close', callback)" in ws
    assert "renderToolButton('stop', '停止训练', 'stop'" in view


def test_queue_page_exposes_classic_core_controls():
    queue = _read("js/dragon-ui/pages/queue.js")
    queue_view = _read("js/dragon-ui/pages/queue-view.js")
    pages = _read("css/dragon/06-dragon-pages.css")
    assert "failure_policy" in queue
    assert "auto_retry" in queue
    assert "cancel-waiting" in queue
    assert "abort-after-current" in queue
    assert "force-abort" in queue
    assert "clear-completed" in queue
    assert "clear-canceled" in queue
    assert "/move" in queue
    assert "done" in queue
    assert "state.draft" in queue
    assert "root.addEventListener('change', settingsHandler)" in queue
    assert "if (options.quiet && state.settingsDirty) return" in queue
    assert "element.classList.add('dragon-in-view')" in queue
    assert "'队列空闲'" in queue_view
    assert "自动调度中" not in queue_view
    queue_css = pages.split("/* Dragon queue manager:", 1)[1].split("/* Dragon weight analysis", 1)[0]
    mobile_queue_css = queue_css.split("@media (max-width: 734px)", 1)[1].split("@media (max-width: 430px)", 1)[0]
    assert ".dragon-queue-filter button" in mobile_queue_css
    assert "min-height: 40px" in mobile_queue_css


def test_system_pages_keep_accessible_core_actions():
    model = _read("js/dragon-ui/pages/model-config.js")
    environment = _read("js/dragon-ui/pages/environment.js")
    settings = _read("js/dragon-ui/pages/global-settings.js")
    assert "type=\"search\"" in model
    assert "beforeLeave" in model
    assert "aria-current" in model
    assert "复制" in environment
    assert "刷新" in environment
    assert "恢复默认" in settings
    assert "payload.path_overrides" in settings
    assert "payload.effective_paths" in settings
    assert "normalizeFormSettings" in settings
    assert "(!state.dirty && !state.migrated)" in model


def test_weight_analysis_and_preview_match_classic_core_contracts():
    weight = _read("js/dragon-ui/pages/weight-analysis.js")
    weight_view = _read("js/dragon-ui/pages/weight-analysis-view.js")
    preview = _read("js/dragon-ui/pages/preview-workspace.js")
    preview_view = _read("js/dragon-ui/pages/preview-workspace-view.js")

    assert "/api/analysis/inspect-upload" in weight
    assert "A / B" in weight_view
    assert "window.print()" in weight
    assert "downloadText" in weight
    assert 'name="${slot}_weight_file"' in weight_view
    assert "例如：output/ckpt/my-lora.safetensors…" in weight_view
    assert "/api/preview/settings" in preview
    assert "/api/preview/weights" in preview
    assert "restore-defaults" in preview_view
    assert "模型与系统 · 数据" in preview_view
    assert 'width="${escapeAttribute(image.width || 1)}"' in preview_view
    assert 'height="${escapeAttribute(image.height || 1)}"' in preview_view
    assert 'loading="lazy"' in preview_view
    assert "preview_task_id" in preview_view
    assert "task_id" in preview
    assert "路径读取失败时不会允许覆盖保存" in preview_view
    assert "preview_group_key" in preview_view
    assert "preview_days" in preview_view
    assert "mode', 'config_group'" in preview
    assert "data-preview-image-select" in preview_view
    assert "deleteSelectedImages" in preview
    assert "配置分组聚合了多个训练目录；请切换到单个任务后删除图片。" in preview
    assert "grouped || selected === 0" in preview
    assert "input.disabled = grouped" in preview
    assert "/api/training/continue-lora/inspect" in preview
    assert "network_weights" in preview


def test_image_test_blocks_failed_dependencies_and_resolves_manual_weight_paths():
    image_test = _read("js/dragon-ui/pages/image-test.js")
    image_view = _read("js/dragon-ui/pages/image-test-view.js")
    assert "readApiResult" in image_test
    assert "blockingError" in image_test
    assert "/api/image-test/resolve-weight" in image_test
    assert 'list="dragon-image-weight-options"' in image_view
    assert "running || blocked" in image_view
    assert "statusSnapshotAvailable" in image_test
    assert "bindWeightDrop" in image_test
    assert "text/uri-list" in image_test


def test_weight_analysis_clears_stale_exports_and_expands_top20():
    weight = _read("js/dragon-ui/pages/weight-analysis.js")
    view = _read("js/dragon-ui/pages/weight-analysis-view.js")
    run_body = weight[weight.index("async function runAnalysis"):weight.index("async function inspectSource")]
    assert run_body.count("setExportEnabled(root, false)") >= 2
    assert "state.primaryResult = null" in run_body
    assert "查看全部 ${rows.length} 项" in view


def test_environment_check_runs_off_the_event_loop():
    route = (Path(__file__).resolve().parents[1] / "web" / "routes" / "environment.py").read_text(encoding="utf-8")
    assert "await asyncio.to_thread(run_environment_check)" in route


def test_shared_tool_styles_include_mobile_and_focus_contracts():
    base = _read("css/dragon/01-dragon-base.css")
    pages = _read("css/dragon/06-dragon-pages.css")
    assert ":focus-visible" in base
    assert "prefers-reduced-motion" in base
    assert ".dragon-tool-hero" in pages
    assert ".dragon-stat-grid" in pages
    assert ".dragon-live-context" in pages


def test_config_and_preview_protect_unsaved_or_unread_state():
    config = _read("js/dragon-ui/pages/config-page.js")
    controls = _read("js/dragon-ui/pages/training-controls.js")
    preview = _read("js/dragon-ui/pages/preview-workspace.js")
    css = _read("css/dragon/04-dragon-config.css")
    assert "renderConfigLoadError" in config
    assert "不会展示默认值，也不会允许保存或启动训练" in config
    assert "beforeLeave: () => confirmConfigDiscard" in config
    assert "beforeunload" in config
    assert "beforeContextChange" in controls
    assert "beforeLeave: () => confirmPreviewDiscard" in preview
    assert "已恢复默认路径；点击“保存路径设置”后生效。" in preview
    restore_body = preview[preview.index("function restoreDefaults"):preview.index("function syncPreviewDirty")]
    assert "saveSettings(" not in restore_body
    assert ".dragon-input:focus-visible" in css
    assert ".dragon-config-load-error" in css


def test_config_fields_expose_native_form_and_live_status_contracts():
    config = _read("js/dragon-ui/pages/config-page.js")
    assert 'role="status" aria-live="polite"' in config
    assert 'name="${name}"' in config
    assert 'autocomplete="off"' in config
    assert 'for="${fieldId}"' in config
    assert 'aria-checked="${value}"' in config
    assert "toggle.setAttribute('aria-checked'" in config
    assert 'aria-expanded="false"' in config
    assert "btn.setAttribute('aria-expanded'" in config


def test_tool_path_inputs_use_example_ellipsis_and_named_uploads():
    dataset_presets = _read("js/dragon-ui/pages/dataset-editor-presets.js")
    preview = _read("js/dragon-ui/pages/preview-workspace-view.js")
    weight = _read("js/dragon-ui/pages/weight-analysis-view.js")
    assert 'name="dataset_preset_import"' in dataset_presets
    assert '例如：output/ckpt/sample…' in preview
    assert '例如：output/tests…' in preview
    assert '例如：output/my-preview…' in preview
    assert '例如：output/ckpt/my-lora.safetensors…' in weight


def test_shell_supports_skip_link_theme_color_and_dark_native_controls():
    index_html = _read("index.html")
    theme = _read("js/dragon-ui/theme.js")
    base = _read("css/dragon/01-dragon-base.css")
    pages = _read("css/dragon/06-dragon-pages.css")
    assert 'name="theme-color"' in index_html
    assert 'class="dragon-skip-link" href="#dragon-main"' in index_html
    assert 'id="dragon-main" tabindex="-1"' in index_html
    assert "document.documentElement.style.colorScheme = theme" in theme
    assert ".dragon-skip-link:focus-visible" in base
    assert "content-visibility: auto" in pages
