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


def test_history_detail_returns_to_the_page_that_opened_it():
    entry = _read("js/dragon-ui/index.js")
    navigation = _read("js/dragon-ui/history-return-navigation.js")
    controller = _read("js/dragon-ui/pages/history.js")
    view = _read("js/dragon-ui/pages/history-view.js")

    assert "trackHistoryDetailEntry(acceptedHash, nextHash)" in entry
    assert "previousTaskId" in navigation
    assert "activeHistoryReturn?.taskId === nextTaskId" in navigation
    assert "'#page/live-training'" in navigation
    assert "返回当前监控" in navigation
    assert "'#page/queue'" in navigation
    assert "返回任务队列" in navigation
    assert "return DEFAULT_HISTORY_RETURN" in navigation
    assert "resolveHistoryReturnNavigation(taskId)" in controller
    assert "window.location.hash = returnNavigation?.hash || '#history'" in controller
    assert "renderHistoryBackButton(model.returnNavigation)" in view
    assert "navigation?.label || '返回历史'" in view


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
    assert "historyFilterOptionAvailable" in view
    assert "${disabled ? ' disabled' : ''}" in view
    assert "expandArchiveScopeForMatches(state)" in controller
    assert "expanded = { ...state.filters, archived: 'all' }" in controller
    collection_controller = _read("js/dragon-ui/pages/history-collections-controller.js")
    assert "/api/training/history/collections/settings" in controller
    assert "/api/training/history/collections/settings" in collection_controller
    assert "/api/training/history/batch" in collection_controller
    assert "renderHistoryCollectionWorkbench" in controller
    assert "handleCollectionDrop" in collection_controller
    assert "data-history-collections" not in view
    assert "resultsHtml: renderHistoryCollectionWorkbench" in controller
    collections = _read("js/dragon-ui/pages/history-collections.js")
    assert "COLLECTION WORKBENCH" in collections
    assert "data-history-drop-collection" in collections
    assert "data-history-select-group" in collections
    assert "item.key === HISTORY_COLLECTION_ALL" in collections
    assert "workspace.selectedTaskIds = new Set" in collections
    assert "state.workspace.selectedTaskIds.clear()" in collection_controller
    assert "workspace.mode" not in controller
    assert "updateHistoryModeButton" not in collection_controller
    assert "scheduleOrderedRowDropTarget" in collection_controller
    assert "rowClassPrefix: 'dragon-history-collection-drop'" in collection_controller
    assert "dragon-dataset-preset-row" in collections
    assert "dragon-dataset-preset-group" in collections
    assert "data-history-collection-dropzone" in collections
    assert "return configGroups.map((group) => renderConfigGroup(group, workspace)).join('')" in collections
    assert "renderFlatCollectionTasks" not in collections
    assert "个任务集" in collections
    assert "Final Loss" in collections
    assert "loss_preview" in collections
    assert "dragon-history-task-sparkline" in collections
    assert "dragon-history-task-sparkline-tooltip" in collections
    assert "Min Loss" in collections and "Max Loss" in collections
    assert 'tabindex="0"' in collections
    assert 'data-count-state="${counts[key] > 0 ? \'nonzero\' : \'zero\'}"' in view
    assert "dragon-history-config-collapsed" not in collections
    assert 'class="dragon-history-advanced"' in view
    assert "高级筛选" in view
    assert "{ renderResults, setStatus }" in collection_controller
    shared_drag = _read("js/dragon-ui/ordered-drag-target.js")
    dataset_editor = _read("js/dragon-ui/pages/dataset-editor.js")
    assert "scheduleOrderedRowDropTarget" in shared_drag
    assert "scheduleOrderedRowDropTarget" in dataset_editor
    assert "PRESET_DRAG_TARGET_OPTIONS" in dataset_editor
    assert "results.querySelectorAll('.dragon-reveal')" in controller
    assert "element.classList.add('dragon-in-view')" in controller
    entry = _read("js/dragon-ui/index.js")
    animation_version = "./animations.js?v=dragon-ui-20260824v69"
    assert animation_version in entry
    assert f"../{animation_version.removeprefix('./')}" in controller
    assert ".dragon-history-controls" in css
    workbench_css = _read("css/dragon/03a-dragon-history-workbench.css")
    assert "max-width:2880px" in workbench_css
    assert "container:dragon-history-content / inline-size" in workbench_css
    assert "@container dragon-history-content (max-width:980px)" in workbench_css
    assert ".dragon-history-workbench-metrics>span" in workbench_css
    assert ".dragon-history-task-sparkline" in workbench_css
    assert ".dragon-history-task-sparkline-tooltip" in workbench_css
    assert ".dragon-history-config-task-list .dragon-history-item-state{white-space:nowrap}" in workbench_css
    assert ".dragon-history-config-task-list .dragon-history-item-state{grid-column:auto;grid-row:auto}" in workbench_css
    assert "@media(max-width:460px)" in workbench_css
    assert ".dragon-history-detail-grid" in css
    assert ".dragon-history-preview-grid" in css


def test_history_detail_tabs_use_fluid_width_and_content_breakpoints():
    workbench_css = _read("css/dragon/03a-dragon-history-workbench.css")

    assert ".dragon-history-page,.dragon-history-detail-page{" in workbench_css
    assert "max-width:2880px" in workbench_css
    assert "container:dragon-history-detail / inline-size" in workbench_css
    assert "@container dragon-history-detail (min-width:1600px)" in workbench_css
    assert 'data-history-detail-panel="metrics"' in workbench_css
    assert "grid-template-columns:minmax(0,2fr) minmax(360px,.8fr)" in workbench_css
    assert "repeat(auto-fit,minmax(min(100%,220px),1fr))" in workbench_css
    assert "repeat(auto-fit,minmax(min(100%,320px),1fr))" in workbench_css
    assert "@container dragon-history-detail (max-width:960px)" in workbench_css
    assert "@container dragon-history-detail (max-width:620px)" in workbench_css


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
    assert "if (options.quiet && queueSnapshotsEqual(state.model, nextModel)) return" in queue
    assert "onMessage('queue', applyQueueEvent)" in queue
    assert "IDLE_FALLBACK_INTERVAL_MS = 60000" in queue
    assert "ACTIVE_FALLBACK_INTERVAL_MS = 5000" in queue
    assert "document.hidden" in queue
    assert "document.addEventListener('visibilitychange', visibilityHandler)" in queue
    assert "document.removeEventListener('visibilitychange', visibilityHandler)" in queue
    assert "state.root.dataset.queueLiveUpdate = 'true'" in queue
    assert "window.scrollTo({ top: scrollTop, behavior: 'auto' })" in queue
    assert "data-queue-live-update=\"true\"" in pages
    assert "element.classList.add('dragon-in-view')" in queue
    assert "'队列空闲'" in queue_view
    assert "自动调度中" not in queue_view
    assert 'data-count-state="${count > 0 ? \'nonzero\' : \'zero\'}"' in queue_view
    assert 'class="dragon-queue-filter"' not in queue_view
    assert 'data-queue-confirm-dialog' in queue_view
    assert 'data-queue-more' in queue_view
    assert '中止后续队列' in queue_view
    assert '强制中止队列' in queue_view
    assert '取消全部队列' in queue_view
    assert '取消全部等待' in queue_view
    assert '清理已完成' in queue_view
    assert '清理已取消' in queue_view
    assert 'renderBulkPanel' not in queue_view
    assert 'dialog.showModal()' in queue
    assert 'state.confirming = true' in queue
    assert "event.key !== 'Escape'" in queue
    assert "document.addEventListener('click', documentClickHandler)" in queue
    assert '去配置页创建新任务' in queue_view
    assert '查看已完成记录' in queue_view
    queue_css = pages.split("/* Dragon queue manager:", 1)[1].split("/* Dragon weight analysis", 1)[0]
    assert '.dragon-queue-stat[data-count-state="nonzero"][data-state="error"]' in queue_css
    assert ".dragon-queue-confirm-dialog" in queue_css
    assert ".dragon-queue-empty-icon" in queue_css
    assert ".dragon-queue-more-menu" in queue_css


def test_system_pages_keep_accessible_core_actions():
    model = _read("js/dragon-ui/pages/model-config.js")
    model_library = _read("js/dragon-ui/pages/model-config-library.js")
    environment = _read("js/dragon-ui/pages/environment.js")
    settings = _read("js/dragon-ui/pages/global-settings.js")
    assert "type=\"search\"" in model
    assert "beforeLeave" in model
    assert "aria-current" in model_library
    assert "复制" in environment
    assert "刷新" in environment
    assert "恢复默认" in settings
    assert "payload.path_overrides" in settings
    assert "payload.effective_paths" in settings
    assert "normalizeFormSettings" in settings
    assert "dragon-global-settings-sidebar" in settings
    assert "dragon-global-settings-main" in settings
    assert "dragon-global-settings-stats" in settings
    assert "ROOT" in settings and "CONFIGS" in settings and "UI" in settings
    assert "data-settings-section=\"${sectionNumber}\"" in settings
    assert "data-global-summary" in settings
    savebar = settings.index('class="dragon-savebar dragon-global-settings-savebar"')
    first_group = settings.index('${SETTING_GROUPS.map((group, index) => renderGroup(group, state, index))')
    assert savebar < first_group
    assert "(!state.dirty && !migrationPending(state))" in model


def test_global_settings_savebar_stays_in_page_layout_at_all_breakpoints():
    css = _read("css/dragon/06-dragon-pages.css")
    scoped = css[css.index(".dragon-global-settings-savebar {"):css.index(".dragon-global-settings-group {")]
    responsive = css[css.index("@media (max-width: 734px)", css.index("@media (max-width: 1068px)")):]

    assert "position: sticky;" in scoped
    assert "bottom: auto;" in scoped
    assert "grid-template-columns: minmax(220px, 1fr) auto;" in scoped
    assert ".dragon-global-settings-savebar" in responsive
    assert "bottom: auto;" in responsive
    assert "grid-template-columns: minmax(0, 1fr);" in responsive


def test_global_settings_uses_fluid_width_and_adaptive_field_columns():
    css = _read("css/dragon/06-dragon-pages.css")
    settings_css = css[css.index(".dragon-global-settings-workspace {"):]

    assert "grid-template-columns: clamp(232px, 14vw, 280px) minmax(0, 1fr);" in settings_css
    assert "width: calc(100% - clamp(32px, 4vw, 72px));" in settings_css
    assert "max-width: 2400px;" in settings_css
    assert "repeat(auto-fit, minmax(min(100%, 300px), 1fr))" in settings_css
    assert ".dragon-settings-entry:first-child .dragon-settings-fields" not in settings_css
    assert "@media (max-width: 734px)" in settings_css
    assert ".dragon-settings-fields" in settings_css
    assert "grid-template-columns: 1fr;" in settings_css


def test_dragon_viewport_sizing_keeps_legacy_fallbacks_and_dynamic_units():
    base = _read("css/dragon/01-dragon-base.css")
    config = _read("css/dragon/04-dragon-config.css")
    pages = _read("css/dragon/06-dragon-pages.css")

    assert base.count("min-height: 100vh;") >= 2
    assert base.count("min-height: 100dvh;") >= 2
    assert "width: min(520px, calc(100% - 32px));" in config
    assert "width: calc(100% - 24px);" in config
    assert "max-height: min(720px, calc(100vh - 48px));" in config
    assert "max-height: min(720px, calc(100dvh - 48px));" in config
    assert "height: min(820px, calc(100vh - var(--dragon-nav-height) - 48px));" in pages
    assert "height: min(820px, calc(100dvh - var(--dragon-nav-height) - 48px));" in pages


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
    help_view = _read("js/dragon-ui/pages/config-field-help.js")
    assert 'role="status" aria-live="polite"' in config
    assert 'name="${name}"' in config
    assert 'autocomplete="off"' in config
    assert 'for="${fieldId}"' in config
    assert 'aria-checked="${value}"' in config
    assert "toggle.setAttribute('aria-checked'" in config
    assert 'aria-haspopup="dialog"' in help_view
    assert 'aria-controls="dragon-config-help-dialog"' in help_view
    assert "dialog.showModal()" in help_view


def test_training_config_uses_fluid_desktop_width_with_mobile_gutters():
    css = _read("css/dragon/04-dragon-config.css")
    page_rule = css[css.index(".dragon-config-page {"):css.index(".dragon-config-runbar {")]

    assert "width: calc(100% - (2 * var(--dragon-content-pad)));" in page_rule
    assert "max-width: none;" in page_rule
    assert "@media (max-width: 833px)" in css


def test_training_runbar_uses_equal_centered_columns_for_context_and_actions():
    css = _read("css/dragon/04-dragon-config.css")
    runbar = css[css.index(".dragon-config-runbar {"):css.index(".dragon-runbar-field,")]
    actions = css[css.index(".dragon-runbar-actions {"):css.index(".dragon-config-page .dragon-btn {")]
    action_buttons = css[css.index(".dragon-runbar-actions .dragon-btn {"):css.index("/* Launch dialogs")]

    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in runbar
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in actions
    assert "width: min(100%, 136px);" in action_buttons
    assert "justify-self: center;" in action_buttons
    assert "justify-content: center;" in action_buttons
    assert "width: calc(100% - (2 * var(--dragon-content-pad-mobile)));" in css


def test_primary_navigation_exposes_centered_workspace_routes_without_duplicate_icons():
    nav = _read("js/dragon-ui/nav.js")
    css = _read("css/dragon/02-dragon-nav.css")

    assert "const PRIMARY_NAV_ITEMS = [" in nav
    assert "{ id: 'training-config', label: '训练配置', hash: '#config/training-config' }" in nav
    assert "{ id: 'datasets', label: '数据集', hash: '#dataset-editor' }" in nav
    assert "{ id: 'history', label: '训练历史', hash: '#history' }" in nav
    assert "{ id: 'model-config', label: '模型配置', hash: '#model-config' }" in nav
    assert "{ id: 'live-training', label: '当前监控', hash: '#page/live-training' }" in nav
    assert "{ id: 'queue', label: '训练队列', hash: '#page/queue' }" in nav
    assert "PRIMARY_NAV_ITEMS.map(renderPrimaryNavItem)" in nav
    assert 'data-primary-nav="${item.id}"' in nav
    shortcut_catalog = nav[nav.index("const NAV_SHORTCUTS = ["):nav.index("];", nav.index("const NAV_SHORTCUTS = ["))]
    assert "{ id: 'global-settings', label: '全局设置', compactLabel: '全局设置', icon: 'settings', hash: '#global-settings' }" in shortcut_catalog
    assert "id: 'datasets'" not in shortcut_catalog
    assert "id: 'history'" not in shortcut_catalog
    assert "id: 'configs'" not in shortcut_catalog
    assert "icon: 'folder'" not in shortcut_catalog
    assert "document.querySelectorAll('.dragon-nav-mobile-link[data-sub-id]')" in nav
    assert "const activePrimaryId" in nav
    assert "'global-settings': hash.startsWith('#global-settings')" in nav
    assert ".dragon-nav-primary-link[data-active=\"true\"]" in css
    assert ".dragon-nav-primary-link[data-active=\"true\"]::after" in css
    assert ".dragon-nav-utility-button[data-active=\"true\"]" in css
    assert ".dragon-nav-mobile-shortcut[data-active=\"true\"]" in css


def test_live_monitor_exposes_current_task_context():
    state = _read("js/dragon-ui/pages/live-training-state.js")
    view = _read("js/dragon-ui/pages/live-training-view.js")
    workspace = _read("js/dragon-ui/pages/live-training-workspace.js")
    assert "currentTask: formatCurrentTaskLabel(status)" in state
    assert "function formatCurrentTaskLabel" in state
    assert "正在训练：${model.currentTask}" in view
    assert "renderLiveSidebar(model)" in view
    assert "renderLiveStatePanel(model, 'idle')" in view
    assert "renderLiveStatePanel(model, 'error')" in view
    assert "liveWorkspaceMode" in workspace


def test_tool_pages_use_fluid_multi_resolution_layouts():
    css = _read("css/dragon/06-dragon-pages.css")
    assert "width: calc(100% - clamp(40px, 6vw, 96px));" in css
    assert "max-width: 1600px;" in css
    assert "@media (max-width: 880px)" in css
    assert ".dragon-live-context { grid-template-columns: repeat(2, minmax(0, 1fr)); }" in css
    assert ".dragon-queue-layout { grid-template-columns: 1fr; }" in css
    assert ".dragon-queue-card { grid-template-columns: 1fr; }" in css
    assert '.dragon-queue-manager .dragon-queue-stats {' in css
    assert ".dragon-live-context { grid-template-columns: 1fr; }" in css


def test_training_config_category_header_is_compact_and_title_first():
    css = _read("css/dragon/04-dragon-config.css")
    start = css.index(".dragon-config-category-page .dragon-config-hero {")
    end = css.index(".dragon-config-index {", start)
    header_rules = css[start:end]

    assert "display: flex;" in header_rules
    assert "align-items: center;" in header_rules
    assert "justify-content: center;" in header_rules
    assert "padding: 12px 0;" in header_rules
    assert ".dragon-config-category-page .dragon-config-hero h1" in header_rules
    assert "order: 1;" in header_rules
    assert "text-overflow: ellipsis;" in header_rules


def test_training_config_category_header_omits_workbench_eyebrow():
    config = _read("js/dragon-ui/pages/config-page.js")
    start = config.index("function renderCategorySubPage")
    end = config.index("function renderConfigNavigation", start)
    assert '<span class="dragon-eyebrow">训练工作台</span>' not in config[start:end]


def test_training_config_category_header_omits_description():
    config = _read("js/dragon-ui/pages/config-page.js")
    assert "训练所需的基础模型、数据行为、适配器、步数和采样设置。" not in config
    assert "${description ? `<p>${description}</p>` : ''}" in config


def test_training_config_category_is_a_real_single_subpage():
    config = _read("js/dragon-ui/pages/config-page.js")
    nav = _read("js/dragon-ui/nav.js")
    css = _read("css/dragon/04-dragon-config.css")

    assert "resolveConfigView(entries, preferredConfigSubId(context.subId, category), category)" in config
    assert "renderCategorySubPage" in config
    assert "renderConfigNavigation" in config
    assert 'href="#config/${category.id}/${item.id}"' in config
    assert 'data-config-entry="${sub.id}"' in config
    assert "data-config-subpage" not in config or "renderCategorySubPage" in config
    assert "#config/training-config" in nav
    assert "function renderCategoryPage" not in config
    assert ".dragon-config-workspace" in css
    assert "grid-template-columns: minmax(220px, 260px) minmax(0, 1fr);" in css
    assert ".dragon-config-section-summary" in css


def test_training_config_navigation_is_merged_into_five_workflow_groups():
    category_map = _read("js/dragon-ui/category-map.js")
    section_groups = _read("js/dragon-ui/pages/section-groups.js")
    router = _read("js/dragon-ui/router.js")
    training = category_map.split("id: 'training-config'", 1)[1].split("id: 'memory-optimization'", 1)[0]

    assert "items: TRAINING_CONFIG_ITEMS" in training
    assert "FORM_CATEGORY_DEFS.map" in category_map
    assert "required: ['base-models', 'data-behavior', 'dataset-filter']" in category_map
    assert "common: ['adapter-basics', 'lokr', 'optimizer', 'steps-volume', 'timestep', 'output-save', 'logging']" in category_map
    assert "preview: ['train-sampling']" in category_map
    assert "item.legacyIds?.includes(id)" in category_map
    assert "matchedSub?.categoryId === route.categoryId ? matchedSub.id : route.subId" in router
    assert "FORM_CATEGORY_DEFS.map" in section_groups
    assert "FORM_SECTION_DEFS.find" in section_groups
    assert "collapsible: true" in section_groups


def test_training_config_matches_classic_scope_and_keeps_unclassified_fields():
    page = _read("js/dragon-ui/pages/config-page.js")
    for shared_filter in (
        "CONFIG_FORM_INTERNAL_KEYS",
        "CONFIG_FORM_MERGED_FIELDS",
        "DATASET_BLUEPRINT_FIELDS",
        "DEPRECATED_CONFIG_FORM_FIELDS",
        "RETIRED_CONFIG_FORM_FIELDS",
        "SPD_UI_DEFAULT_FIELDS",
        "CHIMERA_UI_DEFAULT_FIELDS",
        "IP_ADAPTER_UI_DEFAULT_FIELDS",
    ):
        assert shared_filter in page
    assert "CONVROT_FIELD_KEYS.has(key)" in page
    assert "['w8a16_convrot', 'w8a8_convrot'].includes(baseCompute)" in page
    assert "function unclassifiedConfigKeys(values, knownKeys)" in page
    assert "entry.sub.id === 'advanced'" in page
    assert "unclassifiedConfigKeys(values, knownKeys)" in page


def test_training_config_long_sections_are_collapsible_and_fluid():
    page = _read("js/dragon-ui/pages/config-page.js")
    css = _read("css/dragon/04-dragon-config.css")
    assert "function renderConfigSection(section, keys, currentValues," in page
    assert '<details class="dragon-config-section dragon-config-section-collapsible"' in page
    assert 'class="dragon-config-section-count">${keys.length} 项' in page
    assert ".dragon-config-section-summary::after" in css
    assert ".dragon-config-section-collapsible[open]" in css
    assert "repeat(auto-fit, minmax(min(100%, 250px), 1fr))" in css
    assert "repeat(auto-fit, minmax(min(100%, 240px), 1fr))" in css


def test_training_config_recomputes_scoped_fields_and_supports_search():
    page = _read("js/dragon-ui/pages/config-page.js")
    css = _read("css/dragon/04-dragon-config.css")
    assert "function buildCategoryEntries(category, rawEntries, trainingContext, values)" in page
    assert "buildCategoryEntries(category, rawEntries, context, values)" in page
    assert "entries = nextEntries" in page
    assert "const activeSub = activeView.sub || sub" in page
    assert "function renderConfigFieldFilter(total)" in page
    assert "function bindConfigFieldFilter(root, state)" in page
    assert "data-config-field-search" in page
    assert "bindConfigFieldFilter(root, pageState)" in page
    assert ".dragon-config-field-filter" in css


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


def test_training_config_preset_library_supports_cross_group_and_in_group_drag_drop():
    page = _read("js/dragon-ui/pages/config-page.js")
    library = _read("js/dragon-ui/pages/training-preset-library.js")
    controls = _read("js/dragon-ui/pages/training-controls.js")
    css = _read("css/dragon/04a-dragon-training-presets.css")

    assert "renderTrainingPresetLibrary(trainingContext)" in page
    assert "bindTrainingPresetLibrary(root, committed.context" in page
    assert 'aria-label="训练配置预设管理"' in library
    assert 'draggable="${movable}"' in library
    assert "trainingPresetDropPosition" in library
    assert "targetOrder(state, file, groupId" in library
    assert "'/api/config/file-groups/place'" in library
    assert "JSON.stringify({ target: 'file', file, group: groupId, order })" in library
    assert "dragon-training-preset-drop-before" in library
    assert "dragon-training-preset-drop-after" in library
    assert "将放置到" in library
    assert "groups," in controls
    assert "selectTrainingConfigFile" in controls
    assert ".dragon-config-shell-layout" in css
    assert ".dragon-training-preset-library" in css
    assert ".dragon-training-preset-drop-before::before" in css
    assert ".dragon-training-preset-drop-after::after" in css
    assert "visibleTrainingGroups" in library
    assert "HIDDEN_TRAINING_GROUP_IDS = new Set(['gui_methods', 'presets'])" in library
    assert "!HIDDEN_TRAINING_GROUP_IDS.has(group.id)" in library
    assert "group.methods_subdir !== 'gui-methods'" in library
    assert "height: var(--dragon-training-preset-height" in css
    assert "box-sizing: border-box;" in css
    assert "container: training-preset-library / inline-size;" in css
    assert "display: flex;" in css
    assert "flex-direction: column;" in css
    assert "flex: 1 1 0;" in css
    assert "align-content: start;" in css
    assert "grid-auto-rows: max-content;" in css
    assert "@container training-preset-library (max-width: 300px)" in css
    assert "overflow-y: scroll;" in css
    assert "scrollbar-gutter: stable;" in css
    assert ".dragon-training-preset-groups::-webkit-scrollbar { width: 11px; }" in css
    assert "padding: 9px 14px;" in css
    assert "body[data-dragon-ui] #dragon-root .dragon-training-preset-group > header" in css
    assert "padding: 8px 10px 10px;" in css
    assert "bindTrainingPresetViewport(library)" in library
    assert "window.innerHeight - top - 16" in library
    assert "matchesDragonViewport(DRAGON_VIEWPORT_QUERIES.trainingPresetSidebar)" in library
    assert "window.addEventListener('resize', schedule" in library
    assert "window.addEventListener('scroll', schedule" in library
    assert "destroy: () => viewportLayout.destroy()" in library
    assert "libraryController?.destroy?.()" in page


def test_dragon_viewport_contracts_are_named_and_have_browser_fallbacks():
    responsive = _read("js/dragon-ui/responsive.js")
    nav = _read("js/dragon-ui/nav.js")
    library = _read("js/dragon-ui/pages/training-preset-library.js")

    assert "mobileNavigation: '(max-width: 833px)'" in responsive
    assert "trainingPresetSidebar: '(min-width: 1001px)'" in responsive
    assert "typeof window.matchMedia !== 'function'" in responsive
    assert "DRAGON_VIEWPORT_QUERIES.mobileNavigation" in nav
    assert "window.innerWidth > 833" not in nav
    assert "DRAGON_VIEWPORT_QUERIES.trainingPresetSidebar" in library
    assert "window.matchMedia('(min-width: 1001px)')" not in library


def test_preset_sidebars_use_shared_theme_spacing_and_clear_boundaries():
    training = _read("js/dragon-ui/pages/training-preset-library.js")
    training_css = _read("css/dragon/04a-dragon-training-presets.css")
    dataset_css = _read("css/dragon/06-dragon-pages.css")

    assert '<span class="dragon-eyebrow">预设库</span>' in training
    assert '>PRESETS<' not in training
    assert "padding: var(--dragon-sp-4);" in training_css
    assert "border: 1px solid var(--dragon-border);" in training_css
    assert ".dragon-training-preset-row:focus-within" in training_css
    assert "padding: 9px 14px;" in training_css
    assert ".dragon-training-preset-feedback:empty" in training_css
    assert ".dragon-dataset-library {\n    border-color: var(--dragon-border);" in dataset_css
    assert ".dragon-dataset-preset-item {\n    min-height: 52px;\n    padding: 10px 12px;" in dataset_css


def test_training_preset_library_exports_current_file_and_group_archive():
    library = _read("js/dragon-ui/pages/training-preset-library.js")
    css = _read("css/dragon/04a-dragon-training-presets.css")

    assert 'data-training-preset-action="export"' in library
    assert "exportCurrentConfig(library, state)" in library
    assert "/api/config/raw?file=${encodeURIComponent(file)}" in library
    assert "downloadTextFile(payload.content || ''" in library
    assert "/api/config/file-groups/${encodeURIComponent(group.id)}/export?kind=training" in library
    assert 'title="导出分组 ZIP"' in library
    assert ".dragon-training-preset-group-actions a" in css


def test_training_preset_library_imports_and_saves_as_editable_config():
    page = _read("js/dragon-ui/pages/config-page.js")
    library = _read("js/dragon-ui/pages/training-preset-library.js")
    css = _read("css/dragon/04a-dragon-training-presets.css")

    assert 'data-training-preset-action="import"' in library
    assert 'data-training-preset-action="save-as"' in library
    assert 'data-training-preset-action="save-updates"' in library
    assert 'type="file" accept=".toml,text/plain,application/toml"' in library
    assert "importTrainingConfig(library, state" in library
    assert "saveCurrentConfigAs(library, state" in library
    assert "'/api/config/raw/save-as'" in library
    assert "JSON.stringify({ file, content })" in library
    assert "`configs/imported/${normalizeImportedFilename(answer)}`" in library
    assert "await refreshLibrary(library, state, beforeContextChange)" in library
    assert "activateConfigFile(state, file)" in library
    assert "selectTrainingConfigFile(state.context, file" in library
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in css
    assert '[data-training-preset-action="save-updates"]' in css
    assert "onSaveChanges: () => saveCurrentChanges?.() ?? false" in page
    assert "await state.onSaveChanges()" in library


def test_training_preset_manager_refreshes_independently_from_atomic_editable_pane():
    page = _read("js/dragon-ui/pages/config-page.js")
    library = _read("js/dragon-ui/pages/training-preset-library.js")
    controls = _read("js/dragon-ui/pages/training-controls.js")
    css = _read("css/dragon/04-dragon-config.css")

    assert 'data-config-editable-pane' in page
    assert "renderEditableConfigPane" in page
    assert "transitionEditable" in page
    assert "committed = { context, ...nextView }" in page
    assert "currentPane.outerHTML = renderEditableConfigPane" in page
    assert "libraryController?.updateContext(committed.context)" in page
    assert "selectTrainingConfigFile(committed.context, file, { notify: false, persist: false })" in page
    assert "selectTrainingPreset(committed.context, preset, { notify: false, persist: false })" in page
    assert "return {\n        updateContext(nextContext)" in library
    assert "const scrollTop = library.querySelector('[data-training-preset-groups]')?.scrollTop || 0" in library
    assert "if (groups) groups.scrollTop = scrollTop" in library
    assert "{ notify = true, persist = true }" in controls
    assert "if (persist) storeContext(nextContext)" in controls
    assert "commitTrainingContext(context)" in page
    assert "const sequence = ++transitionSequence" in page
    assert "if (sequence !== transitionSequence) return true" in page
    assert "configFile: state.context.configFile" in library
    assert '[data-config-editable-pane][aria-busy="true"]' in css


def test_config_subpage_routes_keep_independent_preset_manager_mounted():
    router = _read("js/dragon-ui/router.js")
    page = _read("js/dragon-ui/pages/config-page.js")

    assert "async function updateMountedConfigCategory" in router
    assert "currentPage.onRouteUpdate" in router
    assert "await updateMountedConfigCategory(route.categoryId, subId)" in router
    assert "await updateMountedConfigCategory(sub.categoryId, sub.id)" in router
    assert "onRouteUpdate = content.onRouteUpdate" in router
    assert "currentPage = { pageType, context, beforeLeave, onUnmount, onRouteUpdate }" in router
    assert "routeUpdater = async ({ subId })" in page
    assert "currentPane.outerHTML = renderEditableConfigPane" in page
    assert "onRouteUpdate: (route) => routeUpdater?.(route) ?? false" in page
    assert "renderTrainingPresetLibrary" not in page[page.index("routeUpdater = async"):page.index("onRouteUpdate:", page.index("routeUpdater = async"))]


def test_training_context_transition_does_not_publish_mixed_render_state():
    page = _read("js/dragon-ui/pages/config-page.js")
    library = _read("js/dragon-ui/pages/training-preset-library.js")
    controls = _read("js/dragon-ui/pages/training-controls.js")

    commit_at = page.index("commitTrainingContext(context)")
    response_at = page.index("const res = await api(mergedConfigUrl(context))")
    state_at = page.index("committed = { context, ...nextView }")
    render_at = page.index("currentPane.outerHTML = renderEditableConfigPane")
    assert response_at < state_at < commit_at < render_at
    assert "persist: false" in page
    assert "if (persist) storeContext(nextContext)" in controls
    assert "event.target.value = context.configFile" in controls
    assert "event.target.value = context.preset" in controls
    callback_branch = library[library.index("if (state.onConfigFileChange)"):library.index("library.querySelectorAll('[data-training-group-action]')")]
    assert "state.context = nextContext" not in callback_branch.split("else {", 1)[0]
    assert "configFile: state.context.configFile" in library
    assert "preset: state.context.preset" in library


def test_atomic_config_pane_registers_fresh_reveal_nodes_after_partial_render():
    page = _read("js/dragon-ui/pages/config-page.js")
    router = _read("js/dragon-ui/router.js")
    entry = _read("js/dragon-ui/index.js")
    animations = _read("js/dragon-ui/animations.js")

    render_at = page.index("currentPane.outerHTML = renderEditableConfigPane")
    bind_at = page.index("bindEditablePane();", render_at)
    reveal_at = page.index("scanForReveal();", bind_at)
    library_at = page.index("libraryController?.updateContext(committed.context)", reveal_at)

    assert "import { scanForReveal } from '../animations.js" in page
    assert render_at < bind_at < reveal_at < library_at
    animation_token = page.split("../animations.js?v=", 1)[1].split("'", 1)[0]
    assert f"./animations.js?v={animation_token}" in router
    assert f"./animations.js?v={animation_token}" in entry
    assert "threshold: 0.01" in animations
    assert "threshold: 0.15" not in animations


def test_config_transition_cancels_stale_route_and_preserves_dirty_state_on_failure():
    page = _read("js/dragon-ui/pages/config-page.js")
    router = _read("js/dragon-ui/router.js")
    controls = _read("js/dragon-ui/pages/training-controls.js")

    assert "const sequence = ++mountedRouteUpdateSequence" in router
    assert "if (sequence !== mountedRouteUpdateSequence) return true" in router
    assert "mountedRouteUpdateSequence += 1" in router
    unmount_at = router.index("currentPage.onUnmount?.();", router.index("async function renderPage"))
    fade_at = router.index("// Fade out current content", unmount_at)
    loader_at = router.index("content = await loader(context)", fade_at)
    assert unmount_at < fade_at < loader_at
    assert "onRouteUpdate: null" in router[unmount_at:fade_at]
    dispose_at = page.index("transitionSequence += 1;", page.index("disposeMountedPage ="))
    library_destroy_at = page.index("libraryController?.destroy?.();", dispose_at)
    cleanup_at = page.index("cleanupConfigPage(pageState);", library_destroy_at)
    assert dispose_at < library_destroy_at < cleanup_at
    assert "if (contextChanged) resetConfigFormState(pageState, values, nextEntries);" in page
    assert "pageState.dirty = false;" not in page
    confirm_body = page[page.index("function confirmConfigDiscard"):page.index("function cleanupConfigPage")]
    assert "state.dirty = false" not in confirm_body
    assert "const stored = readStoredContext();" in controls[controls.index("await Promise.all"):]


def test_training_config_defaults_to_editable_file_and_hides_spd_ghost_fields():
    controls = _read("js/dragon-ui/pages/training-controls.js")
    page = _read("js/dragon-ui/pages/config-page.js")

    # Default selection must prefer an unlocked/editable file so the config page
    # save path can actually persist changes (gui-methods files are system-locked).
    assert "const editableFiles = files.filter((file) => !file.locked && !file.readonly);" in controls
    assert "const fallback = editableFiles[0]" in controls
    assert "editableFiles," in controls
    assert "export function isEditableConfigFile(context)" in controls
    assert "return !file.locked && !file.readonly;" in controls
    assert 'data-locked="true"' in controls
    assert "（只读）" in controls
    assert "if (!isEditableConfigFile(trainingContext))" in page
    assert "当前训练配置为系统只读，无法保存修改" in page
    # SPD fields are CLI-only ghosts for every selectable Web config; the SPD
    # sub-page must be suppressed unless the current method family is spd.
    assert "const method = activeMethodFamily(trainingContext, values);" in page
    assert "(entry.sub.id !== 'spd' || method === 'spd')" in page
    assert "SPD 是 CLI 实验配置，当前训练配置不会使用这些参数" in page


def test_training_queue_action_stays_on_config_page_and_uses_dragon_dialog_surface():
    controls = _read("js/dragon-ui/pages/training-controls.js")
    css = _read("css/dragon/04-dragon-config.css")

    assert "if (payload.ok && action !== 'queue') window.location.hash = '#live-training';" in controls
    assert "window.location.hash = action === 'queue' ? '#queue' : '#live-training';" not in controls
    assert "title: queued ? '已加入训练队列'" in controls
    assert "当前仍停留在训练配置页" in controls
    assert 'class="dragon-training-dialog-shell"' in controls
    assert 'class="dragon-training-dialog-header"' in controls
    assert ".dragon-training-dialog::backdrop" in css
    assert "width: min(520px, calc(100% - 32px));" in css
    assert ".dragon-training-dialog-actions" in css
    assert "@media (max-width: 520px)" in css
