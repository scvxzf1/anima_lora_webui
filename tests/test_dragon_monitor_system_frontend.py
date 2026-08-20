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
    assert "{ renderResults, setStatus }" in collection_controller
    shared_drag = _read("js/dragon-ui/ordered-drag-target.js")
    dataset_editor = _read("js/dragon-ui/pages/dataset-editor.js")
    assert "scheduleOrderedRowDropTarget" in shared_drag
    assert "scheduleOrderedRowDropTarget" in dataset_editor
    assert "PRESET_DRAG_TARGET_OPTIONS" in dataset_editor
    assert "results.querySelectorAll('.dragon-reveal')" in controller
    assert "element.classList.add('dragon-in-view')" in controller
    entry = _read("js/dragon-ui/index.js")
    animation_version = "./animations.js?v=dragon-ui-20260816v67"
    assert animation_version in entry
    assert f"../{animation_version.removeprefix('./')}" in controller
    assert ".dragon-history-controls" in css
    workbench_css = _read("css/dragon/03a-dragon-history-workbench.css")
    assert "max-width:2880px" in workbench_css
    assert "container:dragon-history-content / inline-size" in workbench_css
    assert "@container dragon-history-content (max-width:980px)" in workbench_css
    assert "grid-template-columns:repeat(6,minmax(0,1fr))" in workbench_css
    assert "@media(max-width:460px)" in workbench_css
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
    queue_css = pages.split("/* Dragon queue manager:", 1)[1].split("/* Dragon weight analysis", 1)[0]
    mobile_queue_css = queue_css.split("@media (max-width: 734px)", 1)[1].split("@media (max-width: 430px)", 1)[0]
    assert ".dragon-queue-filter button" in mobile_queue_css
    assert "min-height: 40px" in mobile_queue_css


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
    assert "(!state.dirty && !migrationPending(state))" in model


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


def test_training_config_uses_fluid_desktop_width_with_mobile_gutters():
    css = _read("css/dragon/04-dragon-config.css")
    page_rule = css[css.index(".dragon-config-page {"):css.index(".dragon-config-runbar {")]

    assert "width: calc(100% - (2 * var(--dragon-content-pad)));" in page_rule
    assert "max-width: none;" in page_rule
    assert "@media (max-width: 833px)" in css
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
    assert "id: 'datasets'" not in shortcut_catalog
    assert "id: 'history'" not in shortcut_catalog
    assert "id: 'configs'" not in shortcut_catalog
    assert "icon: 'folder'" not in shortcut_catalog
    assert "document.querySelectorAll('.dragon-nav-mobile-link[data-sub-id]')" in nav
    assert "const activePrimaryId" in nav
    assert ".dragon-nav-primary-link[data-active=\"true\"]" in css
    assert ".dragon-nav-primary-link[data-active=\"true\"]::after" in css


def test_live_monitor_exposes_current_task_context():
    state = _read("js/dragon-ui/pages/live-training-state.js")
    view = _read("js/dragon-ui/pages/live-training-view.js")
    assert "currentTask: formatCurrentTaskLabel(status)" in state
    assert "function formatCurrentTaskLabel" in state
    assert "contextItem('activity', '当前任务', model.currentTask, 'task')" in view
    assert "title: '当前监控'" in view


def test_tool_pages_use_fluid_multi_resolution_layouts():
    css = _read("css/dragon/06-dragon-pages.css")
    assert "width: calc(100% - clamp(40px, 6vw, 96px));" in css
    assert "max-width: 1600px;" in css
    assert "@media (max-width: 880px)" in css
    assert ".dragon-live-context { grid-template-columns: repeat(2, minmax(0, 1fr)); }" in css
    assert ".dragon-queue-layout { grid-template-columns: 1fr; }" in css
    assert ".dragon-queue-card { grid-template-columns: 1fr; }" in css
    assert ".dragon-queue-filter { width: 100%; justify-content: flex-start; overflow-x: auto;" in css
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


def test_training_config_category_is_a_real_single_subpage():
    config = _read("js/dragon-ui/pages/config-page.js")
    nav = _read("js/dragon-ui/nav.js")
    css = _read("css/dragon/04-dragon-config.css")

    assert "entries.find((entry) => entry.sub.id === context.subId) || entries[0]" in config
    assert "renderCategorySubPage" in config
    assert "renderConfigNavigation" in config
    assert 'href="#config/${category.id}/${item.id}"' in config
    assert 'data-config-entry="${sub.id}"' in config
    assert "data-config-subpage" not in config or "renderCategorySubPage" in config
    assert "#config/training-config" in nav
    assert "function renderCategoryPage" not in config
    assert ".dragon-config-workspace" in css
    assert "grid-template-columns: minmax(220px, 260px) minmax(0, 1fr);" in css
    assert "data-config-subpage=\"base-models\"" in css


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
    assert "window.matchMedia('(min-width: 1001px)')" in library
    assert "window.addEventListener('resize', schedule" in library
    assert "window.addEventListener('scroll', schedule" in library
    assert "destroy: () => viewportLayout.destroy()" in library
    assert "libraryController?.destroy?.()" in page


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
    assert "committed = { context, sub: targetSub, keys: targetKeys, values }" in page
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
    assert "await updateMountedConfigCategory(route.categoryId, route.subId)" in router
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
    state_at = page.index("committed = { context, sub: targetSub, keys: targetKeys, values }")
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
    assert "pageState.dirty = false;\n                    cleanupConfigPage(pageState);" in page
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
    assert "const currentMethodFamily = activeMethodFamily(trainingContext, currentValues);" in page
    assert "(entry.sub.id !== 'spd' || currentMethodFamily === 'spd')" in page
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
    assert "width: min(520px, calc(100vw - 32px));" in css
    assert ".dragon-training-dialog-actions" in css
    assert "@media (max-width: 520px)" in css
