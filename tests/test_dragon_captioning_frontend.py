from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / "web" / "static" / relative).read_text(encoding="utf-8")


def test_dragon_captioning_is_a_lazy_top_level_workspace() -> None:
    nav = _read("js/dragon-ui/nav.js")
    loaders = _read("js/dragon-ui/page-loaders.js")
    styles = _read("js/dragon-ui/route-styles.js")
    router = _read("js/dragon-ui/router.js")

    assert "{ id: 'captioning', label: '打标', hash: '#page/captioning' }" in nav
    assert "captioning: styledPage('captioning'" in loaders
    assert "pages/captioning.js" in loaders
    assert "06c-dragon-captioning.css" in styles
    assert "captioning: '外部 API 打标'" in router


def test_captioning_frontend_exposes_provider_and_scheduler_controls() -> None:
    settings = _read("js/features/captioning/settings-panel.js")
    routing = _read("js/features/captioning/routing-panel.js")
    controls = _read("js/features/captioning/control-bar.js")
    page = _read("js/features/captioning/page.js")
    for field in (
        'name="base_url"',
        'name="api_key"',
        'name="model"',
        'name="retry_count"',
        'name="retry_interval_seconds"',
        'name="concurrency"',
        'name="timeout_seconds"',
        'name="allow_private_network"',
    ):
        assert field in settings
    assert 'data-caption-test="ping"' in settings
    assert 'data-caption-test="actual"' in settings
    assert "留空以保留已保存密钥" in settings
    assert "data-caption-routing-form" in routing
    assert "data-channel-field=\"_apiKeys\"" in routing
    assert "data-routing-action=\"move-step-up\"" in routing
    assert "data-routing-action=\"test-ping\"" in routing
    assert "data-routing-action=\"test-actual\"" in routing
    assert "reasoning_effort" in routing
    assert "clear_api_keys" in routing
    assert "data-channel-feedback" in routing and "Ping 中…" in routing
    assert "dragon-caption-settings-summary" in settings
    assert "querySelectorAll('[data-caption-settings-close]')" in settings
    assert 'name="schedule_id"' in controls
    assert "mixed_70tag_30nl" in controls and "pure_nl" in controls
    assert "captioningApi('/routing')" in page


def test_captioning_frontend_reviews_before_committing() -> None:
    inspector = _read("js/features/captioning/inspector.js")
    workbench = _read("js/features/captioning/workbench.js")
    dataset_fields = _read("js/dragon-ui/pages/dataset-editor-fields.js")
    dataset_page = _read("js/dragon-ui/pages/dataset-editor.js")

    assert "data-caption-save-next" in inspector
    assert "data-caption-pill" in inspector
    assert "/commit" in inspector and "write_mode" in inspector
    assert "renderGallery" in workbench
    assert "已选 ${selectedCount} 项" in workbench
    assert "将覆盖写回 ${count}" in workbench
    assert "data-dataset-caption" in dataset_fields
    assert "dragon-captioning-prefill" in dataset_page


def test_captioning_frontend_guards_async_job_interactions() -> None:
    jobs = _read("js/features/captioning/workbench.js")
    page = _read("js/features/captioning/page.js")

    assert "state.selectedJobId !== jobId" in jobs
    assert "requestId !== state.jobRequestId" in jobs
    assert "state.active = false" in page


def test_captioning_layout_has_desktop_and_mobile_constraints() -> None:
    css = _read("css/dragon/06c-dragon-captioning.css")
    assert "grid-template-columns: minmax(360px, 44%) minmax(480px, 56%)" in css
    assert "grid-template-columns: repeat(auto-fill, minmax(116px, 1fr))" in css
    assert "@media (max-width: 760px)" in css
    assert ".dragon-caption-main { grid-template-columns: 1fr; overflow: visible; }" in css
    assert "max-height: min(250px, 30dvh)" in css
    assert ".dragon-caption-command > .dragon-caption-engine { grid-column: 1; }" in css


def test_captioning_workbench_covers_presets_governance_and_local_engines() -> None:
    controls = _read("js/features/captioning/control-bar.js")
    presets = _read("js/features/captioning/presets.js")
    governance = _read("js/features/captioning/governance.js")
    inspector = _read("js/features/captioning/inspector.js")

    for engine in ("anima_tagger", "wd14", "local_caption", "hybrid"):
        assert engine in controls or "engineOptions" in controls
    assert "Flux / SD3 长描述" in presets
    assert "Danbooru / Anime 标签" in presets
    assert "主体解耦 / 姿态动作" in presets
    assert "data-caption-frequency-tag" in governance
    assert "action: 'replace'" in governance
    assert "action: 'blacklist'" in governance
    assert "data-caption-zoom" in inspector
    assert "draggable=\"true\"" in inspector
    assert "data-caption-use-variant" in inspector
    assert "API 尝试记录" in inspector


def test_captioning_suite_covers_the_full_embedded_labeler_workflow() -> None:
    suite = _read("js/features/captioning/suite.js")
    gallery = _read("js/features/captioning/gallery.js")
    files = _read("js/features/captioning/workspace/files-panel.js")
    retry = _read("js/features/captioning/workspace/retry-panel.js")
    export = _read("js/features/captioning/workspace/export-panel.js")
    config = _read("js/features/captioning/workspace/config-panel.js")
    logs = _read("js/features/captioning/workspace/logs-panel.js")
    role = _read("js/features/captioning/workspace/role-panel.js")

    for panel in ("审阅台", "角色 Tag", "打标补全", "目录浏览", "失败重试", "目录组", "Tag 管理", "Caption 导出", "数据集生成", "提示词预设", "打标日志", "配置中心"):
        assert panel in suite
    for filter_name in ("全部", "可导出", "失败", "解析失败", "已选择"):
        assert filter_name in gallery
    assert "data-file-grid" in files and "data-file-open-workbench" in files
    assert "fileScanning" in files and "fileScanned" in files and "fileQuery" in files
    assert "aria-pressed" in files and "data-file-search-empty" in files
    assert "data-retry-all" in retry and "failure_kind" in retry
    assert "retryLoading" in retry and "retryFilter" in retry
    assert "筛选不改变批量范围" in retry and "重试中…" in retry
    assert "manual_directory" in export and "download-captions" in export
    assert "captions-json" in export and "image-txt" in export
    assert "data-export-inspect" in export and "exportJobDetail" in export
    assert "项因状态不符将跳过" in export and "resolveTarget" in export
    assert "data-config-clear" in config and config.count("window.confirm") >= 2
    assert "/workspace/config" in config and "#global-settings" in config
    assert "validateConfigJson" in config and "data-config-validation" in config
    assert "dragon-caption-danger-zone" in config and "configBusy" in config
    assert "/workspace/logs?page=" in logs and "data-log-page" in logs
    assert "Number.isFinite(entry.duration_ms)" in logs
    assert "确认清空全部打标日志" in logs
    assert "logSearchTimer" in logs and "addEventListener('input'" in logs
    assert "logsRequestId" in logs
    assert "logsLoading" in logs and "dragon-caption-log-summary" in logs
    assert "formatTimestamp" in logs and "清空全部日志" in logs
    assert "data-role-drop" in role and "rolePreview" in role
    assert "roleSource" in role and "正在生成…" in role
    assert "roleScheduleId" in role and "rolePromptId" in role
    assert 'dragon-caption-role-details"><summary>' in role
    assert "已复制角色 Tag" in role
    groups = _read("js/features/captioning/workspace/groups-panel.js")
    assert "groupsDirty" in groups and "groupScans" in groups
    assert "删除目录组“" in groups and "扫描中…" in groups
    tags = _read("js/features/captioning/workspace/tag-manager-panel.js")
    assert "tagDirtyName" in tags and "discardDirty" in tags
    assert "matches.length" in tags and "替换中" in tags and "tagQuery" in tags
    prompts = _read("js/features/captioning/workspace/prompts-panel.js")
    assert "promptQuery" in prompts and "promptDirtyId" in prompts
    assert "discardPromptDraft" in prompts and "内置预设只读" in prompts


def test_captioning_completion_and_dataset_preserve_running_drafts() -> None:
    completion = _read("js/features/captioning/workspace/completion-panel.js")
    dataset = _read("js/features/captioning/workspace/dataset-panel.js")

    assert "completionDraft" in completion
    assert "if (state.workspaceData.completionRunning) return" in completion
    assert "data-completion-scan ${running ? 'disabled'" in completion
    assert "data-completion-retry" in completion and "停止后续任务" in completion
    assert "datasetDraft" in dataset
    assert "if (state.workspaceData.datasetRunning || !candidates.length) return" in dataset
    assert "status === 'stopped'" in dataset or "'failed', 'stopped'" in dataset
    assert "无可用 Gemini 调度" in dataset
    assert "data-dataset-plan" in dataset and "次 API 调用" in dataset
    assert "至少选择生成图或参考图中的一项" in dataset
    assert "if (state.workspace) state.workspace.dataset_results" in dataset


def test_captioning_cache_tokens_are_consistent_for_the_embedded_suite() -> None:
    loaders = _read("js/dragon-ui/page-loaders.js")
    page = _read("js/dragon-ui/pages/captioning.js")
    styles = _read("js/dragon-ui/route-styles.js")
    assert "captioning.js?v=dragon-ui-20260829v11" in loaders
    assert "page.js?v=dragon-ui-20260829v11" in page
    assert "06c-dragon-captioning.css?v=dragon-ui-20260829v11" in styles
