from __future__ import annotations

from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "web" / "static" / "app.js"
INDEX_HTML = Path(__file__).resolve().parents[1] / "web" / "static" / "index.html"


def _section(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_new_training_launch_enters_live_monitoring() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    helper = _section(source, "function enterLiveTrainingForNewRun()", "function showPreflightDialog")

    assert "returnToLiveTraining({ refresh: false });" in helper
    assert 'document.querySelector(\'[data-tab="training"]\')?.click();' in helper
    assert "pollStatus();" in helper
    assert "replayTrainingLogs();" in helper

    start_path = _section(source, "async function startTrainingUnchecked", "function enterLiveTrainingForNewRun")
    preprocess_path = _section(source, "async function startPreprocessFromPreflight", "function currentTrainingConfigFile")

    assert "enterLiveTrainingForNewRun();" in start_path
    assert "enterLiveTrainingForNewRun();" in preprocess_path
    assert 'document.querySelector(\'[data-tab="training"]\').click();' not in start_path
    assert 'document.querySelector(\'[data-tab="training"]\').click();' not in preprocess_path


def test_return_to_live_training_clears_runtime_cursor() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    body = _section(source, "function returnToLiveTraining", "async function loadResumeOptionsForTask")

    for snippet in (
        "viewingHistoryTaskId = '';",
        "historyViewMode = 'live';",
        "trainingRuntime.lastLogId = 0;",
        "trainingRuntime.logLineCount = 0;",
        "stepCounter = 0;",
        "lossChart?.clear();",
    ):
        assert snippet in body


def test_training_queue_frontend_hooks_are_present() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")

    queue_section = _section(source, "async function loadTrainingQueue()", "// ── 状态轮询 ──")
    assert "function renderTrainingQueue()" in queue_section
    assert "function renderTrainingQueueManager()" in queue_section
    assert "function createTrainingQueueItem" in queue_section
    assert "function createTrainingQueueManagerItem" in queue_section
    assert "async function toggleTrainingQueuePause()" in queue_section
    assert "retryQueueItem" in queue_section
    assert "cancelWaitingQueueItems" in queue_section
    assert "clearFinishedQueueItems" in queue_section
    assert "trainingQueueFilter" in queue_section
    assert "/api/training/queue" in queue_section
    assert "/api/training/queue/settings" in queue_section
    assert "/api/training/queue/cancel-waiting" in queue_section
    assert "/api/training/queue/clear" in queue_section

    assert "training-queue-manager" in html
    assert "btn-training-queue-view" in html
    assert "training-queue-failure-policy" in html

    ws_section = _section(source, "function handleWsMessage", "function appendLog")
    assert "case 'queue':" in ws_section
    assert "updateTrainingQueueFromPayload(msg);" in ws_section

    start_section = _section(source, "async function startTrainingUnchecked", "function enterLiveTrainingForNewRun")
    assert "enqueueTrainingFromConfig" in start_section
    assert "chooseTrainingLaunchMode" in source


def test_resume_queue_button_is_wired() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    resume_section = _section(source, "function renderResumePanelState", "function optionNode")
    assert "btn-queue-resume-training" in resume_section
    assert "queueBtn.disabled" in resume_section

    listener_section = _section(source, "function setupEventListeners", "function installBeginnerTooltips")
    assert "queueResumeTrainingFromCheckpoint" in source
    assert "btn-queue-resume-training" in listener_section


def test_history_list_marks_queue_tasks() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    queue_label = _section(source, "function historyQueueLabel", "function historyContinueLabel")
    task_item = _section(source, "function createHistoryTaskItem", "function createHistoryActionButton")

    assert "来自队列" in queue_label
    assert "queue_attempt" in queue_label
    assert "historyQueueLabel(task)" in task_item


def test_output_scope_group_opens_undefined_dialog() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")

    section = _section(source, "title: '输出格式与训练范围'", "title: '方法内部与实验架构'")
    create_group = _section(source, "function createGroup", "function createOpenUndefinedDialogButton")
    button_factory = _section(source, "function createOpenUndefinedDialogButton", "function createFillGlobalModelPathsButton")

    assert "className: 'config-group-output-scope'" in section
    assert "if (extraClass === 'config-group-output-scope')" in create_group
    assert source.count("header.appendChild(createOpenUndefinedDialogButton());") == 1

    assert "btn-open-undefined-dialog" in button_factory
    assert "btn.textContent = '未定义';" in button_factory
    assert "btn.addEventListener('click', openUndefinedDialog);" in button_factory
    assert "function openUndefinedDialog()" in button_factory
    assert "undefined-dialog" in button_factory
    assert "showModal" in button_factory

    assert 'id="undefined-dialog"' in html
    assert 'class="preview-dialog undefined-dialog"' in html
    assert "<h2>未定义</h2>" in html
    assert "undefined-dialog-body" in html
