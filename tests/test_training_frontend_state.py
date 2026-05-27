from __future__ import annotations

from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "web" / "static" / "app.js"


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

    queue_section = _section(source, "async function loadTrainingQueue()", "// ── 状态轮询 ──")
    assert "function renderTrainingQueue()" in queue_section
    assert "function createTrainingQueueItem" in queue_section
    assert "async function toggleTrainingQueuePause()" in queue_section
    assert "/api/training/queue" in queue_section

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
