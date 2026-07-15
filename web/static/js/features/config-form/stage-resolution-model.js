/**
 * Stage schedule model and pure helpers.
 */
import { getAppContext } from '../anima-app/helpers/app-context-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';

const ctx = getAppContext();
const configState = getConfigState();
const stageResolutionState = configState.stageResolutionState;

export const STAGE_COLORS = ['#5B8DEF', '#2DD4BF', '#A78BFA', '#F59E0B', '#F472B6', '#34D399'];

export function clamp01(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(1, n));
}

export function toFraction(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return n > 1 ? clamp01(n / 100) : clamp01(n);
}

export function pctLabel(fraction) {
    return `${Math.round(clamp01(fraction) * 1000) / 10}%`;
}

export function readTotalSteps() {
    const draft = configState.configFormState?.draftValues;
    const current = configState.currentConfig || {};
    const raw = draft?.has?.('max_train_steps')
        ? draft.get('max_train_steps')
        : current.max_train_steps;
    const steps = Math.round(Number(raw) || 0);
    return steps > 0 ? steps : 0;
}

export function activeStageScheduleDatasetState(datasetState = getDatasetState()) {
    const datasetTabActive = Boolean(
        typeof document !== 'undefined'
        && document.getElementById('tab-datasets')?.classList.contains('active'),
    );
    return datasetTabActive
        ? (datasetState.datasetPresetState || {})
        : (datasetState.datasetEditorState || {});
}

export function pickDatasetRows(datasetState) {
    const active = activeStageScheduleDatasetState(datasetState);
    if (Array.isArray(active.datasets) && active.datasets.length) return active.datasets;
    // During initial rendering the active state may not be hydrated yet. Fall back
    // to the other panel, but never let an empty array mask a populated one.
    const candidates = [
        datasetState.datasetPresetState?.datasets,
        datasetState.datasetEditorState?.datasets,
    ];
    for (const rows of candidates) {
        if (Array.isArray(rows) && rows.length) return rows;
    }
    return [];
}

export function listSubsetOptions() {
    const datasetState = getDatasetState();
    const rows = pickDatasetRows(datasetState);
    if (!Array.isArray(rows) || !rows.length) {
        return [{ index: 0, label: 'SUBSET 1（当前数据集）', resolution: 1024 }];
    }
    return rows.map((row, index) => {
        const settings = row?.settings || {};
        const resolution = Number(settings.resolution) || 1024;
        const path = String(row?.source_dir || row?.image_dir || row?.path || '').trim();
        const short = path ? path.split(/[\\/]/).filter(Boolean).slice(-2).join('/') : `子集 ${index + 1}`;
        return {
            index,
            label: `SUBSET ${index + 1} · ${resolution}px · ${short || '未命名'}`,
            resolution,
        };
    });
}

export function defaultStageScheduleStages(options = listSubsetOptions()) {
    const subsetCount = Math.max(1, Array.isArray(options) ? options.length : 0);
    return [
        { name: '阶段1', subset_index: 0, start_pct: 0, end_pct: 0.5 },
        { name: '阶段2', subset_index: Math.min(1, subsetCount - 1), start_pct: 0.5, end_pct: 1 },
    ];
}

export function resolveStageScheduleSource(config = configState.currentConfig || {}) {
    const draft = configState.configFormState?.draftValues;
    const hasDraft = (key) => Boolean(draft?.has?.(key));
    if (hasDraft('stage_schedule_enabled') || hasDraft('stage_schedule')) {
        return {
            stage_schedule_enabled: hasDraft('stage_schedule_enabled')
                ? draft.get('stage_schedule_enabled')
                : config.stage_schedule_enabled,
            stage_schedule: hasDraft('stage_schedule')
                ? draft.get('stage_schedule')
                : config.stage_schedule,
        };
    }
    // The active dataset configuration owns stage_schedule.
    const datasetState = getDatasetState();
    const activeDataset = activeStageScheduleDatasetState(datasetState);
    if (
        activeDataset.selectedFile
        || activeDataset.dataset_config
        || Array.isArray(activeDataset.stage_schedule)
        || activeDataset.stage_schedule_enabled != null
    ) {
        return {
            stage_schedule_enabled: activeDataset.stage_schedule_enabled,
            stage_schedule: activeDataset.stage_schedule,
        };
    }
    return {
        stage_schedule_enabled: config.stage_schedule_enabled,
        stage_schedule: config.stage_schedule,
    };
}

export function hydrateStageScheduleFromConfig(config = {}) {
    const source = resolveStageScheduleSource(config);
    const enabled = Boolean(source.stage_schedule_enabled);
    let stages = normalizeRawStages(source.stage_schedule);
    if (!stages.length) stages = defaultStageScheduleStages();
    stageResolutionState.enabled = enabled;
    stageResolutionState.stages = stages;
    stageResolutionState.selectedIndex = 0;
    stageResolutionState._hydratedFromConfig = true;
}

export function hydrateStageScheduleFromDatasetPreset(preset = {}, options = {}) {
    const enabled = Boolean(preset?.stage_schedule_enabled);
    let stages = normalizeRawStages(preset?.stage_schedule);
    if (!stages.length) stages = defaultStageScheduleStages();
    stageResolutionState.enabled = enabled;
    stageResolutionState.stages = stages;
    stageResolutionState.selectedIndex = 0;
    stageResolutionState._hydratedFromConfig = true;
    // Mirror into currentConfig for summaries; only touch draft when requested.
    if (configState.currentConfig && typeof configState.currentConfig === 'object') {
        configState.currentConfig.stage_schedule_enabled = enabled;
        configState.currentConfig.stage_schedule = stages;
    }
    if (options.touchDraft) {
        const draft = configState.configFormState?.draftValues;
        if (draft && typeof draft.set === 'function') {
            draft.set('stage_schedule_enabled', enabled);
            draft.set('stage_schedule', stages);
        }
    }
}

export function normalizeRawStages(raw) {
    if (!Array.isArray(raw)) return [];
    return raw.map((stage, index) => ({
        name: String(stage?.name || `阶段${index + 1}`).trim() || `阶段${index + 1}`,
        subset_index: Math.max(0, Math.round(Number(stage?.subset_index ?? stage?.subsetIndex ?? index) || 0)),
        start_pct: toFraction(stage?.start_pct ?? stage?.startPct ?? 0),
        end_pct: toFraction(stage?.end_pct ?? stage?.endPct ?? 1),
    }));
}

export function normalizedStageResolutionStages() {
    if (!Array.isArray(stageResolutionState.stages) || !stageResolutionState.stages.length) {
        stageResolutionState.stages = defaultStageScheduleStages();
    }
    // Keep order; re-normalize pct and names only.
    stageResolutionState.stages = stageResolutionState.stages.map((stage, index) => ({
        name: String(stage.name || `阶段${index + 1}`).trim() || `阶段${index + 1}`,
        subset_index: Math.max(0, Math.round(Number(stage.subset_index) || 0)),
        start_pct: toFraction(stage.start_pct),
        end_pct: toFraction(stage.end_pct),
    }));
    // Snap cover: first start 0, last end 1, adjacent seams.
    if (stageResolutionState.stages.length) {
        stageResolutionState.stages[0].start_pct = 0;
        stageResolutionState.stages[stageResolutionState.stages.length - 1].end_pct = 1;
        for (let i = 1; i < stageResolutionState.stages.length; i += 1) {
            const prev = stageResolutionState.stages[i - 1];
            const cur = stageResolutionState.stages[i];
            // Prefer previous end as seam when close; otherwise leave and flag in metrics.
            if (Math.abs(prev.end_pct - cur.start_pct) < 1e-6) {
                cur.start_pct = prev.end_pct;
            }
        }
    }
    stageResolutionState.selectedIndex = Math.max(
        0,
        Math.min(stageResolutionState.selectedIndex || 0, stageResolutionState.stages.length - 1),
    );
    return stageResolutionState.stages;
}

export function stageSchedulePayload() {
    const stages = normalizedStageResolutionStages().map((stage) => ({
        name: stage.name,
        subset_index: stage.subset_index,
        start_pct: stage.start_pct,
        end_pct: stage.end_pct,
    }));
    return {
        stage_schedule_enabled: Boolean(stageResolutionState.enabled),
        stage_schedule: stages,
    };
}

export function stageResolutionMetrics() {
    stageResolutionState.enabled = Boolean(stageResolutionState.enabled);
    const stages = normalizedStageResolutionStages();
    const totalSteps = readTotalSteps();
    const options = listSubsetOptions();
    const optionByIndex = new Map(options.map((item) => [item.index, item]));
    const ranges = stages.map((stage, index) => {
        const problems = [];
        const warnings = [];
        if (!(stage.end_pct > stage.start_pct + 1e-9)) problems.push('区间为空');
        if (!optionByIndex.has(stage.subset_index) && options.length) {
            problems.push('子集索引超出当前数据集');
        }
        if (index > 0) {
            const prev = stages[index - 1];
            if (Math.abs(prev.end_pct - stage.start_pct) > 1e-6) {
                if (stage.start_pct < prev.end_pct - 1e-6) problems.push('与上一段重叠');
                else problems.push('与上一段未贴齐');
            }
        }
        if (index === 0 && Math.abs(stage.start_pct) > 1e-6) problems.push('须从 0% 开始');
        if (index === stages.length - 1 && Math.abs(stage.end_pct - 1) > 1e-6) problems.push('须到 100%');
        const opt = optionByIndex.get(stage.subset_index);
        const startStep = totalSteps ? Math.floor(totalSteps * stage.start_pct) : null;
        const endStep = totalSteps ? Math.floor(totalSteps * stage.end_pct) : null;
        return {
            ...stage,
            index,
            resolution: opt?.resolution ?? null,
            subsetLabel: opt?.label || `SUBSET ${stage.subset_index + 1}`,
            startStep,
            endStep,
            steps: startStep != null && endStep != null ? Math.max(0, endStep - startStep) : null,
            problems,
            warnings,
            color: STAGE_COLORS[index % STAGE_COLORS.length],
        };
    });
    const problemCount = ranges.filter((item) => item.problems.length).length;
    const warningCount = ranges.filter((item) => item.warnings.length).length;
    return {
        enabled: stageResolutionState.enabled,
        stages: ranges,
        totalSteps,
        problemCount,
        warningCount,
        selected: ranges[stageResolutionState.selectedIndex] || ranges[0],
        subsetOptions: options,
    };
}

export function stageResolutionStatus(stage) {
    if (stage.problems?.length) return { tone: 'error', text: stage.problems[0] };
    if (stage.warnings?.length) return { tone: 'warning', text: stage.warnings[0] };
    return { tone: 'ok', text: '就绪' };
}
