import { createPreviewFeature } from '../preview/index.js?v=module-bootstrap-20260608-10';
import { createQueueFeature } from '../queue/index.js?v=module-bootstrap-20260608-10';
import { createHistoryDetailFeature } from '../history-detail/index.js?v=module-bootstrap-20260608-10';
import { createWeightAnalysisFeature } from '../weight-analysis/index.js?v=module-bootstrap-20260608-10';
import { createGpuPicker } from '../app-shell/gpu-picker.js?v=module-bootstrap-20260608-10';
import { createTabController } from '../app-shell/tabs.js?v=module-bootstrap-20260608-10';
import { createThemeController } from '../app-shell/theme.js?v=module-bootstrap-20260608-10';
import {
    blankSamplePromptRow,
    parseSamplePromptRows,
    samplePromptsContentNeedsTextMode,
    serializeSamplePromptsEditor,
} from '../sample-prompts/model.js?v=module-bootstrap-20260608-10';
import { readTomlGroupState, writeTomlGroupState } from '../toml-manager/group-state.js?v=module-bootstrap-20260608-10';
import {
    formatSystemPercent,
    formatSystemTemperature,
    formatSystemVram,
    historySystemSummary,
} from '../history-detail/system.js?v=module-bootstrap-20260608-10';
import { formatCompactNumber, numberOrNull } from '../history-detail/ui.js?v=module-bootstrap-20260608-10';

const ctx = globalThis.ctx;

Object.assign(globalThis, {
    createPreviewFeature,
    createQueueFeature,
    createHistoryDetailFeature,
    createWeightAnalysisFeature,
    createGpuPicker,
    createTabController,
    createThemeController,
    blankSamplePromptRow,
    parseSamplePromptRows,
    samplePromptsContentNeedsTextMode,
    serializeSamplePromptsEditor,
    readTomlGroupState,
    writeTomlGroupState,
    formatSystemPercent,
    formatSystemTemperature,
    formatSystemVram,
    historySystemSummary,
    formatCompactNumber,
    numberOrNull,
});

globalThis.MetricsChart = ctx.MetricsChart;
globalThis.formatLossValue = function formatLossValue(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(5) : '-';
};
