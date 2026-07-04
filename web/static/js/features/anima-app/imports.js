import { createPreviewFeature } from '../preview/index.js?v=module-bootstrap-20260704-1';
import { createQueueFeature } from '../queue/index.js?v=module-bootstrap-20260704-1';
import { createHistoryDetailFeature } from '../history-detail/index.js?v=module-bootstrap-20260704-1';
import { createWeightAnalysisFeature } from '../weight-analysis/index.js?v=module-bootstrap-20260704-1';
import { createEnvironmentCheckFeature } from '../environment-check/index.js?v=module-bootstrap-20260704-1';
import { createImageTestFeature } from '../image-test/index.js?v=module-bootstrap-20260704-1';
import { createGpuPicker } from '../app-shell/gpu-picker.js?v=module-bootstrap-20260704-1';
import { createTabController } from '../app-shell/tabs.js?v=module-bootstrap-20260704-1';
import { createThemeController } from '../app-shell/theme.js?v=module-bootstrap-20260704-1';
import { createUIScaleController } from '../app-shell/ui-scale.js?v=module-bootstrap-20260704-1';
import {
    blankSamplePromptRow,
    parseSamplePromptRows,
    samplePromptsContentNeedsTextMode,
    serializeSamplePromptsEditor,
} from '../sample-prompts/model.js?v=module-bootstrap-20260704-1';
import { readTomlGroupState, writeTomlGroupState } from '../toml-manager/group-state.js?v=module-bootstrap-20260704-1';
import {
    formatSystemPercent,
    formatSystemTemperature,
    formatSystemVram,
    historySystemSummary,
} from '../history-detail/system.js?v=module-bootstrap-20260704-1';
import { formatCompactNumber, numberOrNull } from '../history-detail/ui.js?v=module-bootstrap-20260704-1';

const ctx = globalThis.ctx;

Object.assign(globalThis, {
    createPreviewFeature,
    createQueueFeature,
    createHistoryDetailFeature,
    createWeightAnalysisFeature,
    createEnvironmentCheckFeature,
    createImageTestFeature,
    createGpuPicker,
    createTabController,
    createThemeController,
    createUIScaleController,
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
    const text = String(value ?? '').trim();
    if (/^[+\-]?nan$/i.test(text)) return 'NaN';
    if (/^[+\-]?inf(?:inity)?$/i.test(text)) return text.startsWith('-') ? '-Infinity' : 'Infinity';
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(5) : '-';
};
