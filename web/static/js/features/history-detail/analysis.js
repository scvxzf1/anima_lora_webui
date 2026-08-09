import { createHistoryCurveRenderer } from './curve/index.js?v=module-bootstrap-20260809-nf4-v2';
import { createHistorySystemRenderer } from './system.js?v=module-bootstrap-20260809-nf4-v2';

export function createHistoryAnalysisRenderer({ state, deps, renderHistoryDetailContent }) {
    const system = createHistorySystemRenderer();
    const curve = createHistoryCurveRenderer({
        state,
        deps,
        renderHistoryDetailContent,
        renderHistoryDetailSystem: system.renderHistoryDetailSystem,
    });

    return {
        renderHistoryDetailAnalysis: curve.renderHistoryDetailAnalysis,
        renderHistoryDetailChart: curve.renderHistoryDetailChart,
        renderHistoryDetailSystem: system.renderHistoryDetailSystem,
    };
}
