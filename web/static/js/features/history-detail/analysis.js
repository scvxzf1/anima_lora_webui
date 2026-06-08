import { createHistoryCurveRenderer } from './curve/index.js?v=module-bootstrap-20260608-3';
import { createHistorySystemRenderer } from './system.js?v=module-bootstrap-20260608-3';

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
