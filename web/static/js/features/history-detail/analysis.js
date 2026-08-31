import { createHistoryCurveRenderer } from './curve/index.js?v=module-bootstrap-20260831-release-v1';
import { createHistorySystemRenderer } from './system.js?v=module-bootstrap-20260831-release-v1';

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
