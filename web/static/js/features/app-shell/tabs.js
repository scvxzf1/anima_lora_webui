export function createTabController({
    loadDatasetPresets,
    loadGlobalSettings,
    ensureWeightAnalysisFeature,
    resetTrainingExpandedStateOnLeave,
    resizeLiveChart,
} = {}) {
    function normalizeTopLevelTabState() {
        const activeButton = document.querySelector('.tab-btn.active');
        const activeName = activeButton?.dataset.tab || '';
        const hasUsableActiveTab =
            activeName &&
            activeName !== 'preview' &&
            document.getElementById(`tab-${activeName}`);
        const fallbackButton = document.querySelector('[data-tab="training"]') || document.querySelector('[data-tab="config"]');
        const nextButton = hasUsableActiveTab ? activeButton : fallbackButton;
        const nextName = nextButton?.dataset.tab || '';
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn === nextButton);
        });
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.toggle('active', tab.id === `tab-${nextName}`);
        });
        document.getElementById('tab-preview')?.classList.remove('active');
    }

    function setupTabs() {
        normalizeTopLevelTabState();
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const previousTab = document.querySelector('.tab-btn.active')?.dataset.tab || '';
                const nextTab = btn.dataset.tab || '';
                if (previousTab === 'training' && nextTab !== 'training') {
                    resetTrainingExpandedStateOnLeave();
                }
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('tab-' + nextTab).classList.add('active');
                if (nextTab === 'datasets') {
                    loadDatasetPresets({ manage: true });
                }
                if (nextTab === 'training') {
                    resizeLiveChart?.();
                }
                if (nextTab === 'weight-analysis') {
                    ensureWeightAnalysisFeature().loadAnalysisWeights();
                }
                if (nextTab === 'settings') {
                    loadGlobalSettings();
                }
            });
        });
    }

    return {
        normalizeTopLevelTabState,
        setupTabs,
    };
}
