const TEXT_SELECTORS = Object.freeze({
    state: '[data-live-state-text]',
    headerTitle: '[data-live-header-title]',
    headerMeta: '[data-live-header-meta]',
    errorMessage: '[data-live-error-message]',
    progressText: '[data-live-progress-text]',
    stepText: '[data-live-step-text]',
    epochText: '[data-live-epoch-text]',
    lossValue: '[data-live-metric="live-loss"] strong',
    lossDetail: '[data-live-metric="live-loss"] small',
    lrValue: '[data-live-metric="live-lr"] strong',
    rateValue: '[data-live-metric="live-rate"] strong',
    eta: '[data-live-metric-detail="live-eta"]',
    vramValue: '[data-live-metric="live-vram"] strong',
    vramDetail: '[data-live-metric="live-vram"] small',
    gpuValue: '[data-live-metric="live-gpu-util"] strong',
    gpuDetail: '[data-live-metric="live-gpu-util"] small',
    temperatureValue: '[data-live-metric="live-gpu-temp"] strong',
    temperatureDetail: '[data-live-metric="live-gpu-temp"] small',
    chartCount: '[data-live-chart-count]',
    chartSmoothing: '[data-live-chart-smoothing-value]',
    connectionLabel: '[data-live-connection-label]',
    connectionDetail: '[data-live-connection-detail]',
});

export function createLiveDomBindings(root) {
    const query = (selector) => root.querySelector(selector);
    return {
        root,
        sections: [...root.querySelectorAll('[data-live-section]')],
        text: Object.fromEntries(Object.entries(TEXT_SELECTORS).map(([key, selector]) => [key, query(selector)])),
        sidebar: query('[data-live-sidebar-body]'),
        chart: query('[data-live-chart]'),
        stateBadge: query('[data-live-state]'),
        stateDot: query('[data-live-state] .dragon-nav-status-dot'),
        progressFill: query('[data-live-progress-fill]'),
        progress: query('[data-live-progress]'),
        stop: query('[data-tool-action="stop"]'),
        connection: query('[data-live-connection]'),
        temperatureCard: query('[data-live-metric="live-gpu-temp"]'),
        meters: {
            vram: query('[data-live-metric="live-vram"] .dragon-live-meter i'),
            gpu: query('[data-live-metric="live-gpu-util"] .dragon-live-meter i'),
            temperature: query('[data-live-metric="live-gpu-temp"] .dragon-live-meter i'),
        },
    };
}

export function setLiveText(dom, key, value) {
    const node = dom.text[key];
    const next = String(value ?? '');
    if (node && node.textContent !== next) node.textContent = next;
}

export function setLiveWidth(node, value) {
    if (!node) return;
    const next = `${Math.max(0, Math.min(100, Number(value) || 0))}%`;
    if (node.style.width !== next) node.style.width = next;
}

export function setLiveDataset(node, key, value) {
    if (!node) return;
    const next = String(value ?? '');
    if (node.dataset[key] !== next) node.dataset[key] = next;
}

export function setLiveProperty(node, key, value) {
    if (node && node[key] !== value) node[key] = value;
}

export function setLiveAttribute(node, key, value) {
    if (!node) return;
    const next = String(value ?? '');
    if (node.getAttribute(key) !== next) node.setAttribute(key, next);
}
