const STYLE_ROOT = '/static/css/dragon/';

const ROUTE_STYLES = Object.freeze({
    dashboard: [
        '03-dragon-dashboard.css?v=dragon-ui-20260826v121',
        '05-dragon-animations.css?v=dragon-ui-20260824v44',
        '06a-dragon-shared-dialogs.css?v=dragon-ui-20260902v78',
    ],
    config: [
        '04-dragon-config.css?v=dragon-ui-20260902-lokr-availability-v1',
        '04b-dragon-model-quick-picker.css?v=dragon-ui-20260824v2',
        '04a-dragon-training-presets.css?v=dragon-ui-20260817v84',
        '04c-dragon-config-all.css?v=dragon-ui-20260902-lokr-availability-v1',
        '04d-dragon-training-data.css?v=dragon-ui-20260825v5',
        '04e-dragon-sample-prompts.css?v=dragon-ui-20260902-sample-prompts-v2',
        '05-dragon-animations.css?v=dragon-ui-20260824v44',
        '06a-dragon-shared-dialogs.css?v=dragon-ui-20260902v78',
    ],
    dataset: [
        '05-dragon-animations.css?v=dragon-ui-20260824v44',
        '06a-dragon-shared-dialogs.css?v=dragon-ui-20260902v78',
        '06-dragon-pages.css?v=dragon-ui-20260902v91',
    ],
    live: [
        '03-dragon-dashboard.css?v=dragon-ui-20260826v121',
        '05-dragon-animations.css?v=dragon-ui-20260824v44',
        '06a-dragon-shared-dialogs.css?v=dragon-ui-20260902v78',
        '06-dragon-pages.css?v=dragon-ui-20260902v91',
        '07-dragon-training-polish.css?v=dragon-ui-20260825v2',
        '07a-dragon-live-workbench.css?v=dragon-ui-20260902-training-nav-v3',
    ],
    'history-list': [
        '03-dragon-dashboard.css?v=dragon-ui-20260826v121',
        '03a-dragon-history-workbench.css?v=dragon-ui-20260826v18',
        '05-dragon-animations.css?v=dragon-ui-20260824v44',
        '06a-dragon-shared-dialogs.css?v=dragon-ui-20260902v78',
        '07-dragon-training-polish.css?v=dragon-ui-20260825v2',
    ],
    'history-detail': [
        '03-dragon-dashboard.css?v=dragon-ui-20260826v121',
        '03a-dragon-history-workbench.css?v=dragon-ui-20260826v18',
        '03b-dragon-history-detail.css?v=dragon-ui-20260824v1',
        '05-dragon-animations.css?v=dragon-ui-20260824v44',
        '06a-dragon-shared-dialogs.css?v=dragon-ui-20260902v78',
        '06b-dragon-history-sample-dialog.css?v=dragon-ui-20260824v3',
        '06-dragon-pages.css?v=dragon-ui-20260902v91',
        '07-dragon-training-polish.css?v=dragon-ui-20260825v2',
    ],
    pages: [
        '05-dragon-animations.css?v=dragon-ui-20260824v44',
        '06a-dragon-shared-dialogs.css?v=dragon-ui-20260902v78',
        '06-dragon-pages.css?v=dragon-ui-20260902v91',
    ],
    captioning: [
        '05-dragon-animations.css?v=dragon-ui-20260824v44',
        '06a-dragon-shared-dialogs.css?v=dragon-ui-20260902v78',
        '06-dragon-pages.css?v=dragon-ui-20260902v91',
        '06c-dragon-captioning.css?v=dragon-ui-20260902v18',
    ],
});

let activeKey = '';
let activeLinks = [];
let pending = null;
let requestSequence = 0;

export async function ensureDragonRouteStyles(styleKey) {
    if (typeof document === 'undefined') return;
    if (styleKey === activeKey && activeLinks.length) return;
    if (pending?.key === styleKey) return pending.promise;
    const files = ROUTE_STYLES[styleKey] || ROUTE_STYLES.pages;
    const sequence = ++requestSequence;
    const nextLinks = [];
    const promise = Promise.all(files.map((file) => loadRouteStylesheet(file, nextLinks)))
        .then(() => {
            if (sequence !== requestSequence) {
                nextLinks.forEach((link) => link.remove());
                return;
            }
            activeLinks.forEach((link) => link.remove());
            activeLinks = nextLinks;
            activeKey = styleKey;
        })
        .catch((error) => {
            nextLinks.forEach((link) => link.remove());
            throw error;
        })
        .finally(() => {
            if (pending?.sequence === sequence) pending = null;
        });
    pending = { key: styleKey, sequence, promise };
    return promise;
}

export function clearDragonRouteStyles() {
    requestSequence += 1;
    pending = null;
    activeKey = '';
    activeLinks = [];
    document.querySelectorAll('link[data-dragon-route-style]').forEach((link) => link.remove());
}

function loadRouteStylesheet(file, links) {
    return new Promise((resolve, reject) => {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = `${STYLE_ROOT}${file}`;
        link.dataset.dragonRouteStyle = 'true';
        link.addEventListener('load', resolve, { once: true });
        link.addEventListener('error', () => reject(new Error(`Dragon route stylesheet failed: ${file}`)), { once: true });
        links.push(link);
        document.head.appendChild(link);
    });
}
