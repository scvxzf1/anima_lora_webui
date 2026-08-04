/**
 * Anima LoRA Web UI — ES module bootstrap.
 */
import { MetricsChart } from './chart.js?v=module-bootstrap-20260714-stage-dataset5';
import { createCatalog } from './js/config/catalog.js?v=module-bootstrap-20260714-stage-dataset5';
import { createAnimaApp } from './js/features/anima-app/index.js?v=module-bootstrap-20260804-multigpu1';
import { createApiClient } from './js/shared/api.js?v=module-bootstrap-20260714-stage-dataset5';
import * as dom from './js/shared/dom.js?v=module-bootstrap-20260714-stage-dataset5';
import * as download from './js/shared/download.js?v=module-bootstrap-20260714-stage-dataset5';
import * as format from './js/shared/format.js?v=module-bootstrap-20260714-stage-dataset5';
import { createAppContext } from './js/state/create-app-context.js?v=module-bootstrap-20260714-stage-dataset5';

const ctx = createAppContext({
    api: createApiClient(),
    catalog: createCatalog(),
    dom,
    download,
    format,
    MetricsChart,
});

createAnimaApp(ctx).catch((error) => {
    globalThis.__animaBootstrapError = error;
    console.error('[webui-bootstrap] failed to start Anima app', error);
    document.documentElement.dataset.appBoot = 'error';
    const status = dom.optionalById('status-indicator');
    const text = dom.optionalById('status-text');
    status?.classList.add('error');
    if (text) text.textContent = '启动失败，请查看控制台日志';
});
