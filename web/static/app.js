/**
 * Anima LoRA Web UI — ES module bootstrap.
 */
import { MetricsChart } from './chart.js?v=module-bootstrap-20260627-3';
import { createCatalog } from './js/config/catalog.js?v=module-bootstrap-20260627-3';
import { createAnimaApp } from './js/features/anima-app/index.js?v=module-bootstrap-20260627-3';
import { createApiClient } from './js/shared/api.js?v=module-bootstrap-20260627-3';
import * as dom from './js/shared/dom.js?v=module-bootstrap-20260627-3';
import * as download from './js/shared/download.js?v=module-bootstrap-20260627-3';
import * as format from './js/shared/format.js?v=module-bootstrap-20260627-3';
import { createAppContext } from './js/state/create-app-context.js?v=module-bootstrap-20260627-3';

const ctx = createAppContext({
    api: createApiClient(),
    catalog: createCatalog(),
    dom,
    download,
    format,
    MetricsChart,
});

createAnimaApp(ctx);
