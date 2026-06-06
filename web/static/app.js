/**
 * Anima LoRA Web UI — ES module bootstrap.
 */
import { MetricsChart } from './chart.js?v=module-bootstrap-20260604-10';
import { createCatalog } from './js/config/catalog.js?v=module-bootstrap-20260604-10';
import { createLegacyApp } from './js/features/legacy-app.js?v=module-bootstrap-20260604-10';
import { createApiClient } from './js/shared/api.js?v=module-bootstrap-20260604-10';
import * as dom from './js/shared/dom.js?v=module-bootstrap-20260604-10';
import * as download from './js/shared/download.js?v=module-bootstrap-20260604-10';
import * as format from './js/shared/format.js?v=module-bootstrap-20260604-10';
import { createAppContext } from './js/state/create-app-context.js?v=module-bootstrap-20260604-10';

const ctx = createAppContext({
    api: createApiClient(),
    catalog: createCatalog(),
    dom,
    download,
    format,
    MetricsChart,
});

createLegacyApp(ctx);
