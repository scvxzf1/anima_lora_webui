/**
 * Dataset experimental/advanced dialog for the currently selected subset.
 */
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir2';
import { normalizeDatasetDefaults } from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260711-ir2';
import { createDatasetExperimentalFeaturesEditor } from './row.js?v=module-bootstrap-20260711-ir2';

const datasetState = getDatasetState();

export function openDatasetExperimentalDialog(index) {
    const dialog = document.getElementById('dataset-experimental-dialog');
    const body = document.getElementById('dataset-experimental-dialog-body');
    const title = document.getElementById('dataset-experimental-dialog-title');
    if (!dialog || !body) return;

    const rows = datasetState.datasetPresetState?.datasets
        || datasetState.datasetEditorState?.datasets
        || [];
    if (!rows.length) return;

    index = Math.max(0, Math.min(Number(index) || 0, rows.length - 1));
    datasetState.selectedDatasetIndex = index;
    const row = rows[index];
    const settings = normalizeDatasetDefaults(row.settings || {});
    if (title) {
        title.textContent = `实验性 · SUBSET ${index + 1} · ${settings.resolution || '?'}px`;
    }

    body.innerHTML = '';
    // Reuse existing advanced editor; force open state for dialog context.
    const editor = createDatasetExperimentalFeaturesEditor(row, index);
    editor.open = true;
    editor.classList.add('is-dialog-embedded');
    body.appendChild(editor);

    if (dialog.showModal && !dialog.open) dialog.showModal();
    else if (!dialog.open) dialog.setAttribute('open', 'open');
}
