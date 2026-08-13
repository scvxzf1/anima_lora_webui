/* Dragon weight analysis controller: list, upload, compare, and export. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { downloadText } from '../../shared/download.js?v=dragon-ui-20260812v35';
import {
    renderAnalysisBundle,
    renderAnalysisEmpty,
    renderAnalysisError,
    renderWeightAnalysisPage,
    renderWeightOptions,
} from './weight-analysis-view.js?v=dragon-ui-20260814v43';

const api = createApiClient();

export async function loadWeightAnalysis() {
    const listing = await loadWeightListing();
    const model = normalizeListing(listing);
    const state = {
        ...model,
        compare: false,
        files: { primary: null, secondary: null },
        primaryResult: null,
        secondaryResult: null,
        requestSeq: 0,
        printTimer: null,
        originalTitle: '',
    };
    return {
        html: renderWeightAnalysisPage(model),
        onMount(root) {
            const page = root.querySelector('[data-weight-root]') || root;
            bindWeightAnalysis(page, state);
        },
        onUnmount() { cleanupPrintState(state); },
    };
}

function bindWeightAnalysis(root, state) {
    const page = root;
    bindSourceSlot(root, state, 'primary');
    bindSourceSlot(root, state, 'secondary');
    root.querySelector('[data-weight-action="run"]')?.addEventListener('click', () => runAnalysis(root, state));
    root.querySelector('[data-weight-action="toggle-compare"]')?.addEventListener('click', () => toggleCompare(root, state));
    page.querySelector('[data-tool-action="refresh-weights"]')?.addEventListener('click', () => refreshListing(root, state));
    page.querySelector('[data-tool-action="export-json"]')?.addEventListener('click', () => exportJson(root, state));
    page.querySelector('[data-tool-action="export-pdf"]')?.addEventListener('click', () => exportPrint(root, state));
}

function bindSourceSlot(root, state, slot) {
    const select = root.querySelector(`[data-weight-select="${slot}"]`);
    const path = root.querySelector(`[data-weight-path="${slot}"]`);
    const input = root.querySelector(`[data-weight-file="${slot}"]`);
    const dropzone = root.querySelector(`[data-weight-dropzone="${slot}"]`);
    select?.addEventListener('change', () => {
        if (path) path.value = select.value;
        clearFile(root, state, slot);
    });
    path?.addEventListener('input', () => {
        if (select && select.value !== path.value) select.value = '';
        clearFile(root, state, slot);
    });
    path?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            runAnalysis(root, state);
        }
    });
    input?.addEventListener('change', () => {
        const file = input.files?.[0];
        if (file) setFile(root, state, slot, file);
        input.value = '';
    });
    dropzone?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            input?.click();
        }
    });
    for (const type of ['dragenter', 'dragover']) {
        dropzone?.addEventListener(type, (event) => {
            event.preventDefault();
            dropzone.dataset.dragging = 'true';
        });
    }
    for (const type of ['dragleave', 'drop']) {
        dropzone?.addEventListener(type, (event) => {
            event.preventDefault();
            delete dropzone.dataset.dragging;
        });
    }
    dropzone?.addEventListener('drop', (event) => {
        const file = Array.from(event.dataTransfer?.files || []).find(isSafetensorsFile);
        if (!file) {
            setStatus(root, '没有识别到 .safetensors 文件。', 'error');
            return;
        }
        setFile(root, state, slot, file);
    });
}

function setFile(root, state, slot, file) {
    if (!isSafetensorsFile(file)) {
        setStatus(root, '只支持 .safetensors 权重文件。', 'error');
        return;
    }
    state.files[slot] = file;
    const path = root.querySelector(`[data-weight-path="${slot}"]`);
    const select = root.querySelector(`[data-weight-select="${slot}"]`);
    const label = root.querySelector(`[data-weight-file-label="${slot}"]`);
    if (path) path.value = `uploaded://${file.name}`;
    if (select) select.value = '';
    if (label) label.textContent = `${file.name} · 临时上传`;
    setStatus(root, `已选择权重 ${slot === 'primary' ? 'A' : 'B'}：${file.name}`, 'success');
}

function clearFile(root, state, slot) {
    if (!state.files[slot]) return;
    state.files[slot] = null;
    const label = root.querySelector(`[data-weight-file-label="${slot}"]`);
    if (label) label.textContent = '本地文件仅临时上传分析';
}

function toggleCompare(root, state) {
    state.compare = !state.compare;
    root.dataset.compare = String(state.compare);
    const slot = root.querySelector('[data-weight-slot="secondary"]');
    const button = root.querySelector('[data-weight-action="toggle-compare"]');
    const runLabel = root.querySelector('[data-weight-action="run"] span');
    if (slot) slot.hidden = !state.compare;
    if (button) {
        button.setAttribute('aria-pressed', String(state.compare));
        const label = button.querySelector('span');
        if (label) label.textContent = state.compare ? '关闭 A / B 对比' : '开启 A / B 对比';
    }
    if (runLabel) runLabel.textContent = state.compare ? '分析并对比' : '分析权重';
    if (!state.compare) {
        state.secondaryResult = null;
        if (state.primaryResult) renderResults(root, state);
    }
    setStatus(root, state.compare ? '已开启 A / B 对比，请选择权重 B。' : '已关闭对比模式。', 'info');
}

async function runAnalysis(root, state) {
    const primary = selectedSource(root, state, 'primary');
    const secondary = state.compare ? selectedSource(root, state, 'secondary') : null;
    if (!primary) {
        setStatus(root, '请先选择、填写或上传主权重 A。', 'error');
        root.querySelector('[data-weight-path="primary"]')?.focus();
        return;
    }
    if (state.compare && !secondary) {
        setStatus(root, '对比模式需要选择、填写或上传权重 B。', 'error');
        root.querySelector('[data-weight-path="secondary"]')?.focus();
        return;
    }
    const requestSeq = ++state.requestSeq;
    state.primaryResult = null;
    state.secondaryResult = null;
    setExportEnabled(root, false);
    setBusy(root, true, state.compare);
    setStatus(root, state.compare ? '正在分析 A / B 并计算差异…' : '正在读取权重并重建静态 ΔW…', 'info');
    try {
        const [primaryResult, secondaryResult] = await Promise.all([
            inspectSource(primary),
            secondary ? inspectSource(secondary) : Promise.resolve(null),
        ]);
        if (requestSeq !== state.requestSeq) return;
        if (primaryResult?.ok === false) throw new Error(primaryResult.error || '权重 A 分析失败');
        if (secondaryResult?.ok === false) throw new Error(secondaryResult.error || '权重 B 分析失败');
        state.primaryResult = primaryResult;
        state.secondaryResult = secondaryResult;
        renderResults(root, state);
        setExportEnabled(root, true);
        const unsupported = primaryResult?.unsupported?.unsupported || secondaryResult?.unsupported?.unsupported;
        setStatus(root, unsupported ? '分析完成，但至少一个权重结构不支持静态 ΔW 重建。' : (state.compare ? 'A / B 权重对比完成。' : '权重分析完成。'), unsupported ? 'error' : 'success');
    } catch (error) {
        if (requestSeq !== state.requestSeq) return;
        state.primaryResult = null;
        state.secondaryResult = null;
        setExportEnabled(root, false);
        root.querySelector('[data-weight-results]').innerHTML = renderAnalysisError(error.message || '权重分析失败');
        setStatus(root, error.message || '权重分析失败', 'error');
    } finally {
        if (requestSeq === state.requestSeq) setBusy(root, false, state.compare);
    }
}

async function inspectSource(source) {
    if (!source.file) {
        return api('/api/analysis/inspect', { method: 'POST', body: JSON.stringify({ path: source.path }) });
    }
    const form = new FormData();
    form.append('file', source.file, source.file.name || 'uploaded.safetensors');
    return api('/api/analysis/inspect-upload', { method: 'POST', headers: {}, body: form });
}

async function refreshListing(root, state) {
    const button = root.querySelector('[data-tool-action="refresh-weights"]');
    if (button) button.disabled = true;
    setStatus(root, '正在重新扫描可分析权重…', 'info');
    try {
        const model = normalizeListing(await loadWeightListing());
        state.weights = model.weights;
        state.listMessage = model.listMessage;
        state.listError = model.listError;
        const options = renderWeightOptions(model.weights, model.listMessage || model.listError);
        root.querySelectorAll('[data-weight-select]').forEach((select) => { select.innerHTML = options; });
        const count = root.querySelector('[data-weight-count]');
        if (count) count.textContent = `${model.weights.length} 个权重`;
        setStatus(root, model.listError || model.listMessage || `已找到 ${model.weights.length} 个可分析权重。`, model.listError ? 'error' : 'success');
        if (!state.primaryResult) root.querySelector('[data-weight-results]').innerHTML = renderAnalysisEmpty(model.weights.length);
    } catch (error) {
        setStatus(root, error.message || '刷新权重列表失败', 'error');
    } finally {
        if (button) button.disabled = false;
    }
}

function exportJson(root, state) {
    if (!state.primaryResult) return;
    const generatedAt = new Date().toISOString();
    const report = {
        report_kind: 'weight_analysis_report',
        report_version: 1,
        generated_at: generatedAt,
        mode: state.secondaryResult ? 'compare' : 'single',
        comparison_basis: state.secondaryResult ? 'secondary_minus_primary' : null,
        primary: state.primaryResult,
        secondary: state.secondaryResult,
    };
    const base = safeFilename(state.primaryResult.file?.name || 'weight-analysis');
    downloadText(`${JSON.stringify(report, null, 2)}\n`, `dw-analysis-${base}-${fileTimestamp(generatedAt)}.json`, 'application/json;charset=utf-8');
    setStatus(root, 'JSON 分析报告已导出。', 'success');
}

function exportPrint(root, state) {
    if (!state.primaryResult) return;
    cleanupPrintState(state);
    state.originalTitle = document.title;
    document.title = `权重分析-${state.primaryResult.file?.name || 'report'}`;
    document.documentElement.classList.add('dragon-weight-print-mode');
    setStatus(root, '已打开打印对话框；选择“保存为 PDF”即可导出。', 'success');
    state.printTimer = window.setTimeout(() => {
        window.print();
        state.printTimer = window.setTimeout(() => cleanupPrintState(state), 300);
    }, 50);
}

function cleanupPrintState(state) {
    if (state.printTimer) window.clearTimeout(state.printTimer);
    state.printTimer = null;
    document.documentElement.classList.remove('dragon-weight-print-mode');
    if (state.originalTitle) document.title = state.originalTitle;
    state.originalTitle = '';
}

function selectedSource(root, state, slot) {
    if (state.files[slot]) return { file: state.files[slot] };
    const path = String(root.querySelector(`[data-weight-path="${slot}"]`)?.value || '').trim();
    return path && !path.startsWith('uploaded://') ? { path } : null;
}

function renderResults(root, state) {
    root.querySelector('[data-weight-results]').innerHTML = renderAnalysisBundle(state.primaryResult, state.secondaryResult);
}

function setBusy(root, busy, compare) {
    const button = root.querySelector('[data-weight-action="run"]');
    if (button) {
        button.disabled = busy;
        const label = button.querySelector('span');
        if (label) label.textContent = busy ? (compare ? '对比中…' : '分析中…') : (compare ? '分析并对比' : '分析权重');
    }
    root.querySelectorAll('[data-weight-dropzone]').forEach((dropzone) => { dropzone.dataset.busy = String(busy); });
}

function setExportEnabled(root, enabled) {
    for (const action of ['export-json', 'export-pdf']) {
        const button = root.querySelector(`[data-tool-action="${action}"]`);
        if (button) button.disabled = !enabled;
    }
}

function setStatus(root, message, tone) {
    const status = root.querySelector('[data-weight-status]');
    if (!status) return;
    status.textContent = message;
    status.dataset.tone = tone;
    status.classList.toggle('dragon-config-feedback-visible', Boolean(message));
}


async function loadWeightListing() {
    try { return await api('/api/analysis/weights'); }
    catch (error) { return { ok: false, error: error.message || '读取权重列表失败', weights: [] }; }
}

function normalizeListing(payload = {}) {
    return {
        weights: Array.isArray(payload.weights) ? payload.weights : [],
        listMessage: payload.message || '',
        listError: payload.ok === false ? (payload.error || '读取权重列表失败') : '',
    };
}

function isSafetensorsFile(file) { return Boolean(file && String(file.name || '').toLowerCase().endsWith('.safetensors')); }
function safeFilename(value) { return String(value || 'weight-analysis').replace(/\.safetensors$/i, '').replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || 'weight-analysis'; }
function fileTimestamp(value) { const date = new Date(value); const pad = (number) => String(number).padStart(2, '0'); return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`; }
