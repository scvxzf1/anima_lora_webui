import { fetchAnalysisWeights, inspectAnalysisWeight, inspectAnalysisWeightFile } from './api.js?v=module-bootstrap-20260604-10';
import { createWeightAnalysisRenderer } from './render.js?v=module-bootstrap-20260604-10';
import { createWeightAnalysisState } from './state.js?v=module-bootstrap-20260604-10';

export function createWeightAnalysisFeature(ctx) {
    const state = createWeightAnalysisState();
    const renderer = createWeightAnalysisRenderer({ ctx, state });

    function bindWeightAnalysisEvents() {
        document.getElementById('btn-refresh-analysis-weights')?.addEventListener('click', () => loadAnalysisWeights({ force: true }));
        document.getElementById('btn-run-weight-analysis')?.addEventListener('click', runWeightAnalysis);
        document.getElementById('btn-toggle-weight-compare')?.addEventListener('click', toggleCompareMode);
        document.getElementById('btn-export-weight-analysis')?.addEventListener('click', exportWeightAnalysisReport);
        document.getElementById('btn-weight-analysis-toggle-candidates')?.addEventListener('click', renderer.toggleCandidateExpanded);
        document.querySelectorAll('.weight-analysis-candidate-tab').forEach((button) => {
            button.addEventListener('click', () => renderer.showCandidateKind(button.dataset.candidateKind || 'style'));
        });
        document.getElementById('weight-analysis-select')?.addEventListener('change', syncSelectedWeightPath);
        document.getElementById('weight-analysis-path')?.addEventListener('input', (event) => {
            state.selectedPath = event.target.value || '';
            state.primaryFile = null;
        });
        document.getElementById('weight-analysis-path')?.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') runWeightAnalysis();
        });
        document.getElementById('weight-analysis-compare-path')?.addEventListener('input', (event) => {
            state.comparePath = event.target.value || '';
            state.compareFile = null;
        });
        document.getElementById('weight-analysis-compare-path')?.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') runWeightAnalysis();
        });
        bindDropzoneEvents();
        renderer.renderEmpty();
    }

    async function loadAnalysisWeights(options = {}) {
        if (location.protocol === 'file:') {
            renderer.renderWeightOptions({ ok: true, weights: [], message: '静态打开没有后端 API。' });
            setAnalysisStatus('静态打开没有后端 API，无法扫描训练权重。', 'error');
            return;
        }
        if (state.loadingWeights && !options.force) return;
        state.loadingWeights = true;
        setAnalysisStatus('正在扫描训练输出权重...', '');
        try {
            const payload = await fetchAnalysisWeights(ctx);
            renderer.renderWeightOptions(payload);
            if (payload.ok === false) {
                setAnalysisStatus(payload.error || '读取权重列表失败', 'error');
            } else {
                setAnalysisStatus(payload.message || `已读取 ${payload.count || 0} 个可分析权重。`, 'ok');
            }
        } catch (e) {
            renderer.renderWeightOptions({ ok: false, weights: [], message: e.message });
            setAnalysisStatus('读取权重列表失败: ' + e.message, 'error');
        } finally {
            state.loadingWeights = false;
        }
    }

    async function runWeightAnalysis() {
        const primarySource = currentPrimarySource();
        if (!primarySource) {
            setAnalysisStatus('请先选择、填写或拖入一个 .safetensors 权重。', 'error');
            return;
        }
        if (location.protocol === 'file:') {
            setAnalysisStatus('静态打开没有后端 API，无法分析权重。', 'error');
            return;
        }
        if (state.compareEnabled) {
            await runWeightComparison(primarySource);
            return;
        }
        const requestSeq = ++state.requestSeq;
        state.analyzing = true;
        setButtonBusy(true);
        setAnalysisStatus('正在 CPU 读取 safetensors 并重建静态 ΔW...', '');
        try {
            const payload = await inspectSource(primarySource);
            if (requestSeq !== state.requestSeq) return;
            handleAnalysisPayload(payload, { uploaded: Boolean(primarySource.file) });
        } catch (e) {
            if (requestSeq !== state.requestSeq) return;
            renderer.renderError(e.message);
            setAnalysisStatus('分析失败: ' + e.message, 'error');
        } finally {
            if (requestSeq === state.requestSeq) {
                state.analyzing = false;
                setButtonBusy(false);
            }
        }
    }

    async function runWeightComparison(primarySource) {
        const secondarySource = currentCompareSource();
        if (!secondarySource) {
            setAnalysisStatus('对比模式需要再填写或拖入第二个 .safetensors 权重。', 'error');
            return;
        }
        const requestSeq = ++state.requestSeq;
        state.analyzing = true;
        setButtonBusy(true);
        setDropzoneBusy(true, 'weight-analysis-compare-dropzone');
        setAnalysisStatus('正在分别分析 A / B 权重并计算 B - A 差值...', '');
        try {
            const [primaryPayload, secondaryPayload] = await Promise.all([
                inspectSource(primarySource),
                inspectSource(secondarySource),
            ]);
            if (requestSeq !== state.requestSeq) return;
            if (primaryPayload.ok === false) {
                renderer.renderError(primaryPayload.error || 'A 权重分析失败');
                setAnalysisStatus(primaryPayload.error || 'A 权重分析失败', 'error');
                return;
            }
            if (secondaryPayload.ok === false) {
                renderer.renderError(secondaryPayload.error || 'B 权重分析失败');
                setAnalysisStatus(secondaryPayload.error || 'B 权重分析失败', 'error');
                return;
            }
            renderer.renderResult(primaryPayload);
            renderer.renderComparison(primaryPayload, secondaryPayload);
            const unsupported = primaryPayload.unsupported?.unsupported || secondaryPayload.unsupported?.unsupported;
            setAnalysisStatus(
                unsupported ? '对比需要两个第一版支持的 LoRA / LoHa / LoKr 权重。' : '对比完成：差值为 B - A，只代表静态权重能量差异。',
                unsupported ? 'error' : 'ok',
            );
        } catch (e) {
            if (requestSeq !== state.requestSeq) return;
            renderer.renderError(e.message);
            setAnalysisStatus('对比失败: ' + e.message, 'error');
        } finally {
            if (requestSeq === state.requestSeq) {
                state.analyzing = false;
                setButtonBusy(false);
                setDropzoneBusy(false, 'weight-analysis-compare-dropzone');
            }
        }
    }

    function inspectSource(source) {
        if (source.file) return inspectAnalysisWeightFile(ctx, source.file);
        return inspectAnalysisWeight(ctx, source.path);
    }

    function bindDropzoneEvents() {
        bindSingleDropzone({
            dropzoneId: 'weight-analysis-dropzone',
            fileInputId: 'weight-analysis-file',
            onPath: (path) => {
                setPathInputValue(path);
                runWeightAnalysis();
            },
            onFile: analyzeDroppedWeightFile,
        });
        bindSingleDropzone({
            dropzoneId: 'weight-analysis-compare-dropzone',
            fileInputId: 'weight-analysis-compare-file',
            onPath: (path) => {
                setComparePathValue(path);
                if (state.compareEnabled && currentPrimarySource()) runWeightAnalysis();
            },
            onFile: (file) => {
                setCompareFile(file);
                if (state.compareEnabled && currentPrimarySource()) runWeightAnalysis();
            },
        });
    }

    function bindSingleDropzone({ dropzoneId, fileInputId, onPath, onFile }) {
        const dropzone = document.getElementById(dropzoneId);
        const fileInput = document.getElementById(fileInputId);
        if (!dropzone) return;
        const prevent = (event) => {
            event.preventDefault();
            event.stopPropagation();
        };
        ['dragenter', 'dragover'].forEach((type) => {
            dropzone.addEventListener(type, (event) => {
                prevent(event);
                dropzone.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach((type) => {
            dropzone.addEventListener(type, (event) => {
                prevent(event);
                if (type === 'dragleave') dropzone.classList.remove('dragover');
            });
        });
        dropzone.addEventListener('drop', (event) => handleWeightDrop(event, { dropzone, onPath, onFile }));
        dropzone.addEventListener('click', () => fileInput?.click());
        dropzone.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                fileInput?.click();
            }
        });
        fileInput?.addEventListener('change', (event) => {
            const file = event.target.files?.[0];
            if (file) onFile(file);
            event.target.value = '';
        });
    }

    function handleWeightDrop(event, { dropzone, onPath, onFile }) {
        dropzone?.classList.remove('dragover');
        const file = firstSafetensorsFile(event.dataTransfer?.files);
        if (file) {
            onFile(file);
            return;
        }
        const droppedPath = extractDroppedPath(event.dataTransfer);
        if (!droppedPath) {
            setAnalysisStatus('没有识别到 .safetensors 文件或路径。', 'error');
            return;
        }
        onPath(droppedPath);
    }

    async function analyzeDroppedWeightFile(file) {
        if (!isSafetensorsFile(file)) {
            setAnalysisStatus('请拖入 .safetensors 权重文件。', 'error');
            return;
        }
        if (location.protocol === 'file:') {
            setAnalysisStatus('静态打开没有后端 API，无法分析拖入文件。', 'error');
            return;
        }
        if (state.compareEnabled) {
            setPrimaryFile(file);
            if (currentCompareSource()) runWeightAnalysis();
            return;
        }
        const requestSeq = ++state.requestSeq;
        state.analyzing = true;
        state.uploading = true;
        setButtonBusy(true);
        setDropzoneBusy(true);
        setAnalysisStatus(`正在临时读取拖入文件 ${file.name} 并重建静态 ΔW...`, '');
        try {
            const payload = await inspectAnalysisWeightFile(ctx, file);
            if (requestSeq !== state.requestSeq) return;
            handleAnalysisPayload(payload, { uploaded: true });
        } catch (e) {
            if (requestSeq !== state.requestSeq) return;
            renderer.renderError(e.message);
            setAnalysisStatus('拖入文件分析失败: ' + e.message, 'error');
        } finally {
            if (requestSeq === state.requestSeq) {
                state.analyzing = false;
                state.uploading = false;
                setButtonBusy(false);
                setDropzoneBusy(false);
            }
        }
    }

    function handleAnalysisPayload(payload, { uploaded }) {
        if (payload.ok === false) {
            renderer.renderError(payload.error || '分析失败');
            setAnalysisStatus(payload.error || '分析失败', 'error');
            return;
        }
        renderer.renderResult(payload);
        const suffix = payload.unsupported?.unsupported ? '该权重结构暂不在第一版支持范围内。' : '分析完成。';
        const source = uploaded ? '拖入文件已临时分析，未写入权重目录。' : '这是权重能量推断，不是 prompt 激活图。';
        setAnalysisStatus(`${suffix} ${source}`, payload.unsupported?.unsupported ? 'error' : 'ok');
    }

    function toggleCompareMode() {
        state.compareEnabled = !state.compareEnabled;
        const slot = document.getElementById('weight-analysis-compare-slot');
        const button = document.getElementById('btn-toggle-weight-compare');
        const panel = document.querySelector('.weight-analysis-import-panel');
        if (slot) slot.hidden = !state.compareEnabled;
        panel?.classList.toggle('compare-active', state.compareEnabled);
        if (button) {
            button.classList.toggle('active', state.compareEnabled);
            button.setAttribute('aria-pressed', state.compareEnabled ? 'true' : 'false');
            button.textContent = state.compareEnabled ? '关闭对比' : '开启对比';
        }
        const runButton = document.getElementById('btn-run-weight-analysis');
        if (runButton && !state.analyzing) runButton.textContent = state.compareEnabled ? '分析并对比' : '分析权重';
        if (!state.compareEnabled) {
            renderer.clearComparison();
            setAnalysisStatus('已关闭对比模式。', '');
        } else {
            setAnalysisStatus('已开启对比模式：请填写或拖入第二个权重 B。', '');
        }
    }

    function exportWeightAnalysisReport() {
        if (!state.result || state.result.ok === false) {
            setAnalysisStatus('请先完成一次权重分析，再导出报告。', 'error');
            return;
        }
        const originalTitle = document.title;
        const fileName = state.result.file?.name || state.result.summary?.file_name || 'weight-analysis';
        document.title = `ΔW分析-${fileName}`;
        document.documentElement.classList.add('weight-analysis-print-mode');
        setAnalysisStatus('正在打开打印导出；在系统对话框里选择“保存为 PDF”。', 'ok');
        window.setTimeout(() => {
            window.print();
            window.setTimeout(() => {
                document.documentElement.classList.remove('weight-analysis-print-mode');
                document.title = originalTitle;
            }, 300);
        }, 50);
    }

    function syncSelectedWeightPath(event) {
        const value = String(event.target.value || '').trim();
        if (value) setPathInputValue(value);
    }

    function currentPrimarySource() {
        if (state.primaryFile) return { file: state.primaryFile };
        const input = document.getElementById('weight-analysis-path');
        const path = String(input?.value || state.selectedPath || '').trim();
        return path ? { path } : null;
    }

    function currentCompareSource() {
        if (state.compareFile) return { file: state.compareFile };
        const input = document.getElementById('weight-analysis-compare-path');
        const path = String(input?.value || state.comparePath || '').trim();
        return path ? { path } : null;
    }

    function setPathInputValue(value) {
        state.selectedPath = value || '';
        state.primaryFile = null;
        const input = document.getElementById('weight-analysis-path');
        if (input) input.value = state.selectedPath;
    }

    function setPrimaryFile(file) {
        if (!isSafetensorsFile(file)) {
            setAnalysisStatus('请拖入 .safetensors 主权重。', 'error');
            return;
        }
        state.primaryFile = file;
        state.selectedPath = '';
        const input = document.getElementById('weight-analysis-path');
        if (input) input.value = `uploaded://${file.name}`;
        setAnalysisStatus(`已载入主权重 A：${file.name}。继续放入 B 或点击“分析并对比”。`, 'ok');
    }

    function setComparePathValue(value) {
        state.comparePath = value || '';
        state.compareFile = null;
        const input = document.getElementById('weight-analysis-compare-path');
        if (input) input.value = state.comparePath;
    }

    function setCompareFile(file) {
        if (!isSafetensorsFile(file)) {
            setAnalysisStatus('请拖入 .safetensors 对比权重。', 'error');
            return;
        }
        state.compareFile = file;
        state.comparePath = '';
        const input = document.getElementById('weight-analysis-compare-path');
        if (input) input.value = `uploaded://${file.name}`;
        setAnalysisStatus(`已载入对比权重 B：${file.name}。点击“分析并对比”或拖入主权重 A。`, 'ok');
    }

    function firstSafetensorsFile(files) {
        return Array.from(files || []).find(isSafetensorsFile) || null;
    }

    function isSafetensorsFile(file) {
        return Boolean(file && String(file.name || '').toLowerCase().endsWith('.safetensors'));
    }

    function extractDroppedPath(dataTransfer) {
        if (!dataTransfer) return '';
        const raw = dataTransfer.getData('text/uri-list') || dataTransfer.getData('text/plain') || '';
        return String(raw || '')
            .split(/\r?\n/)
            .map((line) => line.trim())
            .find((line) => line && !line.startsWith('#')) || '';
    }

    function setButtonBusy(busy) {
        const button = document.getElementById('btn-run-weight-analysis');
        if (!button) return;
        button.disabled = Boolean(busy);
        if (busy) {
            button.textContent = state.compareEnabled ? '对比中...' : '分析中...';
        } else {
            button.textContent = state.compareEnabled ? '分析并对比' : '分析权重';
        }
    }

    function setDropzoneBusy(busy, id = 'weight-analysis-dropzone') {
        const dropzone = document.getElementById(id);
        if (!dropzone) return;
        dropzone.classList.toggle('busy', Boolean(busy));
    }

    function setAnalysisStatus(text, stateName = '') {
        const el = document.getElementById('weight-analysis-status');
        if (!el) return;
        el.textContent = text || '';
        el.className = ['preview-status', stateName || ''].filter(Boolean).join(' ');
    }

    return {
        bindWeightAnalysisEvents,
        loadAnalysisWeights,
        runWeightAnalysis,
    };
}
