export function createPreviewWeights({ ctx, state, deps, syncPreviewPanelSubtitle }) {
    const { formatBytes } = ctx.format;

    function renderPreviewWeights(payload) {
        const list = document.getElementById('preview-weights-list');
        const empty = document.getElementById('preview-weights-empty');
        const subtitle = document.getElementById('preview-weights-subtitle');
        if (!list || !empty || !subtitle) return;

        const weights = payload.weights || [];
        const sortedWeights = sortPreviewWeights(weights);
        subtitle.textContent = payload.directory
            ? `目录: ${payload.directory}${payload.mode === 'config_group' && payload.group_task_count != null
                ? ` · ${payload.group_task_count} 次训练`
                : payload.task_count
                    ? ` · 本任务 ${payload.task_count} 个`
                    : ''}`
            : '选择训练任务后显示保存轮次、步数和对应权重。';
        updatePreviewWeightSortButton();
        list.innerHTML = '';
        if (!sortedWeights.length) {
            empty.textContent = payload.error || payload.message || '未找到权重文件。';
            empty.hidden = false;
            syncPreviewPanelSubtitle();
            return;
        }
        empty.hidden = true;
        for (const item of sortedWeights) {
            list.appendChild(createPreviewWeightItem(item));
        }
        syncPreviewPanelSubtitle();
    }

    function togglePreviewWeightSort(loadPreviewWeights) {
        state.weightSortDirection = state.weightSortDirection === 'asc' ? 'desc' : 'asc';
        updatePreviewWeightSortButton();
        loadPreviewWeights();
    }

    function sortPreviewWeights(weights) {
        const direction = state.weightSortDirection === 'desc' ? -1 : 1;
        return [...(weights || [])].sort((a, b) => comparePreviewWeight(a, b, direction));
    }

    function comparePreviewWeight(a, b, direction) {
        const epochDiff = compareOptionalNumber(a.epoch, b.epoch, direction);
        if (epochDiff !== 0) return epochDiff;
        const stepDiff = compareOptionalNumber(a.steps, b.steps, direction);
        if (stepDiff !== 0) return stepDiff;
        return String(a.name || '').localeCompare(String(b.name || ''), 'zh-CN');
    }

    function compareOptionalNumber(a, b, direction) {
        const aNumber = sortableNumber(a);
        const bNumber = sortableNumber(b);
        if (aNumber === null && bNumber === null) return 0;
        if (aNumber === null) return 1;
        if (bNumber === null) return -1;
        return direction * (aNumber - bNumber);
    }

    function sortableNumber(value) {
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function updatePreviewWeightSortButton() {
        const btn = document.getElementById('btn-sort-weights');
        if (!btn) return;
        const isDesc = state.weightSortDirection === 'desc';
        btn.textContent = isDesc ? '反序' : '正序';
        btn.title = isDesc ? '当前按 Epoch/Step 从大到小排列，点击切换为正序。' : '当前按 Epoch/Step 从小到大排列，点击切换为反序。';
    }

    function createPreviewWeightItem(item) {
        const row = document.createElement('article');
        row.className = `preview-weight-item preview-weight-${item.kind || 'weight'}`;
        if (item.scope === 'task') {
            row.classList.add('preview-weight-task');
        }
        if (item.source_task?.id) {
            row.dataset.sourceTaskId = item.source_task.id;
        }

        const main = document.createElement('div');
        main.className = 'preview-weight-main';
        main.title = previewWeightFullPath(item);
        const titleLine = document.createElement('div');
        titleLine.className = 'preview-weight-title-line';
        const name = document.createElement('strong');
        name.textContent = previewWeightName(item);
        name.title = previewWeightFullPath(item);
        const badges = document.createElement('div');
        badges.className = 'preview-weight-badges';
        const badge = document.createElement('em');
        badge.textContent = previewWeightScopeLabel(item);
        if (badge.textContent) badges.appendChild(badge);
        if (item.kind === 'final') {
            const finalBadge = document.createElement('em');
            finalBadge.textContent = '最终权重';
            badges.appendChild(finalBadge);
        }
        titleLine.append(name, badges);
        main.appendChild(titleLine);

        const actions = document.createElement('div');
        actions.className = 'preview-weight-actions';
        const download = document.createElement('a');
        download.className = 'btn btn-small btn-primary preview-weight-download';
        download.href = previewWeightDownloadUrl(item);
        download.download = item.name || 'weight.safetensors';
        download.textContent = '下载';
        download.title = '通过浏览器下载这个权重文件。';
        const copy = document.createElement('button');
        copy.type = 'button';
        copy.className = 'btn btn-small preview-weight-copy';
        copy.textContent = '复制';
        copy.title = '复制这个权重文件的完整路径。';
        copy.addEventListener('click', () => copyPreviewWeightPath(item, copy));
        const continueBtn = document.createElement('button');
        continueBtn.type = 'button';
        continueBtn.className = 'btn btn-small preview-weight-continue';
        continueBtn.textContent = '继续训练';
        continueBtn.title = '把这个权重设置为新的 LoRA/LoHa/LoKr/GLoRA 补充训练来源。';
        continueBtn.addEventListener('click', () => deps.selectContinueLoraWeight(item.abs_path || item.file || ''));
        actions.append(download, copy, continueBtn);

        const stats = document.createElement('div');
        stats.className = 'preview-weight-stats';
        stats.append(
            createWeightStat('Epoch', item.epoch ?? '-'),
            createWeightStat('Step', item.steps ?? '-'),
            createWeightStat('计划', weightPlanText(item)),
            createWeightStat('保存', item.mtime_text || '-'),
            createWeightStat('大小', formatBytes(item.size_bytes)),
            createWeightStat('类型', weightKindLabel(item.kind)),
        );
        const source = createPreviewWeightSource(item);
        if (source) stats.append(source);

        row.append(main, stats, actions);
        return row;
    }

    function previewWeightName(item) {
        const name = String(item?.name || '').trim();
        if (name) return name;
        return fileNameFromPath(item?.file || item?.abs_path || '') || '未命名权重';
    }

    function previewWeightFullPath(item) {
        return String(item?.abs_path || item?.file || item?.name || '').trim();
    }

    function previewWeightScopeLabel(item) {
        return String(item?.scope_label || '').split(' · ')[0].trim();
    }

    function fileNameFromPath(value) {
        const text = String(value || '').trim();
        if (!text) return '';
        return text.split(/[\\/]/).filter(Boolean).pop() || text;
    }

    function previewWeightDownloadUrl(item) {
        if (item.download_url) return item.download_url;
        const params = new URLSearchParams({ file: item.file || '' });
        const taskId = item.source_task?.id || '';
        if (taskId) params.set('task_id', taskId);
        return `/api/preview/weight?${params.toString()}`;
    }

    function createPreviewWeightSource(item) {
        if (!item.source_task?.label) return null;
        const box = document.createElement('div');
        box.className = 'preview-weight-source';
        const sourceText = `来源 ${item.source_task.label}`;
        const text = document.createElement('span');
        text.className = 'preview-weight-source-text';
        text.textContent = sourceText;
        text.title = '双击复制来源文本；也可以像普通文本一样拖选。';
        text.addEventListener('dblclick', async () => {
            await copyPreviewWeightSource(sourceText, text);
        });
        box.appendChild(text);
        return box;
    }

    async function copyPreviewWeightSource(text, el) {
        selectElementText(el);
        try {
            await ctx.dom.copyText(text);
            selectElementText(el);
            el.classList.add('copied');
            const originalTitle = el.title;
            el.title = '已复制来源文本。';
            setTimeout(() => {
                el.classList.remove('copied');
                el.title = originalTitle;
            }, 1000);
        } catch (e) {
            selectElementText(el);
            alert('复制来源失败: ' + e.message);
        }
    }

    function selectElementText(el) {
        if (!el || !window.getSelection || !document.createRange) return;
        const range = document.createRange();
        range.selectNodeContents(el);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
    }

    async function copyPreviewWeightPath(item, button) {
        const path = item.abs_path || item.file || '';
        if (!path) return;
        try {
            await ctx.dom.copyText(path);
            const original = button.textContent;
            button.textContent = '已复制';
            button.classList.add('btn-primary');
            setTimeout(() => {
                button.textContent = original;
                button.classList.remove('btn-primary');
            }, 1200);
        } catch (e) {
            alert('复制权重路径失败: ' + e.message);
        }
    }

    function createWeightStat(label, value) {
        const box = document.createElement('div');
        const key = document.createElement('span');
        key.textContent = label;
        const valEl = document.createElement('strong');
        valEl.textContent = value;
        box.append(key, valEl);
        return box;
    }

    function weightKindLabel(kind) {
        return {
            epoch: '按轮保存',
            step: '按步保存',
            resume: '续训检查点',
            final: '最终权重',
            weight: '权重',
        }[kind] || '权重';
    }

    function weightPlanText(item) {
        const epochs = item.num_epochs ? `${item.num_epochs}ep` : '';
        const steps = item.max_steps ? `${item.max_steps}步` : '';
        return [epochs, steps].filter(Boolean).join(' / ') || '-';
    }

    return {
        renderPreviewWeights,
        togglePreviewWeightSort,
        updatePreviewWeightSortButton,
    };
}
