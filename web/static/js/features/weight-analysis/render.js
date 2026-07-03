import {
    blockBenchmark,
    escapeText,
    fileNameFromPath,
    formatInteger,
    formatNumber,
    formatPercent,
    froBenchmark,
    paramBenchmark,
    ratioBenchmark,
    shortComponent,
} from './render-utils.js?v=module-bootstrap-20260703-7';

export function createWeightAnalysisRenderer({ ctx, state }) {
    const { formatBytes } = ctx.format;
    const COLLAPSED_CANDIDATE_LIMIT = 5;

    function renderEmpty() {
        const summary = document.getElementById('weight-analysis-summary');
        if (summary) {
            summary.innerHTML = [
                '<div class="weight-analysis-empty weight-analysis-empty-hero">',
                '<strong>选择一个 .safetensors 权重后开始静态 ΔW 分析。</strong>',
                '<span>结果只代表权重能量分布，不代表 prompt 激活或真实语义结论。</span>',
                '</div>',
            ].join('');
        }
        clearComparison();
        setEmptyList('weight-analysis-layer-list', '尚未分析层类型贡献。');
        setEmptyList('weight-analysis-block-list', '尚未分析 block 贡献。');
        setEmptyList('weight-analysis-style-top', '尚未生成风格层候选。');
        setEmptyList('weight-analysis-character-top', '尚未生成角色/结构层候选。');
        renderBlockStructure([]);
        renderHeatmap(null);
        showCandidateKind(state.activeCandidateKind || 'style');
        updateCandidateToggleButton();
    }

    function renderWeightOptions(payload = {}) {
        const select = document.getElementById('weight-analysis-select');
        if (!select) return;
        const previous = select.value;
        const weights = Array.isArray(payload.weights) ? payload.weights : [];
        state.weights = weights;
        select.innerHTML = '';
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = weights.length ? '选择训练输出权重…' : (payload.message || '未找到训练权重');
        select.appendChild(placeholder);
        for (const item of weights) {
            const option = document.createElement('option');
            option.value = item.abs_path || item.file || '';
            option.textContent = [
                item.name || fileNameFromPath(option.value),
                item.epoch != null ? `ep ${item.epoch}` : '',
                item.steps != null ? `${item.steps} step` : '',
                item.scope_label || '',
            ].filter(Boolean).join(' · ');
            option.title = item.abs_path || item.file || '';
            select.appendChild(option);
        }
        select.disabled = !weights.length;
        if (previous && Array.from(select.options).some((option) => option.value === previous)) {
            select.value = previous;
        }
    }

    function renderResult(payload) {
        state.result = payload || null;
        clearComparison();
        if (!payload || payload.ok === false) {
            renderError(payload?.error || '分析失败');
            return;
        }
        renderSummary(payload);
        if (payload.unsupported?.unsupported) {
            setEmptyList('weight-analysis-layer-list', payload.unsupported.reason || '该权重结构暂不支持。');
            setEmptyList('weight-analysis-block-list', '未生成 block 统计。');
            setEmptyList('weight-analysis-style-top', '未生成风格候选。');
            setEmptyList('weight-analysis-character-top', '未生成角色/结构候选。');
            renderBlockStructure([]);
            renderHeatmap(null);
            showCandidateKind(state.activeCandidateKind || 'style');
            return;
        }
        renderRankList('weight-analysis-layer-list', payload.component_summary || [], {
            kind: 'component',
            title: (item) => item.label || item.component || 'unknown',
            meta: (item) => `${item.layer_count || 0} 层 · top ${item.top_layer || '-'}`,
        });
        renderBlockStructure(payload.block_summary || []);
        renderRankList('weight-analysis-block-list', payload.block_summary || [], {
            kind: 'block',
            title: (item) => `Block ${item.label ?? item.block ?? '其他'}`,
            meta: (item) => `${item.layer_count || 0} 层 · top ${item.top_component || '-'}`,
        });
        renderCandidateLists(payload);
        renderHeatmap(payload.heatmap || null);
    }

    function renderError(message) {
        const summary = document.getElementById('weight-analysis-summary');
        if (summary) {
            summary.innerHTML = `<div class="weight-analysis-empty error"><strong>${escapeText(message)}</strong></div>`;
        }
    }

    function renderSummary(payload) {
        const summary = document.getElementById('weight-analysis-summary');
        if (!summary) return;
        const info = payload.summary || {};
        const file = payload.file || {};
        const unsupported = payload.unsupported?.unsupported;
        const ratio = Number(info.mid_late_vs_early_ratio);
        const ratioText = Number.isFinite(ratio) ? `${ratio.toFixed(2)}×` : '-';
        summary.innerHTML = '';

        const head = document.createElement('div');
        head.className = ['weight-analysis-summary-head', unsupported ? 'unsupported' : ''].filter(Boolean).join(' ');
        const title = document.createElement('div');
        title.className = 'weight-analysis-current-file';
        title.innerHTML = `<span>当前权重</span><strong>${escapeText(file.name || info.file_name || '-')}</strong><em>${escapeText(payload.adapter_type || '-')}</em>`;
        const note = document.createElement('p');
        note.textContent = unsupported
            ? (payload.unsupported?.reason || '该权重暂不在第一版支持范围内。')
            : '静态 ΔW 范数分析完成：以下贡献、候选层与热力图均为启发式推断。';
        head.append(title, note);
        summary.appendChild(head);

        const stats = document.createElement('div');
        stats.className = 'weight-analysis-stat-grid';
        stats.append(
            statCard('总 Fro 范数', formatNumber(info.total_fro_norm), {
                icon: 'Σ',
                featured: true,
                tone: 'blue',
                note: froBenchmark(info),
            }),
            statCard('参数量', formatInteger(info.total_param_count), {
                icon: '#',
                featured: true,
                tone: 'green',
                note: paramBenchmark(info.total_param_count),
            }),
            statCard('层数', info.layer_count ?? 0, { icon: 'L', note: '参与 ΔW 统计的层' }),
            statCard('Block', info.block_count ?? 0, { icon: 'B', note: blockBenchmark(info.block_count) }),
            statCard('层类型', info.component_count ?? 0, { icon: 'T', note: 'component 覆盖数' }),
            statCard('中后/早段', ratioText, { icon: 'M', note: ratioBenchmark(ratio) }),
        );
        summary.appendChild(stats);

        const meta = document.createElement('div');
        meta.className = 'weight-analysis-file-meta';
        meta.append(
            metaPill('路径', file.path || file.abs_path || '-'),
            metaPill('大小', formatBytes(file.size_bytes || 0)),
            metaPill('保存时间', file.mtime_text || '-'),
            metaPill('Top', [info.top_layer, info.top_component].filter(Boolean).join(' · ') || '-'),
        );
        summary.appendChild(meta);
    }

    function statCard(label, value, options = {}) {
        const card = document.createElement('div');
        card.className = ['weight-analysis-stat-card', options.featured ? 'featured' : '', options.tone ? `tone-${options.tone}` : ''].filter(Boolean).join(' ');
        const icon = document.createElement('span');
        icon.className = 'weight-analysis-stat-icon';
        icon.textContent = options.icon || '·';
        const body = document.createElement('div');
        const strong = document.createElement('strong');
        strong.textContent = String(value);
        const labelNode = document.createElement('span');
        labelNode.textContent = label;
        const note = document.createElement('small');
        note.textContent = options.note || '';
        body.append(strong, labelNode, note);
        card.append(icon, body);
        return card;
    }

    function metaPill(label, value) {
        const pill = document.createElement('div');
        pill.className = 'weight-analysis-meta-pill';
        const key = document.createElement('span');
        key.textContent = label;
        const val = document.createElement('strong');
        val.textContent = String(value || '-');
        val.title = val.textContent;
        pill.append(key, val);
        return pill;
    }

    function renderRankList(id, items, config) {
        const list = document.getElementById(id);
        if (!list) return;
        list.innerHTML = '';
        if (!items.length) {
            setEmptyList(id, '暂无数据。');
            return;
        }
        const topValue = Math.max(...items.map((item) => Number(item.fro_norm || 0)), 1e-12);
        items.slice(0, 20).forEach((item, index) => {
            const row = document.createElement('article');
            row.className = ['weight-analysis-rank-row', `rank-${config.kind}`].join(' ');
            const component = String(item.component || item.label || '');
            const block = item.block ?? item.label ?? '';
            const intensity = Math.max(0.04, Math.min(1, Number(item.fro_norm || 0) / topValue));
            row.style.setProperty('--rank-intensity', intensity.toFixed(3));
            row.style.setProperty('--bar-pct', `${(intensity * 100).toFixed(2)}%`);
            if (config.kind === 'component') {
                row.dataset.component = component;
                row.tabIndex = 0;
                row.role = 'button';
                row.title = '点击高亮该层类型的候选层';
                row.addEventListener('click', () => setActiveComponent(component));
                row.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        setActiveComponent(component);
                    }
                });
            } else if (config.kind === 'block') {
                row.dataset.block = String(block);
                row.tabIndex = 0;
                row.role = 'button';
                row.title = '点击高亮该 block 的候选层';
                row.addEventListener('click', () => setActiveBlock(block));
                row.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        setActiveBlock(block);
                    }
                });
            }

            const main = document.createElement('div');
            main.className = 'weight-analysis-rank-main';
            const title = document.createElement('strong');
            title.textContent = `${index + 1}. ${config.title(item)}`;
            const meta = document.createElement('span');
            meta.textContent = config.meta(item);
            const bar = document.createElement('div');
            bar.className = 'weight-analysis-rank-bar';
            bar.appendChild(document.createElement('i'));
            main.append(title, meta, bar);

            const value = document.createElement('div');
            value.className = 'weight-analysis-rank-value';
            value.innerHTML = `<strong>${formatNumber(item.fro_norm)}</strong><span>${formatPercent(item.contribution)}</span>`;
            row.append(main, value);
            list.appendChild(row);
        });
        applyCandidateHighlight();
    }

    function renderBlockStructure(items) {
        const target = document.getElementById('weight-analysis-block-structure');
        if (!target) return;
        target.innerHTML = '';
        const rows = (items || [])
            .filter((item) => Number.isInteger(Number(item.block)))
            .map((item) => ({ ...item, block: Number(item.block) }))
            .sort((a, b) => a.block - b.block);
        if (!rows.length) {
            target.innerHTML = '<div class="weight-analysis-structure-empty">暂无 block 结构数据</div>';
            return;
        }
        const maxValue = Math.max(...rows.map((item) => Number(item.fro_norm || 0)), 1e-12);
        const labels = document.createElement('div');
        labels.className = 'weight-analysis-structure-labels';
        labels.innerHTML = '<span>输入端 0–8</span><span>中段 9–18</span><span>输出端 19+</span>';
        const strip = document.createElement('div');
        strip.className = 'weight-analysis-structure-strip';
        for (const item of rows) {
            const intensity = Math.max(0.05, Math.min(1, Number(item.fro_norm || 0) / maxValue));
            const segment = document.createElement('button');
            segment.type = 'button';
            segment.className = 'weight-analysis-structure-segment';
            segment.dataset.block = String(item.block);
            segment.style.setProperty('--block-alpha', Math.min(0.96, 0.12 + intensity * 0.78).toFixed(3));
            segment.textContent = String(item.block);
            segment.title = `Block ${item.block}\nFro: ${formatNumber(item.fro_norm)}\n贡献: ${formatPercent(item.contribution)}\nTop: ${item.top_component || '-'}`;
            segment.addEventListener('click', () => setActiveBlock(item.block));
            strip.appendChild(segment);
        }
        target.append(labels, strip);
        applyCandidateHighlight();
    }

    function renderCandidateLists(payload) {
        renderCandidateList('weight-analysis-style-top', payload.style_top20 || [], 'style');
        renderCandidateList('weight-analysis-character-top', payload.character_top20 || [], 'character');
        showCandidateKind(state.activeCandidateKind || 'style');
        updateCandidateToggleButton();
        applyCandidateHighlight();
    }

    function renderCandidateList(id, items, kind) {
        const list = document.getElementById(id);
        if (!list) return;
        list.innerHTML = '';
        if (!items.length) {
            setEmptyList(id, '暂无启发式候选。');
            return;
        }
        const expanded = Boolean(state.candidateExpanded?.[kind]);
        const visibleItems = items.slice(0, expanded ? 20 : COLLAPSED_CANDIDATE_LIMIT);
        const topValue = Math.max(...items.map((item) => Number(item.fro_norm || 0)), 1e-12);
        for (const item of visibleItems) {
            const row = document.createElement('article');
            row.className = `weight-analysis-candidate ${kind}`;
            row.dataset.component = String(item.component || '');
            row.dataset.block = String(item.block ?? '');
            const intensity = Math.max(0.04, Math.min(1, Number(item.fro_norm || 0) / topValue));
            row.style.setProperty('--candidate-intensity', intensity.toFixed(3));

            const head = document.createElement('div');
            head.className = 'weight-analysis-candidate-head';
            const title = document.createElement('strong');
            title.textContent = `${item.rank}. ${item.name || '-'}`;
            title.title = item.name || '';
            const badge = document.createElement('span');
            badge.textContent = `Block ${item.block ?? '其他'} · ${item.component || '-'}`;
            head.append(title, badge);

            const reason = document.createElement('p');
            reason.textContent = item.reason || '按 ΔW 范数与启发式权重排序。';

            const bottom = document.createElement('div');
            bottom.className = 'weight-analysis-candidate-bottom';
            const stats = document.createElement('div');
            stats.className = 'weight-analysis-candidate-stats';
            stats.append(
                miniStat('Fro', formatNumber(item.fro_norm)),
                miniStat('贡献', formatPercent(item.contribution)),
                miniStat('MeanAbs', formatNumber(item.mean_abs)),
                miniStat('MaxAbs', formatNumber(item.max_abs)),
            );
            bottom.append(stats, sparkline(item));
            row.append(head, reason, bottom);
            list.appendChild(row);
        }
    }

    function miniStat(label, value) {
        const box = document.createElement('span');
        box.innerHTML = `<em>${escapeText(label)}</em><strong>${escapeText(value)}</strong>`;
        return box;
    }

    function sparkline(item) {
        const contribution = Math.max(0, Math.min(1, Number(item.contribution || 0)));
        const rank = Math.max(1, Number(item.rank || 1));
        const phase = (rank % 5) * 0.35;
        const points = Array.from({ length: 9 }, (_, index) => {
            const x = index * 8;
            const wave = 0.35 + 0.65 * Math.sin((index / 8) * Math.PI + phase);
            const y = 17 - Math.max(2, (contribution * 22 + 3 / rank) * wave);
            return `${x},${Math.max(2, Math.min(17, y)).toFixed(1)}`;
        }).join(' ');
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'weight-analysis-sparkline');
        svg.setAttribute('viewBox', '0 0 64 20');
        svg.setAttribute('aria-label', '层能量小波形');
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        line.setAttribute('points', points);
        line.setAttribute('fill', 'none');
        line.setAttribute('pathLength', '1');
        svg.appendChild(line);
        return svg;
    }

    function renderHeatmap(heatmap) {
        const target = document.getElementById('weight-analysis-heatmap');
        if (!target) return;
        target.innerHTML = '';
        const blocks = heatmap?.blocks || [];
        const components = heatmap?.components || [];
        if (!blocks.length || !components.length) {
            target.innerHTML = '<div class="weight-analysis-empty">暂无热力图数据。</div>';
            return;
        }
        const grid = document.createElement('div');
        grid.className = 'weight-analysis-heatmap-grid';
        grid.style.gridTemplateColumns = `minmax(58px, auto) repeat(${components.length}, minmax(74px, 1fr))`;
        grid.appendChild(heatmapHeader('Block'));
        components.forEach((component) => grid.appendChild(heatmapHeader(shortComponent(component))));
        const maxValue = Number(heatmap.max_value || 0) || 1;
        const cells = new Map((heatmap.cells || []).map((cell) => [`${cell.block}|${cell.component}`, cell]));
        blocks.forEach((block, rowIndex) => {
            const blockCell = document.createElement('div');
            blockCell.className = 'weight-analysis-heatmap-block';
            blockCell.textContent = String(block);
            grid.appendChild(blockCell);
            components.forEach((component) => {
                const cell = cells.get(`${block}|${component}`) || {};
                const value = Number(cell.fro_norm || 0);
                const heat = maxValue > 0 ? Math.max(0, Math.min(1, value / maxValue)) : 0;
                const node = document.createElement('div');
                node.className = 'weight-analysis-heatmap-cell';
                node.dataset.component = component;
                node.dataset.block = String(block);
                node.style.setProperty('--heat', heat.toFixed(3));
                node.textContent = value > 0 ? formatNumber(value, { compact: true }) : '';
                node.title = `Block ${block} / ${component}\nFro: ${formatNumber(value)}\n层数: ${cell.layer_count || 0}\nTop: ${cell.top_layer || '-'}`;
                if ((rowIndex % 2) === 1) node.classList.add('odd-row');
                grid.appendChild(node);
            });
        });
        target.appendChild(grid);
        applyCandidateHighlight();
    }

    function renderComparison(primary, secondary) {
        state.compareResult = { primary, secondary };
        const target = document.getElementById('weight-analysis-compare-summary');
        if (!target) return;
        if (!primary || !secondary || primary.ok === false || secondary.ok === false) {
            target.innerHTML = '<div class="weight-analysis-empty error"><strong>对比失败：A / B 权重至少有一个分析结果不可用。</strong></div>';
            return;
        }
        const aName = primary.file?.name || primary.summary?.file_name || 'A';
        const bName = secondary.file?.name || secondary.summary?.file_name || 'B';
        if (primary.unsupported?.unsupported || secondary.unsupported?.unsupported) {
            target.innerHTML = `<div class="weight-analysis-empty error"><strong>对比需要两个第一版支持的 LoRA / LoHa / LoKr 权重。</strong><span>A: ${escapeText(aName)} · B: ${escapeText(bName)}</span></div>`;
            return;
        }
        const componentRows = diffRows(primary.component_summary || [], secondary.component_summary || [], 'label').slice(0, 8);
        const blockRows = diffRows(primary.block_summary || [], secondary.block_summary || [], 'label').slice(0, 8);
        target.innerHTML = '';
        const box = document.createElement('section');
        box.className = 'panel weight-analysis-compare-card';
        const head = document.createElement('div');
        head.className = 'weight-analysis-compare-head';
        head.innerHTML = `<div><span>COMPARE MODE</span><strong>B - A 静态 ΔW 差值</strong></div><p>A: ${escapeText(aName)} · B: ${escapeText(bName)}</p>`;
        const grid = document.createElement('div');
        grid.className = 'weight-analysis-compare-grid';
        grid.append(
            renderDiffPanel('层类型差异', componentRows, 'component'),
            renderDiffPanel('Block 差异', blockRows, 'block'),
        );
        box.append(head, grid);
        target.appendChild(box);
    }

    function renderDiffPanel(title, rows, kind) {
        const panel = document.createElement('div');
        panel.className = 'weight-analysis-diff-panel';
        const heading = document.createElement('h4');
        heading.textContent = title;
        panel.appendChild(heading);
        if (!rows.length) {
            const empty = document.createElement('div');
            empty.className = 'weight-analysis-empty';
            empty.textContent = '暂无可对比数据。';
            panel.appendChild(empty);
            return panel;
        }
        const maxDelta = Math.max(...rows.map((row) => Math.abs(row.delta)), 1e-12);
        for (const row of rows) {
            const item = document.createElement('div');
            item.className = ['weight-analysis-diff-row', row.delta >= 0 ? 'positive' : 'negative'].join(' ');
            item.dataset[kind] = row.label;
            item.style.setProperty('--diff-pct', `${Math.min(100, Math.abs(row.delta) / maxDelta * 100).toFixed(2)}%`);
            const label = document.createElement('strong');
            label.textContent = kind === 'block' && row.label !== '其他' ? `Block ${row.label}` : row.label;
            const value = document.createElement('span');
            value.textContent = `${row.delta >= 0 ? '+' : ''}${formatNumber(row.delta)} · ${row.deltaContribution >= 0 ? '+' : ''}${formatPercent(row.deltaContribution)}`;
            item.append(label, value);
            panel.appendChild(item);
        }
        return panel;
    }

    function diffRows(primaryRows, secondaryRows, key) {
        const primaryMap = new Map(primaryRows.map((item) => [String(item[key] ?? item.label ?? ''), item]));
        const secondaryMap = new Map(secondaryRows.map((item) => [String(item[key] ?? item.label ?? ''), item]));
        const labels = new Set([...primaryMap.keys(), ...secondaryMap.keys()].filter(Boolean));
        return Array.from(labels).map((label) => {
            const a = primaryMap.get(label) || {};
            const b = secondaryMap.get(label) || {};
            return {
                label,
                a: Number(a.fro_norm || 0),
                b: Number(b.fro_norm || 0),
                delta: Number(b.fro_norm || 0) - Number(a.fro_norm || 0),
                deltaContribution: Number(b.contribution || 0) - Number(a.contribution || 0),
            };
        }).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
    }

    function clearComparison() {
        state.compareResult = null;
        const target = document.getElementById('weight-analysis-compare-summary');
        if (target) target.innerHTML = '';
    }

    function showCandidateKind(kind) {
        const normalized = kind === 'character' ? 'character' : 'style';
        state.activeCandidateKind = normalized;
        document.querySelectorAll('.weight-analysis-candidate-tab').forEach((button) => {
            const active = button.dataset.candidateKind === normalized;
            button.classList.toggle('active', active);
            button.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        document.querySelectorAll('[data-candidate-panel]').forEach((panel) => {
            panel.hidden = panel.dataset.candidatePanel !== normalized;
        });
        updateCandidateToggleButton();
        applyCandidateHighlight();
    }

    function toggleCandidateExpanded() {
        const kind = state.activeCandidateKind || 'style';
        state.candidateExpanded[kind] = !state.candidateExpanded[kind];
        if (state.result && state.result.ok !== false && !state.result.unsupported?.unsupported) {
            renderCandidateLists(state.result);
        }
        showCandidateKind(kind);
    }

    function updateCandidateToggleButton() {
        const button = document.getElementById('btn-weight-analysis-toggle-candidates');
        if (!button) return;
        const kind = state.activeCandidateKind || 'style';
        const expanded = Boolean(state.candidateExpanded?.[kind]);
        const label = kind === 'style' ? '风格层' : '角色/结构层';
        button.textContent = expanded ? `收起${label}到前 5` : `查看${label}全部 Top20`;
        button.hidden = !state.result || state.result.ok === false || Boolean(state.result.unsupported?.unsupported);
    }

    function setActiveComponent(component) {
        state.activeComponent = state.activeComponent === component ? '' : component;
        state.activeBlock = '';
        applyCandidateHighlight();
    }

    function setActiveBlock(block) {
        const value = String(block ?? '');
        state.activeBlock = state.activeBlock === value ? '' : value;
        state.activeComponent = '';
        applyCandidateHighlight();
    }

    function applyCandidateHighlight() {
        const component = String(state.activeComponent || '');
        const block = String(state.activeBlock || '');
        document.querySelectorAll('.weight-analysis-rank-row').forEach((row) => {
            row.classList.toggle('active', Boolean(component) && row.dataset.component === component);
            row.classList.toggle('active-block', Boolean(block) && row.dataset.block === block);
        });
        document.querySelectorAll('.weight-analysis-candidate').forEach((row) => {
            const matched = (component && row.dataset.component === component) || (block && row.dataset.block === block);
            row.classList.toggle('highlighted', Boolean(matched));
            row.classList.toggle('dimmed', Boolean(component || block) && !matched);
        });
        document.querySelectorAll('.weight-analysis-heatmap-cell').forEach((cell) => {
            const matched = (component && cell.dataset.component === component) || (block && cell.dataset.block === block);
            cell.classList.toggle('highlighted', Boolean(matched));
            cell.classList.toggle('dimmed', Boolean(component || block) && !matched);
        });
        document.querySelectorAll('.weight-analysis-structure-segment').forEach((segment) => {
            segment.classList.toggle('active', Boolean(block) && segment.dataset.block === block);
        });
    }

    function heatmapHeader(text) {
        const node = document.createElement('div');
        node.className = 'weight-analysis-heatmap-header';
        node.textContent = text;
        node.title = text;
        return node;
    }

    function setEmptyList(id, message) {
        const list = document.getElementById(id);
        if (!list) return;
        list.innerHTML = `<div class="weight-analysis-empty">${escapeText(message)}</div>`;
    }

    return {
        renderEmpty,
        renderWeightOptions,
        renderResult,
        renderComparison,
        renderError,
        showCandidateKind,
        toggleCandidateExpanded,
        clearComparison,
    };
}
