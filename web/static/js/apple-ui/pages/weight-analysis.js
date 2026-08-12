/* Weight analysis page: list real training weights, then inspect one on demand. */

import { createApiClient } from '../../shared/api.js?v=apple-ui-20260812v33';
import { escapeHtml, formatBytes } from '../../shared/format.js?v=apple-ui-20260812v33';

const api = createApiClient();

export async function loadWeightAnalysis() {
    const listing = await api('/api/analysis/weights');
    const weights = Array.isArray(listing?.weights) ? listing.weights : [];
    const options = weights.map((item) => {
        const path = item.abs_path || item.file || '';
        const details = [item.name || basename(path), item.steps != null ? `${item.steps} 步` : '', item.scope_label || '']
            .filter(Boolean)
            .join(' · ');
        return `<option value="${escapeAttribute(path)}">${escapeHtml(details)}</option>`;
    }).join('');

    return {
        html: `
            <div class="apple-page apple-page-wide apple-weight-analysis-page">
                <div class="apple-page-hero apple-reveal">
                    <h1>权重分析</h1>
                    <p>读取训练权重，检查层、组件与区块的静态权重能量分布。</p>
                </div>

                <section class="apple-config-section apple-reveal" data-stagger="1">
                    <div class="apple-config-section-header">
                        <span class="apple-eyebrow">分析对象</span>
                        <h2 class="apple-config-section-title">选择训练权重</h2>
                        <p class="apple-config-section-desc">分析在 CPU 上执行，不加载底模，也不会改变训练文件。</p>
                    </div>
                    <div class="apple-weight-analysis-picker">
                        <label class="apple-field">
                            <span class="apple-field-label-text">已找到的权重</span>
                            <select class="apple-select" data-weight-select ${weights.length ? '' : 'disabled'}>
                                <option value="">${escapeHtml(weights.length ? '选择一个权重' : (listing?.message || '未找到训练权重'))}</option>
                                ${options}
                            </select>
                        </label>
                        <label class="apple-field apple-weight-analysis-path-field">
                            <span class="apple-field-label-text">权重路径</span>
                            <input class="apple-input apple-text-mono" data-weight-path type="text" inputmode="url" placeholder="训练输出目录中的 .safetensors 文件">
                        </label>
                        <button class="apple-btn apple-btn-primary" type="button" data-weight-run>开始分析</button>
                    </div>
                    <p class="apple-config-feedback apple-config-feedback-visible" data-weight-status data-tone="${listing?.ok === false ? 'error' : 'info'}" role="status" aria-live="polite">
                        ${escapeHtml(listing?.ok === false ? (listing.error || '读取权重列表失败') : (listing?.message || `已找到 ${weights.length} 个可分析权重`))}
                    </p>
                </section>

                <div data-weight-results>
                    ${renderEmptyResult(weights.length)}
                </div>
            </div>
        `,
        onMount(root) {
            bindWeightAnalysis(root);
        },
    };
}

function bindWeightAnalysis(root) {
    const select = root.querySelector('[data-weight-select]');
    const pathInput = root.querySelector('[data-weight-path]');
    const runButton = root.querySelector('[data-weight-run]');
    const status = root.querySelector('[data-weight-status]');
    const results = root.querySelector('[data-weight-results]');

    select?.addEventListener('change', () => {
        if (select.value && pathInput) pathInput.value = select.value;
    });
    pathInput?.addEventListener('input', () => {
        if (select && select.value !== pathInput.value) select.value = '';
    });
    pathInput?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') runButton?.click();
    });
    runButton?.addEventListener('click', async () => {
        const path = String(pathInput?.value || select?.value || '').trim();
        if (!path) {
            setStatus(status, '请先选择或填写一个 .safetensors 权重路径。', 'error');
            pathInput?.focus();
            return;
        }
        runButton.disabled = true;
        runButton.textContent = '正在分析';
        setStatus(status, '正在读取权重并计算静态分布...', 'info');
        try {
            const payload = await api('/api/analysis/inspect', {
                method: 'POST',
                body: JSON.stringify({ path }),
            });
            if (payload?.ok === false) {
                results.innerHTML = renderError(payload.error || '权重分析失败');
                setStatus(status, payload.error || '权重分析失败', 'error');
                return;
            }
            results.innerHTML = renderAnalysis(payload);
            setStatus(status, payload?.unsupported?.unsupported ? '已读取权重，但该权重结构暂不支持能量重建。' : '权重分析完成。', payload?.unsupported?.unsupported ? 'error' : 'success');
        } catch (error) {
            results.innerHTML = renderError(error.message || '权重分析失败');
            setStatus(status, error.message || '权重分析失败', 'error');
        } finally {
            runButton.disabled = false;
            runButton.textContent = '开始分析';
        }
    });
}

function renderAnalysis(payload) {
    if (payload?.unsupported?.unsupported) {
        return renderError(payload.unsupported.reason || '该权重结构暂不支持静态分析');
    }
    const summary = payload?.summary || {};
    const components = Array.isArray(payload?.component_summary) ? payload.component_summary : [];
    const blocks = Array.isArray(payload?.block_summary) ? payload.block_summary : [];
    const conclusion = Array.isArray(summary.conclusion) ? summary.conclusion : [];
    return `
        <section class="apple-section apple-reveal apple-weight-analysis-summary">
            <div class="apple-section-header-row">
                <div>
                    <span class="apple-eyebrow">分析结果</span>
                    <h2 class="apple-section-title">${escapeHtml(summary.file_name || payload?.file?.name || '训练权重')}</h2>
                </div>
                <p class="apple-section-desc">${escapeHtml(payload?.adapter_type || summary.adapter_type || '未知类型')}</p>
            </div>
            <div class="apple-metrics-grid apple-weight-analysis-metrics">
                ${metricTile('可分析层', formatInteger(summary.layer_count))}
                ${metricTile('覆盖组件', formatInteger(summary.component_count))}
                ${metricTile('覆盖区块', formatInteger(summary.block_count))}
                ${metricTile('参数数量', formatInteger(summary.total_param_count))}
                ${metricTile('总权重范数', formatNumber(summary.total_fro_norm))}
                ${metricTile('中后段 / 前段', formatNumber(summary.mid_late_vs_early_ratio))}
            </div>
            ${conclusion.length ? `<div class="apple-analysis-note">${conclusion.map((line) => `<p>${escapeHtml(line)}</p>`).join('')}</div>` : ''}
        </section>
        ${renderContributionSection('组件贡献', '按组件聚合的权重范数与能量占比。', components, 'component')}
        ${renderContributionSection('区块贡献', '按模型区块聚合，可快速定位权重能量集中位置。', blocks, 'block')}
    `;
}

function renderContributionSection(title, description, rows, kind) {
    const body = rows.map((item) => {
        const label = kind === 'block' ? `区块 ${item.label ?? item.block ?? '其他'}` : componentLabel(item.label || item.component);
        const top = kind === 'block' ? componentLabel(item.top_component) : basename(item.top_layer || '');
        return `
            <tr>
                <td><strong>${escapeHtml(label)}</strong></td>
                <td>${formatInteger(item.layer_count)}</td>
                <td>${formatNumber(item.fro_norm)}</td>
                <td>${formatPercent(item.energy_contribution)}</td>
                <td class="apple-text-mono">${escapeHtml(top || '-')}</td>
            </tr>
        `;
    }).join('');
    return `
        <section class="apple-section apple-reveal apple-analysis-table-section">
            <h2 class="apple-section-title">${title}</h2>
            <p class="apple-section-desc">${description}</p>
            <div class="apple-table-wrapper">
                <table class="apple-table">
                    <thead><tr><th>位置</th><th>层数</th><th>权重范数</th><th>能量占比</th><th>最高贡献项</th></tr></thead>
                    <tbody>${body || '<tr><td colspan="5">没有可显示的分析结果</td></tr>'}</tbody>
                </table>
            </div>
        </section>
    `;
}

function renderEmptyResult(hasWeights) {
    return `<div class="apple-empty-state apple-reveal" data-stagger="2"><p>${hasWeights ? '选择一个权重开始分析。' : '训练完成或导入权重后，可在这里查看静态权重分布。'}</p></div>`;
}

function renderError(message) {
    return `<div class="apple-empty-state apple-reveal"><p>${escapeHtml(message)}</p></div>`;
}

function metricTile(label, value) {
    return `<div class="apple-metric-tile"><div class="apple-metric-value">${escapeHtml(value)}</div><div class="apple-metric-label">${label}</div></div>`;
}

function setStatus(element, message, tone) {
    if (!element) return;
    element.textContent = message;
    element.dataset.tone = tone;
}

function componentLabel(value) {
    const labels = {
        mlp_layer1: '多层感知机输入',
        mlp_layer2: '多层感知机输出',
        self_attn_q_proj: '自注意力查询',
        self_attn_k_proj: '自注意力键',
        self_attn_v_proj: '自注意力值',
        self_attn_output_proj: '自注意力输出',
        cross_attn_q_proj: '交叉注意力查询',
        cross_attn_k_proj: '交叉注意力键',
        cross_attn_v_proj: '交叉注意力值',
        cross_attn_output_proj: '交叉注意力输出',
    };
    return labels[value] || value || '其他';
}

function basename(value) {
    return String(value || '').replace(/\\/g, '/').split('/').filter(Boolean).pop() || '';
}

function formatInteger(value) {
    const number = Number(value);
    return Number.isFinite(number) ? Math.round(number).toLocaleString('zh-CN') : '-';
}

function formatNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '-';
    if (Math.abs(number) >= 1000) return number.toLocaleString('zh-CN', { maximumFractionDigits: 1 });
    if (Math.abs(number) >= 1) return number.toFixed(2);
    return number.toPrecision(3);
}

function formatPercent(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : '-';
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
