/* Presentational helpers for Dragon's static weight analysis workspace. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { renderStatusRegion, renderToolButton, renderToolHero } from './tool-page.js?v=dragon-ui-20260814v43';
import { escapeHtml, formatBytes } from '../../shared/format.js?v=dragon-ui-20260812v35';

export function renderWeightAnalysisPage(model) {
    const options = renderWeightOptions(model.weights, model.listMessage);
    const actions = [
        renderToolButton('refresh', '刷新列表', 'refresh-weights', 'dragon-btn-ghost'),
        renderToolButton('download', '导出 JSON', 'export-json', 'dragon-btn-secondary', 'disabled'),
        renderToolButton('filePlus', '打印 / PDF', 'export-pdf', 'dragon-btn-secondary', 'disabled'),
    ].join('');
    return `
        <div class="dragon-page dragon-page-wide dragon-tool-page dragon-weight-analysis-workspace" data-weight-root data-compare="false">
            ${renderToolHero({
                eyebrow: '模型与系统',
                title: '权重分析',
                description: '在 CPU 上读取 safetensors，重建静态 ΔW，并按组件、区块和候选层解释权重能量。',
                badge: `<span class="dragon-tool-count-badge" data-weight-count>${model.weights.length} 个权重</span>`,
                actions,
            })}
            ${renderStatusRegion(
                'data-weight-status',
                model.listError || model.listMessage || `已找到 ${model.weights.length} 个可分析权重。`,
                model.listError ? 'error' : 'info',
            )}

            <section class="dragon-tool-panel dragon-weight-analysis-import dragon-reveal" data-stagger="1">
                <div class="dragon-tool-panel-head">
                    <div>
                        <span class="dragon-eyebrow">分析对象</span>
                        <h2>选择、填写或上传权重</h2>
                        <p class="dragon-tool-note dragon-weight-analysis-note">上传文件只做临时分析，不会写入模型或训练目录。</p>
                    </div>
                    <button class="dragon-btn dragon-btn-secondary" type="button" data-weight-action="toggle-compare" aria-pressed="false">
                        ${renderIcon('layers', 'dragon-btn-icon')}<span>开启 A / B 对比</span>
                    </button>
                </div>
                <div class="dragon-weight-source-grid">
                    ${renderSourceSlot('primary', 'A', '主权重', options)}
                    ${renderSourceSlot('secondary', 'B', '对比权重', options, true)}
                </div>
                <div class="dragon-weight-runbar">
                    <p>支持 LoRA、LoHa、LoKr；差值按 B − A 计算，仅代表静态权重能量差异。</p>
                    <button class="dragon-btn dragon-btn-primary" type="button" data-weight-action="run">
                        ${renderIcon('chart', 'dragon-btn-icon')}<span>分析权重</span>
                    </button>
                </div>
            </section>

            <div class="dragon-weight-results" data-weight-results>
                ${renderAnalysisEmpty(model.weights.length)}
            </div>
        </div>
    `;
}

function renderSourceSlot(slot, letter, title, options, hidden = false) {
    return `
        <section class="dragon-weight-source-card" data-weight-slot="${slot}" ${hidden ? 'hidden' : ''} aria-label="${title}">
            <header><span>${letter}</span><div><strong>${title}</strong><small>${slot === 'primary' ? '分析基准' : '用于比较差异'}</small></div></header>
            <label class="dragon-field">
                <span class="dragon-field-label-text">训练权重列表</span>
                <select class="dragon-select" data-weight-select="${slot}" name="${slot}_weight_select" autocomplete="off">
                    ${options}
                </select>
            </label>
            <label class="dragon-field">
                <span class="dragon-field-label-text">权重路径</span>
                <input class="dragon-input dragon-text-mono" data-weight-path="${slot}" name="${slot}_weight_path" type="text" autocomplete="off" spellcheck="false" placeholder="例如：output/ckpt/my-lora.safetensors…">
            </label>
            <input class="visually-hidden" data-weight-file="${slot}" id="dragon-weight-file-${slot}" name="${slot}_weight_file" type="file" accept=".safetensors,application/octet-stream">
            <label class="dragon-weight-upload" data-weight-dropzone="${slot}" for="dragon-weight-file-${slot}" tabindex="0">
                ${renderIcon('upload')}
                <span><strong>选择或拖入 .safetensors</strong><small data-weight-file-label="${slot}">本地文件仅临时上传分析</small></span>
            </label>
        </section>
    `;
}

export function renderWeightOptions(weights, message = '') {
    const items = weights.map((item) => {
        const path = item.abs_path || item.file || '';
        const label = [item.name || basename(path), item.steps != null ? `${item.steps} 步` : '', item.scope_label || '']
            .filter(Boolean)
            .join(' · ');
        return `<option value="${escapeAttribute(path)}">${escapeHtml(label)}</option>`;
    }).join('');
    const placeholder = weights.length ? '选择一个已扫描权重' : (message || '未找到训练权重');
    return `<option value="">${escapeHtml(placeholder)}</option>${items}`;
}

export function renderAnalysisBundle(primary, secondary = null) {
    if (!primary || primary.ok === false) return renderAnalysisError(primary?.error || '权重分析失败');
    const summaries = secondary
        ? `<div class="dragon-weight-result-pair">${renderResultSummary(primary, 'A')}${renderResultSummary(secondary, 'B')}</div>`
        : renderResultSummary(primary, 'A');
    return `
        ${summaries}
        ${secondary ? renderComparison(primary, secondary) : ''}
        ${renderContributionSection('组件贡献', '按模块类型汇总主权重 A 的范数与能量占比。', primary.component_summary, 'component')}
        ${renderContributionSection('区块贡献', '按模型区块汇总主权重 A，可快速定位能量集中区域。', primary.block_summary, 'block')}
        ${renderCandidateSection(primary)}
        ${renderHeatmap(primary.heatmap)}
    `;
}

function renderResultSummary(payload, label) {
    const summary = payload.summary || {};
    const unsupported = payload.unsupported?.unsupported;
    const conclusion = Array.isArray(summary.conclusion) ? summary.conclusion : [];
    return `
        <section class="dragon-tool-panel dragon-weight-result-summary dragon-reveal" data-result-label="${label}">
            <div class="dragon-tool-panel-head">
                <div><span class="dragon-eyebrow">权重 ${label}</span><h2>${escapeHtml(summary.file_name || payload.file?.name || '未命名权重')}</h2></div>
                <span class="dragon-analysis-type" data-tone="${unsupported ? 'error' : 'success'}">${escapeHtml(payload.adapter_type || '未知结构')}</span>
            </div>
            <p class="dragon-weight-file-meta">${escapeHtml(payload.file?.path || '')}${payload.file?.size_bytes != null ? ` · ${formatBytes(payload.file.size_bytes)}` : ''}</p>
            ${unsupported ? `<div class="dragon-analysis-warning">${escapeHtml(payload.unsupported.reason || '该权重暂不支持静态 ΔW 重建。')}</div>` : `
                <div class="dragon-stat-grid dragon-weight-analysis-metrics">
                    ${metricTile('可分析层', formatInteger(summary.layer_count))}
                    ${metricTile('覆盖组件', formatInteger(summary.component_count))}
                    ${metricTile('覆盖区块', formatInteger(summary.block_count))}
                    ${metricTile('参数数量', formatInteger(summary.total_param_count))}
                    ${metricTile('总权重范数', formatNumber(summary.total_fro_norm))}
                    ${metricTile('中后段 / 前段', formatNumber(summary.mid_late_vs_early_ratio))}
                </div>
                ${conclusion.length ? `<div class="dragon-analysis-note">${conclusion.map((line) => `<p>${escapeHtml(line)}</p>`).join('')}</div>` : ''}
            `}
        </section>
    `;
}

function renderComparison(primary, secondary) {
    if (primary.unsupported?.unsupported || secondary?.unsupported?.unsupported) {
        return `<section class="dragon-tool-panel dragon-analysis-warning dragon-reveal"><strong>A / B 对比不可用</strong><p>两个权重都必须是可重建的 LoRA、LoHa 或 LoKr。</p></section>`;
    }
    const componentRows = diffRows(primary.component_summary, secondary.component_summary, 'component');
    const blockRows = diffRows(primary.block_summary, secondary.block_summary, 'block');
    return `
        <section class="dragon-tool-panel dragon-analysis-table-section dragon-reveal">
            <div class="dragon-tool-panel-head"><div><span class="dragon-eyebrow">B − A</span><h2>静态能量差异</h2></div><span class="dragon-tool-note">正值表示 B 更强，负值表示 A 更强</span></div>
            <div class="dragon-analysis-compare-grid">
                ${renderDiffTable('组件差异', componentRows)}
                ${renderDiffTable('区块差异', blockRows)}
            </div>
        </section>
    `;
}

function renderDiffTable(title, rows) {
    return `<div><h3>${title}</h3><div class="dragon-table-wrapper"><table class="dragon-table"><thead><tr><th>位置</th><th>范数差</th><th>占比差</th></tr></thead><tbody>${rows.slice(0, 12).map((row) => `<tr><td>${escapeHtml(row.label)}</td><td data-delta="${toneForDelta(row.norm)}">${formatSigned(row.norm)}</td><td data-delta="${toneForDelta(row.ratio)}">${formatSignedPercent(row.ratio)}</td></tr>`).join('') || '<tr><td colspan="3">没有可比较数据</td></tr>'}</tbody></table></div></div>`;
}

function renderContributionSection(title, description, rows = [], kind) {
    const body = rows.map((item) => {
        const label = kind === 'block' ? `区块 ${item.label ?? item.block ?? '其他'}` : componentLabel(item.label || item.component);
        const top = kind === 'block' ? componentLabel(item.top_component) : basename(item.top_layer || '');
        return `<tr><td><strong>${escapeHtml(label)}</strong></td><td>${formatInteger(item.layer_count)}</td><td>${formatNumber(item.fro_norm)}</td><td>${formatPercent(item.energy_contribution ?? item.contribution)}</td><td class="dragon-text-mono">${escapeHtml(top || '-')}</td></tr>`;
    }).join('');
    return `<section class="dragon-tool-panel dragon-analysis-table-section dragon-reveal"><div class="dragon-tool-panel-head"><div><span class="dragon-eyebrow">主权重 A</span><h2>${title}</h2></div><span class="dragon-tool-note">${description}</span></div><div class="dragon-table-wrapper"><table class="dragon-table"><thead><tr><th>位置</th><th>层数</th><th>权重范数</th><th>能量占比</th><th>最高贡献项</th></tr></thead><tbody>${body || '<tr><td colspan="5">没有可显示的分析结果</td></tr>'}</tbody></table></div></section>`;
}

function renderCandidateSection(payload) {
    const groups = [['风格候选', payload.style_top20], ['角色 / 结构候选', payload.character_top20]];
    if (!groups.some(([, rows]) => rows?.length)) return '';
    return `<section class="dragon-tool-panel dragon-reveal"><div class="dragon-tool-panel-head"><div><span class="dragon-eyebrow">启发式排序</span><h2>重点层候选</h2></div><span class="dragon-tool-note">用于缩小排查范围，不等同于实际 prompt 激活</span></div><div class="dragon-analysis-candidates">${groups.map(([title, rows = []]) => renderCandidateGroup(title, rows)).join('')}</div></section>`;
}

function renderCandidateGroup(title, rows) {
    const renderRows = (items) => items.map((row) => `<li><strong>${escapeHtml(basename(row.name))}</strong><span>区块 ${row.block ?? '-'} · ${escapeHtml(componentLabel(row.component))} · ${formatPercent(row.contribution)}</span><small>${escapeHtml(row.reason || '')}</small></li>`).join('');
    const first = rows.slice(0, 8);
    const rest = rows.slice(8);
    return `<div><h3>${title}</h3><ol>${renderRows(first)}</ol>${rest.length ? `<details class="dragon-analysis-candidate-more"><summary>查看全部 ${rows.length} 项</summary><ol start="9">${renderRows(rest)}</ol></details>` : ''}</div>`;
}

function renderHeatmap(heatmap = {}) {
    const blocks = Array.isArray(heatmap.blocks) ? heatmap.blocks : [];
    const components = Array.isArray(heatmap.components) ? heatmap.components : [];
    const matrix = Array.isArray(heatmap.matrix) ? heatmap.matrix : [];
    if (!blocks.length || !components.length) return '';
    const max = Number(heatmap.max_value) || 1;
    return `<section class="dragon-tool-panel dragon-analysis-heatmap-section dragon-reveal"><div class="dragon-tool-panel-head"><div><span class="dragon-eyebrow">分布图</span><h2>区块 × 组件热力图</h2></div><span class="dragon-tool-note">颜色越深表示静态 ΔW 范数越高</span></div><div class="dragon-table-wrapper"><table class="dragon-analysis-heatmap"><thead><tr><th>区块</th>${components.map((item) => `<th title="${escapeAttribute(componentLabel(item))}">${escapeHtml(shortComponentLabel(item))}</th>`).join('')}</tr></thead><tbody>${blocks.map((block, rowIndex) => `<tr><th>${block}</th>${components.map((component, columnIndex) => { const value = Number(matrix[rowIndex]?.[columnIndex]) || 0; const intensity = Math.max(0, Math.min(1, value / max)); return `<td style="--dragon-analysis-intensity:${intensity.toFixed(3)}" title="区块 ${block} · ${escapeAttribute(componentLabel(component))} · ${escapeAttribute(formatNumber(value))}"><span class="visually-hidden">${escapeHtml(formatNumber(value))}</span></td>`; }).join('')}</tr>`).join('')}</tbody></table></div></section>`;
}

export function renderAnalysisEmpty(hasWeights) {
    return `<div class="dragon-empty-state dragon-weight-analysis-empty dragon-reveal"><p>${hasWeights ? '选择权重 A 开始分析，也可以开启 A / B 对比。' : '扫描目录暂无权重；仍可填写允许目录内的路径或临时上传文件。'}</p></div>`;
}

export function renderAnalysisError(message) {
    return `<div class="dragon-empty-state dragon-weight-analysis-empty dragon-reveal"><p>${escapeHtml(message)}</p></div>`;
}

function metricTile(label, value) { return `<div class="dragon-stat-tile"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`; }

function diffRows(aRows = [], bRows = [], kind) {
    const keyFor = (item) => String(item.label ?? item[kind] ?? '其他');
    const aMap = new Map(aRows.map((item) => [keyFor(item), item]));
    const bMap = new Map(bRows.map((item) => [keyFor(item), item]));
    return Array.from(new Set([...aMap.keys(), ...bMap.keys()])).map((key) => {
        const a = aMap.get(key) || {};
        const b = bMap.get(key) || {};
        return {
            label: kind === 'block' ? `区块 ${key}` : componentLabel(key),
            norm: Number(b.fro_norm || 0) - Number(a.fro_norm || 0),
            ratio: Number(b.energy_contribution ?? b.contribution ?? 0) - Number(a.energy_contribution ?? a.contribution ?? 0),
        };
    }).sort((left, right) => Math.abs(right.norm) - Math.abs(left.norm));
}

function componentLabel(value) {
    const labels = { mlp_layer1: '多层感知机输入', mlp_layer2: '多层感知机输出', self_attn_q_proj: '自注意力查询', self_attn_k_proj: '自注意力键', self_attn_v_proj: '自注意力值', self_attn_output_proj: '自注意力输出', cross_attn_q_proj: '交叉注意力查询', cross_attn_k_proj: '交叉注意力键', cross_attn_v_proj: '交叉注意力值', cross_attn_output_proj: '交叉注意力输出' };
    return labels[value] || value || '其他';
}

function shortComponentLabel(value) { return componentLabel(value).replace('多层感知机', 'MLP ').replace('自注意力', '自注意 ').replace('交叉注意力', '交叉 '); }
function basename(value) { return String(value || '').replace(/\\/g, '/').split('/').filter(Boolean).pop() || ''; }
function formatInteger(value) { const number = Number(value); return Number.isFinite(number) ? Math.round(number).toLocaleString('zh-CN') : '-'; }
function formatNumber(value) { const number = Number(value); if (!Number.isFinite(number)) return '-'; if (Math.abs(number) >= 1000) return number.toLocaleString('zh-CN', { maximumFractionDigits: 1 }); if (Math.abs(number) >= 1) return number.toFixed(2); return number === 0 ? '0' : number.toPrecision(3); }
function formatPercent(value) { const number = Number(value); return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : '-'; }
function formatSigned(value) { const number = Number(value); if (!Number.isFinite(number)) return '-'; return `${number > 0 ? '+' : ''}${formatNumber(number)}`; }
function formatSignedPercent(value) { const number = Number(value); if (!Number.isFinite(number)) return '-'; return `${number > 0 ? '+' : ''}${(number * 100).toFixed(1)}%`; }
function toneForDelta(value) { return Number(value) > 0 ? 'positive' : Number(value) < 0 ? 'negative' : 'neutral'; }
function escapeAttribute(value) { return escapeHtml(value); }
