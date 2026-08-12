/* Environment check page: render the service's grouped check contract. */

import { createApiClient } from '../../shared/api.js?v=apple-ui-20260812v33';
import { escapeHtml } from '../../shared/format.js?v=apple-ui-20260812v33';

const api = createApiClient();

export async function loadEnvironment() {
    const data = await api('/api/environment/check');
    if (!data || data.ok === false && !Array.isArray(data.groups)) {
        return renderError(data?.error || '环境检测失败');
    }

    const platform = data.platform || {};
    const summary = data.summary || {};
    const groups = Array.isArray(data.groups) ? data.groups : [];
    return `
        <div class="apple-page apple-page-wide apple-environment-page">
            <div class="apple-page-hero apple-reveal">
                <h1>环境检测</h1>
                <p>核对项目解释器、依赖、模型路径、GPU 与训练输出目录。</p>
            </div>

            <div class="apple-metrics-grid apple-reveal" data-stagger="1">
                ${metricTile('检查项', summary.checks)}
                ${metricTile('错误', summary.errors, Number(summary.errors) ? 'error' : 'ok')}
                ${metricTile('警告', summary.warnings, Number(summary.warnings) ? 'warning' : 'ok')}
                ${metricTile('Python', platform.python_version || '-')}
                ${metricTile('计算环境', cudaTrackLabel(platform.cuda_track))}
                ${metricTile('系统', systemLabel(platform.system, platform.platform))}
            </div>

            <section class="apple-section apple-reveal" data-stagger="2">
                <div class="apple-section-header-row">
                    <div>
                        <span class="apple-eyebrow">检测明细</span>
                        <h2 class="apple-section-title">${data.ok ? '训练环境已通过必要检查' : '训练环境需要处理'}</h2>
                    </div>
                    <p class="apple-section-desc">${escapeHtml(platform.project_python || platform.web_executable || '')}</p>
                </div>
                <div class="apple-environment-groups">
                    ${groups.map(renderGroup).join('') || '<div class="apple-empty-state"><p>没有收到环境检测结果。</p></div>'}
                </div>
            </section>
        </div>
    `;
}

function renderGroup(group) {
    const checks = Array.isArray(group.checks) ? group.checks : [];
    return `
        <section class="apple-environment-group">
            <h3>${escapeHtml(group.title || group.key || '其他')}</h3>
            <div class="apple-environment-checks">
                ${checks.map(renderCheck).join('')}
            </div>
        </section>
    `;
}

function renderCheck(check) {
    const level = ['ok', 'warning', 'error'].includes(check.level) ? check.level : 'warning';
    const details = [check.path, check.detail, check.hint].filter(Boolean);
    return `
        <div class="apple-environment-check" data-level="${level}">
            <span class="apple-environment-indicator" aria-hidden="true"></span>
            <div>
                <strong>${escapeHtml(check.message || check.key || '检测项')}</strong>
                ${details.map((detail) => `<small>${escapeHtml(detail)}</small>`).join('')}
            </div>
        </div>
    `;
}

function metricTile(label, value, tone = '') {
    return `<div class="apple-metric-tile" ${tone ? `data-tone="${tone}"` : ''}><div class="apple-metric-value">${escapeHtml(value ?? '-')}</div><div class="apple-metric-label">${label}</div></div>`;
}

function cudaTrackLabel(value) {
    const track = String(value || '').trim();
    if (!track || track === 'unknown') return '未识别';
    return track.toUpperCase();
}

function systemLabel(system, platform) {
    if (system === 'Darwin') return 'macOS';
    if (system === 'Windows') return 'Windows';
    if (system === 'Linux') return 'Linux';
    return system || platform || '-';
}

function renderError(message) {
    return `<div class="apple-page"><div class="apple-page-hero"><h1>环境检测</h1></div><div class="apple-empty-state"><p>${escapeHtml(message)}</p></div></div>`;
}
