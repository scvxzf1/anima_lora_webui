/* Environment check page: refreshable grouped diagnostics and copyable report. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';

const api = createApiClient();

export async function loadEnvironment() {
    try {
        const data = await fetchEnvironment();
        return {
            html: renderPage(data),
            onMount: (root) => bindPage(root, data),
        };
    } catch (error) {
        return {
            html: renderError(error.message || '环境检测失败'),
            onMount: bindRetry,
        };
    }
}

async function fetchEnvironment() {
    const data = await api('/api/environment/check');
    if (!data || data.ok === false && !Array.isArray(data.groups)) {
        throw new Error(data?.error || '环境检测失败');
    }
    return data;
}

function renderPage(data) {
    const platform = data.platform || {};
    const summary = data.summary || {};
    const groups = sortGroups(Array.isArray(data.groups) ? data.groups : []);
    return `
        <div class="dragon-page dragon-page-wide dragon-tool-page dragon-environment-page">
            <header class="dragon-tool-hero dragon-reveal">
                <div>
                    <span class="dragon-eyebrow">模型与系统</span>
                    <h1>环境检测</h1>
                    <p>核对项目解释器、依赖、模型路径、GPU 与训练输出目录。错误和警告会优先显示。</p>
                </div>
                <div class="dragon-tool-actions">
                    <button class="dragon-btn dragon-btn-secondary" type="button" data-environment-action="copy">复制完整报告</button>
                    <button class="dragon-btn dragon-btn-primary" type="button" data-environment-action="refresh">刷新检测</button>
                </div>
            </header>

            <div class="dragon-environment-summary dragon-reveal" data-stagger="1">
                ${metricTile('检查项', summary.checks)}
                ${metricTile('错误', summary.errors, Number(summary.errors) ? 'error' : 'ok')}
                ${metricTile('警告', summary.warnings, Number(summary.warnings) ? 'warning' : 'ok')}
                ${metricTile('Python', platform.python_version || '-')}
                ${metricTile('计算环境', cudaTrackLabel(platform.cuda_track))}
                ${metricTile('系统', systemLabel(platform.system, platform.platform))}
            </div>

            <section class="dragon-tool-panel dragon-environment-results dragon-reveal" data-stagger="2">
                <div class="dragon-tool-panel-head">
                    <div><span class="dragon-eyebrow">检测明细</span><h2>${data.ok ? '必要检查已通过' : '需要处理环境问题'}</h2></div>
                    <span class="dragon-tool-note" data-environment-status role="status" aria-live="polite">检测完成</span>
                </div>
                <p class="dragon-environment-executable">${escapeHtml(platform.project_python || platform.web_executable || '')}</p>
                <div class="dragon-environment-groups" data-environment-groups>
                    ${groups.map(renderGroup).join('') || '<div class="dragon-empty-state"><p>没有收到环境检测结果。</p></div>'}
                </div>
            </section>
        </div>
    `;
}

function bindPage(root, initialData) {
    let reportData = initialData;
    const refreshButton = root.querySelector('[data-environment-action="refresh"]');
    const copyButton = root.querySelector('[data-environment-action="copy"]');
    if (refreshButton?.dataset.bound === 'true') return;
    if (refreshButton) refreshButton.dataset.bound = 'true';
    if (copyButton) copyButton.dataset.bound = 'true';
    refreshButton?.addEventListener('click', async () => {
        const button = root.querySelector('[data-environment-action="refresh"]');
        const currentCopyButton = root.querySelector('[data-environment-action="copy"]');
        setEnvironmentStatus(root, '正在重新检测…', 'info');
        if (button) {
            button.disabled = true;
            button.textContent = '检测中…';
        }
        if (currentCopyButton) currentCopyButton.disabled = true;
        try {
            const data = await fetchEnvironment();
            reportData = data;
            replaceEnvironmentContent(root, data);
            bindPage(root, reportData);
            setEnvironmentStatus(root, '检测完成', data.ok ? 'success' : 'warning');
        } catch (error) {
            setEnvironmentStatus(root, `${error.message || '环境检测失败'}。请检查服务连接后重试。`, 'error');
        } finally {
            if (button) {
                button.disabled = false;
                button.textContent = '刷新检测';
            }
            if (currentCopyButton) currentCopyButton.disabled = false;
        }
    });
    copyButton?.addEventListener('click', async () => {
        try {
            await copyText(formatReport(reportData));
            setEnvironmentStatus(root, '完整检测报告已复制', 'success');
        } catch {
            setEnvironmentStatus(root, '复制失败，请允许浏览器访问剪贴板后重试。', 'error');
        }
    });
}

function replaceEnvironmentContent(root, data) {
    const replacement = document.createElement('div');
    replacement.innerHTML = renderPage(data);
    const nextPage = replacement.firstElementChild;
    const currentPage = root.querySelector('.dragon-environment-page');
    if (!nextPage || !currentPage) return;
    const currentActions = currentPage.querySelector('.dragon-tool-actions');
    const nextActions = nextPage.querySelector('.dragon-tool-actions');
    if (currentActions && nextActions) nextActions.replaceWith(currentActions);
    currentPage.replaceChildren(...nextPage.childNodes);
}

function bindRetry(root) {
    root.querySelector('[data-environment-action="refresh"]')?.addEventListener('click', () => {
        window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
    });
}

function sortGroups(groups) {
    return groups.map((group) => ({
        ...group,
        checks: [...(Array.isArray(group.checks) ? group.checks : [])].sort((a, b) => levelRank(a.level) - levelRank(b.level)),
    })).sort((a, b) => groupRank(a) - groupRank(b));
}

function groupRank(group) {
    const levels = (group.checks || []).map((check) => levelRank(check.level));
    return levels.length ? Math.min(...levels) : 3;
}

function levelRank(level) {
    return level === 'error' ? 0 : level === 'warning' ? 1 : level === 'ok' ? 2 : 3;
}

function renderGroup(group) {
    const checks = Array.isArray(group.checks) ? group.checks : [];
    const issueCount = checks.filter((check) => check.level === 'error' || check.level === 'warning').length;
    return `
        <section class="dragon-environment-group">
            <header><h3>${escapeHtml(group.title || group.key || '其他')}</h3><span>${issueCount ? `${issueCount} 个需关注项` : '全部正常'}</span></header>
            <div class="dragon-environment-checks">${checks.map(renderCheck).join('')}</div>
        </section>
    `;
}

function renderCheck(check) {
    const level = ['ok', 'warning', 'error'].includes(check.level) ? check.level : 'warning';
    const details = [check.path, check.detail, check.hint].filter(Boolean);
    const levelLabel = level === 'error' ? '错误' : level === 'warning' ? '警告' : '正常';
    return `
        <article class="dragon-environment-check" data-level="${level}">
            <span class="dragon-environment-indicator" aria-hidden="true"></span>
            <div><span class="visually-hidden">${levelLabel}：</span><strong>${escapeHtml(check.message || check.key || '检测项')}</strong>${details.map((detail) => `<small>${escapeHtml(detail)}</small>`).join('')}</div>
        </article>
    `;
}

function metricTile(label, value, tone = '') {
    return `<div class="dragon-environment-stat" ${tone ? `data-tone="${tone}"` : ''}><strong>${escapeHtml(value ?? '-')}</strong><span>${label}</span></div>`;
}

function formatReport(data = {}) {
    const platform = data.platform || {};
    const summary = data.summary || {};
    const lines = [
        'Dragon trainer 环境检测报告',
        `结果：${data.ok ? '通过' : '需要处理'}`,
        `检查项：${summary.checks ?? 0}；错误：${summary.errors ?? 0}；警告：${summary.warnings ?? 0}`,
        `系统：${systemLabel(platform.system, platform.platform)}`,
        `Python：${platform.python_version || '-'}`,
        `计算环境：${cudaTrackLabel(platform.cuda_track)}`,
        `解释器：${platform.project_python || platform.web_executable || '-'}`,
        '',
    ];
    for (const group of sortGroups(Array.isArray(data.groups) ? data.groups : [])) {
        lines.push(`[${group.title || group.key || '其他'}]`);
        for (const check of group.checks || []) {
            const level = check.level === 'error' ? '错误' : check.level === 'warning' ? '警告' : '正常';
            lines.push(`- ${level}：${check.message || check.key || '检测项'}`);
            for (const detail of [check.path, check.detail, check.hint].filter(Boolean)) lines.push(`  ${detail}`);
        }
        lines.push('');
    }
    return lines.join('\n').trim();
}

async function copyText(text) {
    if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(text);
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand('copy');
    textarea.remove();
    if (!copied) throw new Error('copy failed');
}

function setEnvironmentStatus(root, message, tone = '') {
    const status = root.querySelector('[data-environment-status]');
    if (!status) return;
    status.textContent = message;
    status.dataset.tone = tone;
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
    return `
        <div class="dragon-page dragon-tool-page dragon-environment-page">
            <header class="dragon-tool-hero"><div><span class="dragon-eyebrow">模型与系统</span><h1>环境检测</h1><p>无法完成本机环境检测。</p></div><div class="dragon-tool-actions"><button class="dragon-btn dragon-btn-primary" type="button" data-environment-action="refresh">刷新检测</button></div></header>
            <div class="dragon-empty-state"><p>${escapeHtml(message)}</p><p>请确认 WebUI 服务仍在运行，然后重新检测。</p></div>
        </div>
    `;
}
