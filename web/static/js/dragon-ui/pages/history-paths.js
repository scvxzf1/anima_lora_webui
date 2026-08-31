import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';

const HISTORY_PATH_SPECS = [
    ['基础目录', (task) => task.run_dir_abs || task.run_dir],
    ['历史目录', (task) => task.history_dir_abs || task.history_dir],
    ['本次运行目录', (task) => task.run_dir_abs || task.run_dir],
    ['实际运行配置', (task) => task.runtime_config_file],
    ['原始配置副本', (task) => task.original_config_file],
    ['运行时数据集配置', (task) => task.dataset_config_file],
    ['模型缓存目录', (task) => task.model_cache_dir],
    ['数据集缓存目录', (task) => task.dataset_cache_dir],
    ['训练结果目录', (task) => task.training_output_dir || task.output_dir],
    ['样张目录', (task) => task.sample_dir],
    ['日志目录', (task) => task.logs_dir],
    ['历史日志文件', (task) => task.logs_path],
    ['历史指标文件', (task) => task.metrics_path],
    ['系统指标文件', (task) => task.system_path],
    ['历史 TOML 快照', (task) => task.config_snapshot],
];

export function historyPathEntries(task = {}) {
    return HISTORY_PATH_SPECS.map(([label, getValue]) => ({
        label,
        path: absoluteHistoryPath(task, getValue(task)),
    })).filter((item) => item.path);
}

export function renderHistoryPathsPanel(task = {}) {
    const entries = historyPathEntries(task);
    if (!entries.length) return '';
    return `
        <details class="dragon-history-panel dragon-history-paths-panel" data-history-paths>
            <summary>
                <span class="dragon-history-paths-heading">${renderIcon('folder')}<span><small>运行文件</small><strong>文件路径</strong></span></span>
                <span class="dragon-history-paths-summary-meta"><span>${entries.length} 项</span>${renderIcon('chevronDown')}</span>
            </summary>
            <div class="dragon-history-paths-body">
                <dl class="dragon-history-path-list">${entries.map(renderHistoryPathRow).join('')}</dl>
            </div>
        </details>
    `;
}

function renderHistoryPathRow(item) {
    const path = escapeHtml(item.path);
    return `<div><dt>${escapeHtml(item.label)}</dt><dd><code title="${path}">${path}</code><button class="dragon-icon-button dragon-history-path-copy" type="button" data-history-path-copy="${path}" aria-label="复制${escapeHtml(item.label)}" title="复制路径">${renderIcon('copy')}<span class="visually-hidden">复制路径</span></button></dd></div>`;
}

function absoluteHistoryPath(task, value) {
    const path = String(value || '').trim();
    if (!path || /^(?:[A-Za-z]:[\\/]|\\\\|\/)/.test(path)) return path;
    const root = String(task.project_root_abs || '').trim().replace(/[\\/]+$/, '');
    if (!root) return path;
    const separator = root.includes('\\') && !root.includes('/') ? '\\' : '/';
    return `${root}${separator}${path.replace(/^[\\/]+/, '')}`;
}
