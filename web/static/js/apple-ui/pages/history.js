/* Training history page: list of past training tasks with detail.
 * Fetches from /api/training/history.
 */

import { createApiClient } from '../../shared/api.js?v=apple-ui-20260812v33';

const api = createApiClient();

export async function loadHistory() {
    let data = {};
    try { data = await api('/api/training/history?limit=50'); } catch { /* server unavailable */ }
    const tasks = Array.isArray(data?.tasks) ? data.tasks : [];

    if (!tasks.length) {
        return `
            <div class="apple-page">
                <div class="apple-page-hero apple-reveal">
                    <h1>训练历史</h1>
                    <p>查看历史训练任务的配置、损失曲线和样张。</p>
                </div>
                <div class="apple-empty-state apple-reveal" data-stagger="1">
                    <p>暂无训练记录</p>
                </div>
            </div>
        `;
    }

    // Group by config group or date
    const groups = {};
    for (const task of tasks) {
        const group = task.config_group || task.variant || '\u672a\u5206\u7ec4';
        if (!groups[group]) groups[group] = [];
        groups[group].push(task);
    }

    const groupSections = Object.entries(groups).map(([name, items]) => `
        <div class="apple-section apple-reveal" data-stagger="${Math.min(Object.keys(groups).indexOf(name) + 1, 6)}">
            <h2 class="apple-section-title">${name}</h2>
            <p class="apple-section-desc">${items.length} 个任务</p>
            <ul class="apple-history-list">
                ${items.map(renderHistoryItem).join('')}
            </ul>
        </div>
    `).join('');

    return `
        <div class="apple-page apple-page-wide">
            <div class="apple-page-hero apple-reveal">
                <h1>训练历史</h1>
                <p>查看历史训练任务的配置、损失曲线和样张。共 ${tasks.length} 条记录。</p>
            </div>
            ${groupSections}
        </div>
    `;
}

function renderHistoryItem(task) {
    const name = task.output_name || task.task_name || task.id || '-';
    const state = task.status || task.state || 'unknown';
    const time = task.started_at || task.created_at || '';
    const steps = task.total_steps || task.steps || '';
    const loss = task.final_loss != null ? Number(task.final_loss).toFixed(4) : '';

    return `
        <li class="apple-history-item" data-task-id="${task.id || ''}">
            <span class="apple-history-item-name">${name}</span>
            ${loss ? `<span class="apple-history-item-loss">损失: ${loss}</span>` : ''}
            ${steps ? `<span class="apple-history-item-steps">${steps} 步</span>` : ''}
            <span class="apple-history-item-state" data-state="${state}">${stateText(state)}</span>
            <span class="apple-history-item-meta">${time}</span>
        </li>
    `;
}

function stateText(state) {
    const map = {
        idle: '空闲', running: '训练中', training: '训练中',
        queued: '排队中', completed: '已完成', error: '错误',
        stopped: '已停止', unknown: '未知',
    };
    return map[state] || state;
}
