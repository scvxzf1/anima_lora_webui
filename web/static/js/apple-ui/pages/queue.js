/* Training queue page backed by the existing queue control routes. */

import { createApiClient } from '../../shared/api.js?v=apple-ui-20260812v33';

const api = createApiClient();

export async function loadQueue() {
    let data = {};
    let error = '';
    try {
        data = await api('/api/training/queue');
        if (data.ok === false) error = data.error || '读取训练队列失败';
    } catch (cause) {
        error = cause.message || '读取训练队列失败';
    }
    const items = Array.isArray(data.items) ? data.items : [];
    const paused = Boolean(data.paused);
    const html = renderQueue(items, paused, error);
    return { html, onMount: (root) => bindQueue(root) };
}

function renderQueue(items, paused, error) {
    const counts = countStates(items);
    const sections = [
        ['running', '运行中'], ['queued', '等待中'], ['error', '异常'],
        ['completed', '已完成'], ['canceled', '已取消'],
    ].filter(([state]) => items.some((item) => item.state === state));
    return `
        <div class="apple-page apple-page-wide">
            <div class="apple-page-hero apple-reveal">
                <span class="apple-eyebrow">训练编排</span>
                <h1>训练队列</h1>
                <p>${items.length ? `共 ${counts.total} 个任务，${counts.running} 个运行中，${counts.queued} 个等待` : '当前没有排队任务。配置完成后可以从训练配置页加入队列。'}</p>
            </div>
            ${error ? `<div class="apple-config-feedback apple-config-feedback-visible" data-tone="error">${escapeHtml(error)}</div>` : ''}
            <div class="apple-queue-actions apple-reveal" data-stagger="1">
                <button class="apple-btn apple-btn-secondary" type="button" data-queue-action="pause">${paused ? '继续队列' : '暂停队列'}</button>
                <button class="apple-btn apple-btn-ghost" type="button" data-queue-action="refresh">刷新队列</button>
            </div>
            ${sections.map(([state, label], index) => `
                <section class="apple-section apple-reveal" data-stagger="${Math.min(index + 2, 6)}">
                    <div class="apple-section-header-row"><div><span class="apple-eyebrow">任务状态</span><h2 class="apple-section-title">${label}</h2></div><span class="apple-section-desc">${counts[state]} 个任务</span></div>
                    <div class="apple-queue-list">${items.filter((item) => item.state === state).map(renderItem).join('')}</div>
                </section>
            `).join('')}
            ${!items.length ? '<div class="apple-empty-state apple-reveal" data-stagger="2"><p>队列为空</p></div>' : ''}
        </div>
    `;
}

function renderItem(item) {
    const state = String(item.state || 'unknown');
    return `
        <article class="apple-queue-item" data-item-id="${escapeAttribute(item.id || '')}">
            <div class="apple-queue-item-info"><strong class="apple-queue-item-name">${escapeHtml(item.output_name || item.variant || item.id || '未命名任务')}</strong><span class="apple-queue-item-config">${escapeHtml(item.config_file || '')}</span><span class="apple-history-item-meta">${escapeHtml(item.created_at || item.started_at || '')}</span></div>
            <div class="apple-queue-item-actions"><span class="apple-history-item-state" data-state="${escapeAttribute(state)}">${stateText(state)}</span>${state === 'error' ? '<button class="apple-btn apple-btn-ghost apple-btn-sm" type="button" data-item-action="retry">重试</button>' : ''}${state === 'queued' || state === 'running' ? '<button class="apple-btn apple-btn-ghost apple-btn-sm" type="button" data-item-action="cancel">取消</button>' : ''}</div>
        </article>
    `;
}

function bindQueue(root) {
    root.querySelector('[data-queue-action="refresh"]')?.addEventListener('click', () => window.dispatchEvent(new CustomEvent('apple-refresh-route')));
    root.querySelector('[data-queue-action="pause"]')?.addEventListener('click', async (event) => {
        const button = event.currentTarget;
        button.disabled = true;
        try {
            const data = await api('/api/training/queue/pause', { method: 'POST', body: JSON.stringify({ paused: button.textContent.includes('暂停') }) });
            if (data.ok === false) throw new Error(data.error || '更新队列状态失败');
            window.dispatchEvent(new CustomEvent('apple-refresh-route'));
        } catch (error) {
            showFeedback(root, error.message || '更新队列状态失败', 'error');
            button.disabled = false;
        }
    });
    root.querySelectorAll('[data-item-action]').forEach((button) => button.addEventListener('click', () => handleItemAction(root, button)));
}

async function handleItemAction(root, button) {
    const item = button.closest('[data-item-id]');
    const itemId = item?.dataset.itemId;
    if (!itemId) return;
    const action = button.dataset.itemAction;
    if (action === 'cancel' && !window.confirm('确认取消这个训练任务吗？运行中的任务会停止，运行文件会保留。')) return;
    button.disabled = true;
    const request = action === 'retry'
        ? api(`/api/training/queue/${encodeURIComponent(itemId)}/retry`, { method: 'POST' })
        : api(`/api/training/queue/${encodeURIComponent(itemId)}`, { method: 'DELETE' });
    try {
        const data = await request;
        if (data.ok === false) throw new Error(data.error || '队列操作失败');
        window.dispatchEvent(new CustomEvent('apple-refresh-route'));
    } catch (error) {
        showFeedback(root, error.message || '队列操作失败', 'error');
        button.disabled = false;
    }
}

function countStates(items) {
    const count = { total: items.length, running: 0, queued: 0, error: 0, completed: 0, canceled: 0 };
    items.forEach((item) => { if (Object.hasOwn(count, item.state)) count[item.state] += 1; });
    return count;
}

function stateText(state) {
    return { queued: '等待', running: '运行中', completed: '完成', error: '异常', canceled: '已取消' }[state] || '未知';
}

function showFeedback(root, message, tone) {
    const feedback = root.querySelector('.apple-config-feedback');
    if (!feedback) return;
    feedback.textContent = message;
    feedback.dataset.tone = tone;
    feedback.classList.add('apple-config-feedback-visible');
}

function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])); }
function escapeAttribute(value) { return escapeHtml(value); }
