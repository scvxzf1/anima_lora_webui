/* Main Dragon tagging workspace view. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import {
    observeTaggingImageSentinel,
    renderTaggingSource,
    syncTaggingSourceState,
} from './tagging-source-view.js?v=dragon-ui-20260831v6';

export function renderTaggingView(root, state) {
    const gridScrollTop = root.querySelector?.('[data-tagging-image-grid]')?.scrollTop || state.gridScrollTop || 0;
    root.innerHTML = renderPage(state);
    const grid = root.querySelector?.('[data-tagging-image-grid]');
    if (grid) grid.scrollTop = gridScrollTop;
    state.reconnectImageObserver?.();
}

export function mountTaggingView(root, state, actions) {
    renderTaggingView(root, state);
    const controller = new AbortController();
    const options = { signal: controller.signal };
    let stopObserver = () => {};
    state.reconnectImageObserver = () => {
        stopObserver();
        stopObserver = observeTaggingImageSentinel(root, () => actions.loadMoreImages?.());
    };
    state.reconnectImageObserver();
    root.addEventListener('click', (event) => handleClick(root, actions, event), options);
    root.addEventListener('change', (event) => handleChange(actions, event), options);
    root.addEventListener('input', (event) => handleInput(actions, event), options);
    root.addEventListener('submit', (event) => handleSubmit(actions, event), options);
    root.addEventListener('toggle', (event) => handleToggle(actions, event), { ...options, capture: true });
    return () => {
        stopObserver();
        state.reconnectImageObserver = null;
        controller.abort();
    };
}

export function syncTaggingSelectionView(root, state) {
    syncTaggingSourceState(root, state);
    root?.querySelectorAll?.('[data-tagging-selected-count]').forEach((node) => {
        node.textContent = String(state.selectedFiles.size);
    });
    const note = root?.querySelector?.('.dragon-tagging-selection-note');
    if (note) note.textContent = state.selectedFiles.size ? `将发送 ${state.selectedFiles.size} 张图片` : '尚未选择图片';
    const submit = root?.querySelector?.('[data-tagging-submit]');
    if (submit) submit.disabled = !state.selectedFiles.size || state.submitting || ['queued', 'running'].includes(state.job?.state) || !state.systemPrompt.trim() || !state.userPrompt.trim();
}

function renderPage(state) {
    const selectedCount = state.selectedFiles.size;
    const jobBusy = state.submitting || ['queued', 'running'].includes(state.job?.state);
    return `<div class="dragon-page dragon-page-wide dragon-caption-page dragon-tagging-page" data-caption-page data-tagging-page>
        <header class="dragon-tagging-hero dragon-reveal">
            <div><span class="dragon-eyebrow">外部模型工作台</span><h1>打标工作台</h1><p>选择图片，生成候选标注，审阅后写入同名 TXT。</p></div>
            <div class="dragon-tagging-hero-actions">
                <a class="dragon-btn dragon-btn-secondary" href="#dataset-editor">${renderIcon('database', 'dragon-btn-icon')}<span>回到数据集</span></a>
                <button class="dragon-icon-button" type="button" data-tagging-settings-open aria-label="打开外部 API 设置" title="外部 API 设置" ${jobBusy ? 'disabled' : ''}>${renderIcon('settings')}</button>
            </div>
        </header>
        <section class="dragon-tagging-context dragon-reveal" data-stagger="1" aria-label="打标上下文">
            <div><span>当前数据集</span><strong class="dragon-text-mono" title="${escapeAttribute(state.datasetFile)}">${escapeHtml(shortName(state.datasetFile) || '未选择')}</strong></div>
            <div><span>图片组</span><strong>${state.datasetFile ? `第 ${Number(state.datasetIndex) + 1} 组` : '未选择'}</strong></div>
            <div><span>图片</span><strong><b data-tagging-selected-count>${selectedCount}</b> 已选 · <span data-tagging-image-load-status>${imageLoadStatusText(state)}</span></strong></div>
            <div><span>外部 API</span><strong class="dragon-status-badge" data-state="${providerReady(state) ? 'active' : 'idle'}"><i aria-hidden="true"></i><b>${escapeHtml(state.settings?.model || '未配置模型')}</b></strong></div>
        </section>
        ${renderFeedback(state)}
        <div class="dragon-tagging-workspace" data-stagger="2">
            ${renderTaggingSource(state, { jobBusy })}
            <section class="dragon-tagging-run dragon-section" aria-labelledby="tagging-run-title">
                <header class="dragon-tagging-panel-head">
                    <div><span class="dragon-eyebrow">PROMPT</span><h2 id="tagging-run-title">生成与审阅</h2><p>${escapeHtml(providerState(state))}</p></div>
                    <div class="dragon-tagging-tool-actions">
                        <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-tagging-open-tool="captioning-prompts">${renderIcon('settings', 'dragon-btn-icon')}<span>提示词预设</span></button>
                        <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-tagging-open-tool="captioning-results" ${state.job?.id ? '' : 'disabled'}>${renderIcon('list', 'dragon-btn-icon')}<span>最终结果</span></button>
                        <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-tagging-open-tool="captioning-logs">${renderIcon('activity', 'dragon-btn-icon')}<span>打标日志</span></button>
                    </div>
                </header>
                <form class="dragon-tagging-prompt-form" data-tagging-submit-form>
                    <div class="dragon-tagging-preset-row">
                        <label class="dragon-field"><span>提示词预设</span><select class="dragon-select" data-tagging-prompt-preset><option value="">自定义</option>${promptPresetOptions(state)}</select></label>
                        <button class="dragon-icon-button" type="button" data-tagging-open-tool="captioning-prompts" aria-label="管理提示词预设" title="管理提示词预设">${renderIcon('settings')}</button>
                    </div>
                    <div class="dragon-tagging-prompt-grid">
                        <label class="dragon-field"><span>系统提示词</span><textarea class="dragon-textarea" name="system_prompt" rows="6" data-tagging-system-prompt required>${escapeHtml(state.systemPrompt)}</textarea></label>
                        <label class="dragon-field"><span>用户提示词</span><textarea class="dragon-textarea" name="user_prompt" rows="6" data-tagging-user-prompt required>${escapeHtml(state.userPrompt)}</textarea></label>
                    </div>
                    <div class="dragon-tagging-run-actions"><span class="dragon-tagging-selection-note">${selectedCount ? `将发送 ${selectedCount} 张图片` : '尚未选择图片'}</span><button class="dragon-btn dragon-btn-primary" type="submit" data-tagging-submit ${selectedCount && !jobBusy && state.systemPrompt.trim() && state.userPrompt.trim() ? '' : 'disabled'}>${renderIcon('wand', 'dragon-btn-icon')}<span>${jobBusy ? '正在调用…' : '发送到外部模型'}</span></button></div>
                </form>
                ${renderJobSummary(state)}
            </section>
        </div>
        ${renderSettingsDialog(state)}
    </div>`;
}

function renderFeedback(state) {
    return `${state.error ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="error" role="alert">${escapeHtml(state.error)}</div>` : ''}${state.notice ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="success" role="status">${escapeHtml(state.notice)}</div>` : ''}`;
}

function renderJobSummary(state) {
    const job = state.job;
    if (!job) return '<div class="dragon-tagging-job-empty"><span class="dragon-empty-state-icon">&#183;</span><strong>暂无打标任务</strong></div>';
    const total = Number(job.total || job.items?.length || 0);
    const completed = Number(job.completed || 0);
    const failed = Number(job.failed || 0);
    const canceled = Number(job.canceled || 0);
    const progress = total ? Math.round((completed + failed + canceled) * 100 / total) : 0;
    const busy = ['queued', 'running'].includes(job.state);
    return `<details class="dragon-tagging-job-summary" data-tagging-job-details>
        <summary><span><span class="dragon-eyebrow">JOB ${escapeHtml(job.id || '')}</span><strong>${escapeHtml(jobStateLabel(job.state))}</strong><small>${completed}/${total} 完成${failed ? ` · ${failed} 失败` : ''}</small></span><span class="dragon-tagging-progress" role="progressbar" aria-label="打标任务进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress}"><i style="width:${progress}%"></i></span>${renderIcon('chevronDown')}</summary>
        <div class="dragon-tagging-job-summary-body">
            ${job.error ? `<p class="dragon-tagging-job-error" role="alert">${escapeHtml(job.error)}</p>` : ''}
            <dl><div><dt>状态</dt><dd>${escapeHtml(jobStateLabel(job.state))}</dd></div><div><dt>候选结果</dt><dd>${completed}</dd></div><div><dt>失败</dt><dd>${failed}</dd></div><div><dt>写回格式</dt><dd>同名 .txt</dd></div></dl>
            <div class="dragon-tagging-job-actions">${busy ? `<button class="dragon-btn dragon-btn-danger dragon-btn-sm" type="button" data-tagging-cancel>${renderIcon('stop', 'dragon-btn-icon')}<span>停止任务</span></button>` : `<button class="dragon-icon-button" type="button" data-tagging-refresh-job aria-label="刷新任务" title="刷新任务">${renderIcon('refresh')}</button>`}<button class="dragon-btn dragon-btn-primary dragon-btn-sm" type="button" data-tagging-open-tool="captioning-results">${renderIcon('list', 'dragon-btn-icon')}<span>打开最终结果</span></button></div>
        </div>
    </details>`;
}

function renderSettingsDialog(state) {
    const settings = state.settings || {};
    return `<dialog class="dragon-tagging-settings-dialog" data-tagging-settings-dialog aria-labelledby="tagging-settings-title"><form method="dialog" class="dragon-tagging-settings-form" data-tagging-settings-form aria-busy="${state.testingProvider ? 'true' : 'false'}">
        <header><div><span class="dragon-eyebrow">PROVIDER</span><h2 id="tagging-settings-title">外部 API 设置</h2><p>API Key 仅在服务端保存。</p></div><button class="dragon-icon-button" type="button" data-tagging-settings-close aria-label="关闭设置" title="关闭">${renderIcon('x')}</button></header>
        <div class="dragon-tagging-settings-grid">
            <label class="dragon-field dragon-tagging-field-wide"><span>兼容 API 地址</span><input class="dragon-input" type="url" name="base_url" value="${escapeAttribute(settings.base_url || '')}" required></label>
            <label class="dragon-field"><span>模型</span><input class="dragon-input" type="text" name="model" value="${escapeAttribute(settings.model || '')}" required></label>
            <label class="dragon-field"><span>API Key ${settings.api_key_configured ? '<small>（留空保持）</small>' : ''}</span><input class="dragon-input" type="password" name="api_key" value="" autocomplete="new-password"></label>
            <label class="dragon-field"><span>请求超时（秒）</span><input class="dragon-input" type="number" name="timeout_seconds" min="5" max="900" step="1" value="${Number(settings.timeout_seconds || 120)}"></label>
            <label class="dragon-field"><span>失败重试次数</span><input class="dragon-input" type="number" name="retry_count" min="0" max="6" step="1" value="${Number(settings.retry_count ?? 2)}"></label>
            <label class="dragon-field"><span>重试间隔（秒）</span><input class="dragon-input" type="number" name="retry_interval_seconds" min="0" max="60" step="0.1" value="${Number(settings.retry_interval_seconds ?? 1.5)}"></label>
            <label class="dragon-field"><span>并发上限</span><input class="dragon-input" type="number" name="concurrency" min="1" max="8" step="1" value="${Number(settings.concurrency || 2)}"></label>
            <label class="dragon-tagging-check"><input type="checkbox" name="allow_private_network" ${settings.allow_private_network ? 'checked' : ''}><span>允许私有网络 API</span></label>
            <label class="dragon-tagging-check"><input type="checkbox" name="clear_api_key"><span>清除已保存 Key</span></label>
        </div>
        <footer><span class="dragon-tagging-settings-feedback" data-tagging-settings-feedback role="status" aria-live="polite"></span><button class="dragon-btn dragon-btn-secondary" type="button" data-tagging-test="ping" ${state.testingProvider ? 'disabled' : ''}>${renderIcon('activity', 'dragon-btn-icon')}<span>测试连通</span></button><button class="dragon-btn dragon-btn-secondary" type="button" data-tagging-test="actual" ${state.testingProvider ? 'disabled' : ''}><span>实际调用</span></button><button class="dragon-btn dragon-btn-primary" type="submit" ${state.testingProvider ? 'disabled' : ''}>${renderIcon('save', 'dragon-btn-icon')}<span>保存设置</span></button></footer>
    </form></dialog>`;
}

function handleClick(root, actions, event) {
    const target = event.target.closest?.('[data-tagging-settings-open], [data-tagging-settings-close], [data-tagging-refresh], [data-tagging-load-more], [data-tagging-select-all], [data-tagging-clear], [data-tagging-cancel], [data-tagging-refresh-job], [data-tagging-test], [data-tagging-open-tool]');
    if (!target || !root.contains(target)) return;
    if (target.matches('[data-tagging-settings-open]')) return openDialog(root.querySelector('[data-tagging-settings-dialog]'));
    if (target.matches('[data-tagging-settings-close]')) return closeDialog(root.querySelector('[data-tagging-settings-dialog]'));
    if (target.matches('[data-tagging-refresh]')) return run(actions.refreshImages);
    if (target.matches('[data-tagging-load-more]')) return run(actions.loadMoreImages);
    if (target.matches('[data-tagging-select-all]')) return run(actions.selectAll);
    if (target.matches('[data-tagging-clear]')) return run(actions.clearSelection);
    if (target.matches('[data-tagging-cancel]')) return run(actions.cancelJob);
    if (target.matches('[data-tagging-refresh-job]')) return run(actions.refreshJob);
    if (target.matches('[data-tagging-test]')) return run(() => actions.testProvider(target.dataset.taggingTest));
    if (target.matches('[data-tagging-open-tool]')) return actions.openTool?.(target.dataset.taggingOpenTool);
}

function handleChange(actions, event) {
    const target = event.target;
    if (target.matches('[data-tagging-dataset]')) return run(() => actions.selectDataset(target.value));
    if (target.matches('[data-tagging-index]')) return run(() => actions.selectIndex(Number(target.value)));
    if (target.matches('[data-tagging-source]')) return run(() => actions.selectSource(target.value));
    if (target.matches('[data-tagging-image]')) return run(() => actions.toggleImage(target.dataset.file, target.checked));
    if (target.matches('[data-tagging-prompt-preset]')) return run(() => actions.applyPromptPreset(target.value));
}

function handleInput(actions, event) {
    if (event.target.matches('[data-tagging-system-prompt]')) actions.updatePromptDraft?.('systemPrompt', event.target.value);
    if (event.target.matches('[data-tagging-user-prompt]')) actions.updatePromptDraft?.('userPrompt', event.target.value);
}

function handleToggle(actions, event) {
    if (event.target.matches?.('[data-tagging-source-details]')) actions.setSourceExpanded?.(event.target.open);
}

function handleSubmit(actions, event) {
    const form = event.target;
    if (form.matches('[data-tagging-submit-form]')) {
        event.preventDefault();
        return run(() => actions.submitJob({ systemPrompt: form.elements.system_prompt?.value || '', userPrompt: form.elements.user_prompt?.value || '' }));
    }
    if (!form.matches('[data-tagging-settings-form]')) return;
    event.preventDefault();
    const data = Object.fromEntries(new FormData(form).entries());
    for (const key of ['timeout_seconds', 'retry_count', 'retry_interval_seconds', 'concurrency']) data[key] = Number(data[key]);
    data.allow_private_network = form.elements.allow_private_network?.checked === true;
    data.clear_api_key = form.elements.clear_api_key?.checked === true;
    run(() => actions.saveSettings(data));
}

function promptPresetOptions(state) {
    return (state.promptPresets || []).map((preset) => `<option value="${escapeAttribute(preset.id)}" ${preset.id === state.currentPresetId ? 'selected' : ''}>${escapeHtml(preset.name)}</option>`).join('');
}

function providerReady(state) {
    return Boolean(state.settings?.base_url && state.settings?.model);
}

function providerState(state) {
    if (!providerReady(state)) return '请先配置外部 API';
    return state.settings.api_key_configured ? `${state.settings.model} · Key 已配置` : `${state.settings.model} · 未配置 Key`;
}

function jobStateLabel(value) {
    return { queued: '任务排队中', running: '正在调用外部 API', completed: '全部完成', partial: '部分完成', failed: '任务失败', canceled: '任务已停止' }[value] || value || '未知状态';
}

function openDialog(dialog) {
    if (!dialog) return;
    dialog.__dragonOpener = globalThis.document?.activeElement || null;
    const restore = () => {
        const opener = dialog.__dragonOpener;
        dialog.__dragonOpener = null;
        opener?.focus?.({ preventScroll: true });
    };
    dialog.addEventListener('close', restore, { once: true });
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
}

function closeDialog(dialog) {
    if (!dialog) return;
    if (typeof dialog.close === 'function') dialog.close();
    else {
        dialog.removeAttribute('open');
        dialog.__dragonOpener?.focus?.({ preventScroll: true });
        dialog.__dragonOpener = null;
    }
}

function run(fn) {
    if (typeof fn !== 'function') return;
    Promise.resolve().then(fn).catch((error) => console.error('[dragon-tagging]', error));
}

function shortName(value) {
    const clean = String(value || '').replaceAll('\\', '/');
    return clean.split('/').pop() || clean;
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}

function imageLoadStatusText(state) {
    if (state.loadingImages) return '正在扫描…';
    return state.imagesLoaded ? `${Number(state.total || 0)} 张` : '展开后扫描';
}
