import { captioningApi, jsonOptions } from './api.js?v=dragon-ui-20260829v12';
import { escapeAttribute, escapeHtml, showFeedback, splitTags, withBusy } from './utils.js?v=dragon-ui-20260829v12';

export function renderGovernance(job) {
    const frequencies = tagFrequencies(job);
    return `<section class="dragon-caption-governance" data-caption-governance>
        <header><div><span class="dragon-eyebrow">DATA GOVERNANCE</span><h2>批量清洗</h2></div><button class="dragon-icon-button" type="button" data-caption-governance-close aria-label="关闭">×</button></header>
        <div class="dragon-caption-governance-grid">
            <section><h3>标签词频</h3><div class="dragon-caption-frequency">${frequencies.length ? frequencies.slice(0, 80).map(([tag, count]) => `<button type="button" data-caption-frequency-tag="${escapeAttribute(tag)}"><span>${escapeHtml(tag)}</span><b>${count}</b></button>`).join('') : '<p>当前模式没有可统计的标签。</p>'}</div></section>
            <form data-caption-replace-form><h3>全局查找替换</h3><label><span>查找</span><input class="dragon-input" name="find" required></label><label><span>替换为</span><input class="dragon-input" name="replace"></label><label class="dragon-caption-check"><input type="checkbox" name="regex"><span>正则表达式</span></label><button class="dragon-btn dragon-btn-secondary" type="submit">应用到所有候选</button></form>
            <form data-caption-blacklist-form><h3>黑名单过滤</h3><label><span>标签列表</span><textarea class="dragon-textarea" name="tags" rows="5" placeholder="blurry, watermark, high quality"></textarea></label><button class="dragon-btn dragon-btn-secondary" type="submit">从所有图片删除</button></form>
        </div>
    </section>`;
}

export function bindGovernance(root, state, actions) {
    root.querySelector('[data-caption-governance-close]')?.addEventListener('click', () => { state.governanceOpen = false; actions.renderWorkspace(); });
    root.querySelector('[data-caption-replace-form]')?.addEventListener('submit', (event) => transform(event, root, state, actions, {
        action: 'replace', find: event.currentTarget.elements.find.value, replace: event.currentTarget.elements.replace.value, regex: event.currentTarget.elements.regex.checked,
    }));
    root.querySelector('[data-caption-blacklist-form]')?.addEventListener('submit', (event) => transform(event, root, state, actions, {action: 'blacklist', tags: event.currentTarget.elements.tags.value}));
    root.querySelectorAll('[data-caption-frequency-tag]').forEach((button) => button.addEventListener('contextmenu', (event) => {
        event.preventDefault();
        if (window.confirm(`从所有图片中删除“${button.dataset.captionFrequencyTag}”？`)) transform(event, root, state, actions, {action: 'remove_tag', tag: button.dataset.captionFrequencyTag});
    }));
}

function tagFrequencies(job) {
    if (!job || !['tags', 'mixed'].includes(job.output_mode)) return [];
    const counts = new Map();
    (job.results || []).forEach((item) => {
        const source = job.output_mode === 'mixed' ? String(item.proposed_caption || '').split(/\nTags:\s*/i).slice(1).join(',') : item.proposed_caption;
        splitTags(source).forEach((tag) => counts.set(tag, (counts.get(tag) || 0) + 1));
    });
    return [...counts.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]));
}

async function transform(event, root, state, actions, payload) {
    event.preventDefault();
    const button = event.currentTarget?.querySelector?.('button[type="submit"]');
    await withBusy(button, async () => {
        try {
            const response = await captioningApi(`/jobs/${encodeURIComponent(state.selectedJobId)}/transform`, jsonOptions('POST', payload));
            state.selectedJob = response.job;
            actions.renderWorkspace();
            showFeedback(root, `已更新 ${response.changed} 张图片`, 'success');
        } catch (error) { showFeedback(root, error.message, 'error'); }
    });
}
