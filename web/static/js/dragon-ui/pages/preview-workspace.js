/* Preview workspace page: browse real training sample images. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { escapeHtml, formatBytes } from '../../shared/format.js?v=dragon-ui-20260812v35';

const api = createApiClient();

export async function loadPreviewWorkspace() {
    const [settings, listing] = await Promise.all([
        api('/api/preview/settings'),
        api('/api/preview/images?source=training&limit=200'),
    ]);
    const images = Array.isArray(listing?.images) ? listing.images : [];
    const message = listing?.error || listing?.message || '暂无训练样张。';
    const directory = listing?.directory || settings?.effective_training_dir || '';

    return `
        <div class="dragon-page dragon-page-wide dragon-preview-page">
            <div class="dragon-page-hero dragon-reveal">
                <h1>预览工作区</h1>
                <p>查看训练过程实际生成的样张与采样信息。</p>
            </div>

            <div class="dragon-preview-summary dragon-reveal" data-stagger="1">
                <div><span>图片数量</span><strong>${Number(listing?.count || images.length).toLocaleString('zh-CN')}</strong></div>
                <div><span>预览来源</span><strong>${escapeHtml(sourceLabel(settings?.effective_training_source))}</strong></div>
                <div><span>样张目录</span><strong class="dragon-text-mono">${escapeHtml(directory || '未设置')}</strong></div>
            </div>

            ${images.length ? `
                <section class="dragon-section dragon-reveal" data-stagger="2">
                    <div class="dragon-section-header-row">
                        <div><span class="dragon-eyebrow">训练样张</span><h2 class="dragon-section-title">最近生成</h2></div>
                        <p class="dragon-section-desc">共 ${Number(listing?.total || images.length).toLocaleString('zh-CN')} 张</p>
                    </div>
                    <div class="dragon-image-grid">
                        ${images.map(renderImage).join('')}
                    </div>
                </section>
            ` : `
                <div class="dragon-empty-state dragon-reveal" data-stagger="2"><p>${escapeHtml(message)}</p></div>
            `}
        </div>
    `;
}

function renderImage(image) {
    const sample = image.sample || {};
    const title = sample.step != null ? `步数 ${sample.step}` : (sample.epoch != null ? `轮次 ${sample.epoch}` : image.name || '训练样张');
    const parameters = sample.parameters || {};
    const details = [
        image.width && image.height ? `${image.width} x ${image.height}` : '',
        image.size_bytes != null ? formatBytes(image.size_bytes) : '',
        parameters.sample_sampler || parameters.sampler || '',
    ].filter(Boolean).join(' · ');
    const prompt = sample.prompt || sample.caption || '';
    return `
        <figure class="dragon-image-card">
            <img src="${escapeAttribute(image.url || '')}" alt="${escapeAttribute(image.name || title)}" loading="lazy">
            <figcaption class="dragon-image-card-caption"><strong>${escapeHtml(title)}</strong>${details ? `<span>${escapeHtml(details)}</span>` : ''}</figcaption>
            ${prompt ? `<div class="dragon-image-card-prompt">${escapeHtml(prompt)}</div>` : ''}
        </figure>
    `;
}

function sourceLabel(value) {
    const labels = {
        current_task: '当前训练',
        latest_run: '最近训练',
        saved_default: '默认目录',
        selected_task_missing: '所选任务未记录',
    };
    return labels[value] || '训练目录';
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
