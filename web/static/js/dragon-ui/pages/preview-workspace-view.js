/* Presentational helpers for Dragon's image and training-weight preview workspace. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { renderStatusRegion, renderToolButton, renderToolHero } from './tool-page.js?v=dragon-ui-20260814v43';
import { escapeHtml, formatBytes } from '../../shared/format.js?v=dragon-ui-20260812v35';

export function renderPreviewWorkspacePage(model) {
    const actions = [
        renderToolButton('refresh', '刷新内容', 'refresh-preview'),
        renderToolButton('settings', '路径设置', 'toggle-settings', 'dragon-btn-secondary', 'aria-expanded="false"'),
    ].join('');
    return `
        <div class="dragon-page dragon-page-wide dragon-tool-page dragon-preview-workspace" data-preview-root data-source="training">
            ${renderToolHero({
                eyebrow: '模型与系统 · 数据',
                title: '预览工作区',
                description: '集中浏览训练样张、推理输出和自定义目录，并下载当前训练保存的权重。',
                badge: '<span class="dragon-tool-count-badge" data-preview-count>0 张图片</span>',
                actions,
            })}
            ${renderStatusRegion('data-preview-status', model.error || '', model.error ? 'error' : '')}

            <section class="dragon-preview-context dragon-reveal" data-stagger="1">
                <div class="dragon-preview-context-grid">
                    <label class="dragon-field"><span class="dragon-field-label-text">浏览范围</span><select class="dragon-select" name="preview_scope" data-preview-scope><option value="latest">当前任务 / 最近一次训练</option><option value="task">单个历史任务</option><option value="group">配置分组全部训练</option></select><small>按当前运行、单次训练或同一配置分组浏览样张与权重。</small></label>
                    <label class="dragon-field" data-preview-task-wrap><span class="dragon-field-label-text">训练任务</span><select class="dragon-select" name="preview_task_id" data-preview-task><option value="">请选择历史任务</option>${renderTaskOptions(model.tasks)}</select><small>单任务模式下，样张与权重同步切换。</small></label>
                    <label class="dragon-field" data-preview-group-wrap hidden><span class="dragon-field-label-text">配置分组</span><select class="dragon-select" name="preview_group_key" data-preview-group><option value="">请选择配置分组</option>${renderGroupOptions(model.groups)}</select><small>聚合这个配置分组下的全部训练产物。</small></label>
                    <label class="dragon-field"><span class="dragon-field-label-text">时间范围</span><select class="dragon-select" name="preview_days" data-preview-days><option value="7">最近 7 天</option><option value="14">最近 14 天</option><option value="30">最近 30 天</option><option value="all" selected>全部时间</option></select><small>时间筛选仅影响图片，不影响权重列表。</small></label>
                </div>
            </section>

            <section class="dragon-preview-controlbar dragon-reveal" data-stagger="2">
                <div class="dragon-preview-source-tabs" role="group" aria-label="选择预览来源">
                    ${sourceButton('training', '训练样张', true)}
                    ${sourceButton('inference', '推理输出')}
                    ${sourceButton('custom', '自定义目录')}
                </div>
                <div class="dragon-preview-directory"><span>当前目录</span><strong class="dragon-text-mono" data-preview-directory>${escapeHtml(model.directory || '未设置')}</strong></div>
            </section>

            <section class="dragon-tool-panel dragon-preview-settings dragon-reveal" data-preview-settings-panel hidden>
                <div class="dragon-tool-panel-head"><div><span class="dragon-eyebrow">本机路径</span><h2>预览来源设置</h2></div><span class="dragon-tool-note">相对路径以项目根目录为基准</span></div>
                <form data-preview-settings-form>
                    <div class="dragon-preview-settings-grid">
                        ${pathField('training_dir', '训练样张目录', model.settings.training_dir, '例如：output/ckpt/sample…')}
                        ${pathField('inference_dir', '推理输出目录', model.settings.inference_dir, '例如：output/tests…')}
                        ${pathField('custom_dir', '自定义目录', model.settings.custom_dir, '例如：output/my-preview…')}
                    </div>
                    <div class="dragon-preview-settings-actions">
                        <button class="dragon-btn dragon-btn-ghost" type="button" data-preview-action="restore-defaults">恢复默认</button>
                        <button class="dragon-btn dragon-btn-primary" type="submit" ${model.error ? 'disabled' : ''}>保存路径设置</button>
                    </div>
                    ${model.error ? '<p class="dragon-tool-note dragon-preview-settings-blocked">路径读取失败时不会允许覆盖保存；请刷新页面重试。</p>' : ''}
                </form>
            </section>

            <section class="dragon-tool-panel dragon-preview-gallery-panel dragon-reveal" data-stagger="3">
                <div class="dragon-tool-panel-head">
                    <div><span class="dragon-eyebrow" data-preview-source-label>训练样张</span><h2>图片预览</h2></div>
                    <div class="dragon-preview-gallery-head-actions"><span class="dragon-tool-note" data-preview-gallery-meta>正在读取…</span><button class="dragon-btn dragon-btn-danger dragon-btn-sm" type="button" data-preview-action="delete-selected" disabled>删除所选</button></div>
                </div>
                <p class="dragon-tool-note" data-preview-delete-note hidden>配置分组聚合了多个训练目录；请切换到单个任务后删除图片。</p>
                <div class="dragon-image-grid dragon-preview-grid" data-preview-grid></div>
                <div class="dragon-empty-state" data-preview-empty><p>正在读取预览图…</p></div>
            </section>

            <section class="dragon-tool-panel dragon-preview-weight-panel dragon-reveal" data-stagger="4" data-preview-weight-panel>
                <div class="dragon-tool-panel-head">
                    <div><span class="dragon-eyebrow">训练产物</span><h2>保存的权重</h2></div>
                    <span class="dragon-tool-note" data-preview-weight-meta>正在读取…</span>
                </div>
                <div class="dragon-preview-weight-list" data-preview-weight-list></div>
                <div class="dragon-empty-state" data-preview-weight-empty><p>正在读取权重…</p></div>
            </section>
        </div>
    `;
}

function sourceButton(source, label, selected = false) {
    return `<button type="button" data-preview-source="${source}" aria-pressed="${selected}">${label}</button>`;
}

function renderTaskOptions(tasks = []) {
    return tasks.map((task) => {
        const id = task.id || '';
        const label = task.name || task.output_name || `${task.variant || '训练'} · ${task.started_at_text || id}`;
        return id ? `<option value="${escapeAttribute(id)}">${escapeHtml(label)}</option>` : '';
    }).join('');
}

function renderGroupOptions(groups = []) {
    return groups.map((group) => `<option value="${escapeAttribute(group.key)}">${escapeHtml(group.label)} · ${group.count} 次训练</option>`).join('');
}

function pathField(key, label, value, placeholder) {
    return `<label class="dragon-field"><span class="dragon-field-label-text">${label}</span><input class="dragon-input dragon-text-mono" type="text" name="${key}" autocomplete="off" spellcheck="false" value="${escapeAttribute(value)}" placeholder="${escapeAttribute(placeholder)}"></label>`;
}

export function renderPreviewImages(payload = {}) {
    const images = Array.isArray(payload.images) ? payload.images : [];
    return images.map((image) => {
        const sample = image.sample || {};
        const params = sample.parameters || {};
        const title = sample.step != null ? `Step ${sample.step}` : sample.epoch != null ? `Epoch ${sample.epoch}` : (image.name || '预览图');
        const meta = [
            image.width && image.height ? `${image.width} × ${image.height}` : '',
            sample.seed != null ? `seed ${sample.seed}` : '',
            sample.sampler || params.sample_sampler || '',
            image.size_bytes != null ? formatBytes(image.size_bytes) : '',
        ].filter(Boolean).join(' · ');
        const prompt = sample.prompt || sample.caption || '';
        const alt = prompt ? `${title}：${prompt}` : (image.name || title);
        return `
            <figure class="dragon-image-card dragon-preview-image-card" data-preview-file="${escapeAttribute(image.file || '')}">
                <label class="dragon-preview-image-select"><input type="checkbox" name="preview_image" value="${escapeAttribute(image.file || '')}" data-preview-image-select><span>选择图片</span></label>
                <a href="${escapeAttribute(image.url || '')}" target="_blank" rel="noopener" aria-label="打开原图：${escapeAttribute(image.name || title)}">
                    <img src="${escapeAttribute(image.url || '')}" alt="${escapeAttribute(alt)}" width="${escapeAttribute(image.width || 1)}" height="${escapeAttribute(image.height || 1)}" loading="lazy">
                </a>
                <figcaption class="dragon-image-card-caption">
                    <strong>${escapeHtml(title)}</strong>
                    <span>${escapeHtml(image.name || '')}</span>
                    ${meta ? `<small>${escapeHtml(meta)}</small>` : ''}
                    ${prompt ? `<p>${escapeHtml(prompt)}</p>` : ''}
                </figcaption>
            </figure>
        `;
    }).join('');
}

export function renderPreviewWeights(payload = {}) {
    const weights = Array.isArray(payload.weights) ? payload.weights : [];
    return weights.map((item) => {
        const path = item.abs_path || item.file || '';
        const stats = [
            item.epoch != null ? `Epoch ${item.epoch}` : '',
            item.steps != null ? `Step ${item.steps}` : '',
            item.mtime_text || '',
            item.size_bytes != null ? formatBytes(item.size_bytes) : '',
        ].filter(Boolean).join(' · ');
        return `
            <article class="dragon-preview-weight-item">
                <span class="dragon-preview-weight-icon">${renderIcon('layers')}</span>
                <div class="dragon-preview-weight-copy"><strong>${escapeHtml(item.name || basename(path) || '未命名权重')}</strong><span>${escapeHtml(stats || item.scope_label || '训练权重')}</span><small class="dragon-text-mono" title="${escapeAttribute(path)}">${escapeHtml(path || '-')}</small></div>
                <div class="dragon-preview-weight-actions"><button class="dragon-btn dragon-btn-ghost" type="button" data-preview-copy-weight="${escapeAttribute(path)}">复制路径</button><button class="dragon-btn dragon-btn-secondary" type="button" data-preview-hotstart-weight="${escapeAttribute(path)}">热启动</button><a class="dragon-btn dragon-btn-secondary" href="${escapeAttribute(item.download_url || previewWeightUrl(item))}" download="${escapeAttribute(item.name || 'weight.safetensors')}">${renderIcon('download', 'dragon-btn-icon')}<span>下载</span></a></div>
            </article>
        `;
    }).join('');
}

export function sourceLabel(source) {
    return { training: '训练样张', inference: '推理输出', custom: '自定义目录' }[source] || '预览图片';
}

export function effectiveDirectory(settings = {}, source) {
    if (source === 'training') return settings.effective_training_dir || settings.training_dir || '';
    if (source === 'inference') return settings.inference_dir || '';
    return settings.custom_dir || '';
}

function previewWeightUrl(item) {
    return `/api/preview/weight?file=${encodeURIComponent(item.file || '')}`;
}

function basename(value) { return String(value || '').replace(/\\/g, '/').split('/').filter(Boolean).pop() || ''; }
function escapeAttribute(value) { return escapeHtml(value); }
