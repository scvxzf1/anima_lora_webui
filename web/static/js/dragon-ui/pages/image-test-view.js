/* Presentation helpers for Dragon's inference verification workspace. */

import { escapeHtml, formatBytes } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { renderStatusRegion, renderToolHero } from './tool-page.js?v=dragon-ui-20260814v43';

export function renderImageTestPage(state, images, options) {
    const cfg = state.config || {};
    const status = state.status || {};
    const running = Boolean(state.running);
    const stateLabel = imageTestStateLabel(status);
    const weightOptions = state.weightItems.map((item) => {
        const path = item.abs_path || item.path || item.file || item.absolute_path || '';
        return path ? `<option value="${escapeAttribute(path)}" label="${escapeAttribute(item.name || path)}"></option>` : '';
    }).join('');
    const family = String(state.settings?.model_family || cfg.model_family || '').toLowerCase();
    const attnOptions = family === 'krea2_raw'
        ? options.attention.filter(([value]) => ['torch', 'flash'].includes(value))
        : options.attention;
    const gpuOptions = state.gpus.map((gpu) => {
        const index = gpu.index ?? gpu.gpu_index ?? '';
        const label = gpu.label || gpu.name || `GPU ${index}`;
        return `<option value="${escapeAttribute(index)}">${escapeHtml(label)}</option>`;
    }).join('');
    const savePath = state.settings?.image_test_save_root || status.output_dir || 'output/tests';
    const latestRequest = status.last_request || {};
    const blocked = Boolean(state.blockingError);

    return `
        <div class="dragon-page dragon-tool-page dragon-image-test-page">
            ${renderToolHero({
                eyebrow: '模型与系统 · 推理验证',
                title: '生图测试',
                description: '使用当前训练配置与已生成权重运行真实推理，并在同一工作区查看状态、日志与结果。',
                badge: `<span class="dragon-image-test-state" data-state="${escapeAttribute(status.status || 'idle')}"><span aria-hidden="true"></span>${stateLabel}</span>`,
                actions: `${toolButton('refresh', '刷新状态', 'refresh')}${running ? toolButton('stop', '停止推理', 'stop', 'dragon-btn-danger') : ''}`,
            })}
            ${state.blockingError ? renderStatusRegion('data-image-load-error', `无法启动推理：${state.blockingError}`, 'error') : ''}
            ${state.warning ? renderStatusRegion('data-image-load-warning', state.warning, 'warning') : ''}

            <section class="dragon-stat-grid dragon-image-test-stats" aria-label="推理运行摘要">
                ${statTile('当前状态', stateLabel)}
                ${statTile('输出图片', formatInteger(status.output_count ?? images.length))}
                ${statTile('开始时间', status.started_at_text || '尚未启动')}
                ${statTile('结束时间', status.finished_at_text || (running ? '运行中' : '-'))}
                ${statTile('模型族', family || 'Anima')}
                ${statTile('输出目录', savePath, 'dragon-text-mono')}
            </section>

            <div class="dragon-image-test-layout">
                <form class="dragon-tool-panel dragon-image-test-form" data-image-test-form autocomplete="off">
                    <div class="dragon-tool-panel-head">
                        <div><span class="dragon-eyebrow">生成请求</span><h2>图像与采样参数</h2></div>
                        <span class="dragon-tool-note">提交前会校验底模、文本编码器、VAE 与权重路径。</span>
                    </div>

                    <section class="dragon-image-test-group" aria-labelledby="dragon-image-prompt-heading">
                        <header><h3 id="dragon-image-prompt-heading">提示词</h3><p>先描述主体、构图和风格，再按需补充反向约束。</p></header>
                        <div class="dragon-image-test-primary">
                            ${textareaField('prompt', '正向提示词', latestRequest.prompt || '', '例如：cinematic portrait, soft light…')}
                            ${textareaField('negative_prompt', '反向提示词', latestRequest.negative_prompt || '', '例如：low quality, blurry…')}
                        </div>
                    </section>

                    <section class="dragon-image-test-group" aria-labelledby="dragon-image-generation-heading">
                        <header><h3 id="dragon-image-generation-heading">生成参数</h3><p>尺寸、采样器与随机性会直接影响结果和显存占用。</p></header>
                        <div class="dragon-dataset-field-grid">
                            ${numberField('width', '宽度', latestRequest.width || cfg.resolution || 1024, '1', 64, 4096)}
                            ${numberField('height', '高度', latestRequest.height || cfg.resolution || 1024, '1', 64, 4096)}
                            ${numberField('infer_steps', '采样步数', latestRequest.infer_steps || cfg.sample_steps || 28, '1', 1, 1000)}
                            ${numberField('guidance_scale', '引导强度', latestRequest.guidance_scale ?? cfg.guidance_scale ?? 4, '0.1', 0)}
                            ${numberField('flow_shift', 'Flow Shift', latestRequest.flow_shift ?? cfg.flow_shift ?? cfg.discrete_flow_shift ?? 1, '0.1')}
                            ${numberField('seed', '随机种子', latestRequest.seed ?? '', '1')}
                            ${selectField('sampler', '采样器', normalizeChoice(latestRequest.sampler || cfg.sample_sampler, options.sampler, 'euler'), options.sampler)}
                            ${selectField('attn_mode', '注意力后端', normalizeChoice(latestRequest.attn_mode || cfg.attn_mode, attnOptions, family === 'krea2_raw' ? 'torch' : 'flash'), attnOptions)}
                        </div>
                    </section>

                    <section class="dragon-image-test-group" aria-labelledby="dragon-image-runtime-heading">
                        <header><h3 id="dragon-image-runtime-heading">模型、权重与运行设备</h3><p>不选择权重时只运行基础模型；选择权重后可设置 LoRA 强度。</p></header>
                        <div class="dragon-dataset-field-grid">
                            ${selectField('runtime_dtype', '推理精度', normalizeChoice(latestRequest.runtime_dtype || cfg.precision_preference, options.dtype, 'bf16'), options.dtype)}
                            ${selectField('text_encoder_dtype', '文本编码器精度', normalizeChoice(latestRequest.text_encoder_dtype, options.textDtype, 'same'), options.textDtype)}
                            <label class="dragon-field"><span class="dragon-field-label-text">计算设备</span><select class="dragon-select" name="gpu_index" data-image-field="gpu_index"><option value="">自动选择</option>${gpuOptions}</select></label>
                            <label class="dragon-field dragon-field-wide"><span class="dragon-field-label-text">权重文件</span><span class="dragon-image-weight-row"><input class="dragon-input dragon-text-mono" type="text" name="weight_path" list="dragon-image-weight-options" autocomplete="off" spellcheck="false" data-image-field="weight_path" value="${escapeAttribute(latestRequest.weight_path || '')}" placeholder="填写完整路径或从候选中选择…"><button class="dragon-btn dragon-btn-secondary" type="button" data-image-action="resolve-weight">校验路径</button></span><datalist id="dragon-image-weight-options">${weightOptions}</datalist><small>支持直接粘贴或拖入文件路径；开启用户目录扫描后也可只填写文件名。</small></label>
                            ${numberField('lora_multiplier', 'LoRA 强度', latestRequest.lora_multiplier ?? 1, '0.05', 0)}
                            ${textField('save_path', '输出目录', savePath)}
                        </div>
                    </section>

                    ${renderSelectiveLora(options.loraBlocks, options.loraPresets)}

                    <div class="dragon-image-test-submit-row">
                        <button class="dragon-btn dragon-btn-primary" type="submit" data-image-action="start" ${running || blocked ? 'disabled' : ''}>${renderIcon('wand', 'dragon-btn-icon')}<span>${running ? '推理进行中…' : '开始推理'}</span></button>
                        <span>生图任务与训练任务互斥，运行前请确认训练已停止。</span>
                    </div>
                    ${renderStatusRegion('data-image-feedback', status.error || '', status.error ? 'error' : '')}
                </form>

                <aside class="dragon-image-test-runtime">
                    <section class="dragon-tool-panel">
                        <div class="dragon-tool-panel-head"><div><span class="dragon-eyebrow">最近请求</span><h2>运行上下文</h2></div></div>
                        ${renderLastRequest(latestRequest)}
                    </section>
                    <section class="dragon-tool-panel dragon-image-test-log-panel">
                        <div class="dragon-tool-panel-head"><div><span class="dragon-eyebrow">进程输出</span><h2>运行日志</h2></div><span class="dragon-tool-note">最近 ${Array.isArray(status.logs) ? status.logs.length : 0} 行</span></div>
                        <pre class="dragon-image-test-log" data-image-log tabindex="0">${escapeHtml((status.logs || []).join('\n') || '暂无运行日志。')}</pre>
                    </section>
                    <details class="dragon-tool-panel dragon-image-test-command">
                        <summary>查看实际推理命令</summary>
                        <pre tabindex="0">${escapeHtml(Array.isArray(status.command) && status.command.length ? status.command.join(' ') : '当前没有可展示的推理命令。')}</pre>
                    </details>
                </aside>
            </div>

            <section class="dragon-section dragon-image-test-results dragon-reveal">
                <div class="dragon-section-header-row">
                    <div><span class="dragon-eyebrow">输出结果</span><h2 class="dragon-section-title">推理预览</h2></div>
                    <span class="dragon-section-desc">${formatInteger(images.length)} 张图片</span>
                </div>
                ${images.length ? `<div class="dragon-image-grid">${images.map(renderImage).join('')}</div>` : '<div class="dragon-empty-state"><p>暂无生成图片。完成一次推理后会在这里显示结果。</p></div>'}
            </section>
        </div>
    `;
}

function toolButton(icon, label, action, className = 'dragon-btn-secondary') {
    return `<button class="dragon-btn ${className}" type="button" data-image-action="${action}">${renderIcon(icon, 'dragon-btn-icon')}<span>${label}</span></button>`;
}

function statTile(label, value, className = '') {
    return `<div class="dragon-stat-tile"><span>${label}</span><strong class="${className}">${escapeHtml(value)}</strong></div>`;
}

function renderLastRequest(request) {
    if (!request || !Object.keys(request).length) return '<div class="dragon-empty-state dragon-image-test-inline-empty"><p>还没有提交过生图请求。</p></div>';
    const rows = [
        ['尺寸', request.width && request.height ? `${request.width} × ${request.height}` : '-'],
        ['采样器', request.sampler || '-'],
        ['推理精度', request.runtime_dtype || '-'],
        ['GPU', request.gpu_label || request.device || '自动'],
        ['采样步数', request.infer_steps ?? '-'],
        ['种子', request.seed ?? '随机'],
        ['LoRA 强度', request.lora_multiplier ?? '-'],
        ['分层加载', request.anima_selective_lora ? `开启 · ${request.anima_selective_block_count ?? 0} 层` : '关闭'],
    ];
    return `<dl class="dragon-image-test-request">${rows.map(([key, value]) => `<div><dt>${key}</dt><dd>${escapeHtml(value)}</dd></div>`).join('')}</dl>${request.weight_path ? requestBlock('权重路径', request.weight_path) : ''}${request.prompt ? requestBlock('正向提示词', request.prompt) : ''}`;
}

function requestBlock(label, value) {
    return `<div class="dragon-image-test-request-block"><span>${label}</span><p>${escapeHtml(value)}</p></div>`;
}

function renderSelectiveLora(blocks, presets) {
    return `<details class="dragon-dataset-advanced dragon-image-lora-details"><summary><span>分层 LoRA 加载</span><small>高级：按模型层位单独设置倍率</small></summary><div class="dragon-selective-lora-head"><label class="dragon-check-field"><input type="checkbox" name="anima_selective_lora" data-image-field="anima_selective_lora"><span>启用分层加载</span></label>${selectField('anima_selective_preset', '层位预设', 'default', presets)}</div><div data-selective-lora-fields hidden><p class="dragon-section-desc">每个层位可独立关闭或设置 0–2 倍强度。</p><div class="dragon-lora-block-grid">${blocks.map(([key, label]) => `<label class="dragon-lora-block"><span>${label}</span><input class="dragon-input" name="lora_${escapeAttribute(key)}" type="number" min="0" max="2" step="0.05" value="1" data-lora-block="${escapeAttribute(key)}"></label>`).join('')}</div></div></details>`;
}

function numberField(key, label, value, step = '1', min = null, max = null) {
    const bounds = `${min == null ? '' : ` min="${min}"`}${max == null ? '' : ` max="${max}"`}`;
    return `<label class="dragon-field"><span class="dragon-field-label-text">${label}</span><input class="dragon-input" name="${key}" type="number" step="${step}"${bounds} inputmode="decimal" autocomplete="off" data-image-field="${key}" value="${escapeAttribute(value)}"></label>`;
}

function textField(key, label, value) {
    return `<label class="dragon-field"><span class="dragon-field-label-text">${label}</span><input class="dragon-input" name="${key}" type="text" autocomplete="off" spellcheck="false" data-image-field="${key}" value="${escapeAttribute(value)}"></label>`;
}

function selectField(key, label, value, options) {
    return `<label class="dragon-field"><span class="dragon-field-label-text">${label}</span><select class="dragon-select" name="${key}" autocomplete="off" data-image-field="${key}">${options.map(([option, text]) => `<option value="${escapeAttribute(option)}" ${String(option) === String(value) ? 'selected' : ''}>${escapeHtml(text)}</option>`).join('')}</select></label>`;
}

function textareaField(key, label, value, placeholder) {
    return `<label class="dragon-field dragon-field-wide"><span class="dragon-field-label-text">${label}</span><textarea class="dragon-textarea" name="${key}" autocomplete="off" data-image-field="${key}" placeholder="${escapeAttribute(placeholder)}">${escapeHtml(value)}</textarea></label>`;
}

function renderImage(image) {
    const sample = image.sample || {};
    const parameters = sample.parameters || {};
    const title = sample.prompt || image.name || '推理结果';
    const details = [
        image.width && image.height ? `${image.width} × ${image.height}` : '',
        sample.seed != null ? `seed ${sample.seed}` : '',
        parameters.sample_steps ? `${parameters.sample_steps} 步` : '',
        sample.sampler || parameters.sample_sampler || '',
        image.size_bytes != null ? formatBytes(image.size_bytes) : '',
    ].filter(Boolean).join(' · ');
    return `<figure class="dragon-image-card"><img src="${escapeAttribute(image.url || '')}" alt="${escapeAttribute(title)}" width="${Number(image.width) || 1024}" height="${Number(image.height) || 1024}" loading="lazy"><figcaption class="dragon-image-card-caption"><strong>${escapeHtml(image.name || '生成图片')}</strong>${details ? `<span>${escapeHtml(details)}</span>` : ''}</figcaption>${title ? `<div class="dragon-image-card-prompt">${escapeHtml(title)}</div>` : ''}</figure>`;
}

function imageTestStateLabel(status) {
    if (status.running) return '运行中';
    return { done: '已完成', error: '失败', canceled: '已停止', idle: '空闲' }[status.status] || '空闲';
}

function normalizeChoice(value, options, fallback) {
    return options.some(([option]) => String(option) === String(value)) ? String(value) : fallback;
}

function formatInteger(value) {
    const number = Number(value);
    return Number.isFinite(number) ? new Intl.NumberFormat('zh-CN').format(Math.round(number)) : '-';
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
