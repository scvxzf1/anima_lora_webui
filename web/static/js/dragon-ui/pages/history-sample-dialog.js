/* Sample generation-parameter dialog for the Dragon history artifacts tab.
 * Renders the prompt, raw prompt, sampler parameters and source metadata for a
 * training sample image inside a native <dialog>, driven by the already loaded
 * /api/preview/images payload so no extra network request is needed.
 */

import { escapeHtml, formatBytes } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { copyText } from '../../shared/dom.js?v=dragon-ui-20260812v35';
export function bindHistorySampleDialog(root, images) {
    const dialog = root.querySelector('[data-history-sample-dialog]');
    if (!dialog) return () => {};
    const list = Array.isArray(images) ? images : [];
    const openButtons = Array.from(root.querySelectorAll('[data-history-sample-open]'));
    let activeIndex = -1;

    const renderContent = (image) => {
        const sample = image.sample || {};
        const parameters = sample.parameters || {};
        const source = sample.source || {};
        const title = sample.step != null ? `Step ${sample.step}` : (image.name || '训练样张');
        const prompt = String(sample.prompt || '');
        const negativePrompt = String(sample.negative_prompt || '');
        const rawPrompt = String(sample.raw_prompt || prompt);
        const width = parameters.width ?? image.width;
        const height = parameters.height ?? image.height;
        const resolution = width != null && height != null ? `${width} × ${height}` : '';
        dialog.dataset.samplePrompt = rawPrompt;

        const meta = dialog.querySelector('[data-history-sample-meta]');
        if (meta) meta.textContent = [
            title,
            image.mtime_text || '',
            image.size_bytes != null ? formatBytes(image.size_bytes) : '',
        ].filter(Boolean).join(' · ');

        const body = dialog.querySelector('[data-history-sample-body]');
        if (body) body.innerHTML = `
            <div class="dragon-sample-detail-view">
                <img src="${escapeHtml(image.url || '')}" alt="${escapeHtml(title)}">
            </div>
            <div class="dragon-sample-detail-info">
                ${samplePromptBlock('提示词', prompt)}
                ${negativePrompt ? samplePromptBlock('负向提示词', negativePrompt) : ''}
                ${samplePromptBlock('原始提示词', rawPrompt)}
                <section class="dragon-sample-detail-block">
                    <h3>采样参数</h3>
                    <dl class="dragon-sample-detail-params">
                        ${paramRow('分辨率', resolution)}
                        ${paramRow('采样步数', parameters.sample_steps)}
                        ${paramRow('引导系数', parameters.guidance_scale)}
                        ${paramRow('种子', parameters.seed ?? sample.seed)}
                        ${paramRow('采样器', parameters.sample_sampler || sample.sampler)}
                        ${paramRow('生成时间', sample.generated_at_text)}
                    </dl>
                </section>
                ${source.prompt_file ? `<section class="dragon-sample-detail-block"><h3>来源</h3><dl class="dragon-sample-detail-params">${paramRow('提示词文件', source.prompt_file)}${paramRow('文件名', image.name)}</dl></section>` : ''}
            </div>
        `;

        const openLink = dialog.querySelector('[data-history-sample-action="open"]');
        if (openLink) openLink.href = image.url || '#';
        const downloadLink = dialog.querySelector('[data-history-sample-action="download"]');
        if (downloadLink) {
            downloadLink.href = image.url || '#';
            downloadLink.download = image.name || 'sample.png';
        }
    };

    const openAt = (index) => {
        const normalizedIndex = ((Number(index) % list.length) + list.length) % list.length;
        const image = list[normalizedIndex];
        if (!image) return;
        activeIndex = normalizedIndex;
        renderContent(image);
        const position = dialog.querySelector('[data-history-sample-position]');
        if (position) position.textContent = `${activeIndex + 1} / ${list.length}`;
        if (!dialog.open) dialog.showModal();
    };

    const close = () => {
        if (dialog.open) dialog.close('cancel');
    };

    const bindings = [];
    openButtons.forEach((button) => {
        const handler = () => openAt(button.dataset.historySampleOpen);
        button.addEventListener('click', handler);
        bindings.push([button, handler]);
    });
    dialog.querySelectorAll('[data-history-sample-action="close"]').forEach((button) => {
        const handler = () => close();
        button.addEventListener('click', handler);
        bindings.push([button, handler]);
    });
    const previousButton = dialog.querySelector('[data-history-sample-action="previous"]');
    const nextButton = dialog.querySelector('[data-history-sample-action="next"]');
    const showPrevious = () => openAt(activeIndex - 1);
    const showNext = () => openAt(activeIndex + 1);
    previousButton?.addEventListener('click', showPrevious);
    nextButton?.addEventListener('click', showNext);
    if (previousButton) bindings.push([previousButton, showPrevious]);
    if (nextButton) bindings.push([nextButton, showNext]);
    const backdropHandler = (event) => {
        if (event.target === dialog) close();
    };
    dialog.addEventListener('click', backdropHandler);
    const keyHandler = (event) => {
        if (!dialog.open) return;
        if (event.key === 'ArrowLeft') openAt(activeIndex - 1);
        if (event.key === 'ArrowRight') openAt(activeIndex + 1);
    };
    document.addEventListener('keydown', keyHandler);

    const copyButton = dialog.querySelector('[data-history-sample-action="copy"]');
    if (copyButton) {
        const copyHandler = async () => {
            const raw = String(dialog.dataset.samplePrompt || '').trim();
            const label = copyButton.querySelector('span');
            const original = label?.textContent || '复制提示词';
            copyButton.disabled = true;
            try {
                await copyText(raw || '');
                if (label) label.textContent = '已复制';
            } catch {
                if (label) label.textContent = '复制失败';
            } finally {
                window.setTimeout(() => {
                    if (!copyButton.isConnected) return;
                    if (label) label.textContent = original;
                    copyButton.disabled = false;
                }, 1400);
            }
        };
        copyButton.addEventListener('click', copyHandler);
        bindings.push([copyButton, copyHandler]);
    }

    return () => {
        bindings.forEach(([element, handler]) => element.removeEventListener('click', handler));
        dialog.removeEventListener('click', backdropHandler);
        document.removeEventListener('keydown', keyHandler);
    };
}

function samplePromptBlock(title, text) {
    return `
        <section class="dragon-sample-detail-block">
            <h3>${escapeHtml(title)}</h3>
            <pre>${escapeHtml(text || '（未记录）')}</pre>
        </section>
    `;
}

function paramRow(label, value) {
    const text = value == null || String(value).trim() === '' ? '（未记录）' : String(value);
    return `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(text)}</dd></div>`;
}
