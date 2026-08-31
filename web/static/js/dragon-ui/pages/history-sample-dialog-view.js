import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

export function renderHistorySampleDialog() {
    return `
        <dialog class="dragon-sample-detail-dialog" data-history-sample-dialog aria-labelledby="dragon-history-sample-title">
            <div class="dragon-sample-detail-shell">
                <div class="dragon-sample-detail-header">
                    <div><span class="dragon-eyebrow">训练样张</span><h2 id="dragon-history-sample-title">生成参数</h2><p data-history-sample-meta></p></div>
                    <button class="dragon-icon-button" type="button" data-history-sample-action="close" aria-label="关闭生成参数" title="关闭">${renderIcon('x')}</button>
                </div>
                <div class="dragon-sample-detail-body" data-history-sample-body></div>
                <footer class="dragon-sample-detail-footer">
                    <div class="dragon-sample-detail-navigation" aria-label="切换训练样张">
                        <button class="dragon-icon-button" type="button" data-history-sample-action="previous" aria-label="上一张样张" title="上一张">${renderIcon('chevronLeft')}</button>
                        <span data-history-sample-position></span>
                        <button class="dragon-icon-button" type="button" data-history-sample-action="next" aria-label="下一张样张" title="下一张">${renderIcon('chevronRight')}</button>
                    </div>
                    <button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-history-sample-action="copy">${renderIcon('copy', 'dragon-btn-icon')}<span>复制提示词</span></button>
                    <a class="dragon-btn dragon-btn-secondary dragon-btn-sm" href="#" data-history-sample-action="open" target="_blank" rel="noopener">${renderIcon('eye', 'dragon-btn-icon')}<span>打开原图</span></a>
                    <a class="dragon-btn dragon-btn-secondary dragon-btn-sm" href="#" data-history-sample-action="download" download="sample.png">${renderIcon('download', 'dragon-btn-icon')}<span>下载原图</span></a>
                    <button class="dragon-btn dragon-btn-primary dragon-btn-sm" type="button" data-history-sample-action="close">关闭</button>
                </footer>
            </div>
        </dialog>
    `;
}
