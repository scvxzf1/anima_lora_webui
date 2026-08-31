const TRAINING_WORKSPACE_ITEMS = Object.freeze([
    { page: 'live-training', label: '当前监控', hash: '#page/live-training' },
    { page: 'queue', label: '训练队列', hash: '#page/queue' },
    { page: 'history', label: '训练历史', hash: '#history' },
]);

export function renderTrainingWorkspaceNav(pageType) {
    if (!TRAINING_WORKSPACE_ITEMS.some((item) => item.page === pageType)) return '';
    const links = TRAINING_WORKSPACE_ITEMS.map((item) => {
        const active = item.page === pageType;
        return `<a class="dragon-training-workspace-tab" href="${item.hash}" data-active="${active}"${active ? ' aria-current="page"' : ''}>${item.label}</a>`;
    }).join('');
    return `
        <div class="dragon-training-workspace-bar">
            <div class="dragon-training-workspace-bar-inner">
                <strong class="dragon-training-workspace-label">训练任务</strong>
                <nav class="dragon-training-workspace-tabs" aria-label="训练任务视图">${links}</nav>
            </div>
        </div>
    `;
}
