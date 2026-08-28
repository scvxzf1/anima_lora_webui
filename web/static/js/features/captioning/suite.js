import { mountWorkbench } from './workbench.js?v=dragon-ui-20260829v12';
import { bindGroupsPanel, renderGroupsPanel } from './workspace/groups-panel.js?v=dragon-ui-20260829v12';
import { bindRolePanel, renderRolePanel } from './workspace/role-panel.js?v=dragon-ui-20260829v12';
import { bindCompletionPanel, renderCompletionPanel } from './workspace/completion-panel.js?v=dragon-ui-20260829v12';
import { bindTagManagerPanel, renderTagManagerPanel } from './workspace/tag-manager-panel.js?v=dragon-ui-20260829v12';
import { bindExportPanel, renderExportPanel } from './workspace/export-panel.js?v=dragon-ui-20260829v12';
import { bindDatasetPanel, renderDatasetPanel } from './workspace/dataset-panel.js?v=dragon-ui-20260829v12';
import { bindPromptsPanel, renderPromptsPanel } from './workspace/prompts-panel.js?v=dragon-ui-20260829v12';
import { bindLogsPanel, renderLogsPanel } from './workspace/logs-panel.js?v=dragon-ui-20260829v12';
import { bindConfigPanel, renderConfigPanel } from './workspace/config-panel.js?v=dragon-ui-20260829v12';
import { bindFilesPanel, renderFilesPanel } from './workspace/files-panel.js?v=dragon-ui-20260829v12';
import { bindRetryPanel, renderRetryPanel } from './workspace/retry-panel.js?v=dragon-ui-20260829v12';

const PANELS = [
    ['workbench', '审阅台'],
    ['role', '角色 Tag'],
    ['completion', '打标补全'],
    ['files', '目录浏览'],
    ['retry', '失败重试'],
    ['groups', '目录组'],
    ['tags', 'Tag 管理'],
    ['export', 'Caption 导出'],
    ['dataset', '数据集生成'],
    ['prompts', '提示词预设'],
    ['logs', '打标日志'],
    ['config', '配置中心'],
];

export function mountCaptioningSuite(root, state, prefill = {}) {
    state.activePanel ||= 'workbench';
    state.suiteRender = () => renderSuite(root, state, prefill);
    renderSuite(root, state, prefill);
}

function renderSuite(root, state, prefill) {
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    const host = root.querySelector('[data-caption-suite-host]');
    host.innerHTML = `<div class="dragon-caption-suite">
        <nav class="dragon-caption-suite-tabs" aria-label="打标工作流">${PANELS.map(([id, label]) => `<button type="button" data-caption-suite-panel="${id}" data-active="${state.activePanel === id}" ${state.activePanel === id ? 'aria-current="page"' : ''}>${label}</button>`).join('')}<button class="dragon-caption-suite-settings" type="button" data-caption-suite-settings>API / 调度</button></nav>
        <div class="dragon-caption-suite-panel" data-caption-suite-panel-host>${renderPanel(state)}</div>
    </div>`;
    host.querySelectorAll('[data-caption-suite-panel]').forEach((button) => button.addEventListener('click', () => {
        if (state.activePanel === 'groups' && state.workspaceData.groupsDirty && !window.confirm('目录组有未保存修改，放弃并切换界面？')) return;
        state.activePanel = button.dataset.captionSuitePanel;
        renderSuite(root, state, prefill);
    }));
    host.querySelector('[data-caption-suite-settings]')?.addEventListener('click', () => root.querySelector('[data-caption-settings-dialog]')?.showModal());
    bindPanel(root, state, prefill);
}

function renderPanel(state) {
    if (state.activePanel === 'workbench') return '<div data-caption-workbench-host></div>';
    if (state.activePanel === 'groups') return renderGroupsPanel(state);
    if (state.activePanel === 'role') return renderRolePanel(state);
    if (state.activePanel === 'completion') return renderCompletionPanel(state);
    if (state.activePanel === 'files') return renderFilesPanel(state);
    if (state.activePanel === 'retry') return renderRetryPanel(state);
    if (state.activePanel === 'tags') return renderTagManagerPanel(state);
    if (state.activePanel === 'export') return renderExportPanel(state);
    if (state.activePanel === 'dataset') return renderDatasetPanel(state);
    if (state.activePanel === 'prompts') return renderPromptsPanel(state);
    if (state.activePanel === 'logs') return renderLogsPanel(state);
    return renderConfigPanel(state);
}

function bindPanel(root, state, prefill) {
    if (state.activePanel === 'workbench') mountWorkbench(root, state, prefill);
    if (state.activePanel === 'groups') bindGroupsPanel(root, state);
    if (state.activePanel === 'role') bindRolePanel(root, state);
    if (state.activePanel === 'completion') bindCompletionPanel(root, state);
    if (state.activePanel === 'files') bindFilesPanel(root, state);
    if (state.activePanel === 'retry') bindRetryPanel(root, state);
    if (state.activePanel === 'tags') bindTagManagerPanel(root, state);
    if (state.activePanel === 'export') bindExportPanel(root, state);
    if (state.activePanel === 'dataset') bindDatasetPanel(root, state);
    if (state.activePanel === 'prompts') bindPromptsPanel(root, state);
    if (state.activePanel === 'logs') bindLogsPanel(root, state);
    if (state.activePanel === 'config') bindConfigPanel(root, state);
}
