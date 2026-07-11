/**
 * Dataset editor top toolbar: experimental + stage-schedule entries.
 */
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { openStageResolutionDialog } from '../config-form/stage-resolution.js?v=module-bootstrap-20260711-ir1';
import { openDatasetExperimentalDialog } from './experimental-dialog.js?v=module-bootstrap-20260711-ir1';

const datasetState = getDatasetState();

export function createDatasetEditorToolbarActions() {
    const actions = document.createElement('div');
    actions.className = 'dataset-editor-actions dataset-editor-toolbar-actions';

    const experimentalBtn = document.createElement('button');
    experimentalBtn.id = 'btn-dataset-open-experimental';
    experimentalBtn.type = 'button';
    experimentalBtn.className = 'btn btn-small';
    experimentalBtn.textContent = '实验性/高级';
    experimentalBtn.title = '编辑当前选中数据集的低频/旧功能字段';
    experimentalBtn.addEventListener('click', () => {
        const rows = datasetState.datasetPresetState?.datasets
            || datasetState.datasetEditorState?.datasets
            || [];
        if (!rows.length) {
            experimentalBtn.title = '请先添加数据集';
            return;
        }
        const idx = Math.max(0, Math.min(Number(datasetState.selectedDatasetIndex) || 0, rows.length - 1));
        datasetState.selectedDatasetIndex = idx;
        openDatasetExperimentalDialog(idx);
    });

    const stageBtn = document.createElement('button');
    stageBtn.id = 'btn-dataset-open-stage-schedule';
    stageBtn.type = 'button';
    stageBtn.className = 'btn btn-small';
    stageBtn.textContent = '分阶段调度';
    stageBtn.title = '按总训练步数百分比切换数据集子集（写入当前训练配置草稿）';
    stageBtn.addEventListener('click', () => openStageResolutionDialog());

    // 调用方还会追加「添加数据集」按钮；这里只返回双入口容器片段。
    actions.append(experimentalBtn, stageBtn);
    return actions;
}
