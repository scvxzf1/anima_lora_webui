import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useRef, useState } from 'react';

import { trainingContextKeys } from '../../api/trainingContext';
import type { TrainingContextController } from '../../app/useTrainingContext';

import { applyDatasetPreset } from './api';
import { DatasetApplyDialog } from './DatasetApplyDialog';
import { DatasetDeleteDialog } from './DatasetDeleteDialog';
import { DatasetDefaultsEditor } from './DatasetDefaultsEditor';
import { DatasetPreviewDialog } from './DatasetPreviewDialog';
import { DatasetSubsetList } from './DatasetSubsetList';
import { datasetPresetStem } from './datasetForm';
import { DatasetNameDialog, type DatasetNameAction } from './DatasetNameDialog';
import { StageScheduleEditor } from './StageScheduleEditor';
import type { DatasetPresetEditorController } from './useDatasetPresetEditor';

type Props = {
  editor: DatasetPresetEditorController;
  exporting: boolean;
  exportError?: string;
  trainingContext: TrainingContextController;
  onExport: (file: string) => void;
};

export function DatasetPresetEditor({
  editor,
  exporting,
  exportError,
  trainingContext,
  onExport,
}: Props) {
  const queryClient = useQueryClient();
  const [nameAction, setNameAction] = useState<DatasetNameAction | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [applyOpen, setApplyOpen] = useState(false);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);
  const previewTriggerRef = useRef<HTMLElement | null>(null);
  const watchedRows = editor.form.watch('datasets');
  const watchedStageEnabled = editor.form.watch('stage_schedule_enabled');
  const busy = editor.command.isPending;
  const persisted = Boolean(editor.selectedFile);
  const canEdit = Boolean(editor.currentFile) && !editor.readonly;
  const canPreview = persisted && !editor.hasUnsavedChanges;
  const summary = summarizeRows(watchedRows);
  const targetTrainingFile = trainingContext.selectedFile;
  const canApply = persisted
    && !editor.hasUnsavedChanges
    && Boolean(targetTrainingFile)
    && !targetTrainingFile?.locked
    && !busy;
  const applyPreset = useMutation({
    mutationFn: () => applyDatasetPreset(editor.selectedFile, targetTrainingFile!.path),
    onSuccess: async (result) => {
      setApplyOpen(false);
      editor.notify(result.message || '已应用到当前训练配置');
      await queryClient.invalidateQueries({
        queryKey: trainingContextKeys.merged(targetTrainingFile!.path, trainingContext.selectedPreset),
      });
    },
  });

  function openNameDialog(action: DatasetNameAction) {
    setNameAction(action);
  }

  async function confirmName(name: string) {
    let accepted = false;
    if (nameAction === 'new') accepted = editor.startNew(name);
    if (nameAction === 'save-as') accepted = await editor.saveAs(name);
    if (nameAction === 'copy') accepted = await editor.saveAs(name, 'copy');
    if (nameAction === 'rename') accepted = await editor.rename(name);
    if (accepted) setNameAction(null);
  }

  if (
    (!editor.currentFile && editor.selectedPreset.isPending)
    || (editor.selectedFile && !editor.draftFile && editor.hydratedFile !== editor.selectedFile)
  ) {
    return <p className="dataset-empty">正在读取数据集详情</p>;
  }

  if (editor.selectedPreset.isError && !editor.draftFile) {
    return (
      <div className="error-panel" role="alert">
        <h2>无法读取数据集详情</h2>
        <p>{editor.selectedPreset.error.message}</p>
        <button type="button" onClick={() => editor.selectedPreset.refetch()}>重试</button>
      </div>
    );
  }

  if (!editor.currentFile) {
    return (
      <div className="dataset-editor-empty">
        <h2>暂无可用的数据集预设</h2>
        <button type="button" className="primary-command" onClick={() => openNameDialog('new')}>
          新建预设
        </button>
        {nameAction ? (
          <DatasetNameDialog
            action={nameAction}
            initialValue="dataset"
            onCancel={() => setNameAction(null)}
            onConfirm={confirmName}
          />
        ) : null}
      </div>
    );
  }

  return (
    <form className="dataset-editor-form" onSubmit={editor.form.handleSubmit(() => editor.save())}>
      <header className="dataset-detail-header">
        <div>
          <p className="eyebrow">当前预设</p>
          <h2>{datasetPresetStem(editor.currentFile)}.toml</h2>
          <p>{editor.currentFile}</p>
        </div>
        <div className="dataset-detail-status">
          <span className="badge" data-tone={editor.readonly ? 'warning' : 'success'}>
            {editor.readonly ? '只读' : editor.draftFile ? '未保存' : '可编辑'}
          </span>
          <span className="badge">{editor.hasUnsavedChanges ? '有未保存修改' : '已同步'}</span>
          <span className="badge">{watchedStageEnabled ? '阶段调度已启用' : '阶段调度关闭'}</span>
        </div>
      </header>

      <div className="dataset-command-bar" aria-label="预设操作">
        <button type="button" onClick={() => openNameDialog('new')} disabled={busy}>新建</button>
        <button type="submit" className="primary-command" disabled={!canEdit || busy}>保存</button>
        <button type="button" onClick={() => openNameDialog('save-as')} disabled={busy}>另存</button>
        <button type="button" onClick={() => openNameDialog('copy')} disabled={busy}>复制</button>
        <button
          type="button"
          onClick={() => onExport(editor.selectedFile)}
          disabled={!persisted || busy || exporting}
        >
          {exporting ? '导出中' : '导出'}
        </button>
        <button type="button" onClick={() => openNameDialog('rename')} disabled={!persisted || editor.readonly || busy}>
          重命名
        </button>
        <button type="button" onClick={() => setDeleteOpen(true)} disabled={!persisted || editor.readonly || busy}>
          删除
        </button>
        <button type="button" onClick={() => editor.reload()} disabled={!persisted || busy}>重新读取</button>
        <button
          type="button"
          className="apply-command"
          onClick={() => {
            applyPreset.reset();
            setApplyOpen(true);
          }}
          disabled={!canApply}
          title={applyDisabledReason({
            persisted,
            dirty: editor.hasUnsavedChanges,
            hasTrainingFile: Boolean(targetTrainingFile),
            trainingFileLocked: Boolean(targetTrainingFile?.locked),
          })}
        >
          应用到当前训练配置
        </button>
      </div>

      {editor.notice ? <p className="dataset-notice" role="status">{editor.notice}</p> : null}
      {editor.command.isError ? <p className="dataset-command-error" role="alert">{editor.command.error.message}</p> : null}
      {exportError ? <p className="dataset-command-error" role="alert">{exportError}</p> : null}

      <dl className="dataset-summary-grid">
        <div><dt>子集</dt><dd>{watchedRows.length}</dd></div>
        <div><dt>训练集</dt><dd>{summary.train}</dd></div>
        <div><dt>正则集</dt><dd>{summary.reg}</dd></div>
        <div><dt>总重复</dt><dd>{summary.repeats}</dd></div>
        <div><dt>分辨率</dt><dd>{editor.form.watch('defaults.resolution')}</dd></div>
        <div><dt>批大小</dt><dd>{editor.form.watch('defaults.batch_size')}</dd></div>
      </dl>

      <DatasetDefaultsEditor form={editor.form} disabled={editor.readonly || busy} />
      <DatasetSubsetList
        form={editor.form}
        disabled={editor.readonly || busy}
        previewDisabled={!canPreview || busy}
        onPreview={(index, trigger) => {
          previewTriggerRef.current = trigger;
          setPreviewIndex(index);
        }}
      />
      <StageScheduleEditor form={editor.form} disabled={editor.readonly || busy} />

      {nameAction ? (
        <DatasetNameDialog
          action={nameAction}
          initialValue={nameAction === 'copy' ? `${editor.suggestedName}_copy` : editor.suggestedName}
          onCancel={() => setNameAction(null)}
          onConfirm={confirmName}
        />
      ) : null}

      {applyOpen && targetTrainingFile ? (
        <DatasetApplyDialog
          datasetFile={editor.selectedFile}
          trainFile={targetTrainingFile}
          busy={applyPreset.isPending}
          error={applyPreset.error?.message}
          onCancel={() => {
            if (!applyPreset.isPending) setApplyOpen(false);
          }}
          onConfirm={() => applyPreset.mutate()}
        />
      ) : null}

      {deleteOpen ? (
        <DatasetDeleteDialog
          file={editor.currentFile}
          onCancel={() => setDeleteOpen(false)}
          onConfirm={() => {
            setDeleteOpen(false);
            editor.remove();
          }}
        />
      ) : null}

      {previewIndex !== null && editor.selectedFile ? (
        <DatasetPreviewDialog
          file={editor.selectedFile}
          datasetIndex={previewIndex}
          returnFocus={previewTriggerRef.current}
          onClose={() => setPreviewIndex(null)}
        />
      ) : null}
    </form>
  );
}

function summarizeRows(rows: Array<{ is_reg?: boolean; num_repeats?: number }>) {
  return rows.reduce(
    (summary, row) => ({
      train: summary.train + (row.is_reg ? 0 : 1),
      reg: summary.reg + (row.is_reg ? 1 : 0),
      repeats: summary.repeats + (Number(row.num_repeats) || 1),
    }),
    { train: 0, reg: 0, repeats: 0 },
  );
}

function applyDisabledReason({
  persisted,
  dirty,
  hasTrainingFile,
  trainingFileLocked,
}: {
  persisted: boolean;
  dirty: boolean;
  hasTrainingFile: boolean;
  trainingFileLocked: boolean;
}) {
  if (!persisted) return '请先保存数据集预设';
  if (dirty) return '请先保存当前数据集修改';
  if (!hasTrainingFile) return '没有可用的训练配置';
  if (trainingFileLocked) return '当前训练配置为只读，请先选择可编辑配置';
  return '将已保存的数据集预设应用到当前训练配置';
}
