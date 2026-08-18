import { zodResolver } from '@hookform/resolvers/zod';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import { Topbar } from '../../app/Topbar';

import { z } from 'zod';

import { TrainingContextBar } from '../../app/TrainingContextBar';
import { useTrainingContext } from '../../app/useTrainingContext';

import {
  createDatasetGroup,
  datasetKeys,
  datasetLibraryQuery,
  deleteDatasetGroup,
  fetchDatasetPreset,
  importDatasetPreset,
  renameDatasetGroup,
} from './api';
import { DatasetGroupDialog } from './DatasetGroupDialog';
import { DatasetGroupList, datasetPresetName } from './DatasetGroupList';
import { DatasetImportDialog } from './DatasetImportDialog';
import { DatasetPresetEditor } from './DatasetPresetEditor';
import { datasetPresetStem } from './datasetForm';
import { downloadTextFile } from './downloadTextFile';
import type { DatasetLibraryGroup, DatasetPresetSummary } from './types';
import { useDatasetPresetEditor } from './useDatasetPresetEditor';
import { useDatasetLibraryOrdering } from './useDatasetLibraryOrdering';
import './DatasetWorkspace.css';

const groupSchema = z.object({
  label: z.string().trim().min(1, '请输入分组名称').max(80, '分组名称不能超过 80 个字符'),
});

type GroupFormValues = z.infer<typeof groupSchema>;

type GroupDialogState = {
  action: 'rename' | 'delete';
  group: DatasetLibraryGroup;
};

type ImportDraft = {
  sourceName: string;
  name: string;
  content: string;
};

function presetSearchText(preset: DatasetPresetSummary) {
  return `${datasetPresetName(preset)} ${preset.path}`.toLocaleLowerCase();
}

function filterGroups(groups: DatasetLibraryGroup[], search: string) {
  const term = search.trim().toLocaleLowerCase();
  if (!term) return groups;

  return groups
    .map((group) => ({
      ...group,
      files: group.files.filter((preset) => presetSearchText(preset).includes(term)),
    }))
    .filter((group) => group.files.length > 0 || group.label.toLocaleLowerCase().includes(term));
}

export function DatasetWorkspace() {
  const queryClient = useQueryClient();
  const trainingContext = useTrainingContext();
  const library = useQuery(datasetLibraryQuery);
  const [search, setSearch] = useState('');
  const [notice, setNotice] = useState('');
  const [groupDialog, setGroupDialog] = useState<GroupDialogState | null>(null);
  const [importDraft, setImportDraft] = useState<ImportDraft | null>(null);
  const importInputRef = useRef<HTMLInputElement>(null);
  const editor = useDatasetPresetEditor(library.data?.presets ?? []);
  const ordering = useDatasetLibraryOrdering(setNotice);
  const visibleGroups = useMemo(
    () => filterGroups(library.data?.groups ?? [], search),
    [library.data?.groups, search],
  );
  const groupForm = useForm<GroupFormValues>({
    resolver: zodResolver(groupSchema),
    defaultValues: { label: '' },
  });
  const createGroup = useMutation({
    mutationFn: ({ label }: GroupFormValues) => createDatasetGroup(label),
    onSuccess: async (result) => {
      groupForm.reset();
      setNotice(result.message || '分组已创建');
      await queryClient.invalidateQueries({ queryKey: datasetKeys.library() });
    },
  });
  const renameGroup = useMutation({
    mutationFn: ({ groupId, label }: { groupId: string; label: string }) => (
      renameDatasetGroup(groupId, label)
    ),
    onSuccess: async (result) => {
      setGroupDialog(null);
      setNotice(result.message || '分组已重命名');
      await queryClient.invalidateQueries({ queryKey: datasetKeys.library() });
    },
  });
  const deleteGroup = useMutation({
    mutationFn: (groupId: string) => deleteDatasetGroup(groupId),
    onSuccess: async (result) => {
      setGroupDialog(null);
      setNotice(result.message || '分组已删除，TOML 预设已保留');
      await queryClient.invalidateQueries({ queryKey: datasetKeys.library() });
    },
  });
  const importPreset = useMutation({
    mutationFn: ({ name, content }: ImportDraft) => importDatasetPreset(name, content, false),
    onSuccess: async (result) => {
      setImportDraft(null);
      setNotice(result.message || '数据集预设已导入');
      await queryClient.invalidateQueries({ queryKey: datasetKeys.library() });
      editor.selectFile(result.file, true);
    },
  });
  const exportPreset = useMutation({
    mutationFn: (file: string) => fetchDatasetPreset(file),
    onSuccess: (result) => {
      const filename = result.file.split('/').pop() || 'dataset.toml';
      downloadTextFile(filename, result.content || '');
      editor.notify(`已导出 ${filename}`);
    },
  });

  const submitGroup = groupForm.handleSubmit((values) => {
    setNotice('');
    createGroup.mutate(values);
  });

  function chooseImportFile() {
    if (!editor.confirmDiscard('导入预设')) return;
    importPreset.reset();
    importInputRef.current?.click();
  }

  function openGroupDialog(action: 'rename' | 'delete', group: DatasetLibraryGroup) {
    renameGroup.reset();
    deleteGroup.reset();
    setGroupDialog({ action, group });
  }

  function closeGroupDialog() {
    renameGroup.reset();
    deleteGroup.reset();
    setGroupDialog(null);
  }

  async function loadImportFile(file: File | undefined) {
    if (!file) return;
    const content = await file.text();
    setImportDraft({ sourceName: file.name, name: datasetPresetStem(file.name), content });
  }

  return (
    <div className="app-shell">
      <Topbar />

      <main className="dataset-page">
        <header className="dataset-page-header">
          <div>
            <p className="eyebrow">DATASET FORGE</p>
            <h1>数据集蓝图</h1>
            <p>管理训练数据结构、分组和磁盘中的 TOML 预设。</p>
          </div>
          <div className="dataset-metrics" aria-label="数据集统计">
            <span><strong>{library.data?.presets.length ?? 0}</strong>预设</span>
            <span><strong>{library.data?.groups.length ?? 0}</strong>分组</span>
            <span>
              <strong>
                {library.data?.presets.reduce(
                  (total, preset) => total + (preset.summary?.dataset_count ?? 0),
                  0,
                ) ?? 0}
              </strong>
              子集
            </span>
          </div>
        </header>

        <TrainingContextBar context={trainingContext} />

        {library.isError ? (
          <section className="error-panel" role="alert">
            <h2>无法读取数据集预设</h2>
            <p>{library.error.message}</p>
            <button type="button" onClick={() => library.refetch()}>重试</button>
          </section>
        ) : (
          <div className="dataset-workspace" aria-busy={library.isPending}>
            <aside className="dataset-library" aria-label="数据集预设库">
              <div className="dataset-library-toolbar">
                <div>
                  <h2>预设库</h2>
                  <p>按分组整理磁盘中的数据集蓝图。</p>
                </div>
                <div className="dataset-library-toolbar-actions">
                  <button type="button" onClick={chooseImportFile}>导入</button>
                  <button type="button" onClick={() => library.refetch()} disabled={library.isFetching}>
                    {library.isFetching ? '刷新中' : '刷新'}
                  </button>
                  <input
                    ref={importInputRef}
                    className="dataset-hidden-input"
                    type="file"
                    accept=".toml,.txt,text/plain"
                    aria-label="选择要导入的预设"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      event.target.value = '';
                      void loadImportFile(file);
                    }}
                  />
                </div>
              </div>

              <label className="dataset-search">
                <span>搜索预设</span>
                <input
                  type="search"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="名称或路径"
                />
              </label>

              <form className="dataset-group-form" onSubmit={submitGroup}>
                <label>
                  <span>新分组</span>
                  <input {...groupForm.register('label')} placeholder="例如：角色训练" />
                </label>
                <button type="submit" disabled={createGroup.isPending}>创建</button>
                {groupForm.formState.errors.label ? (
                  <p role="alert">{groupForm.formState.errors.label.message}</p>
                ) : null}
                {createGroup.isError ? <p role="alert">{createGroup.error.message}</p> : null}
              </form>

              {notice ? <p className="dataset-notice" role="status">{notice}</p> : null}

              <DatasetGroupList
                groups={visibleGroups}
                pending={library.isPending}
                selectedFile={editor.selectedFile}
                searchActive={Boolean(search.trim())}
                ordering={ordering.isPending}
                orderingError={ordering.error?.message}
                onSelect={editor.selectFile}
                onGroupAction={openGroupDialog}
                onPlaceGroup={(groupId, index) => ordering.mutate({ type: 'group', groupId, index })}
                onPlacePreset={(file, groupId, order) => ordering.mutate({
                  type: 'preset',
                  file,
                  groupId,
                  order,
                })}
              />
            </aside>

            <section className="dataset-detail" aria-live="polite">
              <DatasetPresetEditor
                editor={editor}
                exporting={exportPreset.isPending}
                exportError={exportPreset.error?.message}
                trainingContext={trainingContext}
                onExport={(file) => exportPreset.mutate(file)}
              />
            </section>
          </div>
        )}
      </main>

      {groupDialog ? (
        <DatasetGroupDialog
          action={groupDialog.action}
          group={groupDialog.group}
          busy={renameGroup.isPending || deleteGroup.isPending}
          error={renameGroup.error?.message || deleteGroup.error?.message}
          onCancel={closeGroupDialog}
          onRename={(label) => renameGroup.mutate({ groupId: groupDialog.group.id, label })}
          onDelete={() => deleteGroup.mutate(groupDialog.group.id)}
        />
      ) : null}

      {importDraft ? (
        <DatasetImportDialog
          sourceName={importDraft.sourceName}
          initialName={importDraft.name}
          busy={importPreset.isPending}
          error={importPreset.error?.message}
          onCancel={() => {
            importPreset.reset();
            setImportDraft(null);
          }}
          onConfirm={(name) => importPreset.mutate({ ...importDraft, name })}
        />
      ) : null}
    </div>
  );
}
