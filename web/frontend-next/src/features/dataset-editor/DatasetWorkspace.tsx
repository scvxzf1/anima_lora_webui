import { zodResolver } from '@hookform/resolvers/zod';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';

import { z } from 'zod';

import { TrainingContextBar } from '../../app/TrainingContextBar';
import { useTrainingContext } from '../../app/useTrainingContext';
import { ApiError } from '../../api/client';

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
import { DatasetDiscardDialog } from './DatasetDiscardDialog';
import { DatasetGroupList, datasetPresetName } from './DatasetGroupList';
import { DatasetImportDialog } from './DatasetImportDialog';
import { DatasetPresetEditor } from './DatasetPresetEditor';
import { datasetPresetStem } from './datasetForm';
import { downloadTextFile } from './downloadTextFile';
import type { DatasetLibraryGroup, DatasetPresetSummary } from './types';
import { useDatasetPresetEditor } from './useDatasetPresetEditor';
import { useDatasetLibraryOrdering } from './useDatasetLibraryOrdering';
import './DatasetWorkspace.css';
import './DatasetWorkspaceLayout.css';

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

type DatasetWorkspaceReturn = {
  dataset: string;
  search: string;
  libraryScrollTop: number;
  detailScrollTop: number;
};

export const DATASET_DETAILED_MANAGEMENT_KEY = 'dragon-next:dataset-presets:detailed-management:v1';

function readDetailedManagement() {
  try { return localStorage.getItem(DATASET_DETAILED_MANAGEMENT_KEY) === 'true'; }
  catch { return false; }
}

function presetSearchText(preset: DatasetPresetSummary) {
  return `${datasetPresetName(preset)} ${preset.path}`.toLocaleLowerCase();
}

function filterGroups(groups: DatasetLibraryGroup[], search: string) {
  const term = search.trim().toLocaleLowerCase();
  if (!term) return groups;

  return groups
    .map((group) => ({
      ...group,
      files: group.label.toLocaleLowerCase().includes(term)
        ? group.files
        : group.files.filter((preset) => presetSearchText(preset).includes(term)),
    }))
    .filter((group) => group.files.length > 0 || group.label.toLocaleLowerCase().includes(term));
}

export function DatasetWorkspace() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const libraryScrollRef = useRef<HTMLElement>(null);
  const detailScrollRef = useRef<HTMLElement>(null);
  const restoredReturn = useMemo(
    () => getWorkspaceReturn(location.state, new URLSearchParams(location.search).get('dataset') || ''),
    [location.state, location.search],
  );
  const returnRestored = useRef(false);
  const trainingContext = useTrainingContext();
  const library = useQuery(datasetLibraryQuery);
  const [search, setSearch] = useState(() => restoredReturn?.search || '');
  const [detailedManagement, setDetailedManagement] = useState(readDetailedManagement);
  const [notice, setNotice] = useState('');
  const [groupDialog, setGroupDialog] = useState<GroupDialogState | null>(null);
  const [importDraft, setImportDraft] = useState<ImportDraft | null>(null);
  const importInputRef = useRef<HTMLInputElement>(null);
  const editor = useDatasetPresetEditor(library.data?.presets ?? [], searchParams.get('dataset') || '');
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

  useLayoutEffect(() => {
    if (!restoredReturn || returnRestored.current || !library.data || editor.hydratedFile !== editor.selectedFile) return;
    const frame = requestAnimationFrame(() => {
      if (libraryScrollRef.current) libraryScrollRef.current.scrollTop = restoredReturn.libraryScrollTop;
      if (detailScrollRef.current) detailScrollRef.current.scrollTop = restoredReturn.detailScrollTop;
      returnRestored.current = true;
      const nextState = { ...asRecord(location.state) };
      delete nextState.datasetWorkspaceReturn;
      void navigate(`${location.pathname}${location.search}`, { replace: true, state: nextState });
    });
    return () => cancelAnimationFrame(frame);
  }, [editor.hydratedFile, editor.selectedFile, library.data, location.pathname, location.search, location.state, navigate, restoredReturn]);

  async function openWorkbench(index: number) {
    if (!editor.selectedFile || editor.hasUnsavedChanges) return;
    const returnParams = new URLSearchParams(location.search);
    returnParams.set('dataset', editor.selectedFile);
    const returnTo = `${location.pathname}?${returnParams}`;
    const snapshot: DatasetWorkspaceReturn = {
      dataset: editor.selectedFile,
      search,
      libraryScrollTop: libraryScrollRef.current?.scrollTop || 0,
      detailScrollTop: detailScrollRef.current?.scrollTop || 0,
    };
    const nextState = { ...asRecord(location.state), datasetWorkspaceReturn: snapshot };
    const query = new URLSearchParams({ dataset: editor.selectedFile, subset: String(index) });
    await navigate(returnTo, { replace: true, state: nextState });
    await navigate(`/datasets/workspace/preview?${query}`, { state: { returnTo } });
  }

  const submitGroup = groupForm.handleSubmit((values) => {
    setNotice('');
    createGroup.mutate(values);
  });

  async function chooseImportFile() {
    if (!(await editor.confirmDiscard('导入预设'))) return;
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

      <main className="dataset-page">
        <header className="dataset-page-header">
          <div>
            <p className="eyebrow">DATASET FORGE</p>
            <h1>数据集蓝图</h1>
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
            <aside ref={libraryScrollRef} className="dataset-library" aria-label="数据集预设库">
              <div className="dataset-library-toolbar">
                <div>
                  <h2>预设库</h2>
                  <p>按分组整理磁盘中的数据集蓝图。</p>
                </div>
                <div className="dataset-library-toolbar-actions">
                  <div className="dataset-library-import-actions">
                    <button type="button" onClick={chooseImportFile}>导入</button>
                    <label className="dataset-management-toggle">
                      <input
                        type="checkbox"
                        role="switch"
                        checked={detailedManagement}
                        onChange={(event) => {
                          const enabled = event.target.checked;
                          setDetailedManagement(enabled);
                          try { localStorage.setItem(DATASET_DETAILED_MANAGEMENT_KEY, String(enabled)); }
                          catch { /* Keep the setting for this session. */ }
                        }}
                      />
                      <span className="dataset-management-toggle-track" aria-hidden="true" />
                      <span>详细管理</span>
                    </label>
                  </div>
                  <button type="button" onClick={() => {
                    void library.refetch();
                    void queryClient.invalidateQueries({ queryKey: [...datasetKeys.all, 'cover'] });
                  }} disabled={library.isFetching}>
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
                detailedManagement={detailedManagement}
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

            <section ref={detailScrollRef} className="dataset-detail" aria-live="polite">
              <DatasetPresetEditor
                editor={editor}
                exporting={exportPreset.isPending}
                exportError={exportPreset.error?.message}
                trainingContext={trainingContext}
                onExport={(file) => exportPreset.mutate(file)}
                onOpenWorkbench={(index) => void openWorkbench(index)}
              />
            </section>
          </div>
        )}
      </main>

      {editor.discardDialog && <DatasetDiscardDialog {...editor.discardDialog} />}
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
          resultUnknown={importPreset.error instanceof ApiError && importPreset.error.status === 0}
          onCancel={() => {
            importPreset.reset();
            setImportDraft(null);
          }}
          onReconcile={() => {
            importPreset.reset();
            setImportDraft(null);
            void library.refetch();
          }}
          onConfirm={(name) => importPreset.mutate({ ...importDraft, name })}
        />
      ) : null}
    </div>
  );
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? value as Record<string, unknown> : {};
}

function getWorkspaceReturn(state: unknown, dataset: string): DatasetWorkspaceReturn | null {
  const value = asRecord(state).datasetWorkspaceReturn;
  if (!value || typeof value !== 'object') return null;
  const snapshot = value as Partial<DatasetWorkspaceReturn>;
  if (snapshot.dataset !== dataset || typeof snapshot.search !== 'string') return null;
  return {
    dataset,
    search: snapshot.search,
    libraryScrollTop: Number.isFinite(snapshot.libraryScrollTop) ? Number(snapshot.libraryScrollTop) : 0,
    detailScrollTop: Number.isFinite(snapshot.detailScrollTop) ? Number(snapshot.detailScrollTop) : 0,
  };
}
