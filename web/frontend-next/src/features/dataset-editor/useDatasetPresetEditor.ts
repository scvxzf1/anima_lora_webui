import { zodResolver } from '@hookform/resolvers/zod';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';

import {
  datasetKeys,
  datasetPresetQuery,
  deleteDatasetPreset,
  saveDatasetPreset,
  saveDatasetPresetAs,
} from './api';
import {
  datasetFormFromPreset,
  datasetFormSchema,
  datasetPresetPathFromName,
  datasetPresetStem,
  datasetWritePayload,
  emptyDatasetForm,
  type DatasetFormValues,
} from './datasetForm';
import type {
  DatasetPresetMutationResponse,
  DatasetPresetResponse,
  DatasetPresetSummary,
} from './types';
import { useUnsavedChangesGuard } from './useUnsavedChangesGuard';

const UNSAVED_NAVIGATION_MESSAGE = '当前数据集有未保存修改，离开会丢失这些修改。是否继续？';

type Command =
  | { type: 'save'; file: string; overwrite: boolean; values: DatasetFormValues }
  | { type: 'save-as'; name: string; values: DatasetFormValues }
  | { type: 'copy'; name: string; values: DatasetFormValues }
  | { type: 'rename'; name: string; oldFile: string; values: DatasetFormValues }
  | { type: 'delete'; file: string };

type CommandResult = {
  type: Command['type'];
  message: string;
  file?: string;
  deletedFile?: string;
  preset?: DatasetPresetMutationResponse;
};

class RenamePartialError extends Error {
  constructor(
    readonly preset: DatasetPresetMutationResponse,
    readonly oldFile: string,
    cause: unknown,
  ) {
    super(`新预设已保存，但旧预设删除失败：${errorMessage(cause)}`);
    this.name = 'RenamePartialError';
  }
}

export function useDatasetPresetEditor(presets: DatasetPresetSummary[]) {
  const queryClient = useQueryClient();
  const [selectedFile, setSelectedFile] = useState('');
  const [draftFile, setDraftFile] = useState('');
  const [hydratedFile, setHydratedFile] = useState('');
  const [notice, setNotice] = useState('');
  const form = useForm<DatasetFormValues>({
    resolver: zodResolver(datasetFormSchema),
    defaultValues: emptyDatasetForm(),
    mode: 'onBlur',
  });
  const selectedPreset = useQuery({
    ...datasetPresetQuery(selectedFile),
    enabled: Boolean(selectedFile),
  });
  const hasUnsavedChanges = Boolean(draftFile) || form.formState.isDirty;
  const currentFile = draftFile || selectedFile;
  const readonly = Boolean(selectedPreset.data?.readonly) && !draftFile;
  const presetPaths = useMemo(() => new Set(presets.map((preset) => preset.path)), [presets]);
  useUnsavedChangesGuard(hasUnsavedChanges, UNSAVED_NAVIGATION_MESSAGE);

  useEffect(() => {
    if (draftFile) return;
    if (!presets.length) {
      setSelectedFile('');
      setHydratedFile('');
      return;
    }
    if (!selectedFile) {
      setSelectedFile(presets[0].path);
    }
  }, [draftFile, presets, selectedFile]);

  useEffect(() => {
    if (!selectedPreset.data || draftFile) return;
    form.reset(datasetFormFromPreset(selectedPreset.data));
    setHydratedFile(selectedPreset.data.file);
  }, [draftFile, form.reset, selectedPreset.data]);

  const command = useMutation({
    mutationFn: executeCommand,
    onSuccess: async (result) => {
      if (result.type === 'delete') {
        if (result.deletedFile) {
          queryClient.removeQueries({ queryKey: datasetKeys.preset(result.deletedFile) });
        }
        setDraftFile('');
        setSelectedFile('');
        setHydratedFile('');
        form.reset(emptyDatasetForm());
      } else if (result.preset && result.file) {
        const nextPreset = mutationResponseToPreset(result.preset);
        queryClient.setQueryData(datasetKeys.preset(result.file), nextPreset);
        if (result.deletedFile) {
          queryClient.removeQueries({ queryKey: datasetKeys.preset(result.deletedFile) });
        }
        setDraftFile('');
        setSelectedFile(result.file);
        setHydratedFile(result.file);
        form.reset(datasetFormFromPreset(result.preset));
      }
      setNotice(result.message);
      await queryClient.invalidateQueries({ queryKey: datasetKeys.library() });
    },
    onError: async (error) => {
      if (error instanceof RenamePartialError) {
        const nextPreset = mutationResponseToPreset(error.preset);
        queryClient.setQueryData(datasetKeys.preset(error.preset.file), nextPreset);
        setDraftFile('');
        setSelectedFile(error.preset.file);
        setHydratedFile(error.preset.file);
        form.reset(datasetFormFromPreset(error.preset));
        await queryClient.invalidateQueries({ queryKey: datasetKeys.library() });
      }
    },
  });

  function confirmDiscard(action: string) {
    return !hasUnsavedChanges
      || window.confirm(`当前数据集有未保存修改。${action}会丢弃这些修改，是否继续？`);
  }

  function selectFile(file: string, force = false) {
    if (file === selectedFile && !draftFile) return true;
    if (!force && !confirmDiscard('切换预设')) return false;
    setDraftFile('');
    setSelectedFile(file);
    setHydratedFile('');
    setNotice('');
    form.reset(emptyDatasetForm());
    return true;
  }

  function startNew(name: string) {
    if (!confirmDiscard('新建预设')) return false;
    const file = datasetPresetPathFromName(name);
    if (presetPaths.has(file)) {
      setNotice('数据集预设已存在，请换一个名称或使用复制/重命名');
      return false;
    }
    setSelectedFile('');
    setDraftFile(file);
    setHydratedFile(file);
    setNotice('新预设尚未保存');
    form.reset(emptyDatasetForm());
    return true;
  }

  async function save() {
    if (!currentFile || readonly || !(await form.trigger())) return false;
    command.mutate({
      type: 'save',
      file: currentFile,
      overwrite: !draftFile,
      values: form.getValues(),
    });
    return true;
  }

  async function saveAs(name: string, type: 'save-as' | 'copy' = 'save-as') {
    if (!(await form.trigger())) return false;
    const target = datasetPresetPathFromName(name);
    if (presetPaths.has(target)) {
      setNotice('目标预设已存在，请换一个名称');
      return false;
    }
    command.mutate({ type, name, values: form.getValues() });
    return true;
  }

  async function rename(name: string) {
    if (!selectedFile || readonly || !(await form.trigger())) return false;
    const target = datasetPresetPathFromName(name);
    if (target === selectedFile) return false;
    if (presetPaths.has(target)) {
      setNotice('目标预设已存在，请换一个名称');
      return false;
    }
    command.mutate({ type: 'rename', name, oldFile: selectedFile, values: form.getValues() });
    return true;
  }

  function remove() {
    if (!selectedFile || readonly) return false;
    command.mutate({ type: 'delete', file: selectedFile });
    return true;
  }

  async function reload() {
    if (!selectedFile || !confirmDiscard('重新加载')) return false;
    setDraftFile('');
    const result = await selectedPreset.refetch();
    if (result.data) {
      form.reset(datasetFormFromPreset(result.data));
      setHydratedFile(result.data.file);
    }
    return Boolean(result.data);
  }

  return {
    form,
    selectedPreset,
    selectedFile,
    currentFile,
    draftFile,
    hydratedFile,
    readonly,
    hasUnsavedChanges,
    notice,
    command,
    selectFile,
    startNew,
    save,
    saveAs,
    rename,
    remove,
    reload,
    confirmDiscard,
    notify: setNotice,
    suggestedName: datasetPresetStem(currentFile || 'dataset'),
  };
}

export type DatasetPresetEditorController = ReturnType<typeof useDatasetPresetEditor>;

async function executeCommand(command: Command): Promise<CommandResult> {
  if (command.type === 'save') {
    const preset = await saveDatasetPreset(
      command.file,
      datasetWritePayload(command.values),
      command.overwrite,
    );
    return { type: command.type, preset, file: preset.file, message: preset.message };
  }
  if (command.type === 'save-as' || command.type === 'copy') {
    const preset = await saveDatasetPresetAs(command.name, datasetWritePayload(command.values));
    return {
      type: command.type,
      preset,
      file: preset.file,
      message: command.type === 'copy' ? '已复制数据集预设' : preset.message,
    };
  }
  if (command.type === 'rename') {
    const preset = await saveDatasetPresetAs(command.name, datasetWritePayload(command.values));
    try {
      await deleteDatasetPreset(command.oldFile);
    } catch (error) {
      throw new RenamePartialError(preset, command.oldFile, error);
    }
    return {
      type: command.type,
      preset,
      file: preset.file,
      deletedFile: command.oldFile,
      message: '已重命名数据集预设',
    };
  }
  const deleted = await deleteDatasetPreset(command.file);
  return {
    type: command.type,
    deletedFile: command.file,
    message: deleted.message || '数据集预设已删除，图片和缓存目录未受影响',
  };
}

function mutationResponseToPreset(result: DatasetPresetMutationResponse): DatasetPresetResponse {
  return {
    ...result,
    name: datasetPresetStem(result.file),
    readonly: false,
  };
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}
