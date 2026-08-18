import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useMemo, useState } from 'react';
import { Topbar } from '../../app/Topbar';


import { trainingContextKeys } from '../../api/trainingContext';
import { TrainingContextBar } from '../../app/TrainingContextBar';
import { useTrainingContext } from '../../app/useTrainingContext';
import { datasetKeys } from '../dataset-editor/api';
import { useUnsavedChangesGuard } from '../dataset-editor/useUnsavedChangesGuard';
import {
  fetchRawTrainingConfig,
  previewTrainingConfigPatch,
  runTrainingPreflight,
  saveTrainingConfigAs,
  saveTrainingConfigPatch,
  trainingConfigKeys,
} from './api';
import { TrainingFieldEditor } from './TrainingFieldEditor';
import { TrainingPatchPreview } from './TrainingPatchPreview';
import { TrainingPreflightPanel } from './TrainingPreflightPanel';
import { TrainingSaveAsDialog } from './TrainingSaveAsDialog';
import {
  draftFromMerged,
  importedTrainingPath,
  rawConfigOwnKeys,
  TRAINING_FIELDS,
  trainingPatchValues,
  type TrainingDraft,
} from './trainingForm';
import './TrainingWorkspace.css';

const UNSAVED_MESSAGE = '当前训练配置有未保存修改，离开会丢失这些修改。是否继续？';
const GROUPS = [
  ['identity', '训练身份'],
  ['budget', '训练预算'],
  ['data', '数据集与输出'],
  ['runtime', '显存与执行'],
] as const;

export function TrainingWorkspace() {
  const queryClient = useQueryClient();
  const context = useTrainingContext();
  const selectedFile = context.selectedFile;
  const rawQuery = useQuery({
    queryKey: trainingConfigKeys.raw(selectedFile?.path || ''),
    queryFn: ({ signal }) => fetchRawTrainingConfig(selectedFile!.path, signal),
    enabled: Boolean(selectedFile),
  });
  const [baseline, setBaseline] = useState<TrainingDraft>({});
  const [draft, setDraft] = useState<TrainingDraft>({});
  const [hydratedKey, setHydratedKey] = useState('');
  const [saveAsOpen, setSaveAsOpen] = useState(false);
  const [notice, setNotice] = useState('');
  const mergedConfig = context.mergedConfig || {};
  const hydrationKey = `${selectedFile?.path || ''}\0${context.selectedPreset}`;
  const changes = useMemo(() => trainingPatchValues(draft, baseline), [draft, baseline]);
  const dirty = Object.keys(changes).length > 0;
  const ownKeys = useMemo(() => rawConfigOwnKeys(rawQuery.data?.content || ''), [rawQuery.data?.content]);
  useUnsavedChangesGuard(dirty, UNSAVED_MESSAGE);

  useEffect(() => {
    if (!selectedFile || !rawQuery.data || context.isPending || hydratedKey === hydrationKey) return;
    const next = draftFromMerged(mergedConfig);
    setBaseline(next);
    setDraft(next);
    setHydratedKey(hydrationKey);
    setNotice('');
  }, [context.isPending, hydratedKey, hydrationKey, mergedConfig, rawQuery.data, selectedFile]);

  const preview = useMutation({ mutationFn: () => previewTrainingConfigPatch(selectedFile!.path, changes) });
  const save = useMutation({
    mutationFn: () => saveTrainingConfigPatch(selectedFile!.path, changes),
    onSuccess: async (result) => {
      setBaseline({ ...draft });
      setNotice(result.message || '训练配置已保存');
      preview.reset();
      preflight.reset();
      queryClient.setQueryData(trainingConfigKeys.raw(selectedFile!.path), {
        file: selectedFile!.path,
        content: result.content,
        meta: rawQuery.data!.meta,
      });
      await invalidateTrainingQueries(queryClient, selectedFile!.path, context.selectedPreset);
    },
  });
  const saveAs = useMutation({
    mutationFn: async (name: string) => {
      const target = importedTrainingPath(name);
      if (!target) throw new Error('请输入有效配置名称');
      const patched = await previewTrainingConfigPatch(selectedFile!.path, changes);
      const saved = await saveTrainingConfigAs(target, patched.content);
      return { ...saved, file: target };
    },
    onSuccess: async (result) => {
      setSaveAsOpen(false);
      setNotice(result.message || '训练配置已另存');
      await queryClient.invalidateQueries({ queryKey: trainingContextKeys.files() });
      context.selectConfigFile(result.file);
      setHydratedKey('');
    },
  });
  const preflight = useMutation({
    mutationFn: () => runTrainingPreflight(selectedFile!, context.selectedPreset),
  });

  function confirmDiscard(label: string) {
    return !dirty || window.confirm(`当前训练配置有未保存修改。${label}会丢失这些修改，是否继续？`);
  }

  const guardedContext = {
    ...context,
    selectConfigFile: (file: string) => {
      if (confirmDiscard('切换配置')) {
        setHydratedKey('');
        context.selectConfigFile(file);
      }
    },
    selectPreset: (preset: string) => {
      if (confirmDiscard('切换硬件预设')) {
        setHydratedKey('');
        context.selectPreset(preset);
      }
    },
  };
  const locked = Boolean(selectedFile?.locked);
  const busy = preview.isPending || save.isPending || saveAs.isPending || preflight.isPending;

  return (
    <div className="training-config-shell">
      <Topbar />
      <main className="training-config-page">
        <header className="training-config-header">
          <div><p className="eyebrow">TRAINING BLUEPRINT</p><h1>训练配置</h1><p>编辑字段差异、预览 TOML 变更并运行预检测；本页不会启动训练。</p></div>
          <span className="training-readonly-badge" data-editable={!locked}>{locked ? '系统只读 · 可另存' : dirty ? '有未保存修改' : '已同步'}</span>
        </header>
        <TrainingContextBar context={guardedContext} />

        {context.error || rawQuery.error ? (
          <section className="training-config-error" role="alert"><h2>无法读取训练配置</h2><p>{(context.error || rawQuery.error)?.message}</p></section>
        ) : (
          <>
            <section className="training-config-source" aria-busy={context.isPending || rawQuery.isPending}>
              <div><span>方法文件</span><strong>{selectedFile?.path || '—'}</strong></div>
              <div><span>硬件预设</span><strong>{context.selectedPreset}</strong></div>
              <div><span>文件自有字段</span><strong>{ownKeys.size}</strong></div>
              <div><span>待保存</span><strong>{Object.keys(changes).length}</strong></div>
            </section>

            <div className="training-command-bar" aria-label="训练配置操作">
              <button type="button" onClick={() => preview.mutate()} disabled={!dirty || busy}>预览变更</button>
              <button type="button" className="primary-command" onClick={() => save.mutate()} disabled={!dirty || locked || busy}>保存配置</button>
              <button type="button" onClick={() => { saveAs.reset(); setSaveAsOpen(true); }} disabled={!selectedFile || busy}>另存配置</button>
              <button type="button" onClick={() => preflight.mutate()} disabled={dirty || !selectedFile || busy}>运行预检测</button>
            </div>
            {notice ? <p className="training-notice" role="status">{notice}</p> : null}
            {save.error ? <p className="training-command-error" role="alert">{save.error.message}</p> : null}

            <div className="training-editor-layout">
              <div className="training-edit-groups">
                {GROUPS.map(([group, title]) => (
                  <section className="training-edit-card" key={group}>
                    <header><h2>{title}</h2><span>{TRAINING_FIELDS.filter((field) => field.group === group && ownKeys.has(field.key)).length} 当前文件</span></header>
                    <TrainingFieldEditor
                      fields={TRAINING_FIELDS.filter((field) => field.group === group)}
                      draft={draft}
                      ownKeys={ownKeys}
                      disabled={!selectedFile || context.isPending || rawQuery.isPending || busy}
                      onChange={(key, value) => {
                        setDraft((current) => ({ ...current, [key]: value }));
                        preview.reset();
                        preflight.reset();
                      }}
                    />
                  </section>
                ))}
              </div>
              <aside className="training-validation-column">
                <TrainingPatchPreview preview={preview.data} pending={preview.isPending} error={preview.error?.message} />
                <TrainingPreflightPanel result={preflight.data} pending={preflight.isPending} error={preflight.error?.message} />
              </aside>
            </div>
          </>
        )}
      </main>
      {saveAsOpen && selectedFile ? (
        <TrainingSaveAsDialog
          initialName={`${selectedFile.filename?.replace(/\.toml$/i, '') || selectedFile.method || 'training'}_copy`}
          busy={saveAs.isPending}
          error={saveAs.error?.message}
          onCancel={() => { if (!saveAs.isPending) setSaveAsOpen(false); }}
          onConfirm={(name) => saveAs.mutate(name)}
        />
      ) : null}
    </div>
  );
}

async function invalidateTrainingQueries(queryClient: ReturnType<typeof useQueryClient>, file: string, preset: string) {
  await Promise.all([
    queryClient.invalidateQueries({ queryKey: trainingContextKeys.merged(file, preset) }),
    queryClient.invalidateQueries({ queryKey: trainingConfigKeys.raw(file) }),
    queryClient.invalidateQueries({ queryKey: datasetKeys.library() }),
  ]);
}
