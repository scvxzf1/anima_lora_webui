import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ArrowLeft, Check, Images, RefreshCw, Save, SlidersHorizontal } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { CommandDialog } from '../../components/CommandDialog';
import { datasetKeys, datasetPresetQuery } from '../dataset-editor/api';
import { DatasetDiscardDialog } from '../dataset-editor/DatasetDiscardDialog';
import { useDatasetDiscardGuard } from '../dataset-editor/useDatasetDiscardGuard';
import { applyMasks, fetchMasks } from './api';
import { MaskImageList } from './MaskImageList';
import { IconCommand, MaskTools } from './MaskTools';
import { MaskViewport, type ViewportHandle } from './MaskViewport';
import type { Tool, View } from './maskCanvas';
import { useMaskImage } from './useMaskImage';
import './MaskEditor.css';

export function MaskEditorPage({ embedded = false }: { embedded?: boolean }) {
  const [params, setParams] = useSearchParams();
  const file = params.get('dataset') || '';
  const urlIndex = Math.max(0, Number(params.get('subset')) || 0);
  const [index, setIndex] = useState(urlIndex);
  const pendingUrlIndex = useRef<number | null>(null);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState('');
  const [tool, setTool] = useState<Tool>('brush');
  const [view, setView] = useState<View>('overlay');
  const [size, setSize] = useState(48);
  const [opacity, setOpacity] = useState(.35);
  const [zoom, setZoom] = useState(1);
  const [showImages, setShowImages] = useState(false);
  const [showTools, setShowTools] = useState(false);
  const [applyDialog, setApplyDialog] = useState(false);
  const [applyTargets, setApplyTargets] = useState<number[]>([index]);
  const viewport = useRef<ViewportHandle>(null);
  const client = useQueryClient();
  const preset = useQuery({ ...datasetPresetQuery(file), enabled: Boolean(file) });
  const page = useQuery({ queryKey: ['dataset-masks', file, index, offset],
    queryFn: ({ signal }) => fetchMasks(file, index, offset, signal), enabled: Boolean(file), retry: false });
  const imageFile = selected || page.data?.images[0]?.file || '';
  const editor = useMaskImage(file, index, imageFile, () => { void page.refetch(); });
  useEffect(() => {
    if (pendingUrlIndex.current !== null) {
      if (editor.dirty || editor.loading || index !== pendingUrlIndex.current) return;
      const frame = requestAnimationFrame(() => {
        pendingUrlIndex.current = null;
        setParams(current => {
          const next = new URLSearchParams(current);
          next.set('subset', String(index));
          return next;
        }, { replace: true });
      });
      return () => cancelAnimationFrame(frame);
    } else if (index !== urlIndex) {
      setIndex(urlIndex);
      setOffset(0);
      setSelected('');
    }
  }, [editor.dirty, editor.loading, index, setParams, urlIndex]);
  const apply = useMutation({ mutationFn: () => applyMasks(file, index, page.data!.config_revision, applyTargets),
    onSuccess: async () => {
      setApplyDialog(false);
      await client.invalidateQueries({ queryKey: datasetKeys.all });
      await page.refetch(); editor.reload();
    } });
  const busy = editor.saving || apply.isPending;
  const guard = useDatasetDiscardGuard(editor.dirty, busy);
  const disabled = busy || !editor.engine || Boolean(page.data?.readonly || editor.meta?.readonly);
  async function switchTo(action: () => void, label: string) {
    if (await guard.confirmDiscard(label)) { apply.reset(); action(); }
  }
  const name = page.data?.images.find(image => image.file === imageFile)?.name || '';
  return <main className="mask-editor" data-images-open={showImages} data-tools-open={showTools}>
    <header className="mask-header">
      {!embedded && <Link className="mask-back" to={`/datasets?${new URLSearchParams({ dataset: file })}`} title="返回数据集" aria-label="返回数据集"><ArrowLeft size={20} /></Link>}
      <div className="mask-heading"><h1>蒙版编辑</h1><span title={file}>{preset.data?.name || file || '未选择数据集'}</span></div>
      <span className="mask-save-state" role="status">{editor.saving ? '保存中' : editor.dirty ? '未保存' : editor.notice || (editor.meta?.has_mask ? '已保存' : '尚无蒙版')}</span>
      <button className="primary-command" aria-label="保存" title="保存蒙版" disabled={disabled || (!editor.dirty && Boolean(editor.meta?.has_mask))} onClick={() => void editor.save()}><Save size={16} />{editor.saving ? '保存中' : !editor.dirty && editor.meta?.has_mask ? '已保存' : '保存'}</button>
      <button className="mask-apply" aria-label="应用到子集" title="应用到子集" disabled={busy || editor.dirty || !page.data || page.data.readonly} onClick={() => { setApplyTargets([index]); apply.reset(); setApplyDialog(true); }}><Check size={16} /><span>应用到子集</span></button>
    </header>
    <div className="mask-context">
      <IconCommand icon={Images} label="展开或收起图片列表" active={showImages} className="icon-button mask-list-toggle" onClick={() => setShowImages(value => !value)} />
      <label>子集 <select aria-label="蒙版子集" value={index} disabled={busy} onChange={event => {
        const value = Number(event.target.value);
        void switchTo(() => { pendingUrlIndex.current = value; setIndex(value); setOffset(0); setSelected(''); }, '切换子集');
      }}>{preset.data?.datasets.map((row, i) => <option key={i} value={i}>{i + 1}. {row.source_dir}</option>)}</select></label>
      <span className="mask-filename" title={name}>{name}</span>
      {editor.meta && <span className="mask-dimensions">{editor.meta.width} × {editor.meta.height} · {editor.meta.basis === 'training' ? '训练图' : '预处理预览'}</span>}
      <IconCommand icon={RefreshCw} label="重新加载蒙版" disabled={busy || !imageFile} onClick={() => void switchTo(editor.reload, '重新加载')} />
      <IconCommand icon={SlidersHorizontal} label="展开或收起蒙版工具" active={showTools} className="icon-button mask-list-toggle" onClick={() => setShowTools(value => !value)} />
    </div>
    {(page.error || preset.error || editor.error || apply.error) && <div className="mask-error" role="alert">
      {editor.error || apply.error?.message || page.error?.message || preset.error?.message}
      <button disabled={busy} onClick={() => void switchTo(() => { void page.refetch(); void preset.refetch(); editor.reload(); }, '重新加载')}>重试加载</button>
    </div>}
    {page.data?.readonly && <p className="mask-error" role="status">此数据集已锁定或只读</p>}
    <div className="mask-workspace">
      {page.data && <MaskImageList page={page.data} selected={imageFile} disabled={busy}
        onSelect={value => { if (value !== imageFile) void switchTo(() => { setSelected(value); setShowImages(false); }, '切换图片'); }}
        onPage={value => void switchTo(() => { setOffset(value); setSelected(''); }, '翻页')} />}
      <section className="mask-canvas-area" aria-label="编辑区域" aria-busy={editor.loading}>
        {editor.engine ? <MaskViewport ref={viewport} engine={editor.engine} tool={tool} size={size}
          view={view} opacity={opacity} disabled={disabled} onChange={editor.changed} onZoom={setZoom} />
          : <div className="mask-empty" role="status">{editor.loading || page.isFetching ? '正在加载图片' : page.data?.message || (file ? '暂无可编辑图片' : '未选择数据集')}</div>}
      </section>
      <MaskTools engine={editor.engine} tool={tool} setTool={setTool} size={size} setSize={setSize}
        view={view} setView={setView} opacity={opacity} setOpacity={setOpacity} disabled={disabled}
        changed={editor.changed} viewport={viewport} zoom={zoom} />
    </div>
    <footer className="mask-footer"><span className="mask-mode">{page.data?.mask_mode === 'external' ? '外部蒙版已启用' : '外部蒙版未启用'}</span>
      <span title={page.data?.mask_dir}>{page.data?.mask_dir}</span>
      {apply.isSuccess && <span role="status">已应用，后续启动的训练生效</span>}
    </footer>
    {guard.discardDialog && <DatasetDiscardDialog {...guard.discardDialog} />}
    {applyDialog && <CommandDialog title="应用蒙版到此子集？" onClose={() => { if (!apply.isPending) setApplyDialog(false); }}>
      <p>将当前目录中的蒙版应用到选中的子集。所有引用此数据集的训练配置都会受影响，已运行的训练不会实时更新。</p>
      <fieldset className="unframed-fieldset settings-grid" disabled={apply.isPending}>
        <legend>目标子集（蒙版目录：{page.data?.mask_dir}）</legend>
        {preset.data?.datasets.map((row, targetIndex) => (
          <label key={targetIndex} className="checkbox-row">
            <input type="checkbox" checked={applyTargets.includes(targetIndex)}
              onChange={event => setApplyTargets(current => event.target.checked
                ? [...current, targetIndex].sort((a, b) => a - b)
                : current.filter(value => value !== targetIndex))} />
            <span>{targetIndex + 1}. {row.source_dir}</span>
          </label>
        ))}
      </fieldset>
      <p>选中的子集共用上述目录；仅同名图片使用已有蒙版，未匹配蒙版的图片按整图训练。原有图像 Alpha 模式会被替换。</p>
      <p className="effective-path">{page.data?.mask_dir}</p>
      {apply.error && <p role="alert">{apply.error.message}</p>}
      <footer className="toolbar"><button disabled={apply.isPending} onClick={() => setApplyDialog(false)}>取消</button>
        <button className="primary-command" disabled={apply.isPending || applyTargets.length === 0} onClick={() => apply.mutate()}>确认应用到 {applyTargets.length} 个子集</button></footer>
    </CommandDialog>}
  </main>;
}
