import { useState } from "react";
import { ResilientImage } from "../../components/ResilientImage";
import { previewImageUrl, previewWeightUrl, type PreviewImage, type PreviewWeight } from "./api";

export function PreviewAssets({ images, weights, taskId, readOnlyGroup, onDelete, onHotstart }: {
  images: PreviewImage[]; weights: PreviewWeight[]; taskId?: string; readOnlyGroup: boolean; onDelete: (files: string[]) => void; onHotstart: (path: string) => void;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const [active, setActive] = useState<PreviewImage>();
  const [copyNotice, setCopyNotice] = useState("");
  const toggle = (file: string) => setSelected((current) => current.includes(file) ? current.filter((item) => item !== file) : [...current, file]);
  return <>
    <section className="preview-section">
      <header><div><p className="eyebrow">SAMPLES</p><h2>图片预览</h2></div><button type="button" className="danger-button" disabled={readOnlyGroup || !selected.length} onClick={() => onDelete(selected)}>删除所选（{selected.length}）</button></header>
      {readOnlyGroup && <p className="preview-note">配置组聚合多个目录，只能浏览；切换到单个任务后可删除。</p>}
      {images.length ? <div className="preview-image-grid">{images.map((image) => <figure key={image.file}>
        <label><input type="checkbox" checked={selected.includes(image.file)} disabled={readOnlyGroup} onChange={() => toggle(image.file)} />选择</label>
        <button type="button" className="preview-image-open" onClick={() => setActive(image)}><ResilientImage src={image.url || previewImageUrl(image.file, taskId)} alt={image.name} loading="lazy" /><span>{image.name}</span></button>
        {image.source_task?.label && <small>{image.source_task.label}</small>}
      </figure>)}</div> : <p className="preview-empty">当前筛选条件下没有样张。</p>}
    </section>
    <section className="preview-section">
      <header><div><p className="eyebrow">CHECKPOINTS</p><h2>保存的权重</h2></div><span>{weights.length} 个文件</span></header>
      {weights.length ? <div className="preview-weights">{weights.map((weight) => {
        const path = weight.abs_path || weight.file;
        return <article key={weight.file}><div><strong>{weight.name}</strong><small>{[weight.epoch != null && `Epoch ${weight.epoch}`, weight.steps != null && `Step ${weight.steps}`, weight.size_bytes != null && `${(weight.size_bytes / 1048576).toFixed(1)} MB`, weight.mtime_text].filter(Boolean).join(" · ")}</small><code>{path}</code></div>
          <div className="preview-weight-actions"><button type="button" onClick={async () => {
            try { await navigator.clipboard.writeText(path); setCopyNotice("权重路径已复制。"); }
            catch { setCopyNotice(`无法访问剪贴板，请手动复制：${path}`); }
          }}>复制路径</button><a href={weight.download_url || previewWeightUrl(weight.file, weight.source_task?.id || taskId)} download>下载</a><button type="button" onClick={() => onHotstart(path)}>热启动</button></div>
        </article>;
      })}</div> : <p className="preview-empty">当前任务没有可浏览的权重。</p>}
    </section>
    {copyNotice && <p className="preview-notice" role="status">{copyNotice}</p>}
    {active && <dialog open className="preview-image-dialog" onClick={(event) => { if (event.target === event.currentTarget) setActive(undefined); }}><button type="button" aria-label="关闭图片" onClick={() => setActive(undefined)}>关闭</button><ResilientImage src={active.url || previewImageUrl(active.file, taskId)} alt={active.name} retry /><p>{active.name}</p></dialog>}
  </>;
}
