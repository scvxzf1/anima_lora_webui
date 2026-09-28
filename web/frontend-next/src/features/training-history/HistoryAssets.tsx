import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Copy } from "lucide-react";
import { ResilientImage } from "../../components/ResilientImage";
import { AssetPagination } from "./AssetPagination";
import { HistoryImageDialog } from "./HistoryImageDialog";
import { HistoryArtifacts } from "./HistoryArtifacts";
import { fetchHistoryImages, fetchHistoryWeights, historyAssetUrl, type HistoryImage } from "./api";

export function HistoryAssets({ taskId }: { taskId: string }) {
  return <TaskAssets key={taskId} taskId={taskId} />;
}

function TaskAssets({ taskId }: { taskId: string }) {
  const [imageOffset, setImageOffset] = useState(0);
  const [weightOffset, setWeightOffset] = useState(0);
  const [sort, setSort] = useState("recent");
  const [selected, setSelected] = useState<HistoryImage>();
  const [copyStatus, setCopyStatus] = useState("");
  const images = useQuery({
    queryKey: ["history-images", taskId, imageOffset],
    queryFn: ({ signal }) => fetchHistoryImages(taskId, signal, 60, imageOffset),
  });
  const weights = useQuery({
    queryKey: ["history-weights", taskId, weightOffset, sort],
    queryFn: ({ signal }) => fetchHistoryWeights(taskId, signal, 100, weightOffset, sort),
  });
  return <section className="history-assets">
    <HistoryArtifacts taskId={taskId} />
    <h2>训练样张</h2>
    <p className="data-scope">最新文件优先</p>
    {images.isPending && <p role="status">正在读取样张</p>}
    {images.error && <p role="alert">{images.error.message} <button type="button" onClick={() => images.refetch()}>重试样张</button></p>}
    <div className="history-image-grid">
      {images.data?.images.map((image) => <button key={image.file} type="button" onClick={() => setSelected(image)}>
        <ResilientImage className="history-thumbnail" src={historyAssetUrl(taskId, image.file)} alt={image.name} loading="lazy" /><span>{image.name}</span>
      </button>)}
    </div>
    {images.data?.images.length === 0 && <p>{images.data.message || "本页暂无样张"}</p>}
    <AssetPagination label="样张" offset={imageOffset} count={images.data?.images.length ?? 0} total={images.data?.total ?? 0} size={60} next={images.data?.next_offset} pending={images.isFetching} onChange={setImageOffset} />
    <div className="toolbar"><h2>权重</h2><label>排序 <select value={sort} onChange={(event) => { setSort(event.target.value); setWeightOffset(0); }}><option value="recent">最新文件优先</option><option value="name">文件名</option></select></label></div>
    {weights.isPending && <p role="status">正在读取权重</p>}
    {weights.error && <p role="alert">{weights.error.message} <button type="button" onClick={() => weights.refetch()}>重试权重</button></p>}
    <div className="weight-list">
      {weights.data?.weights.map((weight) => <div key={weight.file}>
        <div className="history-weight-file"><a href={historyAssetUrl(taskId, weight.file, true)} download>{weight.name}</a><code>{(weight as typeof weight & { abs_path?: string }).abs_path || weight.file}</code></div>
        <span>{(weight.size_bytes / 1048576).toFixed(1)} MB · {weight.scope_label}</span>
        <button type="button" aria-label={`复制 ${weight.name} 的本地路径`} title="复制本地路径" onClick={() => void copyWeightPath((weight as typeof weight & { abs_path?: string }).abs_path || weight.file, weight.name, setCopyStatus)}>
          <Copy aria-hidden="true" size={15} />
        </button>
      </div>)}
    </div>
    <p className="history-copy-status" role="status" aria-live="polite">{copyStatus}</p>
    {weights.data?.weights.length === 0 && <p>{weights.data.message || "本页暂无权重"}</p>}
    <AssetPagination label="权重" offset={weightOffset} count={weights.data?.weights.length ?? 0} total={weights.data?.total ?? 0} size={100} next={weights.data?.next_offset} pending={weights.isFetching} onChange={setWeightOffset} />
    {selected && <HistoryImageDialog image={selected} taskId={taskId} onClose={() => setSelected(undefined)} />}
  </section>;
}

async function copyWeightPath(path: string, name: string, setStatus: (message: string) => void) {
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
    await navigator.clipboard.writeText(path);
    setStatus(`${name}的路径已复制。`);
  } catch {
    setStatus(`无法复制${name}的路径，请检查剪贴板权限或手动选择显示的路径。`);
  }
}
