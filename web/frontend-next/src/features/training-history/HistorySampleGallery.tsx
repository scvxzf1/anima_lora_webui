import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { ResilientImage } from "../../components/ResilientImage";
import { AssetPagination } from "./AssetPagination";
import { HistoryImageDialog } from "./HistoryImageDialog";
import { fetchHistoryImages, historyAssetUrl, type HistoryImage } from "./api";
import { groupHistorySamples, sampleStep } from "./historyAssetPresentation";

export function HistorySampleGallery({ taskId }: { taskId: string }) {
  const [offset, setOffset] = useState(0);
  const [view, setView] = useState("recent");
  const [selectedFile, setSelectedFile] = useState<string>();
  const query = useQuery({
    queryKey: ["history-images", taskId, offset],
    queryFn: ({ signal }) => fetchHistoryImages(taskId, signal, 60, offset),
  });
  const images = query.data?.images || [];
  const groups = view === "grouped" ? groupHistorySamples(images) : [];
  const orderedImages = view === "grouped" ? groups.flatMap((group) => group.images) : images;
  const selectedIndex = orderedImages.findIndex((image) => image.file === selectedFile);
  useEffect(() => {
    if (selectedFile && query.isSuccess && selectedIndex < 0) setSelectedFile(undefined);
  }, [query.isSuccess, selectedFile, selectedIndex]);
  const thumbnail = (image: HistoryImage) => <SampleThumbnail key={image.file} taskId={taskId} image={image} onSelect={setSelectedFile} />;
  return <section className="history-asset-section" aria-label="训练样张">
    <header className="history-asset-heading">
      <div><h2>训练样张</h2><p>先看效果，再选择权重 · {view === "grouped" ? "本页按 Prompt / Seed 分组，步数从早到晚" : "最新文件优先"}</p></div>
      <label>浏览方式 <select value={view} onChange={(event) => setView(event.target.value)}>
        <option value="recent">最新样张</option><option value="grouped">Prompt / Seed 演进</option>
      </select></label>
    </header>
    {query.isPending && <p role="status">正在读取样张</p>}
    {query.error && <p role="alert">{query.error.message} <button type="button" onClick={() => query.refetch()}>重试样张</button></p>}
    {view === "grouped" ? groups.map((group) => <section className="history-sample-group" key={group.key}>
      <h3>{group.label}</h3>{group.prompt && <p title={group.prompt}>{group.prompt}</p>}
      <div className="history-image-grid history-sample-evolution">{group.images.map(thumbnail)}</div>
    </section>) : <div className="history-image-grid">{images.map(thumbnail)}</div>}
    {query.data?.images.length === 0 && <p>{query.data.message || "本页暂无样张"}</p>}
    <AssetPagination label="样张" offset={offset} count={images.length} total={query.data?.total ?? 0} size={60}
      next={query.data?.next_offset} pending={query.isFetching} onChange={(next) => { setSelectedFile(undefined); setOffset(next); }} />
    {selectedIndex >= 0 && <HistoryImageDialog image={orderedImages[selectedIndex]} images={orderedImages} index={selectedIndex}
      taskId={taskId} onIndexChange={(index) => setSelectedFile(orderedImages[index]?.file)} onClose={() => setSelectedFile(undefined)} />}
  </section>;
}

function SampleThumbnail({ taskId, image, onSelect }: { taskId: string; image: HistoryImage; onSelect: (file: string) => void }) {
  const step = sampleStep(image);
  const label = step !== undefined ? `Step ${step}` : image.name;
  const accessibleLabel = label === image.name ? image.name : `${label} ${image.name}`;
  return <button type="button" aria-label={`查看 ${accessibleLabel} 的生成参数`} title={image.name} onClick={() => onSelect(image.file)}>
    <ResilientImage className="history-thumbnail" src={historyAssetUrl(taskId, image.file)} alt={image.name} loading="lazy" />
    <span className="history-sample-step">{label}</span>
  </button>;
}
