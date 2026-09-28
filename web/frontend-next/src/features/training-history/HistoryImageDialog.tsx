import { useEffect } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { CommandDialog } from "../../components/CommandDialog";
import { ResilientImage } from "../../components/ResilientImage";
import { HistoryInfo } from "./HistoryInfo";
import { historyAssetUrl, type HistoryImage } from "./api";

const scalar = (value: unknown) => typeof value === "string" || typeof value === "number" ? String(value) : undefined;

export function HistoryImageDialog({ image, images, index, taskId, onIndexChange, onClose }: {
  image: HistoryImage;
  images: HistoryImage[];
  index: number;
  taskId: string;
  onIndexChange: (index: number) => void;
  onClose: () => void;
}) {
  const sample = image.sample || {};
  const previous = index > 0;
  const next = index < images.length - 1;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || target.closest("input, textarea, select"))) return;
      if (event.key === "ArrowLeft" && previous) {
        event.preventDefault();
        onIndexChange(index - 1);
      } else if (event.key === "ArrowRight" && next) {
        event.preventDefault();
        onIndexChange(index + 1);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [index, next, onIndexChange, previous]);

  return <CommandDialog title={image.name} onClose={onClose}>
    <ResilientImage className="history-asset-full" src={historyAssetUrl(taskId, image.file)} alt={image.name} retry />
    <nav className="history-image-navigation" aria-label="样张导航">
      <button type="button" aria-label="上一张样张" title="上一张样张" disabled={!previous} onClick={() => onIndexChange(index - 1)}><ChevronLeft aria-hidden="true" size={18} /></button>
      <span aria-live="polite">{index + 1} / {images.length}</span>
      <button type="button" aria-label="下一张样张" title="下一张样张" disabled={!next} onClick={() => onIndexChange(index + 1)}><ChevronRight aria-hidden="true" size={18} /></button>
    </nav>
    <HistoryInfo rows={[
      ["步数", scalar(sample.step ?? sample.steps)], ["Seed", scalar(sample.seed)],
      ["尺寸", image.width && image.height ? `${image.width} x ${image.height}` : undefined],
      ["Prompt", scalar(sample.prompt)], ["权重", scalar(sample.checkpoint ?? sample.weight)],
    ]} />
    <a href={historyAssetUrl(taskId, image.file)} download>下载样张</a>
    <details><summary>原始样张元数据</summary><pre className="history-detail-toml">{JSON.stringify(sample, null, 2)}</pre></details>
  </CommandDialog>;
}
