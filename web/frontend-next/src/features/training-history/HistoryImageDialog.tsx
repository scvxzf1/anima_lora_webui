import { CommandDialog } from "../../components/CommandDialog";
import { ResilientImage } from "../../components/ResilientImage";
import { HistoryInfo } from "./HistoryInfo";
import { historyAssetUrl, type HistoryImage } from "./api";

const scalar = (value: unknown) => typeof value === "string" || typeof value === "number" ? String(value) : undefined;

export function HistoryImageDialog({ image, taskId, onClose }: { image: HistoryImage; taskId: string; onClose: () => void }) {
  const sample = image.sample || {};
  return <CommandDialog title={image.name} onClose={onClose}>
    <ResilientImage className="history-asset-full" src={historyAssetUrl(taskId, image.file)} alt={image.name} retry />
    <HistoryInfo rows={[
      ["步数", scalar(sample.step ?? sample.steps)], ["Seed", scalar(sample.seed)],
      ["尺寸", image.width && image.height ? `${image.width} x ${image.height}` : undefined],
      ["Prompt", scalar(sample.prompt)], ["权重", scalar(sample.checkpoint ?? sample.weight)],
    ]} />
    <a href={historyAssetUrl(taskId, image.file)} download>下载样张</a>
    <details><summary>原始样张元数据</summary><pre className="history-detail-toml">{JSON.stringify(sample, null, 2)}</pre></details>
  </CommandDialog>;
}
