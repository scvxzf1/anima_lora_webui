import type { CaptionItem } from "./api";
import type { Drafts } from "./captionBatch";
export function CaptionBatchConflict({ items, drafts, disabled, onRebase }: {
  items: CaptionItem[]; drafts: Drafts; disabled: boolean; onRebase: () => void;
}) {
  return <div className="caption-batch">
    <p role="status">服务器候选与草稿不同，草稿已保留。请逐项审阅差异后采用新基线；仍需再次保存，不会自动重试。</p>
    <div className="caption-batch-diff">{Object.entries(drafts).map(([id, local]) => {
      const item = items.find((entry) => entry.id === id);
      return <article key={id}><strong>{item?.name || id}</strong><p>服务器：{item?.proposed_caption ?? "候选已不存在"}</p><p>本地草稿：{local}</p></article>;
    })}</div>
    <button type="button" disabled={disabled || Object.keys(drafts).some((id) => !items.some((item) => item.id === id))} onClick={onRebase}>保留草稿并采用服务器基线</button>
  </div>;
}
