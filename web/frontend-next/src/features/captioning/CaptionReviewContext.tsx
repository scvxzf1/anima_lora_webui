import { LocateFixed } from "lucide-react";

type Item = { id: string; name: string };

export function CaptionReviewContext({ items, visibleItems = items, current, page, drafts, checked, onLocate, mobileView, onView }: {
  items: Item[]; visibleItems?: Item[]; current?: Item; page: number; drafts: Record<string, string>; checked: string[];
  onLocate: (id: string) => void; mobileView: "editor" | "items"; onView: (view: "editor" | "items") => void;
}) {
  const pageFor = (id: string) => Math.floor(visibleItems.findIndex((item) => item.id === id) / 50);
  const pageLabel = (id: string) => pageFor(id) < 0 ? "不在当前筛选中" : `第 ${pageFor(id) + 1} 页`;
  const currentPage = current ? pageFor(current.id) : -1;
  const draftItems = items.filter((item) => drafts[item.id] !== undefined);
  const hiddenChecked = checked.filter((id) => !visibleItems.slice(page * 50, (page + 1) * 50).some((item) => item.id === id)).length;
  return <div className="caption-review-context">
    {current ? <div className="toolbar caption-current-object" role="status">
      <span>当前编辑：{current.name} · {pageLabel(current.id)}</span>
      {currentPage !== page ? <button type="button" onClick={() => onLocate(current.id)}>
        <LocateFixed size={15} />定位当前候选
      </button> : null}
    </div> : null}
    <p className="data-scope">已勾选 {checked.length} 项{hiddenChecked ? `（${visibleItems.length !== items.length ? "当前页外" : "其他页"} ${hiddenChecked} 项）` : ""} · 未保存草稿 {draftItems.length} 项</p>
    {draftItems.length ? <details className="caption-draft-list">
      <summary>未保存草稿 ({draftItems.length})</summary>
      {draftItems.map((item) => <button type="button" key={item.id} onClick={() => onLocate(item.id)}>
        <LocateFixed size={14} />{item.name} · {pageLabel(item.id)}
      </button>)}
    </details> : null}
    <div className="caption-mobile-views" role="group" aria-label="审阅工作区">
      <button type="button" aria-pressed={mobileView === "editor"} onClick={() => onView("editor")}>当前图片</button>
      <button type="button" aria-pressed={mobileView === "items"} onClick={() => onView("items")}>候选列表 ({visibleItems.length}{visibleItems.length !== items.length ? `/${items.length}` : ""})</button>
    </div>
  </div>;
}
