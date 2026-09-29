import { ChevronLeft, ChevronRight } from "lucide-react";
import type { FormEvent } from "react";

import type { DatasetPreviewResponse } from "./types";

export const DATASET_PREVIEW_PAGE_SIZE = 24;

export function hasDatasetPreviewPagination(data: DatasetPreviewResponse) {
  return (
    Number.isInteger(data.offset) &&
    data.offset! >= 0 &&
    Number.isInteger(data.limit) &&
    data.limit > 0 &&
    typeof data.has_more_before === "boolean" &&
    typeof data.has_more_after === "boolean"
  );
}

export function DatasetPreviewPager({
  data,
  offset,
  loading,
  onChange,
}: {
  data: DatasetPreviewResponse;
  offset: number;
  loading: boolean;
  onChange: (offset: number) => void;
}) {
  if (!hasDatasetPreviewPagination(data)) return null;
  const limit = data.limit;
  const pages = Math.max(1, Math.ceil(data.total / limit));
  if (pages <= 1) return null;
  const page = Math.min(pages, Math.floor(offset / limit) + 1);
  const current = !loading && data.offset === offset;
  const previous = current ? data.has_more_before === true : offset > 0;
  const next = current ? data.has_more_after === true : page < pages;

  function jump(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = Number(new FormData(event.currentTarget).get("page"));
    if (!Number.isInteger(value) || value < 1 || value > pages) return;
    onChange((value - 1) * limit);
  }

  return (
    <nav className="dataset-preview-pagination" aria-label="图片分页">
      <button
        type="button"
        aria-label="上一页"
        title="上一页"
        disabled={!previous}
        onClick={() => onChange(Math.max(0, offset - limit))}
      >
        <ChevronLeft aria-hidden="true" size={18} />
      </button>
      <span className="dataset-preview-page-status">
        第 {page} / {pages} 页 · 共 {data.total} 张
      </span>
      <form onSubmit={jump}>
        <label>
          页码{" "}
          <input
            type="number"
            name="page"
            min={1}
            max={pages}
            step={1}
            defaultValue={page}
            aria-label="跳转页码"
          />
        </label>
        <button type="submit">跳转</button>
      </form>
      <button
        type="button"
        aria-label="下一页"
        title="下一页"
        disabled={!next}
        onClick={() => onChange(offset + limit)}
      >
        <ChevronRight aria-hidden="true" size={18} />
      </button>
    </nav>
  );
}
