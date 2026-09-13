import { Check, ChevronLeft, ChevronRight } from 'lucide-react';
import { IconCommand } from './MaskTools';
import type { MaskPage } from './api';

export function MaskImageList({ page, selected, disabled, onSelect, onPage }: {
  page: MaskPage; selected: string; disabled: boolean;
  onSelect: (file: string) => void; onPage: (offset: number) => void;
}) {
  return <aside className="mask-images" aria-label="数据集图片">
    <header><strong>图片</strong><span>{page.total} 张</span></header>
    <div className="mask-image-items">
      {page.images.map(image => <button type="button" key={image.file} title={image.name}
        className="mask-image-item" aria-current={selected === image.file ? 'true' : undefined}
        disabled={disabled} onClick={() => onSelect(image.file)}>
        <img src={image.thumbnail_url || image.url} alt="" loading="lazy" />
        <span>{image.name}</span>
        {image.has_mask && <Check size={15} aria-label="已有蒙版" />}
      </button>)}
    </div>
    <footer><IconCommand icon={ChevronLeft} label="上一页图片" disabled={disabled || page.offset === 0}
      onClick={() => onPage(Math.max(0, page.offset - 48))} />
      <span>{page.total ? Math.floor(page.offset / 48) + 1 : 0} / {Math.ceil(page.total / 48)}</span>
      <IconCommand icon={ChevronRight} label="下一页图片" disabled={disabled || !page.has_more_after}
        onClick={() => onPage(page.next_offset)} /></footer>
  </aside>;
}
