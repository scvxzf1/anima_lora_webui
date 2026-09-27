import { useEffect, useRef, useState, type PointerEventHandler } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ImageOff, Image as ImageIcon } from 'lucide-react';
import { apiRequest } from '../../api/client';
import { datasetKeys } from './api';
import './DatasetCover.css';

type Props = {
  file: string;
  onPointerDown?: PointerEventHandler<HTMLSpanElement>;
};

export function DatasetCover({ file, onPointerDown }: Props) {
  const ref = useRef<HTMLSpanElement>(null);
  const [visible, setVisible] = useState(false);
  const [broken, setBroken] = useState(false);
  useEffect(() => {
    if (!ref.current || typeof IntersectionObserver === 'undefined') return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setVisible(true);
        observer.disconnect();
      }
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  const cover = useQuery({
    queryKey: [...datasetKeys.all, 'cover', file],
    queryFn: ({ signal }) => apiRequest<{ image: string | null; reason: string }>(
      `/api/config/dataset-presets/cover?${new URLSearchParams({ file })}`, { signal },
    ),
    enabled: visible,
    staleTime: 60_000,
    gcTime: 120_000,
    retry: false,
    refetchOnWindowFocus: false,
  });
  useEffect(() => setBroken(false), [cover.data]);
  const failed = broken || cover.isError || (cover.isSuccess && !cover.data.image);
  const reason = broken ? '缩略图读取失败' : cover.isError ? '数据集检测失败' : cover.data?.reason;
  return (
    <span ref={ref} className="dataset-cover" data-drag-enabled={onPointerDown ? 'true' : undefined}
      title={failed ? reason : onPointerDown ? '长按拖动排序预设' : '数据集封面'}
      aria-label={failed ? `无图像：${reason}` : onPointerDown ? '数据集封面，长按拖动排序' : '数据集封面'}
      onPointerDown={onPointerDown}>
      {cover.data?.image && !failed ? (
        <img src={cover.data.image} alt="" width={48} height={48} decoding="async"
          draggable={false}
          onDragStart={(event) => event.preventDefault()}
          onError={() => setBroken(true)} />
      ) : failed ? <><ImageOff size={19} aria-hidden="true" /><small>无图像</small></>
        : <ImageIcon size={20} aria-hidden="true" />}
      {failed && <span className="dataset-cover-tooltip" role="tooltip">{reason}</span>}
    </span>
  );
}
