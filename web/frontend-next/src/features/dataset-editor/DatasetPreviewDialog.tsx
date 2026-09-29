import { useQuery } from '@tanstack/react-query';
import { Copy, Expand, ImageOff, RefreshCw, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { datasetKeys, fetchDatasetPresetImages } from './api';
import { copyText } from './copyText';
import { DatasetImageViewer } from './DatasetImageViewer';
import { DATASET_PREVIEW_PAGE_SIZE, DatasetPreviewPager, hasDatasetPreviewPagination } from './DatasetPreviewPager';
import { trapDialogFocus } from './trapDialogFocus';
import type { DatasetPreviewImage } from './types';
import './DatasetPreview.css';

type Props = {
  file: string;
  datasetIndex: number;
  returnFocus: HTMLElement | null;
  onClose: () => void;
  embedded?: boolean;
};

export function DatasetPreviewDialog({ file, datasetIndex, returnFocus, onClose, embedded = false }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  const viewerImageRef = useRef<DatasetPreviewImage | null>(null);
  const viewerTriggerRef = useRef<HTMLElement | null>(null);
  const [viewerImage, setViewerImage] = useState<DatasetPreviewImage | null>(null);
  const identity = `${file}\0${datasetIndex}`;
  const [page, setPage] = useState({ identity, offset: 0 });
  const offset = page.identity === identity ? page.offset : 0;
  const preview = useQuery({
    queryKey: datasetKeys.preview(file, datasetIndex, offset),
    queryFn: ({ signal }) => fetchDatasetPresetImages(file, datasetIndex, signal, DATASET_PREVIEW_PAGE_SIZE, offset),
    placeholderData: (previous, previousQuery) => previousQuery?.queryKey[2] === file && previousQuery?.queryKey[3] === datasetIndex ? previous : undefined,
    gcTime: 0,
  });
  const currentData = !preview.isPlaceholderData && (preview.data?.offset ?? 0) === offset ? preview.data : undefined;
  const paginationSupported = preview.data && hasDatasetPreviewPagination(preview.data);
  const pageMismatch = paginationSupported && !preview.isPlaceholderData && preview.data!.offset !== offset && preview.data!.total > offset;

  useEffect(() => {
    const data = preview.data;
    if (!data || preview.isPlaceholderData || !hasDatasetPreviewPagination(data) || data.offset === offset) return;
    if (data.total <= offset) {
      setPage({ identity, offset: Math.max(0, Math.floor((data.total - 1) / data.limit) * data.limit) });
    }
  }, [identity, offset, preview.data, preview.isPlaceholderData]);

  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = 0;
    setViewerImage(null);
  }, [identity]);

  useEffect(() => {
    if (embedded) return;
    onCloseRef.current = onClose;
  }, [embedded, onClose]);

  useEffect(() => {
    if (embedded) return;
    viewerImageRef.current = viewerImage;
  }, [viewerImage]);

  useEffect(() => {
    if (embedded) return;
    closeRef.current?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const handleKeyDown = (event: KeyboardEvent) => {
      if (viewerImageRef.current) return;
      trapDialogFocus(event, dialogRef.current);
      if (event.key !== 'Escape') return;
      event.preventDefault();
      onCloseRef.current();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = previousOverflow;
      returnFocus?.focus();
    };
  }, [embedded, returnFocus]);

  function openViewer(image: DatasetPreviewImage, trigger: HTMLElement) {
    viewerTriggerRef.current = trigger;
    setViewerImage(image);
  }

  function changePage(next: number) {
    setPage({ identity, offset: next });
    if (bodyRef.current) bodyRef.current.scrollTop = 0;
  }

  const panel = (
    <div className="dataset-preview-backdrop" role="presentation" onMouseDown={(event) => {
      if (!embedded && event.target === event.currentTarget) onClose();
    }} data-embedded={embedded || undefined}>
      <section ref={dialogRef} className="dataset-preview-dialog" data-embedded={embedded || undefined} role={embedded ? 'region' : 'dialog'} aria-modal={embedded ? undefined : 'true'} aria-labelledby="dataset-preview-title">
        <header className="dataset-preview-header">
          <div>
            <p className="eyebrow">DATASET PREVIEW</p>
            <h3 id="dataset-preview-title">子集 {datasetIndex + 1} 图片与标注</h3>
            <p>{preview.data ? `${preview.data.source_label} · ${preview.data.directory} · ${preview.data.count}/${preview.data.total} 张 · 标注来源 ${preview.data.caption_source_label} · ${preview.data.caption_summary}` : file}</p>
          </div>
          <div className="dataset-preview-actions">
            <button type="button" onClick={() => preview.refetch()} disabled={preview.isFetching}>
              <RefreshCw aria-hidden="true" size={16} />
              {preview.isFetching && preview.data ? '刷新中' : '刷新'}
            </button>
            {!embedded && <button ref={closeRef} type="button" className="dataset-icon-button" aria-label="关闭图片预览" title="关闭" onClick={onClose}>
              <X aria-hidden="true" size={18} />
            </button>}
          </div>
        </header>

        <div ref={bodyRef} className="dataset-preview-body" aria-busy={preview.isPending || preview.isFetching}>
          {preview.isPending || preview.isPlaceholderData ? <p className="dataset-preview-message" role="status" aria-live="polite">正在读取图片与标注</p> : null}
          {preview.isError ? (
            <div className="dataset-preview-message" role="alert">
              <strong>无法读取数据集预览</strong>
              <span>{preview.error.message}</span>
              <button type="button" onClick={() => preview.refetch()}>重试</button>
            </div>
          ) : null}
          {pageMismatch ? <div className="dataset-preview-message" role="alert">
            <strong>分页结果与请求位置不一致</strong>
            <button type="button" onClick={() => changePage(0)}>返回第一页</button>
          </div> : null}
          {paginationSupported ? <DatasetPreviewPager key={`${identity}:${offset}`} data={preview.data!} offset={offset} loading={preview.isFetching || preview.isPlaceholderData} onChange={changePage} /> : null}
          {currentData && !paginationSupported && currentData.total > currentData.images.length ? <p className="dataset-preview-page-warning" role="status">分页接口尚未生效，请重启 WebUI 服务后再翻页。</p> : null}
          {currentData ? (
            <div className="dataset-preview-layout">
              <aside className="dataset-preview-info">
                <dl className="dataset-preview-details">
                  <Detail label="数据集文件" value={currentData.file} />
                  <Detail label="当前目录" value={currentData.directory} />
                  <Detail label="原始路径" value={String(currentData.row.source_dir || '未设置')} />
                  <Detail label="重复次数" value={String(currentData.row.num_repeats || 1)} />
                  <Detail label="分辨率" value={formatResolution(currentData.settings.resolution)} />
                  <Detail label="分桶" value={formatBucket(currentData.settings)} />
                  <Detail label="验证集" value={formatValidation(currentData.settings)} />
                  <Detail label="标注来源" value={currentData.caption_source_label} />
                  <Detail label="识别摘要" value={currentData.caption_summary} />
                </dl>
              </aside>
              <section className="dataset-preview-results" aria-label="预览图片">
                {currentData.images.length ? (
                  <div className="dataset-preview-grid">
                    {currentData.images.map((image) => (
                      <DatasetPreviewCard key={image.file} image={image} onOpen={openViewer} />
                    ))}
                  </div>
                ) : (
                  <p className="dataset-preview-message">{currentData.message || '当前目录没有可预览图片'}</p>
                )}
              </section>
            </div>
          ) : null}
        </div>
      </section>
      {viewerImage ? (
        <DatasetImageViewer
          image={viewerImage}
          returnFocus={viewerTriggerRef.current}
          onClose={() => setViewerImage(null)}
        />
      ) : null}
    </div>
  );
  return embedded ? panel : createPortal(panel, document.body);
}

function DatasetPreviewCard({
  image,
  onOpen,
}: {
  image: DatasetPreviewImage;
  onOpen: (image: DatasetPreviewImage, trigger: HTMLElement) => void;
}) {
  const [imageFailed, setImageFailed] = useState(false);
  const [copyStatus, setCopyStatus] = useState('');

  async function copyCaption() {
    try {
      await copyText(image.caption.text);
      setCopyStatus('已复制');
      window.setTimeout(() => setCopyStatus(''), 1000);
    } catch (error) {
      setCopyStatus(error instanceof Error ? error.message : '复制失败');
    }
  }

  return (
    <article className="dataset-preview-card" data-image-error={imageFailed}>
      <button
        type="button"
        className="dataset-preview-image-button"
        aria-label={`查看大图 ${image.name}`}
        title="查看大图"
        onClick={(event) => onOpen(image, event.currentTarget)}
      >
        {imageFailed ? <ImageOff aria-hidden="true" size={28} /> : (
          <img src={image.url} alt={image.name} loading="lazy" onError={() => setImageFailed(true)} />
        )}
        <span><Expand aria-hidden="true" size={15} />查看大图</span>
      </button>
      <div className="dataset-preview-card-body">
        <strong title={image.file}>{image.name}</strong>
        <span>{image.width && image.height ? `${image.width} x ${image.height}` : '尺寸未知'}</span>
        {image.caption.ok ? (
          <section className="dataset-preview-caption">
            <header>
              <span>{image.caption.format_label || '标注'} · {image.caption.caption_count || 1} 条{image.caption.truncated ? ' · 已截断' : ''}</span>
              <button type="button" aria-label={`复制 ${image.name} 的标注`} onClick={copyCaption}>
                <Copy aria-hidden="true" size={14} />
                {copyStatus || '复制'}
              </button>
            </header>
            <pre>{image.caption.text}</pre>
          </section>
        ) : <p className="dataset-preview-caption-empty">未按当前标注来源找到 caption 文件</p>}
      </div>
    </article>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div><dt>{label}</dt><dd>{value}</dd></div>;
}

function formatResolution(value: unknown) {
  const resolution = Number(value);
  return Number.isFinite(resolution) && resolution > 0 ? `${resolution}px` : '未设置';
}

function formatBucket(settings: Record<string, unknown>) {
  if (settings.enable_bucket === false) return '关闭';
  const min = Number(settings.min_bucket_reso);
  const max = Number(settings.max_bucket_reso);
  const step = Number(settings.bucket_reso_steps);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return '启用';
  return Number.isFinite(step) && step > 0 ? `${min}-${max}px / 步长 ${step}` : `${min}-${max}px`;
}

function formatValidation(settings: Record<string, unknown>) {
  const count = Number(settings.validation_split_num);
  if (Number.isFinite(count) && count > 0) return `固定 ${count} 张`;
  const ratio = Number(settings.validation_split);
  return Number.isFinite(ratio) && ratio > 0 ? `比例 ${ratio}` : '关闭';
}
