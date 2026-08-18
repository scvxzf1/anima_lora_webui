import { useQuery } from '@tanstack/react-query';
import { Copy, Expand, ImageOff, RefreshCw, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { datasetKeys, fetchDatasetPresetImages } from './api';
import { copyText } from './copyText';
import { DatasetImageViewer } from './DatasetImageViewer';
import { trapDialogFocus } from './trapDialogFocus';
import type { DatasetPreviewImage } from './types';
import './DatasetPreview.css';

type Props = {
  file: string;
  datasetIndex: number;
  returnFocus: HTMLElement | null;
  onClose: () => void;
};

export function DatasetPreviewDialog({ file, datasetIndex, returnFocus, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const onCloseRef = useRef(onClose);
  const viewerImageRef = useRef<DatasetPreviewImage | null>(null);
  const viewerTriggerRef = useRef<HTMLElement | null>(null);
  const [viewerImage, setViewerImage] = useState<DatasetPreviewImage | null>(null);
  const preview = useQuery({
    queryKey: datasetKeys.preview(file, datasetIndex),
    queryFn: ({ signal }) => fetchDatasetPresetImages(file, datasetIndex, signal),
  });

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    viewerImageRef.current = viewerImage;
  }, [viewerImage]);

  useEffect(() => {
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
  }, [returnFocus]);

  function openViewer(image: DatasetPreviewImage, trigger: HTMLElement) {
    viewerTriggerRef.current = trigger;
    setViewerImage(image);
  }

  return createPortal(
    <div className="dataset-preview-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section ref={dialogRef} className="dataset-preview-dialog" role="dialog" aria-modal="true" aria-labelledby="dataset-preview-title">
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
            <button ref={closeRef} type="button" className="dataset-icon-button" aria-label="关闭图片预览" title="关闭" onClick={onClose}>
              <X aria-hidden="true" size={18} />
            </button>
          </div>
        </header>

        <div className="dataset-preview-body" aria-busy={preview.isPending || preview.isFetching}>
          {preview.isPending ? <p className="dataset-preview-message">正在读取图片与标注</p> : null}
          {preview.isError ? (
            <div className="dataset-preview-message" role="alert">
              <strong>无法读取数据集预览</strong>
              <span>{preview.error.message}</span>
              <button type="button" onClick={() => preview.refetch()}>重试</button>
            </div>
          ) : null}
          {preview.data ? (
            <div className="dataset-preview-layout">
              <aside className="dataset-preview-info">
                <dl className="dataset-preview-details">
                  <Detail label="数据集文件" value={preview.data.file} />
                  <Detail label="当前目录" value={preview.data.directory} />
                  <Detail label="原始路径" value={String(preview.data.row.source_dir || '未设置')} />
                  <Detail label="重复次数" value={String(preview.data.row.num_repeats || 1)} />
                  <Detail label="分辨率" value={formatResolution(preview.data.settings.resolution)} />
                  <Detail label="分桶" value={formatBucket(preview.data.settings)} />
                  <Detail label="验证集" value={formatValidation(preview.data.settings)} />
                  <Detail label="标注来源" value={preview.data.caption_source_label} />
                  <Detail label="识别摘要" value={preview.data.caption_summary} />
                </dl>
              </aside>
              <section className="dataset-preview-results" aria-label="预览图片">
                {preview.data.images.length ? (
                  <div className="dataset-preview-grid">
                    {preview.data.images.map((image) => (
                      <DatasetPreviewCard key={image.file} image={image} onOpen={openViewer} />
                    ))}
                  </div>
                ) : (
                  <p className="dataset-preview-message">{preview.data.message || '当前目录没有可预览图片'}</p>
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
    </div>,
    document.body,
  );
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
