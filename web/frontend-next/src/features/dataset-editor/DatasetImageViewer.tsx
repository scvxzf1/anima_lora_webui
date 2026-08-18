import { Copy, ImageOff, RefreshCw, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { copyText } from './copyText';
import { trapDialogFocus } from './trapDialogFocus';
import type { DatasetPreviewImage } from './types';

type Props = {
  image: DatasetPreviewImage;
  returnFocus: HTMLElement | null;
  onClose: () => void;
};

export function DatasetImageViewer({ image, returnFocus, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const onCloseRef = useRef(onClose);
  const [copyStatus, setCopyStatus] = useState('');
  const [imageFailed, setImageFailed] = useState(false);
  const [reloadIndex, setReloadIndex] = useState(0);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    closeRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      trapDialogFocus(event, dialogRef.current);
      if (event.key !== 'Escape') return;
      event.preventDefault();
      onCloseRef.current();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      returnFocus?.focus();
    };
  }, [returnFocus]);

  async function copyCaption() {
    try {
      await copyText(image.caption.text);
      setCopyStatus('标注已复制');
    } catch (error) {
      setCopyStatus(error instanceof Error ? error.message : '复制失败');
    }
  }

  return createPortal(
    <div className="dataset-image-viewer-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section ref={dialogRef} className="dataset-image-viewer" role="dialog" aria-modal="true" aria-labelledby="dataset-image-viewer-title">
        <header>
          <div>
            <p className="eyebrow">图片详情</p>
            <h3 id="dataset-image-viewer-title">{image.name}</h3>
          </div>
          <button ref={closeRef} type="button" className="dataset-icon-button" aria-label="关闭大图" title="关闭" onClick={onClose}>
            <X aria-hidden="true" size={18} />
          </button>
        </header>
        <div className="dataset-image-viewer-body">
          <div className="dataset-image-viewer-canvas">
            {imageFailed ? (
              <div className="dataset-image-viewer-error" role="alert">
                <ImageOff aria-hidden="true" size={30} />
                <strong>图片加载失败</strong>
                <button type="button" onClick={() => {
                  setImageFailed(false);
                  setReloadIndex((current) => current + 1);
                }}>
                  <RefreshCw aria-hidden="true" size={15} />
                  重试
                </button>
              </div>
            ) : (
              <img key={reloadIndex} src={image.url} alt={image.name} onError={() => setImageFailed(true)} />
            )}
          </div>
          <aside>
            <dl className="dataset-preview-details">
              <Detail label="图片路径" value={image.file} />
              <Detail label="尺寸" value={image.width && image.height ? `${image.width} x ${image.height}` : '未知'} />
              <Detail label="像素" value={formatPixels(image.total_pixels)} />
              <Detail label="文件大小" value={formatBytes(image.size_bytes)} />
              <Detail label="修改时间" value={image.mtime_text || '未知'} />
              <Detail label="标注文件" value={image.caption.file || '未找到'} />
            </dl>
            <section className="dataset-viewer-caption" aria-label="图片标注">
              <header>
                <strong>{image.caption.ok ? image.caption.format_label || '标注' : '无标注'}</strong>
                {image.caption.ok ? (
                  <button type="button" onClick={copyCaption}>
                    <Copy aria-hidden="true" size={15} />
                    复制
                  </button>
                ) : null}
              </header>
              <pre>{image.caption.ok ? image.caption.text : '未按当前标注来源找到 caption 文件'}</pre>
              {copyStatus ? <p role="status">{copyStatus}</p> : null}
            </section>
          </aside>
        </div>
      </section>
    </div>,
    document.body,
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div><dt>{label}</dt><dd>{value}</dd></div>;
}

function formatBytes(value?: number) {
  if (!value) return '未知';
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

function formatPixels(value?: number) {
  if (!value) return '未知';
  return value >= 1_000_000 ? `${(value / 1_000_000).toFixed(2)} MP` : value.toLocaleString();
}
