import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ImageOff, Trash2, X } from "lucide-react";
import { datasetKeys, fetchDatasetPresetImages } from "../dataset-editor/api";
import type { DatasetPreviewImage, DatasetRow } from "../dataset-editor/types";

type Props = {
  file: string;
  index: number;
  origin: number | null;
  row: DatasetRow;
  previewCurrent: boolean;
  disabled: boolean;
  removeDisabled: boolean;
  onChange: (patch: Partial<DatasetRow>) => void;
  onRemove: () => void;
};

export function TrainingDatasetSubset({
  file,
  index,
  origin,
  row,
  previewCurrent,
  disabled,
  removeDisabled,
  onChange,
  onRemove,
}: Props) {
  const [expanded, setExpanded] = useState<DatasetPreviewImage | null>(null);
  const preview = useQuery({
    queryKey: [...datasetKeys.all, "picker-preview", file, origin],
    queryFn: ({ signal }) => fetchDatasetPresetImages(file, origin!, signal, 8),
    enabled: previewCurrent && origin !== null,
    staleTime: 60_000,
  });
  const count = previewCurrent ? preview.data?.total : undefined;
  return (
    <section
      className="training-dataset-subset"
      aria-label={`子集 ${index + 1}`}
    >
      <header>
        <h4>子集 {index + 1}</h4>
        <span>{row.is_reg ? "正则化" : "训练集"}</span>
        <button
          type="button"
          className="icon-button"
          title={`移除子集 ${index + 1}`}
          aria-label={`移除子集 ${index + 1}`}
          disabled={disabled || removeDisabled}
          onClick={onRemove}
        >
          <Trash2 size={15} />
        </button>
      </header>
      <div className="training-dataset-subset-fields">
        <label>
          原始图片目录
          <input
            aria-label={`子集 ${index + 1} 原始图片目录`}
            value={row.source_dir || ""}
            disabled={disabled}
            onChange={(event) => {
              setExpanded(null);
              onChange({ source_dir: event.target.value });
            }}
          />
        </label>
        <label>
          训练图片目录
          <input
            aria-label={`子集 ${index + 1} 训练图片目录`}
            value={row.image_dir || ""}
            disabled={disabled}
            onChange={(event) => {
              setExpanded(null);
              onChange({ image_dir: event.target.value });
            }}
          />
        </label>
      </div>
      <div className="training-dataset-subset-metrics">
        <div>
          <span>原始图片数量</span>
          <strong>
            {count === undefined ? "—" : count.toLocaleString()} 张
          </strong>
        </div>
        <label>
          重复次数
          <input
            type="number"
            min={1}
            step={1}
            aria-label={`子集 ${index + 1} 重复次数`}
            value={row.num_repeats ?? 1}
            disabled={disabled}
            onChange={(event) =>
              onChange({
                num_repeats:
                  event.target.value === "" ? 0 : Number(event.target.value),
              })
            }
          />
        </label>
        <label className="training-dataset-reg">
          <input
            type="checkbox"
            checked={Boolean(row.is_reg)}
            disabled={disabled}
            onChange={(event) => onChange({ is_reg: event.target.checked })}
          />
          正则化子集
        </label>
      </div>
      {!previewCurrent ? (
        <p role="status">目录修改保存后可预览图像。</p>
      ) : preview.isPending ? (
        <p role="status">正在读取图片数量与预览…</p>
      ) : preview.isError ? (
        <p role="alert">
          图片读取失败{" "}
          <button type="button" onClick={() => void preview.refetch()}>
            重试
          </button>
        </p>
      ) : (
        <>
          <div className="training-dataset-thumbnails">
            {preview.data.images.map((image) => (
              <button
                type="button"
                key={image.file}
                title={image.name}
                aria-label={`预览子集 ${index + 1} 图像 ${image.name}`}
                onClick={() => setExpanded(image)}
              >
                <PreviewImage image={image} />
              </button>
            ))}
          </div>
          {!preview.data.images.length && (
            <p>{preview.data.message || "目录中没有可预览图像"}</p>
          )}
          {preview.data.images.length > 0 && (
            <small>
              预览 {preview.data.images.length} / {preview.data.total} 张
            </small>
          )}
        </>
      )}
      {expanded && (
        <div className="training-dataset-expanded" aria-label="图像详情">
          <header>
            <strong>{expanded.name}</strong>
            <button
              type="button"
              className="icon-button"
              title="收起图像"
              aria-label="收起图像"
              onClick={() => setExpanded(null)}
            >
              <X size={15} />
            </button>
          </header>
          <PreviewImage key={expanded.file} image={expanded} />
          <pre>{expanded.caption?.text || "无标注"}</pre>
        </div>
      )}
    </section>
  );
}

function PreviewImage({ image }: { image: DatasetPreviewImage }) {
  const [failed, setFailed] = useState(false);
  return failed ? (
    <span role="img" aria-label="图像加载失败">
      <ImageOff size={20} />
    </span>
  ) : (
    <img
      src={image.url}
      alt={image.name}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}
