import { useCallback, useState } from 'react';
import type { DatasetFormValues } from './datasetForm';

type ImageCounts = Record<string, number | null>;

export function useDatasetImageCounts() {
  const [imageCounts, setImageCounts] = useState<ImageCounts>({});
  const onImageCount = useCallback((source: string, count: number | null) => {
    if (!source) return;
    setImageCounts((current) => current[source] === count ? current : { ...current, [source]: count });
  }, []);
  return { imageCounts, onImageCount };
}

export function summarizeDatasetImages(rows: DatasetFormValues['datasets'], counts: ImageCounts) {
  const included = rows.filter((row) => row.edit_role !== 'before');
  const total = (items: typeof rows, repeated = false) => {
    if (items.some((row) => counts[row.source_dir.trim()] == null)) return null;
    return items.reduce((sum, row) => sum + counts[row.source_dir.trim()]!
      * (repeated ? Math.max(1, Number(row.num_repeats) || 1) : 1), 0);
  };
  return {
    train: total(included.filter((row) => !row.is_reg)),
    reg: total(included.filter((row) => row.is_reg)),
    repeats: total(included, true),
  };
}

export function DatasetSummary({ rows, counts, resolution, batchSize }: {
  rows: DatasetFormValues['datasets'];
  counts: ImageCounts;
  resolution: number;
  batchSize: number;
}) {
  const summary = summarizeDatasetImages(rows, counts);
  const sourceHint = '按原始目录递归检测的图片数；不含路径筛选、验证拆分及高级规则影响。检测未完成或失败时显示 —。';
  return (
    <dl className="dataset-summary-grid" aria-label="当前蓝图统计">
      <div><dt>子集</dt><dd>{rows.length}</dd></div>
      <div title={sourceHint}><dt>训练图</dt><dd>{summary.train ?? '—'}</dd></div>
      <div title={sourceHint}><dt>正则图</dt><dd>{summary.reg ?? '—'}</dd></div>
      <div title={`原始目录图片数 × 各子集重复次数，非最终训练样本数。${sourceHint}`}>
        <dt>总重复</dt><dd>{summary.repeats ?? '—'}</dd>
      </div>
      <div><dt>分辨率</dt><dd>{resolution}</dd></div>
      <div><dt>批大小</dt><dd>{batchSize}</dd></div>
    </dl>
  );
}
