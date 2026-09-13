export type EstimateBucket = { width: number; height: number; count: number };
export type EstimateDataset = {
  index: number;
  is_reg?: boolean;
  source_dir: string;
  image_dir: string;
  train_image_count: number;
  num_repeats: number;
  sample_ratio: number;
  sampled_image_count: number;
  sampled_weighted_image_count: number;
  trigger_clone_sampled_weighted_image_count: number;
  uses_preprocessed_images: boolean;
  bucket_distribution?: {
    basis: "image_dimensions" | "source_projection";
    status: "ready" | "partial" | "pending";
    image_count: number;
    unreadable_count: number;
    unscanned_count?: number;
    filtered_count?: number;
    buckets: EstimateBucket[];
  };
};
export type TrainingEstimateData = {
  total_steps: number;
  train_image_count: number;
  effective_batch_size: number;
  steps_per_epoch: number;
  repeated_image_count?: number;
  train_batch_size?: number;
  gradient_accumulation_steps?: number;
  duration_mode?: "epochs" | "steps" | "unset";
  max_train_epochs?: number | null;
  datasets?: EstimateDataset[];
};

export const number = (value: number | undefined) => value == null ? "—" : value.toLocaleString("zh-CN");
export const datasetName = (row: EstimateDataset) =>
  (row.source_dir || row.image_dir).replace(/\\/g, "/").replace(/\/$/, "").split("/").pop() || `数据集 ${row.index}`;
export const orientation = (bucket: EstimateBucket) =>
  bucket.width === bucket.height ? "square" : bucket.width > bucket.height ? "landscape" : "portrait";

export function collectBuckets(rows: EstimateDataset[]) {
  const buckets = new Map<string, EstimateBucket>();
  for (const row of rows) {
    for (const bucket of row.bucket_distribution?.buckets || []) {
      const key = `${bucket.width}x${bucket.height}`;
      const previous = buckets.get(key);
      buckets.set(key, { ...bucket, count: (previous?.count || 0) + bucket.count });
    }
  }
  return [...buckets.values()];
}
