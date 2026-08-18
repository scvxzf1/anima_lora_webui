export type DatasetSummary = {
  ok?: boolean;
  dataset_count?: number;
  train_dataset_count?: number;
  reg_dataset_count?: number;
  repeat_total?: number;
  reg_repeat_total?: number;
  source_dir?: string;
  image_dir?: string;
  cache_dir?: string;
  resolution?: number;
  batch_size?: number;
  enable_bucket?: boolean;
  prior_loss_weight?: number;
};

export type DatasetPresetSummary = {
  path: string;
  label?: string;
  filename?: string;
  readonly?: boolean;
  locked?: boolean;
  system_preset?: boolean;
  summary?: DatasetSummary;
};

export type DatasetLibraryGroup = {
  id: string;
  label: string;
  open?: boolean;
  locked?: boolean;
  group_locked?: boolean;
  user_group_locked?: boolean;
  system_locked?: boolean;
  kind?: 'dataset' | 'training' | string;
  user_managed?: boolean;
  renamable?: boolean;
  deletable?: boolean;
  movable?: boolean;
  files: DatasetPresetSummary[];
};

export type DatasetLibraryResponse = {
  ok: true;
  presets: DatasetPresetSummary[];
  groups: DatasetLibraryGroup[];
};

export type DatasetRow = {
  source_dir?: string;
  image_dir?: string;
  cache_dir?: string;
  num_repeats?: number;
  is_reg?: boolean;
  settings?: Record<string, unknown>;
  [key: string]: unknown;
};

export type DatasetPresetResponse = {
  ok: true;
  file: string;
  name: string;
  content: string;
  datasets: DatasetRow[];
  defaults: Record<string, unknown>;
  readonly: boolean;
  summary: DatasetSummary;
  stage_schedule_enabled?: boolean;
  stage_schedule?: Array<Record<string, unknown>>;
};

export type CreateDatasetGroupResponse = {
  ok: true;
  message: string;
  group: DatasetLibraryGroup;
};

export type RenameDatasetGroupResponse = CreateDatasetGroupResponse;

export type DeleteDatasetGroupResponse = {
  ok: true;
  message: string;
};

export type PlaceDatasetLibraryItemResponse = {
  ok: true;
  message: string;
  group?: DatasetLibraryGroup | null;
};

export type DatasetPresetWritePayload = {
  datasets: DatasetRow[];
  defaults: Record<string, unknown>;
  stage_schedule_enabled?: boolean;
  stage_schedule?: Array<Record<string, unknown>>;
};

export type DatasetPresetMutationResponse = DatasetPresetWritePayload & {
  ok: true;
  message: string;
  file: string;
  content: string;
  summary: DatasetSummary;
};

export type DeleteDatasetPresetResponse = {
  ok: true;
  message: string;
  file: string;
};

export type ApplyDatasetPresetResponse = {
  ok: true;
  message: string;
  dataset_config: string;
  datasets: DatasetRow[];
  defaults: Record<string, unknown>;
  train_content: string;
  changed: string[];
  values: Record<string, unknown>;
  summary: DatasetSummary;
};

export type DatasetPreviewCaption = {
  ok: boolean;
  file: string;
  extension: string;
  source_mode: string;
  source_label: string;
  detected_mode: string;
  format_label: string;
  caption_count: number;
  text: string;
  truncated: boolean;
  length: number;
};

export type DatasetPreviewImage = {
  file: string;
  name: string;
  url: string;
  mtime?: number;
  mtime_text?: string;
  size_bytes?: number;
  width?: number;
  height?: number;
  total_pixels?: number;
  caption: DatasetPreviewCaption;
};

export type DatasetPreviewResponse = {
  ok: true;
  file: string;
  dataset_index: number;
  dataset_label: string;
  source: 'source' | 'training';
  source_label: string;
  directory: string;
  directory_exists: boolean;
  caption_extension: string;
  prefer_json_caption: boolean;
  caption_source_mode: string;
  caption_source_label: string;
  caption_summary: string;
  count: number;
  total: number;
  limit: number;
  images: DatasetPreviewImage[];
  row: DatasetRow;
  settings: Record<string, unknown>;
  message: string;
};
