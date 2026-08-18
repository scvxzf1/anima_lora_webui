export type TrainingFieldKind = 'text' | 'number' | 'boolean' | 'select';

export type TrainingFieldSpec = {
  key: string;
  label: string;
  group: 'identity' | 'budget' | 'data' | 'runtime';
  kind: TrainingFieldKind;
  options?: string[];
  min?: number;
  step?: number;
};

export const TRAINING_FIELDS: TrainingFieldSpec[] = [
  { key: 'output_name', label: '输出名称', group: 'identity', kind: 'text' },
  { key: 'model_family', label: '模型族', group: 'identity', kind: 'select', options: ['anima', 'krea2_raw'] },
  { key: 'network_dim', label: 'Network rank', group: 'identity', kind: 'number', min: 1, step: 1 },
  { key: 'network_alpha', label: 'Network alpha', group: 'identity', kind: 'number', min: 0, step: 1 },
  { key: 'max_train_steps', label: '最大训练步数', group: 'budget', kind: 'number', min: 1, step: 1 },
  { key: 'max_train_epochs', label: '最大训练轮数', group: 'budget', kind: 'number', min: 1, step: 1 },
  { key: 'train_batch_size', label: '训练批大小', group: 'budget', kind: 'number', min: 1, step: 1 },
  { key: 'gradient_accumulation_steps', label: '梯度累积', group: 'budget', kind: 'number', min: 1, step: 1 },
  { key: 'learning_rate', label: '学习率', group: 'budget', kind: 'number', min: 0, step: 0.000001 },
  { key: 'dataset_config', label: '数据集配置', group: 'data', kind: 'text' },
  { key: 'output_dir', label: '输出目录', group: 'data', kind: 'text' },
  { key: 'mixed_precision', label: '混合精度', group: 'runtime', kind: 'select', options: ['bf16', 'fp16', 'no'] },
  { key: 'gradient_checkpointing', label: '梯度检查点', group: 'runtime', kind: 'boolean' },
  { key: 'blocks_to_swap', label: 'Block swap', group: 'runtime', kind: 'number', min: 0, step: 1 },
  { key: 'torch_compile', label: 'Torch compile', group: 'runtime', kind: 'boolean' },
  { key: 'attn_mode', label: 'Attention', group: 'runtime', kind: 'select', options: ['torch', 'flash'] },
];

export type TrainingDraft = Record<string, string | number | boolean>;

export function draftFromMerged(config: Record<string, unknown>): TrainingDraft {
  return Object.fromEntries(TRAINING_FIELDS.map((field) => [field.key, formValue(field, config[field.key])])) as TrainingDraft;
}

export function trainingPatchValues(draft: TrainingDraft, baseline: TrainingDraft) {
  return Object.fromEntries(TRAINING_FIELDS.flatMap((field) => (
    Object.is(draft[field.key], baseline[field.key]) ? [] : [[field.key, draft[field.key]]]
  )));
}

export function rawConfigOwnKeys(content: string) {
  const keys = new Set<string>();
  let tableDepth = 0;
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    if (trimmed.startsWith('[')) {
      tableDepth = 1;
      continue;
    }
    if (tableDepth === 0) {
      const match = trimmed.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=/);
      if (match) keys.add(match[1]);
    }
  }
  return keys;
}

export function importedTrainingPath(name: string) {
  const stem = name.trim().replace(/\.toml$/i, '').replace(/[^\w\u3400-\u9fff.-]+/g, '_').replace(/^[._-]+|[._-]+$/g, '');
  return stem ? `configs/imported/${stem}.toml` : '';
}

function formValue(field: TrainingFieldSpec, value: unknown): string | number | boolean {
  if (field.kind === 'boolean') return Boolean(value);
  if (field.kind === 'number') return value === undefined || value === null || value === '' ? '' : Number(value);
  if (value === undefined || value === null) return field.options?.[0] || '';
  return String(value);
}
