import type { FieldPath, UseFormReturn } from 'react-hook-form';

import { captionSourceModes, type DatasetFormValues } from './datasetForm';

type Props = {
  form: UseFormReturn<DatasetFormValues>;
  prefix: 'defaults' | `datasets.${number}.settings`;
  includeKeepTokens?: boolean;
};

const captionModeLabels: Record<typeof captionSourceModes[number], string> = {
  auto: '自动识别',
  txt: '同名 TXT',
  json: 'JSON 标注',
  captions_json: 'captions.json',
};

export function DatasetSettingsFields({ form, prefix, includeKeepTokens = false }: Props) {
  const path = (key: string) => `${prefix}.${key}` as FieldPath<DatasetFormValues>;

  return (
    <div className="dataset-settings-sections">
      <SettingsGroup title="训练尺寸">
        <NumberSetting form={form} path={path('resolution')} label="训练分辨率" min={64} step={64} />
        <NumberSetting form={form} path={path('batch_size')} label="批大小" min={1} />
        <NumberSetting form={form} path={path('prior_loss_weight')} label="正则损失权重" min={0} step={0.05} />
      </SettingsGroup>

      <SettingsGroup title="分桶规则">
        <CheckboxSetting form={form} path={path('enable_bucket')} label="启用分桶" />
        <NumberSetting form={form} path={path('min_bucket_reso')} label="最小桶尺寸" min={64} step={64} />
        <NumberSetting form={form} path={path('max_bucket_reso')} label="最大桶尺寸" min={64} step={64} />
        <NumberSetting form={form} path={path('bucket_reso_steps')} label="桶尺寸步长" min={1} />
        <CheckboxSetting form={form} path={path('bucket_no_upscale')} label="禁止放大小图" />
      </SettingsGroup>

      <SettingsGroup title="验证集">
        <NumberSetting form={form} path={path('validation_split')} label="验证集比例" min={0} max={1} step={0.01} />
        <NumberSetting form={form} path={path('validation_split_num')} label="验证集数量" min={0} />
        <NumberSetting form={form} path={path('validation_seed')} label="验证随机种子" min={0} />
      </SettingsGroup>

      <SettingsGroup title="标注读取">
        <TextSetting form={form} path={path('caption_extension')} label="标注扩展名" placeholder=".txt" />
        {includeKeepTokens ? (
          <NumberSetting form={form} path={path('keep_tokens')} label="保留 Token 数" min={0} />
        ) : null}
        <CheckboxSetting form={form} path={path('prefer_json_caption')} label="优先 JSON 标注" />
        <label>
          <span>标注来源</span>
          <select {...form.register(path('caption_source_mode'))}>
            {captionSourceModes.map((mode) => <option key={mode} value={mode}>{captionModeLabels[mode]}</option>)}
          </select>
          <FieldError form={form} path={path('caption_source_mode')} />
        </label>
      </SettingsGroup>
    </div>
  );
}

function SettingsGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="dataset-settings-group">
      <h4>{title}</h4>
      <div>{children}</div>
    </section>
  );
}

function NumberSetting({
  form,
  path,
  label,
  min,
  max,
  step = 1,
}: {
  form: UseFormReturn<DatasetFormValues>;
  path: FieldPath<DatasetFormValues>;
  label: string;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label>
      <span>{label}</span>
      <input type="number" min={min} max={max} step={step} {...form.register(path, { valueAsNumber: true })} />
      <FieldError form={form} path={path} />
    </label>
  );
}

function TextSetting({
  form,
  path,
  label,
  placeholder,
}: {
  form: UseFormReturn<DatasetFormValues>;
  path: FieldPath<DatasetFormValues>;
  label: string;
  placeholder?: string;
}) {
  return (
    <label>
      <span>{label}</span>
      <input {...form.register(path)} placeholder={placeholder} />
      <FieldError form={form} path={path} />
    </label>
  );
}

function CheckboxSetting({
  form,
  path,
  label,
}: {
  form: UseFormReturn<DatasetFormValues>;
  path: FieldPath<DatasetFormValues>;
  label: string;
}) {
  return (
    <label className="dataset-checkbox-field">
      <input type="checkbox" {...form.register(path)} />
      <span>{label}</span>
    </label>
  );
}

function FieldError({ form, path }: { form: UseFormReturn<DatasetFormValues>; path: FieldPath<DatasetFormValues> }) {
  const error = form.getFieldState(path, form.formState).error;
  return typeof error?.message === 'string' ? <small role="alert">{error.message}</small> : null;
}
