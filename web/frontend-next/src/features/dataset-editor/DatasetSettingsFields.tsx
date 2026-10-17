import type { FieldPath, UseFormReturn } from 'react-hook-form';

import { captionSourceModes, type DatasetFormValues } from './datasetForm';
import { DatasetFieldLabel } from './DatasetFieldLabel';
import type { DatasetFieldHelpKey } from './field-help';
import { SettingSwitch } from '../../components/SettingSwitch';

type Props = {
  form: UseFormReturn<DatasetFormValues>;
  prefix: 'defaults' | `datasets.${number}.settings`;
  includeKeepTokens?: boolean;
  disabled?: boolean;
};

const captionModeLabels: Record<typeof captionSourceModes[number], string> = {
  auto: '自动识别',
  txt: '同名 TXT',
  json: 'JSON 标注',
  captions_json: 'captions.json',
};

export function DatasetSettingsFields({ form, prefix, includeKeepTokens = false, disabled = false }: Props) {
  const path = (key: string) => `${prefix}.${key}` as FieldPath<DatasetFormValues>;
  const settingProps = { form, disabled };

  return (
    <div className="dataset-settings-sections">
      <SettingsGroup title="训练尺寸">
        <NumberSetting {...settingProps} path={path('resolution')} helpField="resolution" label="训练分辨率" min={64} step={64} />
        <NumberSetting {...settingProps} path={path('batch_size')} helpField="batch_size" label="批大小" min={1} />
        <NumberSetting {...settingProps} path={path('prior_loss_weight')} helpField="prior_loss_weight" label="正则损失权重" min={0} step={0.05} />
      </SettingsGroup>

      <SettingsGroup title="分桶规则" controls={(
        <>
          <CheckboxSetting {...settingProps} path={path('enable_bucket')} helpField="enable_bucket" label="启用分桶" />
          <CheckboxSetting {...settingProps} path={path('bucket_no_upscale')} helpField="bucket_no_upscale" label="禁止放大小图" />
        </>
      )}>
        <NumberSetting {...settingProps} path={path('min_bucket_reso')} helpField="min_bucket_reso" label="最小桶尺寸" min={64} step={64} />
        <NumberSetting {...settingProps} path={path('max_bucket_reso')} helpField="max_bucket_reso" label="最大桶尺寸" min={64} step={64} />
        <NumberSetting {...settingProps} path={path('bucket_reso_steps')} helpField="bucket_reso_steps" label="桶尺寸步长" min={1} />
      </SettingsGroup>

      <SettingsGroup title="验证集">
        <NumberSetting {...settingProps} path={path('validation_split')} helpField="validation_split" label="验证集比例" min={0} max={1} step={0.01} />
        <NumberSetting {...settingProps} path={path('validation_split_num')} helpField="validation_split_num" label="验证集数量" min={0} />
        <NumberSetting {...settingProps} path={path('validation_seed')} helpField="validation_seed" label="验证随机种子" min={0} />
      </SettingsGroup>

      <SettingsGroup title="标注读取" controls={(
        <CheckboxSetting {...settingProps} path={path('prefer_json_caption')} helpField="prefer_json_caption" label="优先 JSON 标注" compact />
      )}>
        <TextSetting {...settingProps} path={path('caption_extension')} helpField="caption_extension" label="标注扩展名" placeholder=".txt" />
        {includeKeepTokens ? (
          <NumberSetting {...settingProps} path={path('keep_tokens')} helpField="keep_tokens" label="保留 Token 数" min={0} />
        ) : null}
        <div className="dataset-setting-field">
          <DatasetFieldLabel htmlFor={path('caption_source_mode')} field="caption_source_mode" label="标注来源" />
          <select id={path('caption_source_mode')} disabled={disabled} {...form.register(path('caption_source_mode'))}>
            {captionSourceModes.map((mode) => <option key={mode} value={mode}>{captionModeLabels[mode]}</option>)}
          </select>
          <FieldError form={form} path={path('caption_source_mode')} />
        </div>
      </SettingsGroup>
    </div>
  );
}

function SettingsGroup({ title, children, controls }: { title: string; children: React.ReactNode; controls?: React.ReactNode }) {
  return (
    <section className="dataset-settings-group">
      <header className="dataset-settings-group-heading">
        <h4>{title}</h4>
        {controls ? <div className="dataset-settings-group-controls">{controls}</div> : null}
      </header>
      <div>{children}</div>
    </section>
  );
}

function NumberSetting({
  form,
  path,
  helpField,
  label,
  min,
  max,
  step = 1,
  disabled,
}: {
  form: UseFormReturn<DatasetFormValues>;
  path: FieldPath<DatasetFormValues>;
  helpField?: DatasetFieldHelpKey;
  label: string;
  min?: number;
  max?: number;
  step?: number;
  disabled: boolean;
}) {
  return (
    <div className="dataset-setting-field">
      <DatasetFieldLabel htmlFor={path} field={helpField} label={label} />
      <input id={path} type="number" min={min} max={max} step={step} disabled={disabled} {...form.register(path, { valueAsNumber: true })} />
      <FieldError form={form} path={path} />
    </div>
  );
}

function TextSetting({
  form,
  path,
  helpField,
  label,
  placeholder,
  disabled,
}: {
  form: UseFormReturn<DatasetFormValues>;
  path: FieldPath<DatasetFormValues>;
  helpField?: DatasetFieldHelpKey;
  label: string;
  placeholder?: string;
  disabled: boolean;
}) {
  return (
    <div className="dataset-setting-field">
      <DatasetFieldLabel htmlFor={path} field={helpField} label={label} />
      <input id={path} {...form.register(path)} placeholder={placeholder} disabled={disabled} />
      <FieldError form={form} path={path} />
    </div>
  );
}

function CheckboxSetting({
  form,
  path,
  helpField,
  label,
  disabled,
  compact = false,
}: {
  form: UseFormReturn<DatasetFormValues>;
  path: FieldPath<DatasetFormValues>;
  helpField: DatasetFieldHelpKey;
  label: string;
  disabled: boolean;
  compact?: boolean;
}) {
  return (
    <div className={`dataset-setting-field dataset-checkbox-field${compact ? ' dataset-checkbox-compact' : ''}`}>
      <DatasetFieldLabel htmlFor={path} field={helpField} label={label} />
      <SettingSwitch
        id={path}
        aria-label={label}
        checked={Boolean(form.watch(path))}
        disabled={disabled}
        onCheckedChange={(checked) => form.setValue(path, checked, { shouldDirty: true, shouldValidate: true })}
      />
    </div>
  );
}

function FieldError({ form, path }: { form: UseFormReturn<DatasetFormValues>; path: FieldPath<DatasetFormValues> }) {
  const error = form.getFieldState(path, form.formState).error;
  return typeof error?.message === 'string' ? <small role="alert">{error.message}</small> : null;
}
