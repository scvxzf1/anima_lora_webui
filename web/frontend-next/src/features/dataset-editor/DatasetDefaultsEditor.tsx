import type { UseFormReturn } from 'react-hook-form';

import type { DatasetFormValues } from './datasetForm';
import { DatasetSettingsFields } from './DatasetSettingsFields';

export function DatasetDefaultsEditor({
  form,
  disabled,
}: {
  form: UseFormReturn<DatasetFormValues>;
  disabled: boolean;
}) {
  return (
    <fieldset className="dataset-default-fields" disabled={disabled}>
      <legend>默认训练参数</legend>
      <DatasetSettingsFields form={form} prefix="defaults" includeKeepTokens />
    </fieldset>
  );
}
