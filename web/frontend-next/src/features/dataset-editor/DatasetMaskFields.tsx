import { useWatch, type UseFormReturn } from 'react-hook-form';

import type { DatasetFormValues } from './datasetForm';
import './DatasetMaskFields.css';

export function DatasetMaskFields({ form, index }: {
  form: UseFormReturn<DatasetFormValues>;
  index: number;
}) {
  const modePath = `datasets.${index}.mask_mode` as const;
  const directoryPath = `datasets.${index}.mask_dir` as const;
  const mode = useWatch({ control: form.control, name: modePath });
  const error = form.getFieldState(directoryPath, form.formState).error;
  const modeError = form.getFieldState(modePath, form.formState).error;
  const errorId = `dataset-mask-directory-error-${index}`;

  return (
    <section className="dataset-settings-group dataset-mask-settings">
      <h4>蒙版</h4>
      <div>
        <label>
          <span>蒙版模式</span>
          <select {...form.register(modePath)} aria-invalid={Boolean(modeError)}>
            <option value="none">不使用蒙版</option>
            <option value="external">外部蒙版目录</option>
            <option value="embedded">图像 Alpha</option>
            <option value="auto">兼容自动发现</option>
          </select>
          {modeError && <small role="alert">{modeError.message}</small>}
        </label>
        <label className="dataset-mask-directory">
          <span>外部蒙版目录</span>
          <input
            {...form.register(directoryPath)}
            disabled={mode !== 'external'}
            aria-invalid={mode === 'external' && Boolean(error)}
            aria-describedby={mode === 'external' && error ? errorId : undefined}
            placeholder="post_image_dataset/masks"
          />
          {mode === 'external' && error && <small id={errorId} role="alert">{error.message}</small>}
        </label>
      </div>
    </section>
  );
}
