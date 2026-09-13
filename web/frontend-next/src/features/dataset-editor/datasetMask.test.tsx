import { zodResolver } from '@hookform/resolvers/zod';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useForm } from 'react-hook-form';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { DatasetMaskFields } from './DatasetMaskFields';
import { datasetFormFromPreset, datasetFormSchema, datasetWritePayload, emptyDatasetForm, emptyDatasetRow, type DatasetFormValues } from './datasetForm';
import { datasetMaskFromRow } from './datasetMask';
import type { DatasetRow } from './types';

afterEach(cleanup);

function presetForm(row: DatasetRow) {
  return datasetFormFromPreset({
    ok: true, file: 'configs/datasets/mask-test.toml', name: 'mask-test', content: '',
    readonly: false, summary: {}, defaults: {},
    datasets: [{ source_dir: 'images/train', ...row }],
  });
}

describe('dataset mask contract', () => {
  it.each([
    [{}, 'auto'],
    [{ mask_dir: ' masks/train ' }, 'external'],
    [{ alpha_mask: true }, 'embedded'],
    [{ mask_mode: 'none', mask_dir: 'old', alpha_mask: true }, 'none'],
    [{ mask_mode: 'auto', mask_dir: 'old' }, 'external'],
    [{ mask_mode: 'auto', alpha_mask: true }, 'embedded'],
    [{ mask_mode: 'image-alpha' }, 'embedded'],
  ])('hydrates legacy mask fields %j as %s', (row, mode) => {
    expect(datasetMaskFromRow(row).mask_mode).toBe(mode);
  });

  it('preserves independent regularization masks and unknown fields through save/reload', () => {
    const form = presetForm({ mask_mode: 'external', mask_dir: ' masks/train ', custom_future: 'keep' });
    form.datasets.push({ ...emptyDatasetRow(), source_dir: 'images/reg', is_reg: true, mask_mode: 'external', mask_dir: 'masks/reg' });
    const payload = datasetWritePayload(datasetFormSchema.parse(form));
    expect(payload.datasets[0]).toMatchObject({ mask_mode: 'external', mask_dir: 'masks/train', alpha_mask: true, custom_future: 'keep' });
    expect(payload.datasets[1]).toMatchObject({ is_reg: true, mask_mode: 'external', mask_dir: 'masks/reg' });
    const reloaded = datasetFormFromPreset({ ...payload, ok: true, message: '', file: 'mask-test.toml', content: '', summary: {} });
    expect(reloaded.datasets.map((row) => row.mask_dir)).toEqual(['masks/train', 'masks/reg']);
  });

  it.each(['none', 'embedded', 'auto'] as const)('clears stale external flags when saving %s', (mode) => {
    const form = presetForm({ mask_mode: 'external', mask_dir: 'old', alpha_mask: true });
    form.datasets[0].mask_mode = mode;
    const row = datasetWritePayload(datasetFormSchema.parse(form)).datasets[0];
    expect(row).toMatchObject({ mask_mode: mode, mask_dir: '', alpha_mask: mode === 'embedded' });
    expect(datasetMaskFromRow(row).mask_mode).toBe(mode);
  });

  it('rejects empty external directories and unknown modes, but accepts modes without a directory', () => {
    const form = presetForm({ mask_mode: 'external', mask_dir: '  ' });
    const result = datasetFormSchema.safeParse(form);
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error.issues[0].path).toEqual(['datasets', 0, 'mask_dir']);
    expect(datasetFormSchema.safeParse(presetForm({ mask_mode: 'future_mode' })).success).toBe(false);
    for (const mode of ['none', 'embedded', 'auto']) {
      expect(datasetFormSchema.safeParse(presetForm({ mask_mode: mode })).success).toBe(true);
    }
  });
});

function MaskForm({ onSave, disabled = false }: { onSave: (value: unknown) => void; disabled?: boolean }) {
  const defaults = emptyDatasetForm();
  defaults.datasets[0].source_dir = 'images/train';
  const form = useForm<DatasetFormValues>({ defaultValues: defaults, resolver: zodResolver(datasetFormSchema) });
  return (
    <form onSubmit={form.handleSubmit((value) => onSave(datasetWritePayload(value)))}>
      <fieldset disabled={disabled}><DatasetMaskFields form={form} index={0} /></fieldset>
      <button type="submit">保存</button>
    </form>
  );
}

describe('dataset mask controls', () => {
  it('validates, saves external masks and switches back to auto without stale flags', async () => {
    const user = userEvent.setup();
    const save = vi.fn();
    render(<MaskForm onSave={save} />);
    const mode = screen.getByLabelText('蒙版模式');
    const directory = screen.getByLabelText('外部蒙版目录');
    expect(directory).toBeDisabled();
    await user.selectOptions(mode, 'external');
    expect(directory).toBeEnabled();
    await user.click(screen.getByRole('button', { name: '保存' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('使用外部蒙版时必须填写蒙版目录');
    expect(save).not.toHaveBeenCalled();
    await user.type(directory, ' masks/train ');
    await user.click(screen.getByRole('button', { name: '保存' }));
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1));
    expect(save.mock.calls[0][0].datasets[0]).toMatchObject({ mask_mode: 'external', mask_dir: 'masks/train' });
    await user.selectOptions(mode, 'auto');
    expect(directory).toBeDisabled();
    await user.click(screen.getByRole('button', { name: '保存' }));
    await waitFor(() => expect(save).toHaveBeenCalledTimes(2));
    expect(save.mock.calls[1][0].datasets[0]).toMatchObject({ mask_mode: 'auto', mask_dir: '', alpha_mask: false });
  });

  it('respects read-only subset state', () => {
    render(<MaskForm onSave={vi.fn()} disabled />);
    expect(screen.getByLabelText('蒙版模式')).toBeDisabled();
    expect(screen.getByLabelText('外部蒙版目录')).toBeDisabled();
  });
});
