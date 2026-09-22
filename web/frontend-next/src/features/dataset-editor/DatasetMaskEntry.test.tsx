import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useForm } from 'react-hook-form';
import { afterEach, expect, it, vi } from 'vitest';
import { DatasetSubsetList } from './DatasetSubsetList';
import { emptyDatasetForm, emptyDatasetRow, type DatasetFormValues } from './datasetForm';

afterEach(cleanup);

function Harness({ disabled = false, previewDisabled = false, onEditMasks = vi.fn() }) {
  const values = emptyDatasetForm();
  values.datasets.push(emptyDatasetRow());
  const form = useForm<DatasetFormValues>({ defaultValues: values });
  return <DatasetSubsetList form={form} disabled={disabled} previewDisabled={previewDisabled}
    onPreview={vi.fn()} onEditMasks={onEditMasks} />;
}

it('places mask editing before preview and passes the selected zero-based subset index', async () => {
  const onEditMasks = vi.fn();
  render(<Harness onEditMasks={onEditMasks} />);
  const subset = screen.getByRole('group', { name: '子集 2' });
  const buttons = within(subset.querySelector('legend')!).getAllByRole('button');
  expect(buttons.map(button => button.textContent)).toEqual(['编辑蒙版', '预览']);
  await userEvent.click(buttons[0]);
  expect(onEditMasks).toHaveBeenCalledWith(1);
});

it.each([{ disabled: true }, { previewDisabled: true }])('disables unsafe entry for %j', props => {
  render(<Harness {...props} />);
  expect(screen.getByRole('button', { name: '编辑子集 1 蒙版' })).toBeDisabled();
  expect(screen.getByRole('button', { name: '编辑子集 2 蒙版' })).toBeDisabled();
});
