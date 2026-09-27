import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import { zodResolver } from '@hookform/resolvers/zod';
import userEvent from '@testing-library/user-event';
import { useForm } from 'react-hook-form';
import { afterEach, expect, it, vi } from 'vitest';
import { DatasetSubsetList } from './DatasetSubsetList';
import { datasetFormSchema, emptyDatasetForm, emptyDatasetRow, type DatasetFormValues } from './datasetForm';

afterEach(cleanup);

function Harness({
  disabled = false,
  workbenchDisabled = false,
  qwenEditIssue = null,
  twoRows = true,
  onOpenWorkbench = vi.fn(),
}) {
  const values = emptyDatasetForm();
  if (twoRows) values.datasets.push(emptyDatasetRow());
  const form = useForm<DatasetFormValues>({ defaultValues: values });
  return <DatasetSubsetList form={form} disabled={disabled} workbenchDisabled={workbenchDisabled}
    qwenEditIssue={qwenEditIssue}
    onOpenWorkbench={onOpenWorkbench} />;
}

it('provides one image-workbench entry per subset and passes its zero-based index', async () => {
  const onOpenWorkbench = vi.fn();
  render(<Harness onOpenWorkbench={onOpenWorkbench} />);
  const subset = screen.getByRole('group', { name: '子集 2' });
  const buttons = within(subset.querySelector('legend')!).getAllByRole('button');
  expect(buttons.map(button => button.textContent)).toEqual(['图片工作台']);
  await userEvent.click(buttons[0]);
  expect(onOpenWorkbench).toHaveBeenCalledWith(1);
});

it('disables the entry while the preset is dirty or not saved', () => {
  const props = { workbenchDisabled: true };
  render(<Harness {...props} />);
  expect(screen.getByRole('button', { name: '打开子集 1 图片工作台' })).toBeDisabled();
  expect(screen.getByRole('button', { name: '打开子集 2 图片工作台' })).toBeDisabled();
});

it('keeps the workbench available for a read-only saved preset', () => {
  render(<Harness disabled />);
  expect(screen.getByRole('button', { name: '打开子集 1 图片工作台' })).toBeEnabled();
});

it('shows role selectors and creates a named before/after pair', async () => {
  const user = userEvent.setup();
  render(<Harness twoRows={false} />);
  expect(screen.getAllByRole('combobox', { name: '子集角色' })).toHaveLength(1);
  await user.click(screen.getByRole('button', { name: '添加编辑配对' }));
  expect(screen.getByRole('group', { name: '子集 1' })).toHaveTextContent('编辑前图片目录');
  expect(screen.getByRole('group', { name: '子集 2' })).toHaveTextContent('目标图训练目录（编辑后）');
  expect(within(screen.getByRole('group', { name: '子集 1' })).getByRole('textbox', { name: '配对编号或名称' })).toHaveValue('1');
  expect(within(screen.getByRole('group', { name: '子集 2' })).getByRole('textbox', { name: '配对编号或名称' })).toHaveValue('1');
});

it('keeps edit source and target paths editable when the selected training family cannot apply them', () => {
  function ExistingEditHarness() {
    const values = emptyDatasetForm();
    values.datasets = [
      { ...emptyDatasetRow(), edit_role: 'before', edit_pair_id: '1', source_dir: 'image_dataset/reference' },
      { ...emptyDatasetRow(), edit_role: 'after', edit_pair_id: '1', source_dir: 'image_dataset/target' },
    ];
    const form = useForm<DatasetFormValues>({ defaultValues: values });
    return <DatasetSubsetList form={form} disabled={false} workbenchDisabled={false}
      qwenEditIssue="需要 Qwen Image 2.1 训练配置" />;
  }

  render(<ExistingEditHarness />);
  const reference = screen.getByDisplayValue('image_dataset/reference');
  expect(reference).toBeEnabled();
  expect(screen.getByRole('alert')).toHaveTextContent('当前训练配置不兼容');
});

it('clears an outdated incomplete-pair error when the matching subset is added', async () => {
  function IncompletePairHarness() {
    const values = emptyDatasetForm();
    values.datasets = [{ ...emptyDatasetRow(), edit_role: 'before', edit_pair_id: '1', source_dir: 'references' }];
    const form = useForm<DatasetFormValues>({
      defaultValues: values, resolver: zodResolver(datasetFormSchema), mode: 'onBlur',
    });
    return <>
      <button type="button" onClick={() => form.trigger('datasets')}>校验</button>
      <DatasetSubsetList form={form} disabled={false} workbenchDisabled={false} qwenEditIssue={null} />
    </>;
  }

  const user = userEvent.setup();
  render(<IncompletePairHarness />);
  await user.click(screen.getByRole('button', { name: '校验' }));
  expect(await screen.findByText(/需要各一个编辑前和编辑后子集/)).toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: '添加编辑配对' }));
  await waitFor(() => expect(screen.queryByText(/需要各一个编辑前和编辑后子集/)).not.toBeInTheDocument());
});

it('clears the pair error when an existing subset is manually assigned the matching role', async () => {
  function ManualPairHarness() {
    const values = emptyDatasetForm();
    values.datasets = [
      { ...emptyDatasetRow(), edit_role: 'before', edit_pair_id: '1', source_dir: 'references' },
      { ...emptyDatasetRow(), source_dir: 'targets' },
    ];
    const form = useForm<DatasetFormValues>({
      defaultValues: values, resolver: zodResolver(datasetFormSchema), mode: 'onBlur',
    });
    return <>
      <button type="button" onClick={() => form.trigger('datasets')}>校验</button>
      <DatasetSubsetList form={form} disabled={false} workbenchDisabled={false} qwenEditIssue={null} />
    </>;
  }

  const user = userEvent.setup();
  render(<ManualPairHarness />);
  await user.click(screen.getByRole('button', { name: '校验' }));
  expect(await screen.findByText(/需要各一个编辑前和编辑后子集/)).toBeInTheDocument();
  const second = screen.getByRole('group', { name: '子集 2' });
  await user.selectOptions(within(second).getByRole('combobox', { name: '子集角色' }), 'after');
  await user.type(within(second).getByRole('textbox', { name: /配对编号或名称/ }), '1');
  await waitFor(() => expect(screen.queryByText(/需要各一个编辑前和编辑后子集/)).not.toBeInTheDocument());
});
