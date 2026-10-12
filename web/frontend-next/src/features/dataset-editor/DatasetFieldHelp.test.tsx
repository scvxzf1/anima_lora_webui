import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useForm } from 'react-hook-form';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { emptyDatasetForm, type DatasetFormValues } from './datasetForm';
import { DatasetDefaultsEditor } from './DatasetDefaultsEditor';
import { DatasetFieldHelp } from './DatasetFieldHelp';
import { DatasetFieldLabel } from './DatasetFieldLabel';
import { DatasetSubsetList } from './DatasetSubsetList';
import { StageScheduleEditor } from './StageScheduleEditor';

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    disconnect() {}
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function EditorHarness({ disabled = false }: { disabled?: boolean }) {
  const form = useForm<DatasetFormValues>({ defaultValues: emptyDatasetForm() });
  return <>
    <DatasetDefaultsEditor form={form} disabled={disabled} />
    <DatasetSubsetList form={form} disabled={disabled} workbenchDisabled={disabled} />
    <StageScheduleEditor form={form} disabled={disabled} />
    <output data-testid="dirty">{form.formState.isDirty ? 'modified' : 'clean'}</output>
  </>;
}

describe('DatasetFieldHelp', () => {
  it('keeps labels attached to inputs and does not toggle a checkbox or submit the form', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn((event) => event.preventDefault());
    render(<form onSubmit={onSubmit}>
      <DatasetFieldLabel htmlFor="resolution" field="resolution" label="训练分辨率" />
      <input id="resolution" defaultValue="1024" />
      <input id="bucket" type="checkbox" defaultChecked />
      <DatasetFieldLabel htmlFor="bucket" field="enable_bucket" label="启用分桶" />
    </form>);

    const trigger = screen.getByRole('button', { name: '查看启用分桶帮助' });
    expect(trigger.closest('label')).toBeNull();
    expect(screen.queryByRole('note')).toBeNull();
    await user.click(trigger);
    expect(screen.getByRole('note')).toHaveTextContent('建议');
    expect(screen.getByLabelText('启用分桶')).toBeChecked();
    expect(screen.getByLabelText('训练分辨率')).toHaveValue('1024');
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('offers enum choices with advanced details collapsed initially', async () => {
    const user = userEvent.setup();
    render(<DatasetFieldHelp field="caption_source_mode" label="标注来源" />);
    await user.click(screen.getByRole('button', { name: '查看标注来源帮助' }));
    const note = screen.getByRole('note', { name: '标注来源说明' });
    expect(within(note).getAllByRole('row')).toHaveLength(5);
    const advanced = note.querySelector('details')!;
    expect(advanced.open).toBe(false);
    await user.click(within(note).getByText('进阶'));
    expect(advanced.open).toBe(true);
  });

  it('closes on Escape, the close button or an outside pointer and restores keyboard focus', async () => {
    const user = userEvent.setup();
    render(<>
      <DatasetFieldHelp field="resolution" label="训练分辨率" />
      <button type="button">其他操作</button>
    </>);
    const trigger = screen.getByRole('button', { name: '查看训练分辨率帮助' });
    await user.click(trigger);
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('note')).toBeNull();
    expect(trigger).toHaveFocus();
    await user.click(trigger);
    await user.click(screen.getByRole('button', { name: '关闭字段帮助' }));
    expect(screen.queryByRole('note')).toBeNull();
    expect(trigger).toHaveFocus();
    await user.click(trigger);
    await user.click(screen.getByRole('button', { name: '其他操作' }));
    expect(screen.queryByRole('note')).toBeNull();
  });

  it('allows help on readonly defaults and subsets while preserving disabled inputs and a clean draft', async () => {
    const user = userEvent.setup();
    const view = render(<EditorHarness disabled />);
    expect(Array.from(view.container.querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select'))
      .every((input) => input.disabled)).toBe(true);
    const defaults = screen.getByRole('group', { name: '默认训练参数' });
    await user.click(within(defaults).getByRole('button', { name: '查看训练分辨率帮助' }));
    expect(screen.getByRole('note', { name: '训练分辨率说明' })).toBeInTheDocument();
    const subset = screen.getByRole('group', { name: '子集 1' });
    await user.click(within(subset).getByRole('button', { name: '查看原始图片目录帮助' }));
    expect(screen.getByRole('note', { name: '原始图片目录说明' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '配置阶段' })).toBeDisabled();
    expect(screen.getByTestId('dirty')).toHaveTextContent('clean');
  });

  it('keeps stage configuration open on the first Escape and closes it on the second', async () => {
    const user = userEvent.setup();
    render(<EditorHarness />);
    await user.click(screen.getByRole('button', { name: '配置阶段' }));
    const dialog = screen.getByRole('dialog', { name: '配置分阶段调度' });
    const trigger = within(dialog).getByRole('button', { name: '查看开始百分比帮助' });
    await user.click(trigger);
    expect(within(dialog).getByRole('note', { name: '开始百分比说明' })).toBeInTheDocument();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('note')).toBeNull();
    expect(dialog).toBeInTheDocument();
    expect(trigger).toHaveFocus();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(screen.getByRole('button', { name: '配置阶段' })).toHaveFocus();
    expect(screen.getByTestId('dirty')).toHaveTextContent('clean');
  });
});
