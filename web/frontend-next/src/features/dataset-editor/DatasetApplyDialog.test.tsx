import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { DatasetApplyDialog } from './DatasetApplyDialog';

afterEach(cleanup);

describe('DatasetApplyDialog', () => {
  it('shows a model-neutral task before confirmation', () => {
    render(
      <DatasetApplyDialog
        datasetFile="configs/datasets/edit.toml"
        trainFile={{ path: 'configs/methods/qwen_image_2_1_lora.toml', label: 'Qwen LoRA' }}
        taskMode="edit"
        busy={false}
        onCancel={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );

    expect(screen.getByText('数据集任务')).toBeInTheDocument();
    expect(screen.getByText('编辑 LoRA')).toBeInTheDocument();
    expect(screen.queryByText(/qwen_image_2_1_task/)).not.toBeInTheDocument();
    expect(screen.getByText('这会更新目标训练 TOML 中的数据集引用和兼容字段，不会启动训练。')).toBeInTheDocument();
  });

  it('keeps model compatibility feedback in the apply dialog', () => {
    render(
      <DatasetApplyDialog
        datasetFile="configs/datasets/edit.toml"
        trainFile={{ path: 'configs/methods/anima.toml', label: 'Anima LoRA' }}
        taskMode={null}
        compatibilityIssue="需要支持编辑任务的训练配置"
        busy={false}
        onCancel={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );

    expect(screen.getByRole('alert')).toHaveTextContent('需要支持编辑任务的训练配置');
    expect(screen.getByRole('button', { name: '确认应用' })).toBeDisabled();
  });
});
