import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { DatasetApplyDialog } from './DatasetApplyDialog';

afterEach(cleanup);

describe('DatasetApplyDialog', () => {
  it('shows the Qwen edit task and persisted TOML change before confirmation', () => {
    render(
      <DatasetApplyDialog
        datasetFile="configs/datasets/edit.toml"
        trainFile={{ path: 'configs/methods/qwen_image_2_1_lora.toml', label: 'Qwen LoRA' }}
        qwenTaskMode="edit"
        busy={false}
        onCancel={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );

    expect(screen.getByText('Qwen Image 2.1 任务')).toBeInTheDocument();
    expect(screen.getByText('编辑 LoRA')).toBeInTheDocument();
    expect(screen.getByText(/qwen_image_2_1_task = "edit"/)).toBeInTheDocument();
    expect(screen.getByText('这会更新目标训练 TOML 中的数据集引用和兼容字段，不会启动训练。')).toBeInTheDocument();
  });
});
