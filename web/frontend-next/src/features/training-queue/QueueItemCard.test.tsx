import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { QueueItemCard, queueItemTitle } from './QueueItemCard';
import type { QueueItem, QueueSnapshot } from './api';

function snapshot(): QueueSnapshot {
  return {
    paused: false,
    status: 'idle',
    current_item_id: '',
    summary: { total: 2, queued: 2, running: 0, done: 0, error: 0, canceled: 0 },
    items: [],
  };
}

describe('queueItemTitle', () => {
  it('prefers the resume checkpoint name for resume items', () => {
    const item: QueueItem = { resume_info: { checkpoint_name: 'output/ckpt/lora-01000.safetensors' } };
    expect(queueItemTitle(item)).toBe('续训 · lora-01000.safetensors');
  });

  it('falls back to variant and preset when no checkpoint exists', () => {
    const item: QueueItem = { variant: 'krea2_lora', preset: 'default', id: 'abc' };
    expect(queueItemTitle(item)).toBe('krea2_lora · default');
  });
});

describe('QueueItemCard', () => {
  it('renders move actions for queued items and reports clicks', async () => {
    const user = userEvent.setup();
    const onAction = vi.fn();
    render(
      <QueueItemCard
        item={{ id: 'q1', state: 'queued', variant: 'lora', preset: 'default', queuedPosition: 0, queuedTotal: 1 } as QueueItem}
        snapshot={snapshot()}
        queuedPosition={0}
        queuedTotal={2}
        onAction={onAction}
      />,
    );

    await user.click(screen.getByRole('button', { name: '下移' }));
    expect(onAction).toHaveBeenCalledWith('move', expect.objectContaining({ id: 'q1' }), 'down');
  });

  it('offers retry for finished items and remove for canceled records', async () => {
    const user = userEvent.setup();
    const onAction = vi.fn();
    render(
      <QueueItemCard
        item={{ id: 'q2', state: 'error', variant: 'lora', preset: 'default' } as QueueItem}
        snapshot={snapshot()}
        queuedPosition={-1}
        queuedTotal={1}
        onAction={onAction}
      />,
    );

    expect(screen.getByRole('button', { name: '重试' })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '移出列表' }));
    expect(onAction).toHaveBeenCalledWith('remove', expect.objectContaining({ id: 'q2' }));
  });
});
