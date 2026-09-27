import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { DatasetImageViewer } from './DatasetImageViewer';

const image = {
  file: 'images/studio/photo.png',
  name: 'photo.png',
  url: '/api/config/dataset-presets/image?image=photo.png',
  width: 128,
  height: 128,
  total_pixels: 16_384,
  size_bytes: 2_048,
  mtime_text: '2026-09-27 12:00',
  caption: {
    ok: true,
    file: 'images/studio/photo.txt',
    extension: '.txt',
    source_mode: 'txt',
    source_label: 'TXT',
    detected_mode: 'txt',
    format_label: 'TXT',
    caption_count: 1,
    text: 'studio portrait',
    truncated: false,
    length: 15,
  },
};

describe('DatasetImageViewer', () => {
  afterEach(cleanup);

  it('exposes a pending status until the image load event settles', () => {
    render(<DatasetImageViewer image={image} returnFocus={null} onClose={() => undefined} />);

    const viewer = screen.getByRole('dialog', { name: 'photo.png' });
    const canvas = viewer.querySelector('.dataset-image-viewer-canvas');
    expect(canvas).toHaveAttribute('aria-busy', 'true');
    expect(within(viewer).getByRole('status')).toHaveTextContent('正在读取图片');

    fireEvent.load(within(viewer).getByRole('img', { name: 'photo.png' }));

    expect(canvas).toHaveAttribute('aria-busy', 'false');
    expect(within(viewer).queryByRole('status')).not.toBeInTheDocument();
  });

  it('settles the pending state before exposing an image error and retry', () => {
    render(<DatasetImageViewer image={image} returnFocus={null} onClose={() => undefined} />);

    const viewer = screen.getByRole('dialog', { name: 'photo.png' });
    fireEvent.error(within(viewer).getByRole('img', { name: 'photo.png' }));

    expect(viewer.querySelector('.dataset-image-viewer-canvas')).toHaveAttribute('aria-busy', 'false');
    expect(within(viewer).getByRole('alert')).toHaveTextContent('图片加载失败');
    expect(within(viewer).queryByRole('status')).not.toBeInTheDocument();

    fireEvent.click(within(viewer).getByRole('button', { name: '重试' }));
    expect(viewer.querySelector('.dataset-image-viewer-canvas')).toHaveAttribute('aria-busy', 'true');
    expect(within(viewer).getByRole('status')).toHaveTextContent('正在读取图片');
  });
});
