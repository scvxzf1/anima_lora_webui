import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useRef, useState } from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { MaskTools } from './MaskTools';
import type { ViewportHandle } from './MaskViewport';

afterEach(() => vi.restoreAllMocks());

function Harness() {
  const [zoom, setZoom] = useState(1);
  const viewport = useRef<ViewportHandle | null>(null);
  viewport.current = {
    fit: vi.fn(() => setZoom(1)),
    zoom: (factor) => setZoom((value) => value * factor),
  };
  return (
    <MaskTools
      engine={null}
      tool="brush"
      setTool={vi.fn()}
      size={32}
      setSize={vi.fn()}
      view="overlay"
      setView={vi.fn()}
      opacity={0.5}
      setOpacity={vi.fn()}
      disabled={false}
      changed={vi.fn()}
      viewport={viewport}
      zoom={zoom}
    />
  );
}

it('announces zoom changes and keeps zoom commands keyboard accessible', async () => {
  const user = userEvent.setup();
  render(<Harness />);
  const zoomOutput = screen.getByLabelText('画布缩放');
  expect(zoomOutput).toHaveAttribute('aria-live', 'polite');
  expect(zoomOutput).toHaveTextContent('100%');

  const zoomIn = screen.getByRole('button', { name: '放大' });
  zoomIn.focus();
  await user.keyboard('{Enter}');
  expect(zoomOutput).toHaveTextContent('125%');

  await user.click(screen.getByRole('button', { name: '缩小' }));
  expect(zoomOutput).toHaveTextContent('100%');
});
