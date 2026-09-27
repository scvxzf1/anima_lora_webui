import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider, useLocation } from 'react-router-dom';
import { afterEach, expect, it, vi } from 'vitest';

import { CaptioningPage } from './CaptioningPage';

vi.mock('./CaptionSource', () => ({
  CaptionSource: ({
    onSourceChange,
    onCreated,
  }: {
    onSourceChange: (file: string, index: number) => void;
    onCreated: (id: string) => void;
  }) => (
    <div>
      <button onClick={() => onSourceChange('configs/datasets/next.toml', 1)}>模拟切换子集</button>
      <button onClick={() => onCreated('job-1')}>模拟创建任务</button>
    </div>
  ),
}));

vi.mock('./CaptionReview', () => ({ CaptionReview: () => <div>review</div> }));
vi.mock('./CaptionProviders', () => ({ CaptionProviders: () => <div>providers</div> }));
vi.mock('./CaptionPrompts', () => ({ CaptionPrompts: () => <div>prompts</div> }));
vi.mock('./CaptionAssets', () => ({ CaptionAssets: () => <div>assets</div> }));
vi.mock('./CaptionJobCleanup', () => ({ CaptionJobCleanup: () => null }));

function LocationDisplay() {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}{location.search}</output>;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it('keeps dataset and subset context while changing the source and creating a job', async () => {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === '/api/captioning/profiles') {
      return new Response(JSON.stringify({ profiles: [], active_profile_id: '', provider_types: [] }));
    }
    if (url === '/api/captioning/jobs') return new Response(JSON.stringify({ jobs: [] }));
    return new Response(JSON.stringify({ presets: [] }));
  }));
  const query = new URLSearchParams({ dataset: 'configs/datasets/alpha.toml', subset: '0' }).toString();
  const router = createMemoryRouter([
    { path: '/datasets/workspace/tagging', element: <><CaptioningPage embeddedBasePath="/datasets/workspace/tagging" /><LocationDisplay /></> },
  ], { initialEntries: [`/datasets/workspace/tagging?${query}`] });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const user = userEvent.setup();
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);

  await screen.findByRole('heading', { name: '打标工作台' });
  await user.click(screen.getByRole('button', { name: '模拟切换子集' }));
  await waitFor(() => expect(router.state.location.search).toContain('subset=1'));
  expect(router.state.location.search).toContain('dataset=configs%2Fdatasets%2Fnext.toml');

  await user.click(screen.getByRole('button', { name: '模拟创建任务' }));
  await waitFor(() => expect(router.state.location.search).toContain('job=job-1'));
  expect(router.state.location.search).toContain('dataset=configs%2Fdatasets%2Fnext.toml');
  expect(router.state.location.search).toContain('subset=1');
});
