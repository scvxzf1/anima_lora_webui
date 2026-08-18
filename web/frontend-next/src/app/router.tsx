import { createBrowserRouter, Navigate } from 'react-router-dom';
import { lazy, Suspense, type ReactNode } from 'react';

const TrainingWorkspace = lazy(async () => {
  const module = await import('../features/training-config/TrainingWorkspace');
  return { default: module.TrainingWorkspace };
});

const DatasetWorkspace = lazy(async () => {
  const module = await import('../features/dataset-editor/DatasetWorkspace');
  return { default: module.DatasetWorkspace };
});

const QueuePage = lazy(async () => {
  const module = await import('../features/training-queue/QueuePage');
  return { default: module.QueuePage };
});

const LiveMonitorPage = lazy(async () => {
  const module = await import('../features/live-monitor/LiveMonitorPage');
  return { default: module.LiveMonitorPage };
});

const HistoryPage = lazy(async () => {
  const module = await import('../features/training-history/HistoryPage');
  return { default: module.HistoryPage };
});

const HistoryDetailPage = lazy(async () => {
  const module = await import('../features/training-history/HistoryDetailPage');
  return { default: module.HistoryDetailPage };
});

function lazyPage(children: ReactNode) {
  return (
    <Suspense fallback={<main className="route-loading" aria-busy="true">正在加载工作区</main>}>
      {children}
    </Suspense>
  );
}

export const router = createBrowserRouter(
  [
    {
      path: '/',
      element: <Navigate to="/datasets" replace />,
    },
    {
      path: '/datasets',
      element: lazyPage(<DatasetWorkspace />),
    },
    {
      path: '/training',
      element: lazyPage(<TrainingWorkspace />),
    },
    {
      path: '/queue',
      element: lazyPage(<QueuePage />),
    },
    {
      path: '/monitor',
      element: lazyPage(<LiveMonitorPage />),
    },
    {
      path: '/history',
      element: lazyPage(<HistoryPage />),
    },
    {
      path: '/history/:taskId',
      element: lazyPage(<HistoryDetailPage />),
    },
  ],
  { basename: '/next' },
);
