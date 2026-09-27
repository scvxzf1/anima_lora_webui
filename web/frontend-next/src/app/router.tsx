import { createBrowserRouter, Navigate } from "react-router-dom";
import { lazy, Suspense, type ReactNode } from "react";
import { AppShell, RouteError } from "./AppShell";

const TrainingWorkspace = lazy(async () => {
  const module = await import("../features/training-config/TrainingWorkspace");
  return { default: module.TrainingWorkspace };
});

const DatasetWorkspace = lazy(async () => {
  const module = await import("../features/dataset-editor/DatasetWorkspace");
  return { default: module.DatasetWorkspace };
});

const MaskEditorPage = lazy(async () => {
  const module = await import("../features/mask-editor/MaskEditorPage");
  return { default: module.MaskEditorPage };
});

const DatasetImageWorkspacePage = lazy(async () => {
  const module = await import("../features/dataset-editor/DatasetImageWorkspacePage");
  return { default: module.DatasetImageWorkspacePage };
});

const QueuePage = lazy(async () => {
  const module = await import("../features/training-queue/QueuePage");
  return { default: module.QueuePage };
});

const LiveMonitorPage = lazy(async () => {
  const module = await import("../features/live-monitor/LiveMonitorPage");
  return { default: module.LiveMonitorPage };
});

const HistoryPage = lazy(async () => {
  const module = await import("../features/training-history/HistoryPage");
  return { default: module.HistoryPage };
});

const HistoryDetailPage = lazy(async () => {
  const module = await import("../features/training-history/HistoryDetailPage");
  return { default: module.HistoryDetailPage };
});

const ModelConfigPage = lazy(async () => {
  const module = await import("../features/settings/ModelConfigPage");
  return { default: module.ModelConfigPage };
});

const SettingsPage = lazy(async () => {
  const module = await import("../features/settings/SettingsPage");
  return { default: module.SettingsPage };
});

const CaptioningPage = lazy(async () => {
  const module = await import("../features/captioning/CaptioningPage");
  return { default: module.CaptioningPage };
});

const ImageTestPage = lazy(async () => ({ default: (await import("../features/image-test/ImageTestPage")).ImageTestPage }));
const WeightAnalysisPage = lazy(async () => ({ default: (await import("../features/weight-analysis/WeightAnalysisPage")).WeightAnalysisPage }));
const EnvironmentPage = lazy(async () => ({ default: (await import("../features/environment/EnvironmentPage")).EnvironmentPage }));

function lazyPage(children: ReactNode) {
  return (
    <Suspense
      fallback={
        <main className="route-loading" aria-busy="true">
          <p role="status" aria-live="polite">正在加载工作区</p>
        </main>
      }
    >
      {children}
    </Suspense>
  );
}

export const router = createBrowserRouter(
  [
    {
      element: <AppShell />,
      errorElement: <RouteError />,
      children: [
        {
          path: "/",
          element: <Navigate to="/training" replace />,
        },
        {
          path: "/datasets",
          element: lazyPage(<DatasetWorkspace />),
        },
        { path: "/datasets/masks", element: lazyPage(<MaskEditorPage />) },
        { path: "/datasets/workspace/*", element: lazyPage(<DatasetImageWorkspacePage />) },
        {
          path: "/training",
          element: lazyPage(<TrainingWorkspace />),
        },
        {
          path: "/queue",
          element: lazyPage(<QueuePage />),
        },
        {
          path: "/monitor",
          element: lazyPage(<LiveMonitorPage />),
        },
        {
          path: "/history",
          element: lazyPage(<HistoryPage />),
        },
        {
          path: "/history/:taskId",
          element: lazyPage(<HistoryDetailPage />),
        },
        { path: "/models", element: lazyPage(<ModelConfigPage />) },
        { path: "/captioning/*", element: lazyPage(<CaptioningPage />) },
        { path: "/settings", element: lazyPage(<SettingsPage />) },
        { path: "/image-test", element: lazyPage(<ImageTestPage />) },
        { path: "/weight-analysis", element: lazyPage(<WeightAnalysisPage />) },
        { path: "/environment", element: lazyPage(<EnvironmentPage />) },
        { path: "*", element: <RouteError /> },
      ],
    },
  ],
  { basename: "/next" },
);
