import { ArrowLeft, Brush, Images, ScanText } from 'lucide-react';
import { lazy, Suspense } from 'react';
import { Link, Navigate, useLocation, useNavigate, useSearchParams } from 'react-router-dom';

import { DatasetPreviewDialog } from './DatasetPreviewDialog';
import { datasetPresetQuery } from './api';
import { useQuery } from '@tanstack/react-query';
import './DatasetImageWorkspace.css';

const MaskEditorPage = lazy(async () => {
  const module = await import('../mask-editor/MaskEditorPage');
  return { default: module.MaskEditorPage };
});

const CaptioningPage = lazy(async () => {
  const module = await import('../captioning/CaptioningPage');
  return { default: module.CaptioningPage };
});

const VIEWS = [
  { key: 'preview', label: '预览', path: 'preview', icon: Images },
  { key: 'masks', label: '编辑蒙版', path: 'masks', icon: Brush },
  { key: 'tagging', label: '打标', path: 'tagging', icon: ScanText },
] as const;

export function DatasetImageWorkspacePage() {
  const [params] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const file = params.get('dataset') || '';
  const subset = Math.max(0, Number(params.get('subset')) || 0);
  const preset = useQuery({ ...datasetPresetQuery(file), enabled: Boolean(file) });
  const suffix = location.pathname.replace(/^\/datasets\/workspace\/?/, '');
  const view = suffix.startsWith('masks') ? 'masks' : suffix.startsWith('tagging') ? 'tagging' : 'preview';
  const state = location.state && typeof location.state === 'object'
    ? location.state as { returnTo?: unknown }
    : {};
  const query = `?${new URLSearchParams({ dataset: file, subset: String(subset) })}`;
  const datasetReturn = typeof state.returnTo === 'string' && /^\/datasets(?:[/?#]|$)/.test(state.returnTo)
    ? state.returnTo : null;
  const returnTo = datasetReturn || `/datasets?${new URLSearchParams({ dataset: file })}`;

  function goBack() {
    if (datasetReturn) navigate(-1);
    else navigate(returnTo, { replace: true });
  }

  if (!suffix) {
    return <Navigate to={`/datasets/workspace/preview${query}`} replace state={location.state} />;
  }

  return (
    <main className="dataset-image-workspace">
      <header className="dataset-image-workspace-header">
        <button type="button" className="dataset-image-workspace-back" aria-label="返回数据集" title="返回数据集" onClick={goBack}>
          <ArrowLeft aria-hidden="true" size={19} />
        </button>
        <div className="dataset-image-workspace-context">
          <p className="eyebrow">DATASET IMAGES</p>
          <h1>图片工作台</h1>
          <span title={file}>{preset.data?.name || file || '未选择数据集'} · 子集 {subset + 1}</span>
        </div>
      </header>

      <nav
        className="dataset-image-workspace-tabs"
        role="tablist"
        aria-label="图片工作台功能区"
        aria-orientation="horizontal"
        onKeyDown={(event) => {
          const tabs = Array.from(event.currentTarget.querySelectorAll<HTMLAnchorElement>('[role="tab"]'));
          const activeIndex = tabs.indexOf(document.activeElement as HTMLAnchorElement);
          const nextIndex = event.key === 'ArrowRight'
            ? (activeIndex + 1) % tabs.length
            : event.key === 'ArrowLeft'
              ? (activeIndex - 1 + tabs.length) % tabs.length
              : event.key === 'Home'
                ? 0
                : event.key === 'End'
                  ? tabs.length - 1
                  : -1;
          if (nextIndex < 0 || !tabs.length) return;
          event.preventDefault();
          tabs[nextIndex].focus();
          tabs[nextIndex].click();
        }}
      >
        {VIEWS.map(({ key, label, path, icon: Icon }) => (
          <Link
            key={key}
            id={`dataset-image-workspace-tab-${key}`}
            to={`/datasets/workspace/${path}${query}`}
            replace
            state={location.state}
            role="tab"
            aria-selected={view === key}
            aria-controls="dataset-image-workspace-panel"
            aria-current={view === key ? 'page' : undefined}
          >
            <Icon aria-hidden="true" size={16} />
            {label}
          </Link>
        ))}
      </nav>

      <section
        id="dataset-image-workspace-panel"
        className="dataset-image-workspace-panel"
        data-view={view}
        role="tabpanel"
        aria-labelledby={`dataset-image-workspace-tab-${view}`}
        tabIndex={0}
        aria-label={`${VIEWS.find((item) => item.key === view)?.label || '预览'}区域`}
      >
        {view === 'preview' ? (
          file ? (
            <DatasetPreviewDialog file={file} datasetIndex={subset} returnFocus={null} onClose={goBack} embedded />
          ) : <p className="dataset-empty">未选择数据集</p>
        ) : view === 'masks' ? (
          <Suspense fallback={<p className="route-loading" aria-busy="true" role="status" aria-live="polite">正在加载蒙版编辑器</p>}>
            <MaskEditorPage embedded />
          </Suspense>
        ) : (
          <Suspense fallback={<p className="route-loading" aria-busy="true" role="status" aria-live="polite">正在加载打标工作台</p>}>
            <CaptioningPage embeddedBasePath="/datasets/workspace/tagging" />
          </Suspense>
        )}
      </section>
    </main>
  );
}
