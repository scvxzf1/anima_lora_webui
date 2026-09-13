import { Outlet, useRouteError, Link, useLocation } from "react-router-dom";
import { Topbar } from "./Topbar";
import { useUIPreferences } from "./uiPreferences";

export function AppShell() {
  const location = useLocation();
  const scale = useUIPreferences(location.pathname + location.search);
  return (
    <div
      className="next-layout"
      data-training-workspace={location.pathname === "/training"}
      data-dataset-workspace={location.pathname === "/datasets"}
      data-mask-workspace={location.pathname === "/datasets/masks"}
      style={{
        zoom: scale.global,
        height:
          ["/training", "/datasets", "/datasets/masks"].includes(location.pathname)
            ? `calc(100dvh / ${scale.global})`
            : undefined,
      }}
    >
      <a className="skip-link" href="#workspace">
        跳到工作区
      </a>
      <Topbar />
      <div
        id="workspace"
        className="next-content"
        style={{ zoom: scale.content }}
      >
        <Outlet />
      </div>
    </div>
  );
}
export function RouteError() {
  const error = useRouteError();
  return (
    <main className="error-panel" role="alert">
      <h1>工作区未能加载</h1>
      <p>
        {error instanceof Error
          ? error.message
          : "页面不存在或资源暂时不可用。"}
      </p>
      <div className="toolbar">
        <Link to="/training">训练配置</Link>
        <a href="/?ui=dragon">旧 Dragon 界面</a>
        <a href="/?ui=classic">Classic 界面</a>
      </div>
    </main>
  );
}
