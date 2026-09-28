import { Outlet, useRouteError, Link, useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Topbar } from "./Topbar";
import { useUIPreferences } from "./uiPreferences";

export function AppShell() {
  const location = useLocation();
  const scale = useUIPreferences(location.pathname + location.search);
  useEffect(() => {
    if (location.hash === "#workspace") {
      document.getElementById("workspace")?.focus();
    }
  }, [location.pathname, location.hash]);
  return (
    <div
      className="next-layout"
      data-training-workspace={location.pathname === "/training"}
      data-dataset-workspace={location.pathname === "/datasets"}
      data-mask-workspace={location.pathname === "/datasets/masks"}
      data-dataset-image-workspace={location.pathname.startsWith("/datasets/workspace")}
      style={{
        zoom: scale.global,
        height:
          ["/training", "/datasets", "/datasets/masks"].includes(location.pathname)
            || location.pathname.startsWith("/datasets/workspace")
            ? `calc(100dvh / ${scale.global})`
            : undefined,
      }}
    >
      <Link
        className="skip-link"
        to={{
          pathname: location.pathname,
          search: location.search,
          hash: "#workspace",
        }}
      >
        跳到工作区
      </Link>
      <Topbar />
      <div
        id="workspace"
        className="next-content"
        tabIndex={-1}
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
      </div>
    </main>
  );
}
