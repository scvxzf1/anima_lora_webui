import { useQuery } from "@tanstack/react-query";
import { ClipboardCopy, RefreshCw } from "lucide-react";
import { useState } from "react";
import { QueryFeedback } from "../../components/QueryFeedback";
import { fetchEnvironment } from "./api";
import "../tools.css";

const groupLabels: Record<string, string> = { runtime: "运行环境", project_files: "项目文件", platform_tools: "系统工具", model_paths: "模型路径", python_packages: "Python 依赖", gpu_stack: "GPU / CUDA", web_runtime: "Web 服务" };

export function EnvironmentPage() {
  const query = useQuery({ queryKey: ["environment", "check"], queryFn: ({ signal }) => fetchEnvironment(signal), retry: false, staleTime: Infinity });
  const [notice, setNotice] = useState("");
  const report = query.data;
  const groups = [...new Set(report?.checks.map((check) => check.group || "其他") || [])];
  async function copyReport() {
    if (!report) return;
    try { await navigator.clipboard.writeText(JSON.stringify(report, null, 2)); setNotice("报告已复制"); }
    catch { setNotice("复制失败，请检查浏览器剪贴板权限"); }
  }
  return <main className="tool-page">
    <header className="tool-heading"><div><h1>环境检测</h1><p>当前 Web 服务运行环境</p></div><div className="tool-row">
      <button type="button" onClick={() => query.refetch()} disabled={query.isFetching}><RefreshCw size={16} />{query.isFetching ? "检测中" : "重新检测"}</button>
      <button type="button" onClick={copyReport} disabled={!report}><ClipboardCopy size={16} />复制报告</button>
    </div></header>
    <QueryFeedback query={query} label="环境检测" hasData={Boolean(report)} />
    {notice && <p role="status" className="tool-notice">{notice}</p>}
    {report && <><div className="tool-grid" aria-label="检测摘要">{Object.entries(report.summary).map(([key, value]) => <div className="tool-metric" key={key}><strong>{value}</strong><span>{({ errors: "错误", warnings: "警告", checks: "检查项" } as Record<string, string>)[key] || key}</span></div>)}</div>
      {groups.map((group) => <section className="tool-section" key={group}><h2>{groupLabels[group] || group}</h2><ul className="tool-list">{report.checks.filter((check) => (check.group || "其他") === group).map((check, index) => <li key={`${check.key}-${index}`}><strong className={check.level === "error" ? "tool-error" : check.level === "warning" ? "" : "tool-notice"}>{check.level === "error" ? "错误" : check.level === "warning" ? "警告" : "正常"}</strong> · {check.message}{check.path && !check.message.includes(check.path) && <small>{check.path}</small>}{check.hint && <small>{check.hint}</small>}</li>)}</ul></section>)}
      <details className="tool-section"><summary>平台信息</summary><pre className="tool-log">{JSON.stringify(report.platform, null, 2)}</pre></details>
    </>}
  </main>;
}
