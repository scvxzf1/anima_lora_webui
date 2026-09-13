import type { ReactNode } from "react";
import { HistoryInfo } from "./HistoryInfo";

export function OverviewInfo({ rows }: { rows: [string, ReactNode][] }) {
  const missing = rows.filter(([, value]) => value == null || value === "" || value === "未记录");
  const available = rows.filter((row) => !missing.includes(row));
  return <>
    {available.length > 0 && <HistoryInfo rows={available} />}
    {missing.length > 0 && <details className="history-missing-fields">
      <summary>{missing.length} 项信息未记录</summary>
      <HistoryInfo rows={missing.map(([label]) => [label, "未记录"])} />
    </details>}
  </>;
}
