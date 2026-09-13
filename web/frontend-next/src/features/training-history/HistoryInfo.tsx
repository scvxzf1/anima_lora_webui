import type { ReactNode } from "react";

export function HistoryInfo({ rows }: { rows: [string, ReactNode][] }) {
  return <dl className="history-info">{rows.map(([label, value]) => (
    <div key={label}><dt>{label}</dt><dd>{value ?? "未记录"}</dd></div>
  ))}</dl>;
}
