import { useMemo, useState } from "react";
import { comparisonParameters, type ComparisonEntry } from "./comparisonData";

export function ComparisonParameters({ entries }: { entries: ComparisonEntry[] }) {
  const [differencesOnly, setDifferencesOnly] = useState(true);
  const [page, setPage] = useState(0);
  const all = useMemo(() => comparisonParameters(entries), [entries]);
  const rows = differencesOnly ? all.filter((row) => row.different) : all;
  const pages = Math.max(1, Math.ceil(rows.length / 30));
  const current = Math.min(page, pages - 1);
  return <section className="comparison-parameters" aria-label="历史快照参数对比">
    <div className="chart-heading">
      <h3>历史快照参数</h3>
      <label className="checkbox-row"><input type="checkbox" checked={differencesOnly} onChange={(event) => { setDifferencesOnly(event.target.checked); setPage(0); }} />仅显示差异</label>
    </div>
    {!rows.length ? <p className="empty-state">{all.length ? "已记录参数没有差异" : "没有可比较的配置快照"}</p> : <>
      <div className="comparison-table-scroll" tabIndex={0} role="region" aria-label="配置参数表">
        <table className="comparison-table">
          <thead><tr><th scope="col">参数</th>{entries.map((entry) => <th scope="col" key={entry.id}>{entry.name}</th>)}</tr></thead>
          <tbody>{rows.slice(current * 30, (current + 1) * 30).map((row) => <tr key={row.key} data-different={row.different}>
            <th scope="row"><code>{row.key}</code></th>{row.values.map((value, index) => <td key={entries[index].id}>{value ?? "未记录"}</td>)}
          </tr>)}</tbody>
        </table>
      </div>
      <div className="toolbar comparison-pagination">
        <span>共 {rows.length} 项 · 第 {current + 1} / {pages} 页</span>
        <button type="button" disabled={!current} onClick={() => setPage(current - 1)}>上一页参数</button>
        <button type="button" disabled={current + 1 >= pages} onClick={() => setPage(current + 1)}>下一页参数</button>
      </div>
    </>}
  </section>;
}
