import { useState } from "react";
import { BarChart3 } from "lucide-react";
import { collectBuckets, datasetName, number, orientation, type EstimateDataset } from "./estimateData";

export function EstimateBuckets({ datasets }: { datasets: EstimateDataset[] }) {
  const [scope, setScope] = useState("all");
  const [sort, setSort] = useState("count");
  const rows = scope === "all" ? datasets : datasets.filter((row) => String(row.index) === scope);
  const buckets = collectBuckets(rows).sort((a, b) => sort === "count"
    ? b.count - a.count || a.width / a.height - b.width / b.height
    : a.width / a.height - b.width / b.height || a.width - b.width);
  const total = buckets.reduce((sum, bucket) => sum + bucket.count, 0);
  const maxCount = Math.max(1, ...buckets.map((bucket) => bucket.count));
  const pending = rows.filter((row) => row.bucket_distribution?.status === "pending").length;
  const unavailable = rows.filter((row) => !row.bucket_distribution).length;
  const unreadable = rows.reduce((sum, row) => sum + (row.bucket_distribution?.unreadable_count || 0), 0);
  const unscanned = rows.reduce((sum, row) => sum + (row.bucket_distribution?.unscanned_count || 0), 0);
  const predicted = rows.filter((row) => row.bucket_distribution?.basis === "source_projection").length;
  const measured = rows.filter((row) => row.bucket_distribution?.basis === "image_dimensions" && row.bucket_distribution.status !== "pending").length;
  const filtered = rows.reduce((sum, row) => sum + (row.bucket_distribution?.filtered_count || 0), 0);
  const groups = [
    { key: "landscape", label: "横图" },
    { key: "square", label: "方图" },
    { key: "portrait", label: "竖图" },
  ].map((group) => ({ ...group, count: buckets.filter((bucket) => orientation(bucket) === group.key).reduce((sum, bucket) => sum + bucket.count, 0) }));
  return <section className="estimate-section" aria-label="分桶分布">
    <div className="estimate-section-heading">
      <h3><BarChart3 size={16} />分桶分布 <span>{number(buckets.length)} 个尺寸</span></h3>
      <div className="estimate-controls">
        <select aria-label="分桶数据集" value={scope} onChange={(event) => setScope(event.target.value)}>
          <option value="all">全部数据集</option>
          {datasets.map((row) => <option key={row.index} value={row.index}>{row.index}. {datasetName(row)}{row.is_reg ? " · 正则化" : ""}</option>)}
        </select>
        <select aria-label="分桶排序" value={sort} onChange={(event) => setSort(event.target.value)}>
          <option value="count">数量优先</option><option value="aspect">宽高比排序</option>
        </select>
      </div>
    </div>
    <p className="estimate-caption">{predicted > 0 ? `源图预测 ${predicted} 个子集${measured > 0 ? ` · 训练目录实测 ${measured} 个子集` : ""} · 按已保存预处理配置选桶` : "image_dir 尺寸实测"} · 不含重复、抽样、验证划分和触发词克隆；最终训练分桶以运行结果为准。</p>
    {unavailable > 0 && <p className="estimate-warning" role="status">{unavailable} 个数据集未返回分桶信息，请更新并重启 WebUI 后端。</p>}
    {pending > 0 && <p className="estimate-warning" role="status">{pending} 个数据集的源图与训练目录均无匹配图片，请检查目录和筛选规则。</p>}
    {filtered > 0 && <p className="estimate-caption">按最低像素阈值排除 {number(filtered)} 张源图。</p>}
    {unreadable > 0 && <p className="estimate-warning" role="status">{number(unreadable)} 张图片无法读取尺寸，未计入下方分布。</p>}
    {unscanned > 0 && <p className="estimate-warning" role="status">已达到单次扫描边界，尚有 {number(unscanned)} 张未统计；下方占比仅代表已读取图片。</p>}
    {total > 0 ? <>
      <div className="estimate-aspect-bar" aria-hidden="true">
        {groups.map((group) => <span key={group.key} className={group.key} style={{ width: `${group.count / total * 100}%` }} />)}
      </div>
      <div className="estimate-aspect-legend">
        {groups.map((group) => <span key={group.key}><i className={group.key} />{group.label} <strong>{number(group.count)}</strong><small>{(group.count / total * 100).toFixed(1)}%</small></span>)}
        <span className="estimate-count">共 {number(total)} 张</span>
      </div>
      <div className="estimate-bucket-table" role="table" aria-label="分桶尺寸明细" tabIndex={0}>
        <div className="estimate-bucket-row estimate-table-head" role="row">
          <span role="columnheader">尺寸 / 宽高比</span><span role="columnheader">数量分布</span><span role="columnheader">图片</span><span role="columnheader">占比</span>
        </div>
        {buckets.map((bucket) => <div key={`${bucket.width}x${bucket.height}`} className="estimate-bucket-row" role="row">
          <div className="estimate-bucket-size" role="cell">
            <span className="estimate-shape-space" aria-hidden="true"><i className={orientation(bucket)} style={{ width: `${32 * Math.min(1, bucket.width / bucket.height)}px`, height: `${32 * Math.min(1, bucket.height / bucket.width)}px` }} /></span>
            <span><strong>{bucket.width} × {bucket.height}</strong><small>{(bucket.width / bucket.height).toFixed(2)}:1 · {(bucket.width * bucket.height / 1e6).toFixed(2)} MP</small></span>
          </div>
          <span role="cell" className="estimate-bar-track"><i className={orientation(bucket)} style={{ width: `${bucket.count / maxCount * 100}%` }} /></span>
          <strong role="cell">{number(bucket.count)}</strong>
          <span role="cell">{(bucket.count / total * 100).toFixed(1)}%</span>
        </div>)}
      </div>
    </> : <p className="estimate-empty">暂无可统计的分桶图片</p>}
  </section>;
}
