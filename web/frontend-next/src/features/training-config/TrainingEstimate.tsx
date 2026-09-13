import { useQuery } from "@tanstack/react-query";
import { Layers3, RefreshCw } from "lucide-react";
import { apiRequest } from "../../api/client";
import type { TrainingConfigFile } from "../../api/trainingContext";
import { EstimateBuckets } from "./EstimateBuckets";
import { datasetName, number, type TrainingEstimateData } from "./estimateData";
import "./TrainingEstimate.css";

export function TrainingEstimate({ file, preset, dirty }: { file?: TrainingConfigFile; preset: string; dirty: boolean }) {
  const query = useQuery({
    queryKey: ["training-config", "estimate", file?.path, preset, "buckets"],
    enabled: Boolean(file),
    staleTime: 30_000,
    queryFn: ({ signal }) => {
      const params = new URLSearchParams({ variant: file!.method || "lora", preset,
        methods_subdir: file!.methods_subdir || "gui-methods", config_file: file!.path, include_buckets: "1" });
      return apiRequest<TrainingEstimateData>(`/api/config/steps?${params}`, { signal });
    },
  });
  const data = query.data;
  return <div className="estimate-workspace" aria-busy={query.isFetching}>
    <div className="estimate-context"><span>{file?.label || file?.path || "未选择配置"}<small>{preset}{dirty ? " · 已保存版本（有未保存修改）" : " · 已保存版本"}</small></span>
      <button className="icon-button" type="button" title="重新估算" aria-label="重新估算" disabled={!file || query.isFetching} onClick={() => void query.refetch()}><RefreshCw size={16} /></button>
    </div>
    {query.error && <p className="form-error" role="alert">{query.error.message}</p>}
    {!data ? <p role="status" className="estimate-empty">{query.isFetching ? "正在读取数据集与分桶尺寸…" : "暂无估算"}</p> : <>
      <dl className="estimate-metrics">
        <div><dt>训练图片</dt><dd>{number(data.train_image_count)}<small>张</small></dd></div>
        <div><dt>有效样本 / epoch</dt><dd>{number(data.repeated_image_count)}</dd></div>
        <div><dt>每轮步数 ≈</dt><dd>{number(data.steps_per_epoch)}</dd></div>
        <div className="estimate-total"><dt>预计总步数</dt><dd>{data.duration_mode === "unset" ? "未设置" : number(data.total_steps)}</dd></div>
      </dl>
      <div className="estimate-equation">
        <span>有效批量 <strong>{number(data.effective_batch_size)}</strong> = batch {number(data.train_batch_size)} × 梯度累积 {number(data.gradient_accumulation_steps)}</span>
        <span>{data.duration_mode === "epochs" ? `${number(data.max_train_epochs ?? undefined)} epochs` : data.duration_mode === "steps" ? "固定步数" : "训练时长未设置"}</span>
      </div>
      <p className="estimate-caption">步数按有效样本 ÷ 有效批量向上取整估算；分桶尾批次、分布式训练等可能使实际步数不同。</p>
      <section className="estimate-section" aria-label="样本构成">
        <div className="estimate-section-heading"><h3><Layers3 size={16} />样本构成 <span>{data.datasets?.length || 0} 个子集</span></h3></div>
        {data.datasets?.length ? <div className="estimate-datasets">{data.datasets.map((row) => <div key={row.index} className="estimate-dataset">
          <div className="estimate-dataset-name"><span className={`estimate-tag ${row.is_reg ? "regularization" : ""}`}>{row.is_reg ? "正则化" : "训练"}</span><strong>{datasetName(row)}</strong><small>{row.uses_preprocessed_images ? "训练目录" : "源图估算"}</small></div>
          <p className="estimate-path">{row.uses_preprocessed_images ? row.image_dir : row.source_dir}</p>
          <div className="estimate-dataset-values"><span>{number(row.train_image_count)} 张 · 抽样 {Math.round(row.sample_ratio * 100)}%</span><span>{number(row.sampled_image_count)} × {number(row.num_repeats)} 次重复 <b>= {number(row.sampled_weighted_image_count)}</b></span></div>
          {row.trigger_clone_sampled_weighted_image_count > 0 && <small className="estimate-clone">触发词克隆 +{number(row.trigger_clone_sampled_weighted_image_count)} 有效样本</small>}
        </div>)}</div> : <p className="estimate-empty">暂无数据集明细</p>}
      </section>
      <EstimateBuckets key={file?.path} datasets={data.datasets || []} />
    </>}
  </div>;
}
