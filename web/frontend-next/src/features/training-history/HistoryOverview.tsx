import { Link } from "react-router-dom";
import type { HistoryTaskDetail } from "./api";
import { HistoryInfo } from "./HistoryInfo";
import { HistoryArtifacts } from "./HistoryArtifacts";
import { displayField, historyCheckpointLabel, historySummary, snapshotFields, taskOutcome } from "./historySummary";
import { formatDuration, formatLearningRate, formatLoss, formatStep } from "../../components/trainingNumbers";
import { overviewProgress } from "./overviewProgress";
import { OverviewInfo } from "./OverviewInfo";
import { HistoryResultSummary } from "./HistoryResultSummary";
import type { ReactNode } from "react";

export function HistoryOverview({ detail }: { detail: HistoryTaskDetail }) {
  const task = detail.task;
  const training = task?.job === "training";
  const summary = historySummary(task, detail.metrics);
  const { stepTarget } = overviewProgress(detail);
  return <>
    <section className="history-outcome" aria-label="执行结果" data-state={task?.state}>
      <p>{taskOutcome(task)}</p>
      {task?.state === "running" && <Link to={`/monitor?from_task=${encodeURIComponent(task.id || "")}`}>查看当前监控</Link>}
    </section>
    {training ? <dl className="history-detail-stats" aria-label="训练摘要">
      <div><dt>{task?.state === "running" ? "最新 Loss" : "末次有效 Loss"}</dt><dd>{formatLoss(summary.loss)}</dd></div>
      <div><dt>{stepTarget ? "实际 / 目标步数" : "最后步数"}</dt><dd>{formatStep(summary.step)}{stepTarget ? ` / ${stepTarget}` : ""}</dd></div>
      <div><dt>运行时长</dt><dd>{formatDuration(task?.started_at, task?.finished_at)}</dd></div>
    </dl> : task?.job === "preprocess" ? <section className="history-overview-section">
      <h2>预处理结果</h2>
      <OverviewInfo rows={[
        ["源图片", task?.source_image_dir], ["缩放图片", task?.resized_image_dir],
        ["缓存目录", task?.dataset_cache_dir || task?.lora_cache_dir],
        ["预处理精度", task?.preprocess_precision],
        ["运行时长", formatDuration(task?.started_at, task?.finished_at)],
      ]} />
    </section> : <p role="status">此记录未保存可识别的任务类型。</p>}
    {training && task?.id && <HistoryResultSummary taskId={task.id} />}
    {task?.id && <HistoryArtifacts taskId={task.id} />}
    <div className="history-overview-grid">
      <HistoryFingerprint detail={detail} />
      <section className="history-overview-section">
        <h2>来源与结果</h2>
        <OverviewInfo rows={[
          ["集合", task?.group || "未分类"],
          ["配置组", task?.history_group_label || task?.history_group_key],
          ["来源任务", task?.source_task_id ? <Link to={`/history/${encodeURIComponent(task.source_task_id)}`}>{task.source_task_name || task.source_task_id}</Link> : "无来源任务记录"],
          ...(training ? [
            ["来源检查点", historyCheckpointLabel(task?.resume_from)],
            ["关联预处理", task?.linked_preprocess_task?.id ? <Link to={`/history/${encodeURIComponent(task.linked_preprocess_task.id)}`}>{task.linked_preprocess_task.name || task.linked_preprocess_task.id}</Link> : "未记录"],
            ["检查点", "尚未检查"],
          ] as [string, ReactNode][] : []),
          ["输出目录", task?.output_dir || task?.training_output_dir],
          ["样张目录", task?.sample_dir],
        ]} />
      </section>
    </div>
    <details className="history-diagnostics">
      <summary>诊断信息与完整路径</summary>
      <HistoryInfo rows={[
        ["任务 ID", task?.id], ["源配置", task?.history_source_config_file],
        ["运行目录", task?.run_dir || task?.output_dir],
        ["开始时间", task?.started_at_text], ["结束时间", task?.finished_at_text],
        ["退出码", task?.returncode], ["日志采样", `${task?.log_count ?? detail.logs?.length ?? 0} 行`],
        ["指标采样", `${task?.metric_count ?? detail.metrics?.length ?? 0} 点`],
        ...(training ? [["末次学习率", formatLearningRate(summary.lr)] as [string, string]] : []),
      ]} />
    </details>
  </>;
}

export function HistoryFingerprint({ detail }: { detail: HistoryTaskDetail }) {
  const { values, invalid } = snapshotFields(detail);
  const task = detail.task;
  const training = task?.job === "training";
  const { epochTarget, stepTarget } = overviewProgress(detail);
  const pick = (key: string, fallback?: unknown) => displayField(values[key] ?? fallback);
  return <section className="history-overview-section">
    <h2>配置指纹</h2>
    {invalid && <p role="status">快照格式无法解析，仅显示已保存的任务摘要。</p>}
    {!detail.config_toml && <p className="muted">未保存配置快照</p>}
    <OverviewInfo rows={[
      ["模型族", pick("model_family", task?.model_family)],
      ...(training ? [
        ["训练方法", task?.training_variant || task?.variant || "未记录"],
        ["Rank", pick("network_dim")], ["基础计算", pick("base_compute", task?.base_compute)],
        ["训练精度", pick("mixed_precision", task?.precision_preference)], ["交换精度", task?.block_swap_precision],
        ["训练目标", epochTarget ? `${epochTarget} 轮（总步数未估算）` : stepTarget ? `${stepTarget} 步` : "未记录"], ["梯度检查点", pick("gradient_checkpointing")],
        ["编译", pick("torch_compile")], ["交换层数", pick("blocks_to_swap")],
      ] as [string, string | undefined][] : []),
    ]} />
  </section>;
}
