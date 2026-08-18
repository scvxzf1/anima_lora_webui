import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { Topbar } from '../../app/Topbar';
import { fetchHistoryTaskDetail, historyKeys, type HistoryTaskSummary } from './api';
import './HistoryDetailPage.css';

function taskName(task?: HistoryTaskSummary) {
  return String(task?.name || task?.history_run_label || task?.id || '历史任务');
}

function metricValue(point: Record<string, unknown>, key: string) {
  const value = point[key];
  return value == null ? null : Number(value);
}

export function HistoryDetailPage() {
  const { taskId = '' } = useParams();
  const query = useQuery({
    queryKey: historyKeys.detail(taskId),
    queryFn: ({ signal }) => fetchHistoryTaskDetail(taskId, signal),
    enabled: Boolean(taskId),
  });
  const detail = query.data;
  const task = detail?.task;
  const metrics = detail?.metrics || [];
  const logs = detail?.logs || [];
  const lastMetric = metrics.at(-1) || {};
  const lastLoss = metricValue(lastMetric, 'loss');
  const lastStep = metricValue(lastMetric, 'step');
  const lastLr = metricValue(lastMetric, 'lr');

  return (
    <div className="history-detail-shell">
      <Topbar />
      <main className="history-detail-page">
        <header className="history-detail-header">
          <div>
            <Link to="/history" className="history-back">← 返回历史</Link>
            <p className="eyebrow">{task?.job === 'preprocess' ? 'PREPROCESS' : 'TRAINING'}</p>
            <h1>{taskName(task)}</h1>
            <p>{task?.started_at_text || taskId}</p>
          </div>
          <span className="history-detail-state" data-state={task?.state || 'unknown'}>
            {task?.state === 'idle' ? '完成' : task?.state || '未知'}
          </span>
        </header>

        {query.error ? (
          <section className="history-detail-error" role="alert"><h2>无法读取历史任务</h2><p>{query.error.message}</p></section>
        ) : (
          <>
            <section className="history-detail-stats">
              <div><span>最终损失</span><strong>{lastLoss != null ? lastLoss.toFixed(4) : 'N/A'}</strong></div>
              <div><span>最后步数</span><strong>{lastStep != null ? lastStep.toFixed(0) : (task?.metric_count ?? '—')}</strong></div>
              <div><span>学习率</span><strong>{lastLr != null ? lastLr.toFixed(6) : 'N/A'}</strong></div>
              <div><span>曲线数据</span><strong>{metrics.length || task?.metric_count || 0} 点</strong></div>
            </section>

            <section className="history-detail-grid">
              <div className="history-detail-panel">
                <header><span className="eyebrow">运行信息</span><h2>任务信息</h2></header>
                <dl className="history-info">
                  <InfoRow label="任务 ID" value={task?.id} />
                  <InfoRow label="类型" value={task?.job === 'preprocess' ? '预处理' : '训练'} />
                  <InfoRow label="分组" value={task?.group || task?.history_group_key || '未分组'} />
                  <InfoRow label="源配置" value={task?.history_source_config_file} />
                  <InfoRow label="运行目录" value={task?.run_dir || task?.output_dir || task?.training_output_dir} />
                  <InfoRow label="日志" value={`${task?.log_count ?? logs.length} 行`} />
                  <InfoRow label="指标" value={`${task?.metric_count ?? metrics.length} 点`} />
                </dl>
              </div>
              <div className="history-detail-panel">
                <header><span className="eyebrow">日志</span><h2>日志记录</h2><span>{logs.length} 行</span></header>
                <div className="history-detail-logs">
                  {logs.length ? logs.map((log, index) => (
                    <div key={log.id || index} className="history-detail-log-line" data-type={log.type || 'info'}>
                      {log.line || String(log)}
                    </div>
                  )) : <p className="history-detail-empty">暂无日志。</p>}
                </div>
              </div>
            </section>

            {detail?.config_toml ? (
              <section className="history-detail-panel">
                <header><span className="eyebrow">CONFIG</span><h2>配置快照</h2></header>
                <pre className="history-detail-toml">{detail.config_toml}</pre>
              </section>
            ) : null}
          </>
        )}
      </main>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value?: string | number }) {
  return <div><dt>{label}</dt><dd title={String(value || '')}>{value || '—'}</dd></div>;
}
