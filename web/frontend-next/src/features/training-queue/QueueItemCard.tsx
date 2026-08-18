import type { QueueItem, QueueSnapshot } from './api';

const STATE_LABELS: Record<string, string> = {
  queued: '等待',
  running: '运行中',
  done: '已完成',
  error: '异常',
  canceled: '已取消',
};

function basename(path: string) {
  const clean = path.replace(/\\/g, '/');
  return clean.split('/').pop() || clean;
}

export function queueItemTitle(item: QueueItem) {
  const checkpoint = String(item.resume_info?.checkpoint_name || item.resume_info?.checkpoint || '').trim();
  if (checkpoint) return `续训 · ${basename(checkpoint)}`;
  const variant = String(item.variant || '').trim();
  const preset = String(item.preset || '').trim();
  const config = basename(String(item.source_config_file || item.runtime_config_file || ''));
  return [variant || config, preset].filter(Boolean).join(' · ') || String(item.id || '未命名任务');
}

function queueItemTime(item: QueueItem) {
  if (item.state === 'running') return item.started_at_text || item.created_at_text || '';
  if (['done', 'error', 'canceled'].includes(String(item.state))) {
    return item.finished_at_text || item.started_at_text || '';
  }
  return item.created_at_text || '';
}

type Props = {
  item: QueueItem;
  snapshot: QueueSnapshot;
  queuedPosition: number;
  queuedTotal: number;
  onAction: (action: string, item: QueueItem, direction?: string) => void;
};

export function QueueItemCard({ item, snapshot, queuedPosition, queuedTotal, onAction }: Props) {
  const state = String(item.state || 'unknown');
  const id = String(item.id || '');
  const current = id === String(snapshot.current_item_id || '');
  const attempt = Math.max(1, Number(item.attempt) || 1);
  const maxAttempts = Math.max(1, Number(item.max_attempts ?? snapshot.max_attempts ?? 1) || 1);
  const gpu = item.gpu_whitelist?.length ? `GPU ${item.gpu_whitelist.join(', ')}` : '自动选择 GPU';
  const message = String(item.message || '').trim();
  const finished = ['done', 'error', 'canceled'].includes(state);
  const actionable = state === 'queued' || state === 'running';

  return (
    <article className="queue-card" data-state={state} data-current={current}>
      <div className="queue-card-main">
        <div className="queue-card-heading">
          <span className="queue-state" data-state={state}>{STATE_LABELS[state] || state}</span>
          <h3>{queueItemTitle(item)}</h3>
          {item.requires_preprocess ? <span className="queue-chip">含预处理</span> : null}
          {item.retry_of ? <span className="queue-chip">重试任务</span> : null}
          <time>{queueItemTime(item)}</time>
        </div>
        <div className="queue-card-meta">
          <span>{item.kind === 'resume' ? '续训' : '训练'}</span>
          <span>{item.methods_subdir || 'gui-methods'}</span>
          <span>{gpu}</span>
          <span>尝试 {attempt} / {maxAttempts}</span>
        </div>
        {message ? <p className="queue-card-message">{message}</p> : null}
        <dl className="queue-card-paths">
          {item.runtime_config_file ? <div><dt>运行配置</dt><dd title={item.runtime_config_file}>{item.runtime_config_file}</dd></div> : null}
          {item.source_config_file && item.source_config_file !== item.runtime_config_file ? <div><dt>源配置</dt><dd title={item.source_config_file}>{item.source_config_file}</dd></div> : null}
        </dl>
      </div>
      <div className="queue-card-actions">
        {state === 'queued' ? (
          <div className="queue-move-group">
            <button type="button" disabled={queuedPosition === 0} onClick={() => onAction('move', item, 'top')}>置顶</button>
            <button type="button" disabled={queuedPosition === 0} onClick={() => onAction('move', item, 'up')}>上移</button>
            <button type="button" disabled={queuedPosition === queuedTotal - 1} onClick={() => onAction('move', item, 'down')}>下移</button>
            <button type="button" disabled={queuedPosition === queuedTotal - 1} onClick={() => onAction('move', item, 'bottom')}>置底</button>
          </div>
        ) : null}
        {finished ? <button type="button" onClick={() => onAction('retry', item)}>重试</button> : null}
        {actionable
          ? <button type="button" className="queue-danger" onClick={() => onAction(state === 'running' ? 'stop' : 'cancel', item)}>{state === 'running' ? '停止' : '取消'}</button>
          : <button type="button" onClick={() => onAction('remove', item)}>移出列表</button>}
      </div>
    </article>
  );
}
