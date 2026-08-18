import { useState } from 'react';
import type { QueueSettingsPayload, QueueSnapshot } from './api';

type Props = {
  snapshot: QueueSnapshot;
  onSave: (payload: QueueSettingsPayload) => void;
  disabled?: boolean;
};

export function QueuePolicyForm({ snapshot, onSave, disabled = false }: Props) {
  const [failurePolicy, setFailurePolicy] = useState(snapshot.failure_policy || 'pause');
  const [autoRetry, setAutoRetry] = useState(Boolean(snapshot.auto_retry));
  const [maxAttempts, setMaxAttempts] = useState(String(snapshot.max_attempts ?? 1));
  const [retryBackoff, setRetryBackoff] = useState(String(snapshot.retry_backoff_sec ?? 0));

  return (
    <form
      className="queue-policy"
      onSubmit={(event) => {
        event.preventDefault();
        onSave({
          failure_policy: failurePolicy,
          auto_retry: autoRetry,
          max_attempts: Number(maxAttempts),
          retry_backoff_sec: Number(retryBackoff),
        });
      }}
    >
      <header><div><span className="eyebrow">调度策略</span><h2>失败与重试</h2></div><span>仅覆盖当前队列</span></header>
      <label className="queue-field">
        <span>任务失败后</span>
        <select value={failurePolicy} onChange={(event) => setFailurePolicy(event.target.value)}>
          <option value="pause">暂停队列，等待处理</option>
          <option value="continue">继续执行后续任务</option>
        </select>
      </label>
      <label className="queue-check">
        <input type="checkbox" checked={autoRetry} onChange={(event) => setAutoRetry(event.target.checked)} />
        <span><strong>自动重试可恢复异常</strong><small>最大次数包含首次运行；失败策略为“暂停”时，重试任务会等待手动继续。</small></span>
      </label>
      <div className="queue-policy-grid">
        <label className="queue-field">
          <span>最大尝试次数</span>
          <input type="number" min="1" max="10" step="1" value={maxAttempts} onChange={(event) => setMaxAttempts(event.target.value)} />
        </label>
        <label className="queue-field">
          <span>重试等待（秒）</span>
          <input type="number" min="0" max="3600" step="1" value={retryBackoff} onChange={(event) => setRetryBackoff(event.target.value)} />
        </label>
      </div>
      <button type="submit" className="primary-command" disabled={disabled}>保存策略</button>
    </form>
  );
}
