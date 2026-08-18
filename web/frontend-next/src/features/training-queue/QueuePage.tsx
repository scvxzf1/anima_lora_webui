import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import { Topbar } from '../../app/Topbar';
import {
  abortQueueAfterCurrent,
  cancelAllQueueItems,
  cancelQueueItem,
  cancelWaitingQueueItems,
  clearCanceledQueueItems,
  clearCompletedQueueItems,
  fetchQueueSnapshot,
  forceAbortQueue,
  moveQueueItem,
  queueKeys,
  retryQueueItem,
  saveQueueSettings,
  type QueueItem,
  type QueueSettingsPayload,
  type QueueSnapshot,
} from './api';
import { QueueItemCard } from './QueueItemCard';
import { QueuePolicyForm } from './QueuePolicyForm';
import './QueuePage.css';

const FILTERS = [
  ['active', '待处理'],
  ['all', '全部'],
  ['queued', '等待'],
  ['running', '运行'],
  ['error', '异常'],
  ['done', '完成'],
  ['canceled', '已取消'],
] as const;

function filterItems(items: QueueItem[], filter: string) {
  if (filter === 'all') return items;
  if (filter === 'active') return items.filter((item) => item.state === 'queued' || item.state === 'running' || item.state === 'error');
  return items.filter((item) => item.state === filter);
}

export function QueuePage() {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState('active');
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');

  const query = useQuery({
    queryKey: queueKeys.snapshot,
    queryFn: ({ signal }) => fetchQueueSnapshot(signal),
    refetchInterval: 5000,
  });
  const snapshot = query.data || emptySnapshot();

  function applySnapshot(next: QueueSnapshot) {
    queryClient.setQueryData(queueKeys.snapshot, next);
    setNotice(next.message || '队列状态已更新。');
    setError('');
  }

  function applyFailure(message: string) {
    setError(message);
    setNotice('');
  }

  const policyMutation = useMutation({
    mutationFn: (payload: QueueSettingsPayload) => saveQueueSettings(payload),
    onSuccess: applySnapshot,
    onError: (err: Error) => applyFailure(err.message),
  });

  const actionMutation = useMutation({
    mutationFn: async ({ action, item, direction }: { action: string; item: QueueItem; direction?: string }) => {
      const id = String(item.id || '');
      switch (action) {
        case 'move':
          return moveQueueItem(id, direction as 'up' | 'down' | 'top' | 'bottom');
        case 'retry':
          return retryQueueItem(id);
        case 'cancel':
          return cancelQueueItem(id, false);
        case 'stop':
          return cancelQueueItem(id, false);
        case 'remove':
          return cancelQueueItem(id, false);
        case 'cancel-all':
          return cancelAllQueueItems();
        case 'abort-after-current':
          return abortQueueAfterCurrent();
        case 'force-abort':
          return forceAbortQueue();
        case 'cancel-waiting':
          return cancelWaitingQueueItems();
        case 'clear-completed':
          return clearCompletedQueueItems();
        case 'clear-canceled':
          return clearCanceledQueueItems();
        default:
          throw new Error('未知队列操作');
      }
    },
    onSuccess: applySnapshot,
    onError: (err: Error) => applyFailure(err.message),
  });

  const summary = snapshot.summary || {};
  const visible = useMemo(() => filterItems(snapshot.items || [], filter), [snapshot.items, filter]);
  const queuedItems = useMemo(() => (snapshot.items || []).filter((item) => item.state === 'queued'), [snapshot.items]);
  const queuedPositions = new Map(queuedItems.map((item, index) => [String(item.id || ''), index]));
  const running = (summary.running || 0) > 0 || snapshot.status === 'running';

  function confirmAction(action: string) {
    const messages: Record<string, string> = {
      'cancel-all': '确定取消全部队列任务吗？等待任务会转为已取消，当前运行任务会停止。',
      'abort-after-current': '当前任务完成后停止队列，并取消后续等待任务吗？',
      'force-abort': '立即强制中止运行任务和等待任务吗？训练文件不会删除。',
      'cancel-waiting': '确定取消全部等待中的任务吗？',
      'clear-completed': '清理已完成队列记录吗？训练历史、运行目录和权重不会删除。',
      'clear-canceled': '清理已取消队列记录吗？训练历史、运行目录和权重不会删除。',
    };
    return window.confirm(messages[action] || '确定执行该操作吗？');
  }

  function runAction(action: string, item?: QueueItem, direction?: string) {
    if (action !== 'move' && action !== 'retry' && !confirmAction(action)) return;
    actionMutation.mutate({ action, item: item || {}, direction });
  }

  function runItemAction(action: string, item: QueueItem, direction?: string) {
    if (action === 'cancel' && !window.confirm('确定取消这个等待任务吗？')) return;
    if (action === 'stop' && !window.confirm('确定停止这个正在运行的任务吗？')) return;
    if (action === 'remove' && !window.confirm('只从队列列表移除这条已结束记录吗？')) return;
    actionMutation.mutate({ action, item, direction });
  }

  return (
    <div className="queue-shell">
      <Topbar />
      <main className="queue-page">
        <header className="queue-header">
          <div>
            <p className="eyebrow">TRAINING QUEUE</p>
            <h1>训练队列</h1>
            <p>管理等待顺序、失败策略和重试规则；运行日志与性能指标请前往“当前监控”。</p>
          </div>
          <div className="queue-header-actions">
            <span className="queue-badge" data-state={snapshot.paused ? 'paused' : snapshot.status}>
              {snapshot.paused ? '队列已暂停' : running ? '正在执行' : (summary.queued || 0) > 0 ? '等待调度' : '队列空闲'}
            </span>
            <button type="button" disabled={policyMutation.isPending || actionMutation.isPending} onClick={() => policyMutation.mutate({ paused: !snapshot.paused })}>
              {snapshot.paused ? '继续队列' : '暂停队列'}
            </button>
            <button type="button" disabled={query.isFetching} onClick={() => query.refetch()}>刷新</button>
            <button type="button" onClick={() => runAction('abort-after-current')}>中止后续队列</button>
            <button type="button" onClick={() => runAction('force-abort')}>强制中止</button>
          </div>
        </header>

        {query.error || error ? (
          <section className="queue-error" role="alert"><h2>队列操作失败</h2><p>{error || query.error?.message}</p></section>
        ) : null}

        <section className="queue-stats" aria-label="队列统计">
          <StatButton label="全部" value={summary.total} filter={filter} target="all" onFilter={setFilter} />
          <StatButton label="等待" value={summary.queued} filter={filter} target="queued" onFilter={setFilter} />
          <StatButton label="运行" value={summary.running} filter={filter} target="running" onFilter={setFilter} />
          <StatButton label="异常" value={summary.error} filter={filter} target="error" onFilter={setFilter} />
          <StatButton label="完成" value={summary.done} filter={filter} target="done" onFilter={setFilter} />
          <StatButton label="取消" value={summary.canceled} filter={filter} target="canceled" onFilter={setFilter} />
        </section>

        <div className="queue-layout">
          <section className="queue-worklist">
            <header>
              <div><span className="eyebrow">任务列表</span><h2>{FILTERS.find(([key]) => key === filter)?.[1]}</h2><p>显示 {visible.length} / {(snapshot.items || []).length} 个任务</p></div>
              <div className="queue-bulk-actions">
                <button type="button" onClick={() => runAction('cancel-waiting')}>取消全部等待</button>
                <button type="button" onClick={() => runAction('clear-completed')}>清理已完成</button>
                <button type="button" onClick={() => runAction('clear-canceled')}>清理已取消</button>
              </div>
            </header>
            <div className="queue-list">
              {visible.length ? visible.map((item) => (
                <QueueItemCard
                  key={String(item.id || item.created_at || Math.random())}
                  item={item}
                  snapshot={snapshot}
                  queuedPosition={queuedPositions.get(String(item.id || '')) ?? -1}
                  queuedTotal={queuedItems.length}
                  onAction={runItemAction}
                />
              )) : <p className="queue-empty">当前筛选下没有队列任务。</p>}
            </div>
          </section>
          <QueuePolicyForm key={`${snapshot.failure_policy}-${snapshot.auto_retry}-${snapshot.max_attempts}-${snapshot.retry_backoff_sec}`} snapshot={snapshot} onSave={(payload) => policyMutation.mutate(payload)} disabled={policyMutation.isPending} />
        </div>
        {notice ? <p className="queue-notice" role="status">{notice}</p> : null}
      </main>
    </div>
  );
}

function StatButton({ label, value, filter, target, onFilter }: {
  label: string;
  value?: number;
  filter: string;
  target: string;
  onFilter: (filter: string) => void;
}) {
  return (
    <button type="button" className="queue-stat" data-active={filter === target} onClick={() => onFilter(target)}>
      <span>{label}</span><strong>{Number(value) || 0}</strong>
    </button>
  );
}

function emptySnapshot(): QueueSnapshot {
  return {
    ok: true,
    paused: false,
    failure_policy: 'pause',
    auto_retry: false,
    max_attempts: 1,
    retry_backoff_sec: 0,
    status: 'idle',
    summary: { total: 0, queued: 0, running: 0, done: 0, error: 0, canceled: 0 },
    items: [],
  };
}
