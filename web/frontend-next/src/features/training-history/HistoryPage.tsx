import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Topbar } from '../../app/Topbar';
import { batchUpdateHistoryTasks, fetchHistoryTasks, historyKeys, type HistoryTaskSummary } from './api';
import './HistoryPage.css';

const STATE_LABELS: Record<string, string> = {
  idle: '完成',
  running: '运行中',
  error: '异常',
  interrupted: '已中断',
};

function historyStateLabel(state?: string) {
  return STATE_LABELS[state || ''] || state || '未知';
}

function taskName(task: HistoryTaskSummary) {
  return String(task.name || task.history_run_label || task.id || '未命名任务');
}

function taskGroup(task: HistoryTaskSummary) {
  return String(task.group || task.history_group_key || '未分组');
}

function filterTasks(tasks: HistoryTaskSummary[], search: string, state: string, archived: string) {
  const needle = search.trim().toLowerCase();
  return tasks.filter((task) => {
    if (state !== 'all' && task.state !== state) return false;
    if (archived === 'active' && task.archived) return false;
    if (archived === 'archived' && !task.archived) return false;
    if (!needle) return true;
    const haystack = [task.name, task.history_run_label, task.history_source_config_file, task.run_dir, task.output_dir, task.id]
      .filter(Boolean).join(' ').toLowerCase();
    return haystack.includes(needle);
  });
}

export function HistoryPage() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState('');
  const [state, setState] = useState('all');
  const [archived, setArchived] = useState('active');
  const [selected, setSelected] = useState<string[]>([]);
  const [notice, setNotice] = useState('');

  const query = useQuery({
    queryKey: historyKeys.list,
    queryFn: ({ signal }) => fetchHistoryTasks(200, signal),
  });
  const tasks = query.data?.tasks || [];
  const visible = useMemo(() => filterTasks(tasks, search, state, archived), [tasks, search, state, archived]);

  const batch = useMutation({
    mutationFn: batchUpdateHistoryTasks,
    onSuccess: async (result) => {
      setSelected([]);
      setNotice(result.message || '历史任务已更新。');
      await queryClient.invalidateQueries({ queryKey: historyKeys.list });
    },
  });

  function toggleSelected(taskId: string) {
    setSelected((current) => current.includes(taskId) ? current.filter((id) => id !== taskId) : [...current, taskId]);
  }

  function runBatch(action: 'archive' | 'unarchive' | 'delete') {
    if (!selected.length) return;
    const message = action === 'delete'
      ? `确定彻底删除已选 ${selected.length} 条历史记录吗？该操作不会删除运行目录和权重。`
      : `确定${action === 'archive' ? '归档' : '取消归档'}已选 ${selected.length} 条历史记录吗？`;
    if (!window.confirm(message)) return;
    batch.mutate({ action, task_ids: selected });
  }

  const counts = useMemo(() => ({
    total: tasks.length,
    training: tasks.filter((task) => task.job === 'training').length,
    preprocess: tasks.filter((task) => task.job === 'preprocess').length,
    error: tasks.filter((task) => task.state === 'error').length,
    archived: tasks.filter((task) => task.archived).length,
  }), [tasks]);

  return (
    <div className="history-shell">
      <Topbar />
      <main className="history-page">
        <header className="history-header">
          <div>
            <p className="eyebrow">HISTORY FORGE</p>
            <h1>历史任务</h1>
            <p>检索、筛选和回顾已完成的训练与预处理任务。</p>
          </div>
          <button type="button" onClick={() => query.refetch()}>{query.isFetching ? '刷新中' : '刷新'}</button>
        </header>

        <section className="history-stats" aria-label="历史任务统计">
          <span><strong>{counts.total}</strong>总量</span>
          <span><strong>{counts.training}</strong>训练</span>
          <span><strong>{counts.preprocess}</strong>预处理</span>
          <span><strong>{counts.error}</strong>异常</span>
          <span><strong>{counts.archived}</strong>归档</span>
        </section>

        <div className="history-toolbar">
          <label className="history-search">
            <span>全局搜索</span>
            <input type="search" placeholder="任务 / 配置 / 目录" value={search} onChange={(event) => setSearch(event.target.value)} />
          </label>
          <label>
            <span>状态</span>
            <select value={state} onChange={(event) => setState(event.target.value)}>
              <option value="all">全部</option>
              <option value="idle">完成</option>
              <option value="running">运行中</option>
              <option value="error">异常</option>
              <option value="interrupted">已中断</option>
            </select>
          </label>
          <label>
            <span>归档</span>
            <select value={archived} onChange={(event) => setArchived(event.target.value)}>
              <option value="active">未归档</option>
              <option value="all">全部</option>
              <option value="archived">已归档</option>
            </select>
          </label>
          {selected.length ? (
            <div className="history-bulk-bar">
              <strong>已选 {selected.length} 项</strong>
              <button type="button" onClick={() => runBatch('archive')}>归档已选</button>
              <button type="button" onClick={() => runBatch('unarchive')}>取消归档</button>
              <button type="button" className="history-danger" onClick={() => runBatch('delete')}>彻底删除</button>
            </div>
          ) : null}
        </div>

        {batch.error ? <p className="history-error" role="alert">{batch.error.message}</p> : null}
        {notice ? <p className="history-notice" role="status">{notice}</p> : null}

        <section className="history-list" aria-label="历史任务列表">
          {visible.length ? visible.map((task) => (
            <article key={task.id} className="history-card" data-state={task.state || 'unknown'}>
              <input
                type="checkbox"
                aria-label={`选择 ${taskName(task)}`}
                checked={selected.includes(String(task.id))}
                onChange={() => toggleSelected(String(task.id))}
              />
              <Link to={`/history/${encodeURIComponent(String(task.id))}`} className="history-card-link">
                <span className="history-state" data-state={task.state || 'unknown'}>{historyStateLabel(task.state)}</span>
                <h3>{taskName(task)}</h3>
                <p>{task.history_source_config_file || task.run_dir || task.output_dir || '—'}</p>
              </Link>
              <div className="history-card-meta">
                <span>{task.job === 'preprocess' ? '预处理' : '训练'}</span>
                <span>{taskGroup(task)}</span>
                <span>{task.started_at_text || ''}</span>
                <span>{task.log_count ?? 0} 日志 · {task.metric_count ?? 0} 指标</span>
              </div>
            </article>
          )) : <p className="history-empty">{query.isFetching ? '正在读取历史任务…' : '没有匹配的历史记录。'}</p>}
        </section>
      </main>
    </div>
  );
}
