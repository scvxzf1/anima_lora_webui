import type { TrainingPreflightResponse } from './api';

type Props = {
  result?: TrainingPreflightResponse;
  pending: boolean;
  error?: string;
};

export function TrainingPreflightPanel({ result, pending, error }: Props) {
  return (
    <section className="training-preflight-panel" aria-live="polite">
      <header>
        <div>
          <p className="eyebrow">PREFLIGHT</p>
          <h2>训练预检测</h2>
        </div>
        <span className="training-preflight-state" data-tone={result?.ok ? 'success' : result ? 'danger' : 'neutral'}>
          {pending ? '检测中' : result?.ok ? '可以继续' : result ? '需要处理' : '尚未检测'}
        </span>
      </header>
      {error ? <p className="training-command-error" role="alert">{error}</p> : null}
      {result ? (
        <>
          <div className="training-preflight-summary">
            <span><strong>{result.summary.errors}</strong>错误</span>
            <span><strong>{result.summary.warnings}</strong>警告</span>
            <span><strong>{result.summary.checks}</strong>检查</span>
          </div>
          <ul className="training-preflight-checks">
            {result.checks.map((check, index) => (
              <li key={`${check.key}-${index}`} data-level={check.level}>
                <span>{check.level === 'ok' ? '通过' : check.level === 'warning' ? '警告' : '错误'}</span>
                <div><strong>{check.message}</strong>{check.path ? <code>{check.path}</code> : null}</div>
              </li>
            ))}
          </ul>
        </>
      ) : <p>保存配置后运行预检测；草稿不会被静默保存或用于检测。</p>}
    </section>
  );
}
