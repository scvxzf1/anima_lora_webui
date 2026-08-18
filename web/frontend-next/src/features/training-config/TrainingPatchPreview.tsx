import type { RawPatchResponse } from './api';

type Props = {
  preview?: RawPatchResponse;
  pending: boolean;
  error?: string;
};

export function TrainingPatchPreview({ preview, pending, error }: Props) {
  return (
    <section className="training-patch-preview" aria-live="polite">
      <header>
        <h2>变更预览</h2>
        <span>{pending ? '计算中' : preview ? `${preview.changed.length} 项` : '尚未预览'}</span>
      </header>
      {error ? <p className="training-command-error" role="alert">{error}</p> : null}
      {preview ? (
        <>
          {preview.changed.length ? (
            <ul>{preview.changed.map((key) => <li key={key}><code>{key}</code></li>)}</ul>
          ) : <p>当前草稿不会改变配置文件。</p>}
          {preview.warnings?.length ? (
            <div className="training-warning-list">
              <strong>Schema 警告</strong>
              {preview.warnings.map((warning) => <p key={warning}>{warning}</p>)}
            </div>
          ) : null}
        </>
      ) : <p>编辑字段后点击“预览变更”，后端会验证 TOML 和 schema，但不会写盘。</p>}
    </section>
  );
}
