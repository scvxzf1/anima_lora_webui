import type { TrainingDraft, TrainingFieldSpec } from './trainingForm';

type Props = {
  fields: TrainingFieldSpec[];
  draft: TrainingDraft;
  ownKeys: Set<string>;
  disabled: boolean;
  onChange: (key: string, value: string | number | boolean) => void;
};

export function TrainingFieldEditor({ fields, draft, ownKeys, disabled, onChange }: Props) {
  return (
    <div className="training-edit-fields">
      {fields.map((field) => (
        <label key={field.key}>
          <span>{field.label}</span>
          <code>{field.key}</code>
          {field.kind === 'boolean' ? (
            <input
              type="checkbox"
              aria-label={field.label}
              checked={Boolean(draft[field.key])}
              disabled={disabled}
              onChange={(event) => onChange(field.key, event.target.checked)}
            />
          ) : field.kind === 'select' ? (
            <select aria-label={field.label} value={String(draft[field.key] ?? '')} disabled={disabled} onChange={(event) => onChange(field.key, event.target.value)}>
              {field.options?.map((option) => <option key={option} value={option}>{option}</option>)}
            </select>
          ) : (
            <input
              type={field.kind}
              aria-label={field.label}
              value={String(draft[field.key] ?? '')}
              min={field.min}
              step={field.step}
              disabled={disabled}
              onChange={(event) => onChange(
                field.key,
                field.kind === 'number' && event.target.value !== '' ? Number(event.target.value) : event.target.value,
              )}
            />
          )}
          <small data-source={ownKeys.has(field.key) ? 'file' : 'merged'}>
            {ownKeys.has(field.key) ? '当前文件' : '继承/预设'}
          </small>
        </label>
      ))}
    </div>
  );
}
