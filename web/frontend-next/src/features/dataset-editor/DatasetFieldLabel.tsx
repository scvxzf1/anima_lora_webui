import { DatasetFieldHelp } from './DatasetFieldHelp';
import type { DatasetFieldHelpKey } from './field-help';

export function DatasetFieldLabel({ htmlFor, label, field }: {
  htmlFor: string;
  label: string;
  field?: DatasetFieldHelpKey;
}) {
  return (
    <span className="dataset-field-label">
      <label htmlFor={htmlFor}>{label}</label>
      {field ? <DatasetFieldHelp field={field} label={label} /> : null}
    </span>
  );
}
