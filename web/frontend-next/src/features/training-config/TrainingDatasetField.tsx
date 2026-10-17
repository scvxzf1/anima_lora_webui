import { useState } from "react";
import { FolderOpen } from "lucide-react";
import { TrainingDatasetDialog } from "./TrainingDatasetDialog";
import "./TrainingDatasetField.css";
import { TrainingFieldHeader } from "./TrainingFieldHeader";

type Props = {
  value: string;
  disabled: boolean;
  own: boolean;
  dirty: boolean;
  overridden?: boolean;
  onReset?: () => void;
  onHelp: () => void;
  onChange: (value: string) => void;
  onUndo: () => void;
};

export function TrainingDatasetField({
  value,
  disabled,
  own,
  dirty,
  onHelp,
  onChange,
  onUndo,
  overridden,
  onReset,
}: Props) {
  const [open, setOpen] = useState(false);
  return (
    <div className="training-dataset-field" data-own={own || undefined} data-dirty={dirty || undefined} data-overridden={overridden || undefined} id="training-field-dataset_config" tabIndex={-1}>
      <TrainingFieldHeader label="数据集配置" fieldKey="dataset_config" own={own} dirty={dirty} overridden={overridden} disabled={disabled} onHelp={onHelp} onUndo={onUndo} onReset={onReset} />
      <div className="training-dataset-reference">
        <input
          aria-label="数据集配置"
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
        />
        <button type="button" disabled={disabled} onClick={() => setOpen(true)}>
          <FolderOpen size={16} />
          选择与配置数据集
        </button>
      </div>
      {open && (
        <TrainingDatasetDialog
          value={value}
          onClose={() => setOpen(false)}
          onChoose={(file) => {
            onChange(file);
            setOpen(false);
          }}
        />
      )}
    </div>
  );
}
