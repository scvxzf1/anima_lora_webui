import { useState } from "react";
import { FolderOpen, Undo2 } from "lucide-react";
import { TrainingDatasetDialog } from "./TrainingDatasetDialog";
import "./TrainingDatasetField.css";

type Props = {
  value: string;
  disabled: boolean;
  own: boolean;
  dirty: boolean;
  onChange: (value: string) => void;
  onUndo: () => void;
};

export function TrainingDatasetField({
  value,
  disabled,
  own,
  dirty,
  onChange,
  onUndo,
}: Props) {
  const [open, setOpen] = useState(false);
  return (
    <div className="training-dataset-field" id="training-field-dataset_config" tabIndex={-1}>
      <div className="training-dataset-heading">
        <div>
          <strong>数据集配置</strong>
          <code>dataset_config</code>
        </div>
        <small data-source={own ? "file" : "merged"}>
          {own ? "当前文件" : "继承/预设"}
        </small>
        {dirty && <small className="training-field-dirty" role="status">已修改</small>}
        {dirty && <button type="button" className="training-field-undo" aria-label="撤销数据集配置修改" title="撤销数据集配置修改" disabled={disabled} onClick={onUndo}><Undo2 size={15} aria-hidden="true" /></button>}
      </div>
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
