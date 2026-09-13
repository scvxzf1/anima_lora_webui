import { useState } from "react";
import { FolderOpen } from "lucide-react";
import { TrainingDatasetDialog } from "./TrainingDatasetDialog";
import "./TrainingDatasetField.css";

type Props = {
  value: string;
  disabled: boolean;
  own: boolean;
  onChange: (value: string) => void;
};

export function TrainingDatasetField({
  value,
  disabled,
  own,
  onChange,
}: Props) {
  const [open, setOpen] = useState(false);
  return (
    <div className="training-dataset-field" id="training-field-dataset_config">
      <div className="training-dataset-heading">
        <div>
          <strong>数据集配置</strong>
          <code>dataset_config</code>
        </div>
        <small data-source={own ? "file" : "merged"}>
          {own ? "当前文件" : "继承/预设"}
        </small>
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
