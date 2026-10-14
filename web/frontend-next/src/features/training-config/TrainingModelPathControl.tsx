import { FolderOpen } from "lucide-react";
import { useState } from "react";
import { TrainingModelPathDialog } from "./TrainingModelPathDialog";

export function isTrainingModelPath(key: string) {
  return ["pretrained_model_name_or_path", "qwen3", "vae"].includes(key);
}

export function TrainingModelPathControl({
  fieldKey,
  label,
  value,
  disabled,
  onChange,
}: {
  fieldKey: string;
  label: string;
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <span className="training-model-path-control">
        <input
          id={`training-control-${fieldKey}`}
          type="text"
          aria-label={label}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
        />
        <button
          type="button"
          className="training-model-browse"
          aria-label={`浏览${label}服务器路径`}
          title="浏览服务器模型文件或目录"
          disabled={disabled}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            setOpen(true);
          }}
        >
          <FolderOpen size={17} aria-hidden="true" />
        </button>
      </span>
      {open && (
        <TrainingModelPathDialog
          label={label}
          disabled={disabled}
          onClose={() => setOpen(false)}
          onSelect={(path) => {
            if (!disabled) onChange(path);
            setOpen(false);
          }}
        />
      )}
    </>
  );
}
