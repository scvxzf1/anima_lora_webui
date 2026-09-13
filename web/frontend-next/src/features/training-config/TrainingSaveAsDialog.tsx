import { useState, useRef } from "react";
import { CommandDialog } from "../../components/CommandDialog";

type Props = {
  initialName: string;
  busy: boolean;
  error?: string;
  onCancel: () => void;
  onConfirm: (name: string) => void;
};

export function TrainingSaveAsDialog({
  initialName,
  busy,
  error,
  onCancel,
  onConfirm,
}: Props) {
  const [name, setName] = useState(initialName);
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <CommandDialog
      title="另存训练配置"
      onClose={onCancel}
      busy={busy}
      initialFocusRef={inputRef}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          const next = name.trim();
          if (next && !busy) onConfirm(next);
        }}
      >
        <label>
          <span>配置名称</span>
          <input
            ref={inputRef}
            value={name}
            onChange={(event) => setName(event.target.value)}
            required
            disabled={busy}
          />
        </label>
        {error ? (
          <p className="training-command-error" role="alert">
            {error}
          </p>
        ) : null}
        <footer className="toolbar">
          <button type="button" onClick={onCancel} disabled={busy}>
            取消
          </button>
          <button type="submit" className="primary-command" disabled={busy}>
            {busy ? "另存中" : "确认另存"}
          </button>
        </footer>
      </form>
    </CommandDialog>
  );
}
