import { useRef, useState } from "react";
import { CommandDialog } from "../../components/CommandDialog";

export type LibraryNameAction = {
  title: string;
  initialName: string;
  filename?: boolean;
  submit: (name: string) => Promise<{ file?: string }>;
};

export function TrainingLibraryNameDialog({
  action,
  busy,
  error,
  onClose,
  onSubmit,
}: {
  action: LibraryNameAction;
  busy: boolean;
  error?: string;
  onClose: () => void;
  onSubmit: (name: string) => void;
}) {
  const [name, setName] = useState(action.initialName);
  const ref = useRef<HTMLInputElement>(null);
  const value = name.trim();
  const invalid =
    action.filename &&
    (/[\\/]/.test(value) ||
      !value.replace(/\.toml$/i, "") ||
      value === "." ||
      value === "..");
  return (
    <CommandDialog
      title={action.title}
      busy={busy}
      onClose={onClose}
      initialFocusRef={ref}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (value && !invalid && !busy) onSubmit(value);
        }}
      >
        <label>
          <span>{action.filename ? "文件名称" : "分组名称"}</span>
          <input
            ref={ref}
            value={name}
            onChange={(event) => setName(event.target.value)}
            disabled={busy}
            required
            maxLength={action.filename ? 200 : 80}
          />
        </label>
        {invalid && (
          <p className="form-error" role="alert">
            请输入文件名称，不要包含目录路径。
          </p>
        )}
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <footer className="toolbar">
          <button type="button" onClick={onClose} disabled={busy}>
            取消
          </button>
          <button
            type="submit"
            className="primary-command"
            disabled={
              busy || !value || Boolean(invalid) || value === action.initialName
            }
          >
            {busy ? "处理中" : "确认"}
          </button>
        </footer>
      </form>
    </CommandDialog>
  );
}
