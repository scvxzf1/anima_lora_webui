import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { parse } from "smol-toml";
import { CommandDialog } from "../../components/CommandDialog";
import { apiRequest } from "../../api/client";
import { downloadTextFile } from "../dataset-editor/downloadTextFile";
import { importedTrainingPath } from "./trainingForm";
import { saveTrainingConfigAs, type RawPatchResponse } from "./api";

export function TrainingRawEditor({
  file,
  content,
  locked,
  onSaved,
  onClose,
}: {
  file?: string;
  content: string;
  locked?: boolean;
  onSaved: (file: string) => void;
  onClose: () => void;
}) {
  const [text, setText] = useState(content);
  const [name, setName] = useState("");
  const [localError, setLocalError] = useState("");
  const save = useMutation({
    mutationFn: async () => {
      parse(text);
      const target = file && !locked ? file : importedTrainingPath(name);
      if (!target) throw new Error("请输入配置名称");
      if (file && !locked)
        await apiRequest<RawPatchResponse>("/api/config/raw", {
          method: "PUT",
          body: JSON.stringify({ file, content: text }),
        });
      else await saveTrainingConfigAs(target, text);
      return target;
    },
    retry: false,
    onSuccess: onSaved,
  });
  const dirty = text !== content || Boolean(name);
  function close() {
    if (!dirty || window.confirm("放弃未保存的 TOML 修改？")) onClose();
  }
  return (
    <CommandDialog
      title={file ? "TOML 配置" : "新建训练配置"}
      onClose={close}
      busy={save.isPending}
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          save.mutate();
        }}
      >
        <fieldset
          className="unframed-fieldset settings-grid"
          disabled={save.isPending}
        >
          {(!file || locked) && (
            <label className="full-width">
              <span>配置名称</span>
              <input
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </label>
          )}
          <label className="full-width">
            <span>TOML</span>
            <textarea
              className="raw-config-text"
              rows={16}
              spellCheck={false}
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </label>
          <label className="full-width">
            <span>导入 TOML</span>
            <input
              type="file"
              accept=".toml,text/plain"
              onChange={async (e) => {
                const source = e.target.files?.[0];
                if (!source) return;
                try {
                  const value = await source.text();
                  parse(value);
                  setText(value);
                  setName(source.name.replace(/\.toml$/i, ""));
                  setLocalError("");
                } catch (error) {
                  setLocalError(
                    error instanceof Error ? error.message : "无效 TOML",
                  );
                }
              }}
            />
          </label>
        </fieldset>
        <footer className="toolbar">
          <button
            type="button"
            disabled={save.isPending}
            onClick={() =>
              downloadTextFile(file?.split("/").pop() || "training.toml", text)
            }
          >
            导出 TOML
          </button>
          <button
            type="submit"
            className="primary-command"
            disabled={save.isPending}
          >
            {file && !locked ? "保存 TOML" : "创建配置"}
          </button>
        </footer>
      </form>
      {(save.error || localError) && (
        <p role="alert" className="form-error">
          {save.error?.message || localError}
        </p>
      )}
    </CommandDialog>
  );
}
