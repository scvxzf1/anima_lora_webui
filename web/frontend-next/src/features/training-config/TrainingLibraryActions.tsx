import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, Download, FolderPlus, ListPlus, Pencil, Trash2 } from "lucide-react";
import { apiRequest } from "../../api/client";
import { useState } from "react";
import { createPortal } from "react-dom";
import {
  TrainingLibraryNameDialog,
  type LibraryNameAction,
} from "./TrainingLibraryNameDialog";
import {
  trainingContextKeys,
  type TrainingConfigFile,
  type TrainingConfigGroup,
} from "../../api/trainingContext";
import { downloadTrainingGroup } from "./downloadTrainingGroup";
import { TrainingGroupQueueDialog } from "./TrainingGroupQueueDialog";
import { queueableTrainingFiles } from "./trainingGroupQueue";

const request = (path: string, body?: unknown, method = "POST") =>
  apiRequest<{ file?: string }>(path, {
    method,
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });

export function TrainingLibraryActions({
  groups,
  file,
  disabled,
  onRenamed,
  scope = "library",
  targetGroup,
  detailedManagement,
  onDetailedManagementChange,
  preset = "",
  gpuIds = [],
  deviceSummary = "",
  deviceIssue = "",
}: {
  groups: TrainingConfigGroup[];
  file?: TrainingConfigFile;
  disabled: boolean;
  onRenamed: (file: string) => void;
  scope?: "library" | "group" | "file";
  targetGroup?: TrainingConfigGroup;
  detailedManagement?: boolean;
  onDetailedManagementChange?: (enabled: boolean) => void;
  preset?: string;
  gpuIds?: string[];
  deviceSummary?: string;
  deviceIssue?: string;
}) {
  const qc = useQueryClient();
  const [nameAction, setNameAction] = useState<LibraryNameAction | null>(null);
  const [queueGroup, setQueueGroup] = useState<TrainingConfigGroup | null>(null);
  const mutation = useMutation({
    mutationFn: (operation: () => Promise<{ file?: string }>) => operation(),
    retry: false,
    onSuccess: async (data) => {
      await qc.invalidateQueries({ queryKey: trainingContextKeys.files() });
      if (data.file) onRenamed(data.file);
      setNameAction(null);
    },
  });
  const exportGroup = useMutation({ mutationFn: downloadTrainingGroup, retry: false });
  const busy = disabled || mutation.isPending || exportGroup.isPending || Boolean(nameAction) || Boolean(queueGroup);
  function openNameAction(action: LibraryNameAction) {
    mutation.reset();
    setNameAction(action);
  }
  const group =
    targetGroup ||
    groups.find((entry) =>
      entry.files.some((entry) => entry.path === file?.path),
    );
  const groupLocked = group?.locked || group?.readonly || group?.system;
  function rename() {
    if (!file || busy) return;
    const source = file.path;
    openNameAction({
      title: "重命名文件",
      filename: true,
      initialName: file.filename || source.split("/").pop() || "",
      submit: (name) =>
        request("/api/config/raw/rename", {
          source,
          target:
            source.slice(0, source.lastIndexOf("/") + 1) +
            name.replace(/\.toml$/i, "") +
            ".toml",
        }),
    });
  }
  return (
    <div className="training-library-actions" data-scope={scope}>
      <fieldset disabled={busy} className="unframed-fieldset">
        {scope === "library" && (
          <div className="toolbar">
            <button
              type="button"
              onClick={() => {
                openNameAction({
                  title: "新建分组",
                  initialName: "",
                  submit: (label) =>
                    request("/api/config/file-groups", {
                      label,
                      kind: "training",
                    }),
                });
              }}
            >
              <FolderPlus size={16} />
              新建分组
            </button>
          </div>
        )}
        {scope === "file" && (
          <div className="training-library-file-commands">
            <button
              type="button"
              disabled={!file || file.locked || file.readonly}
              onClick={rename}
              title="重命名文件"
              aria-label={`重命名 ${file?.filename || file?.path}`}
            >
              <Pencil size={16} />
            </button>
            <label>
              <select
                aria-label={`移动 ${file?.filename || file?.path} 到分组`}
                title="移动到分组"
                value={group?.id || ""}
                disabled={!file || file.locked || file.readonly}
                onChange={(e) => {
                  const target = e.currentTarget.value;
                  mutation.mutate(() =>
                    request("/api/config/file-groups/move-file", {
                      file: file!.path,
                      group: target,
                    }),
                  );
                }}
              >
                <option value="" disabled>
                  选择分组
                </option>
                {groups.map((entry) => (
                  <option
                    key={entry.id}
                    value={entry.id}
                    disabled={entry.locked || entry.readonly}
                  >
                    {entry.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
        )}
        {scope === "group" && group && (
          <div className="toolbar">
            <button
              type="button"
              disabled={!queueableTrainingFiles(group).length}
              title="整组加入队列"
              aria-label={`整组加入队列 ${group.label}`}
              onClick={() => setQueueGroup(group)}
            >
              <ListPlus size={16} />
            </button>
            <button
              type="button"
              disabled={!group.files.some((entry) => entry.path.toLowerCase().endsWith(".toml"))}
              title="导出分组"
              aria-label={`导出分组 ${group.label}`}
              onClick={() => exportGroup.mutate(group.id)}
            >
              <Download size={16} />
            </button>
            {detailedManagement && !groupLocked && <button
              type="button"
              disabled={groupLocked}
              title="重命名当前分组"
              aria-label="重命名当前分组"
              onClick={() => {
                openNameAction({
                  title: "重命名分组",
                  initialName: group.label,
                  submit: (label) =>
                    request(
                      `/api/config/file-groups/${encodeURIComponent(group.id)}`,
                      { label },
                      "PATCH",
                    ),
                });
              }}
            >
              <Pencil size={16} />
            </button>}
            {detailedManagement && !groupLocked && <button
              type="button"
              disabled={groupLocked}
              title="删除当前分组"
              aria-label="删除当前分组"
              onClick={() => {
                if (window.confirm(`删除分组 ${group.label}？配置文件保留。`))
                  mutation.mutate(() =>
                    request(
                      `/api/config/file-groups/${encodeURIComponent(group.id)}`,
                      undefined,
                      "DELETE",
                    ),
                  );
              }}
            >
              <Trash2 size={16} />
            </button>}
            {detailedManagement && !groupLocked && (["up", "down"] as const).map((direction) => (
              <button
                key={direction}
                type="button"
                disabled={groupLocked}
                aria-label={direction === "up" ? "上移分组" : "下移分组"}
                title={direction === "up" ? "上移分组" : "下移分组"}
                onClick={() =>
                  mutation.mutate(() =>
                    request("/api/config/file-groups/reorder-group", {
                      group: group.id,
                      direction,
                    }),
                  )
                }
              >
                {direction === "up" ? (
                  <ArrowUp size={16} />
                ) : (
                  <ArrowDown size={16} />
                )}
              </button>
            ))}
          </div>
        )}
        {scope === "file" && group && file && (
          <div className="toolbar">
            {(["up", "down"] as const).map((direction) => (
              <button
                key={direction}
                type="button"
                disabled={
                  file.locked ||
                  file.readonly ||
                  groupLocked ||
                  group.files.findIndex((item) => item.path === file.path) ===
                    (direction === "up" ? 0 : group.files.length - 1)
                }
                title={direction === "up" ? "上移配置" : "下移配置"}
                aria-label={`${direction === "up" ? "上移配置" : "下移配置"} ${file.filename || file.path}`}
                onClick={() =>
                  mutation.mutate(() =>
                    request("/api/config/file-groups/reorder-file", {
                      file: file.path,
                      group: group.id,
                      direction,
                    }),
                  )
                }
              >
                {direction === "up" ? (
                  <ArrowUp size={16} />
                ) : (
                  <ArrowDown size={16} />
                )}
              </button>
            ))}
          </div>
        )}
      </fieldset>
      {scope === "library" && onDetailedManagementChange && (
        <label className="training-library-management-toggle">
          <input
            type="checkbox"
            role="switch"
            checked={Boolean(detailedManagement)}
            onChange={(event) => onDetailedManagementChange(event.target.checked)}
          />
          <span className="training-library-management-track" aria-hidden="true" />
          <span>详细管理</span>
        </label>
      )}
      {mutation.error && !nameAction && (
        <p role="alert" className="form-error">
          {mutation.error.message}
        </p>
      )}
      {exportGroup.error && <p role="alert" className="form-error">{exportGroup.error.message}</p>}
      {nameAction &&
        createPortal(
          <TrainingLibraryNameDialog
            action={nameAction}
            busy={mutation.isPending}
            error={mutation.error?.message}
            onClose={() => {
              if (!mutation.isPending) setNameAction(null);
            }}
            onSubmit={(name) => {
              if (!mutation.isPending)
                mutation.mutate(() => nameAction.submit(name));
            }}
          />,
          document.body,
        )}
      {queueGroup && createPortal(
        <TrainingGroupQueueDialog
          group={queueGroup}
          preset={preset}
          gpuIds={gpuIds}
          deviceSummary={deviceSummary}
          deviceIssue={deviceIssue}
          onClose={() => setQueueGroup(null)}
        />,
        document.body,
      )}
    </div>
  );
}
