import type {
  TrainingConfigFile,
  TrainingConfigGroup,
} from "../../api/trainingContext";
import { useMemo, useState } from "react";
import { Plus, RefreshCw } from "lucide-react";
import { useQueryClient, useIsFetching } from "@tanstack/react-query";
import { trainingContextKeys } from "../../api/trainingContext";
import { TrainingLibraryActions } from "./TrainingLibraryActions";
import { TrainingLibraryGroup } from "./TrainingLibraryGroup";
import { TrainingLibraryDrag } from "./TrainingLibraryDrag";
import "./TrainingLibrary.css";

type Props = {
  expanded: boolean;
  files: TrainingConfigFile[];
  selectedPath?: string;
  disabled?: boolean;
  onSelect: (path: string) => void;
  onCreate: () => void;
  libraryGroups?: TrainingConfigGroup[];
  dirty?: boolean;
};

export function TrainingConfigLibrary({
  expanded,
  files,
  selectedPath,
  disabled,
  onSelect,
  onCreate,
  libraryGroups = [],
  dirty,
}: Props) {
  const [query, setQuery] = useState("");
  const queryClient = useQueryClient();
  const refreshing =
    useIsFetching({ queryKey: trainingContextKeys.files() }) > 0;
  const groups = useMemo(
    () =>
      (libraryGroups.length
        ? libraryGroups
        : groupFiles(files).map(([label, entries]) => ({
            id: label,
            label,
            files: entries,
            readonly: true,
          }))
      )
        .map((group) => ({
          ...group,
          files: group.files.filter((file) => {
            const needle = query.trim().toLowerCase();
            return (
              !needle ||
              group.label.toLowerCase().includes(needle) ||
              `${file.label || ""} ${file.filename || ""} ${file.path}`
                .toLowerCase()
                .includes(needle)
            );
          }),
        }))
        .filter(
          (group) =>
            !query.trim() ||
            group.files.length ||
            group.label.toLowerCase().includes(query.trim().toLowerCase()),
        ),
    [files, query, libraryGroups],
  );
  return (
    <aside
      id="training-config-library"
      hidden={!expanded}
      className="training-config-library"
      aria-label="训练配置库"
      data-expanded={expanded}
    >
      <header>
        <div>
          <p className="eyebrow">CONFIG LIBRARY</p>
          <h2>配置库</h2>
        </div>
        <button
          type="button"
          className="icon-button"
          aria-label="新建配置"
          title="新建配置"
          onClick={onCreate}
          disabled={disabled}
        >
          <Plus size={16} />
        </button>
        <button
          type="button"
          className="icon-button"
          title="刷新配置库"
          aria-label="刷新配置库"
          disabled={disabled || refreshing}
          onClick={() =>
            queryClient.invalidateQueries({
              queryKey: trainingContextKeys.files(),
            })
          }
        >
          <RefreshCw size={16} />
        </button>
      </header>
      <label className="training-library-search">
        <span>搜索配置</span>
        <input
          aria-label="搜索配置"
          placeholder="名称或路径"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>
      <TrainingLibraryActions
        groups={libraryGroups}
        file={files.find((file) => file.path === selectedPath)}
        disabled={Boolean(disabled || dirty)}
        onRenamed={onSelect}
      />
      <div className="training-library-groups">
        <TrainingLibraryDrag
          groups={libraryGroups}
          disabled={Boolean(disabled || dirty || query.trim())}
        >
          {(dragBusy) => (
            <>
              {groups.map((group) => (
                <TrainingLibraryGroup
                  key={group.id}
                  group={
                    libraryGroups.find((entry) => entry.id === group.id) ||
                    group
                  }
                  files={group.files}
                  selectedPath={selectedPath}
                  disabled={disabled || dragBusy}
                  dirty={dirty}
                  onSelect={onSelect}
                  groups={libraryGroups}
                  searchActive={Boolean(query.trim())}
                />
              ))}
              {!groups.length ? (
                <p className="training-library-empty">
                  {query.trim() ? "没有匹配的配置" : "没有可编辑配置"}
                </p>
              ) : null}
            </>
          )}
        </TrainingLibraryDrag>
      </div>
    </aside>
  );
}

function groupFiles(files: TrainingConfigFile[]) {
  const grouped = new Map<string, TrainingConfigFile[]>();
  files.forEach((file) => {
    const group = file.methods_subdir || "配置";
    grouped.set(group, [...(grouped.get(group) || []), file]);
  });
  return [...grouped.entries()];
}
