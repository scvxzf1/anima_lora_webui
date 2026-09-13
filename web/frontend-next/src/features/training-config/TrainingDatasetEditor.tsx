import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Save } from "lucide-react";
import { datasetKeys, saveDatasetPreset } from "../dataset-editor/api";
import type {
  DatasetPresetResponse,
  DatasetRow,
} from "../dataset-editor/types";
import { TrainingDatasetSubset } from "./TrainingDatasetSubset";

export type DatasetEditorStatus = { dirty: boolean; busy: boolean };
type RowDraft = { id: number; origin: number | null; row: DatasetRow };
type Props = {
  preset: DatasetPresetResponse;
  onStatus: (status: DatasetEditorStatus) => void;
  onChoose: (file: string) => void;
};
const draftRows = (rows: DatasetRow[]): RowDraft[] =>
  rows.map((row, index) => ({
    id: index,
    origin: index,
    row: structuredClone(row),
  }));

export function TrainingDatasetEditor({ preset, onStatus, onChoose }: Props) {
  const client = useQueryClient();
  const [baseline, setBaseline] = useState(preset);
  const [rows, setRows] = useState(() => draftRows(preset.datasets));
  const nextId = useRef(preset.datasets.length);
  const saving = useRef(false);
  const [notice, setNotice] = useState("");
  const dirty =
    JSON.stringify(rows.map(({ row }) => row)) !==
    JSON.stringify(baseline.datasets);
  const valid =
    rows.length > 0 &&
    rows.every(
      ({ row }) =>
        Number.isInteger(Number(row.num_repeats ?? 1)) &&
        Number(row.num_repeats ?? 1) >= 1 &&
        Boolean(String(row.source_dir || row.image_dir || "").trim()),
    );
  const save = useMutation({
    mutationFn: () =>
      saveDatasetPreset(
        baseline.file,
        {
          datasets: rows.map(({ row }) => row),
          defaults: baseline.defaults,
          stage_schedule_enabled: baseline.stage_schedule_enabled,
          stage_schedule: baseline.stage_schedule,
        },
        true,
      ),
    onSuccess: async (result) => {
      const updated = { ...baseline, ...result };
      setBaseline(updated);
      setRows(draftRows(updated.datasets));
      nextId.current = updated.datasets.length;
      setNotice("蓝图参数已保存");
      client.setQueryData(datasetKeys.preset(result.file), updated);
      await client.invalidateQueries({ queryKey: datasetKeys.library() });
      await client.invalidateQueries({
        queryKey: [...datasetKeys.all, "picker-preview", result.file],
      });
      await client.invalidateQueries({
        queryKey: [...datasetKeys.all, "preview", result.file],
      });
      await client.invalidateQueries({
        queryKey: [...datasetKeys.all, "cover", result.file],
      });
    },
    onSettled: () => {
      saving.current = false;
    },
  });
  useEffect(() => {
    onStatus({ dirty, busy: save.isPending });
  }, [dirty, save.isPending, onStatus]);
  function update(id: number, patch: Partial<DatasetRow>) {
    setNotice("");
    setRows((current) =>
      current.map((item) =>
        item.id === id ? { ...item, row: { ...item.row, ...patch } } : item,
      ),
    );
  }
  function saveChanges() {
    if (saving.current || !valid || baseline.readonly) return;
    if (
      !window.confirm(
        "保存将更新此共享数据集蓝图，引用它的其他训练配置也会使用新参数。确认保存？",
      )
    )
      return;
    saving.current = true;
    save.mutate();
  }
  return (
    <div className="training-dataset-editor">
      <header>
        <div>
          <h3>{baseline.file.split("/").pop()}</h3>
          <code>{baseline.file}</code>
        </div>
        <span>{baseline.readonly ? "只读蓝图" : `${rows.length} 个子集`}</span>
      </header>
      <div className="training-dataset-editor-actions">
        <span>
          分辨率 {String(baseline.defaults.resolution ?? "—")} · 批量{" "}
          {String(baseline.defaults.batch_size ?? "—")}
        </span>
        <button
          type="button"
          disabled={
            baseline.readonly ||
            save.isPending ||
            Boolean(baseline.stage_schedule_enabled)
          }
          title={
            baseline.stage_schedule_enabled
              ? "阶段调度蓝图请在完整管理中调整子集结构"
              : "添加子集"
          }
          onClick={() =>
            setRows((current) => [
              ...current,
              {
                id: nextId.current++,
                origin: null,
                row: {
                  source_dir: "",
                  image_dir: "",
                  num_repeats: 1,
                  is_reg: false,
                },
              },
            ])
          }
        >
          <Plus size={15} />
          添加子集
        </button>
      </div>
      <div className="training-dataset-subsets">
        {rows.map((item, index) => (
          <TrainingDatasetSubset
            key={item.id}
            file={baseline.file}
            index={index}
            origin={item.origin}
            row={item.row}
            previewCurrent={
              item.origin !== null &&
              item.row.source_dir ===
                baseline.datasets[item.origin]?.source_dir &&
              item.row.image_dir === baseline.datasets[item.origin]?.image_dir
            }
            disabled={baseline.readonly || save.isPending}
            removeDisabled={
              rows.length <= 1 || Boolean(baseline.stage_schedule_enabled)
            }
            onChange={(patch) => update(item.id, patch)}
            onRemove={() => {
              if (
                window.confirm(
                  `移除子集 ${index + 1}？图片和缓存文件不会删除。`,
                )
              )
                setRows((current) =>
                  current.filter((row) => row.id !== item.id),
                );
            }}
          />
        ))}
      </div>
      {!valid && (
        <p role="alert">每个子集需要图片目录，重复次数必须为正整数。</p>
      )}
      {save.isError && <p role="alert">{save.error.message}</p>}
      {notice && <p role="status">{notice}</p>}
      <div className="training-dataset-editor-footer">
        <button
          type="button"
          disabled={!dirty || !valid || baseline.readonly || save.isPending}
          onClick={saveChanges}
        >
          <Save size={16} />
          保存蓝图参数
        </button>
        <button
          type="button"
          className="primary-command"
          disabled={dirty || !valid || save.isPending}
          onClick={() => onChoose(baseline.file)}
        >
          使用此数据集
        </button>
      </div>
    </div>
  );
}
