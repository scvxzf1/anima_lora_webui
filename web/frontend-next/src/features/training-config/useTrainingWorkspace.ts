import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { trainingContextKeys } from "../../api/trainingContext";
import { useTrainingContext } from "../../app/useTrainingContext";
import { datasetKeys } from "../dataset-editor/api";
import { useTrainingDevices } from "./useTrainingDevices";
import { useUnsavedChangesGuard } from "../dataset-editor/useUnsavedChangesGuard";
import { ApiError } from "../../api/client";
import {
  fetchRawTrainingConfig,
  previewTrainingConfigPatch,
  runTrainingPreflight,
  saveTrainingConfigAs,
  saveTrainingConfigPatch,
  trainingConfigKeys,
} from "./api";
import {
  fieldsForConfig,
  filterTrainingFields,
  fetchFieldCapabilities,
} from "./fieldCatalog";
import {
  draftFromMerged,
  importedTrainingPath,
  rawConfigOwnKeys,
  trainingPatchValues,
  sameTrainingValue,
  type TrainingDraft,
} from "./trainingForm";

const UNSAVED_MESSAGE =
  "当前训练配置有未保存修改，离开会丢失这些修改。是否继续？";
export const GROUPS = [
  ["input", "输入准备"],
  ["method", "方法配置"],
  ["training", "训练计划"],
  ["resources", "设备与性能"],
] as const;

export function useTrainingWorkspace() {
  const queryClient = useQueryClient();
  const context = useTrainingContext();
  const deviceState = useTrainingDevices();
  const capabilities = useQuery({
    queryKey: ["training-config", "capabilities"],
    queryFn: ({ signal }) => fetchFieldCapabilities(signal),
    staleTime: Infinity,
  });
  const selectedFile = context.selectedFile;
  const rawQuery = useQuery({
    queryKey: trainingConfigKeys.raw(selectedFile?.path || ""),
    queryFn: ({ signal }) => fetchRawTrainingConfig(selectedFile!.path, signal),
    enabled: Boolean(selectedFile),
  });
  const [baseline, setBaseline] = useState<TrainingDraft>({});
  const [draft, setDraft] = useState<TrainingDraft>({});
  const pipelineUnavailable = draft.pipeline_parallel === true || draft.pipeline_parallel === "true";
  const [hydratedKey, setHydratedKey] = useState("");
  const [saveAsOpen, setSaveAsOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const [activeStage, setActiveStage] = useState("input");
  const [launchMode, setLaunchMode] = useState<"start" | "queue" | null>(null);
  const [rawMode, setRawMode] = useState<"edit" | "new" | null>(null);
  const [fieldSearch, setFieldSearch] = useState("");
  const [fieldView, setFieldView] = useState("applicable");
  const [promptsOpen, setPromptsOpen] = useState(false);
  const mergedConfig = context.mergedConfig || {};
  const fields = useMemo(
    () => fieldsForConfig(mergedConfig),
    [context.mergedConfig],
  );
  const hydrationKey = `${selectedFile?.path || ""}\0${context.selectedPreset}`;
  const patch = useMemo(() => {
    try {
      return {
        values: trainingPatchValues(draft, baseline, fields, mergedConfig),
        error: "",
      };
    } catch (error) {
      return {
        values: {},
        error: error instanceof Error ? error.message : "结构化字段格式错误",
      };
    }
  }, [draft, baseline, fields, mergedConfig]);
  const changes = patch.values;
  const dirty = fields.some(
    (field) =>
      !sameTrainingValue(draft[field.key], baseline[field.key], field.kind),
  );
  const ownKeys = useMemo(
    () => rawConfigOwnKeys(rawQuery.data?.content || ""),
    [rawQuery.data?.content],
  );
  useUnsavedChangesGuard(
    dirty || Boolean(rawMode) || promptsOpen || saveAsOpen,
    UNSAVED_MESSAGE,
  );

  useEffect(() => {
    if (
      !selectedFile ||
      !rawQuery.data ||
      context.isPending ||
      hydratedKey === hydrationKey
    )
      return;
    const next = draftFromMerged(mergedConfig, fields);
    setBaseline(next);
    setDraft(next);
    setHydratedKey(hydrationKey);
    setNotice("");
  }, [
    context.isPending,
    hydratedKey,
    hydrationKey,
    mergedConfig,
    rawQuery.data,
    selectedFile,
  ]);

  const preview = useMutation({
    mutationFn: () => previewTrainingConfigPatch(selectedFile!.path, changes),
  });
  const save = useMutation({
    mutationFn: (submitted: TrainingDraft) =>
      saveTrainingConfigPatch(
        selectedFile!.path,
        trainingPatchValues(submitted, baseline, fields, mergedConfig),
        rawQuery.data?.revision,
      ),
    onSuccess: async (result, submitted) => {
      setBaseline({ ...submitted });
      setNotice(result.message || "训练配置已保存");
      preview.reset();
      preflight.reset();
      queryClient.setQueryData(trainingConfigKeys.raw(selectedFile!.path), {
        file: selectedFile!.path,
        content: result.content,
        revision: result.revision,
        meta: rawQuery.data!.meta,
      });
      await invalidateTrainingQueries(
        queryClient,
        selectedFile!.path,
        context.selectedPreset,
      );
      const normalized = queryClient.getQueryData<Record<string, unknown>>(
        trainingContextKeys.merged(selectedFile!.path, context.selectedPreset),
      );
      if (normalized) {
        const next = draftFromMerged(normalized, fieldsForConfig(normalized));
        setBaseline(next);
        setDraft((current) =>
          JSON.stringify(current) === JSON.stringify(submitted)
            ? next
            : current,
        );
      }
    },
    retry: false,
  });
  const saveConflict = save.error instanceof ApiError && save.error.status === 409;
  async function reloadConflictedConfig() {
    if (!window.confirm("重新加载会放弃当前未保存的配置修改。是否继续？")) return;
    await Promise.all([
      queryClient.refetchQueries({ queryKey: trainingConfigKeys.raw(selectedFile!.path) }),
      queryClient.refetchQueries({ queryKey: trainingContextKeys.merged(selectedFile!.path, context.selectedPreset) }),
    ]);
    setHydratedKey("");
    save.reset();
  }
  const saveAs = useMutation({
    mutationFn: async (name: string) => {
      const target = importedTrainingPath(name);
      if (!target) throw new Error("请输入有效配置名称");
      const patched = await previewTrainingConfigPatch(
        selectedFile!.path,
        changes,
      );
      const saved = await saveTrainingConfigAs(target, patched.content);
      return { ...saved, file: target };
    },
    onSuccess: async (result) => {
      setSaveAsOpen(false);
      setNotice(result.message || "训练配置已另存");
      await queryClient.invalidateQueries({
        queryKey: trainingContextKeys.files(),
      });
      context.selectConfigFile(result.file);
      setHydratedKey("");
    },
  });
  const preflight = useMutation({
    mutationFn: () =>
      runTrainingPreflight(selectedFile!, context.selectedPreset, deviceState.gpuIds),
  });

  useEffect(() => { preflight.reset(); }, [JSON.stringify(deviceState.gpuIds), deviceState.selection.mode]);

  useEffect(() => {
    save.reset();
    saveAs.reset();
    preview.reset();
    preflight.reset();
  }, [hydrationKey]);

  function confirmDiscard(label: string) {
    return (
      !dirty ||
      window.confirm(
        `当前训练配置有未保存修改。${label}会丢失这些修改，是否继续？`,
      )
    );
  }

  async function beforeAction(
    action: "preflight" | "start" | "queue" | "prompts",
  ) {
    if (busy || saveConflict || patch.error || (dirty && locked)) return;
    if (action !== "prompts" && deviceState.issue) return;
    if ((action === "start" || action === "queue") && pipelineUnavailable) return;
    try {
      if (dirty) await save.mutateAsync({ ...draft });
      if (action === "preflight") preflight.mutate();
      else if (action === "prompts") setPromptsOpen(true);
      else setLaunchMode(action);
    } catch {
      /* Save errors remain visible; never continue to execution. */
    }
  }

  const guardedContext = {
    ...context,
    selectConfigFile: (file: string) => {
      if (!busy && confirmDiscard("切换配置")) {
        setHydratedKey("");
        context.selectConfigFile(file);
      }
    },
    selectPreset: (preset: string) => {
      if (!busy && confirmDiscard("切换硬件预设")) {
        setHydratedKey("");
        context.selectPreset(preset);
      }
    },
  };
  const locked = Boolean(selectedFile?.locked || selectedFile?.readonly);
  const busy =
    preview.isPending ||
    save.isPending ||
    saveAs.isPending ||
    saveAsOpen ||
    preflight.isPending ||
    Boolean(launchMode) ||
    Boolean(rawMode) ||
    promptsOpen;
  const visibleFields = filterTrainingFields(
    fields,
    draft,
    baseline,
    fieldSearch,
    fieldView,
    selectedFile?.method || "lora",
  );
  const commandBlocked =
    !selectedFile ||
    busy ||
    context.isPending ||
    rawQuery.isPending ||
    capabilities.isPending ||
    Boolean(capabilities.error) ||
    Boolean(patch.error) ||
    saveConflict ||
    (dirty && locked);

  return {
    queryClient,
    deviceState,
    pipelineUnavailable,
    context,
    capabilities,
    selectedFile,
    rawQuery,
    baseline,
    draft,
    setDraft,
    hydratedKey,
    setHydratedKey,
    saveAsOpen,
    setSaveAsOpen,
    notice,
    activeStage,
    setActiveStage,
    launchMode,
    setLaunchMode,
    rawMode,
    setRawMode,
    fieldSearch,
    setFieldSearch,
    fieldView,
    setFieldView,
    promptsOpen,
    setPromptsOpen,
    mergedConfig,
    fields,
    patch,
    changes,
    dirty,
    ownKeys,
    preview,
    save,
    saveConflict,
    reloadConflictedConfig,
    saveAs,
    preflight,
    confirmDiscard,
    beforeAction,
    guardedContext,
    locked,
    busy,
    visibleFields,
    commandBlocked,
  };
}

export async function invalidateTrainingQueries(
  queryClient: ReturnType<typeof useQueryClient>,
  file: string,
  preset: string,
) {
  await Promise.all([
    queryClient.invalidateQueries({
      queryKey: trainingContextKeys.merged(file, preset),
    }),
    queryClient.invalidateQueries({ queryKey: trainingConfigKeys.raw(file) }),
    queryClient.invalidateQueries({ queryKey: datasetKeys.library() }),
    queryClient.invalidateQueries({
      queryKey: ["training-config", "estimate", file, preset],
    }),
  ]);
}
