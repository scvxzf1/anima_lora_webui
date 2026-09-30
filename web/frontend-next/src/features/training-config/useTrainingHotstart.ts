import { useEffect, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import type { TrainingConfigFile } from "../../api/trainingContext";
import { useTrainingContextStore } from "../../app/trainingContextStore";
import { hotstartTarget, inspectHotstart } from "../training-history/historyHotstartApi";
import { useHotstartIntent } from "./hotstartIntent";
import type { TrainingDraft } from "./trainingForm";

type Args = {
  file?: TrainingConfigFile;
  preset: string;
  hydrated: boolean;
  unavailable: boolean;
  busy: boolean;
  dirty: boolean;
  draft: TrainingDraft;
  mergedConfig: Record<string, unknown>;
  setDraft: Dispatch<SetStateAction<TrainingDraft>>;
};

export function useTrainingHotstart(args: Args) {
  const intent = useHotstartIntent((state) => state.intent);
  const clear = useHotstartIntent((state) => state.clear);
  const [status, setStatus] = useState<"idle" | "checking" | "ready" | "error">("idle");
  const [error, setError] = useState("");
  const [retryVersion, setRetryVersion] = useState(0);
  const inspectedPath = useRef("");
  const target = args.file && hotstartTarget(args.file, args.preset);
  const matches = Boolean(intent && target && intent.configFile === target.configFile &&
    intent.preset === target.preset && intent.variant === target.variant && intent.subdir === target.subdir);
  const latest = useRef({ args, matches, intent, target });
  latest.current = { args, matches, intent, target };

  useEffect(() => {
    const controller = new AbortController();
    inspectedPath.current = "";
    if (!intent) { setStatus("idle"); return () => controller.abort(); }
    if (!matches && args.hydrated) {
      clear(intent.id);
      setStatus("error");
      setError("训练配置或预设已改变，热启动请求已取消");
      return () => controller.abort();
    }
    if (args.unavailable && args.hydrated) {
      clear(intent.id);
      setStatus("error");
      setError("训练配置不可用，热启动请求已取消");
      return () => controller.abort();
    }
    if (!matches || !args.hydrated || args.unavailable || args.busy) {
      setStatus("idle");
      return () => controller.abort();
    }
    setStatus("checking");
    setError("");
    void inspectHotstart(intent.path, target!, controller.signal).then((result) => {
      if (controller.signal.aborted) return;
      const current = useTrainingContextStore.getState();
      const now = latest.current;
      if (useHotstartIntent.getState().intent?.id !== intent.id || !now.matches ||
          !now.args.hydrated || now.args.unavailable || now.args.busy) return;
      if (current.configFile !== intent.configFile || current.preset !== intent.preset) {
        clear(intent.id);
        setError("训练配置或预设已改变，热启动请求已取消");
        setStatus("error");
        return;
      }
      inspectedPath.current = result.abs_path;
      setStatus("ready");
    }).catch((cause) => {
      if (controller.signal.aborted || useHotstartIntent.getState().intent?.id !== intent.id) return;
      setError(cause instanceof Error ? cause.message : "权重检查失败");
      setStatus("error");
    });
    return () => controller.abort();
  }, [intent, matches, args.hydrated, args.unavailable, args.busy, target?.configFile, target?.preset, target?.variant, target?.subdir, clear, retryVersion]);

  function retry() {
    if (!intent) return;
    setStatus("idle");
    setRetryVersion((version) => version + 1);
  }
  function cancel() {
    if (intent) clear(intent.id);
    setStatus("idle");
    setError("");
  }
  function apply() {
    if (!intent || status !== "ready" || !matches || !args.hydrated || args.busy || args.unavailable || !inspectedPath.current) return;
    const current = useTrainingContextStore.getState();
    if (current.configFile !== intent.configFile || current.preset !== intent.preset) { cancel(); return; }
    const path = inspectedPath.current;
    args.setDraft((draft) => ({
      ...draft,
      network_weights: path,
      ...(draft.resume || args.mergedConfig.resume ? { resume: "" } : {}),
    }));
    clear(intent.id);
    setStatus("idle");
  }
  return { intent, status, error, dirty: args.dirty, apply, retry, cancel };
}
