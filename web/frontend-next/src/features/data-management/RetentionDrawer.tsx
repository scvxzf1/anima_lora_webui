import { useMutation } from "@tanstack/react-query";
import { SlidersHorizontal, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CommandDialog } from "../../components/CommandDialog";
import { formatBytes } from "./format";
import { RetentionPolicyForm } from "./RetentionPolicyForm";
import { RetentionPreview } from "./RetentionPreview";
import { cleanupRetention } from "./retentionApi";
import { useRetentionPlan } from "./useRetentionPlan";
import { DEFAULT_RETENTION_POLICY, type RetentionPolicy, type RetentionScopeOption } from "./retentionTypes";
import "./retention.css";

export function RetentionDrawer({
  options,
  onClose,
  onCompleted,
}: {
  options: RetentionScopeOption[];
  onClose: () => void;
  onCompleted: (message: string) => void;
}) {
  const [scopeId, setScopeId] = useState(options[0]?.id ?? "");
  const [policy, setPolicy] = useState<RetentionPolicy>(DEFAULT_RETENTION_POLICY);
  const [patternText, setPatternText] = useState("");
  const [excluded, setExcluded] = useState<string[]>([]);
  const [onlyMatched, setOnlyMatched] = useState(true);
  const [confirmed, setConfirmed] = useState(false);
  const [error, setError] = useState("");
  const scope = options.find((option) => option.id === scopeId) ?? options[0];
  const plan = useRetentionPlan({
    rootId: scope?.rootId ?? "",
    dirs: scope?.dirs ?? [],
    policy,
    excludeUnits: excluded,
    enabled: Boolean(scope),
  });
  const excludedSet = useMemo(() => new Set(excluded), [excluded]);
  const activeGroups = useMemo(
    () => (plan.data?.groups ?? []).filter((group) => !excludedSet.has(group.relative_path)),
    [plan.data, excludedSet],
  );
  const deleteCount = activeGroups.reduce((total, group) => total + group.delete_count, 0);
  const keepCount = activeGroups.reduce((total, group) => total + group.keep_count, 0);
  const deleteSize = activeGroups.reduce((total, group) => total + group.delete_size_bytes, 0);
  const deleteItems = useMemo(
    () => activeGroups.flatMap((group) => group.items
      .filter((item) => item.action === "delete")
      .map((item) => item.relative_path)),
    [activeGroups],
  );
  const cleanup = useMutation({
    mutationFn: async () => {
      if (!scope || !plan.data) throw new Error("预览尚未就绪");
      return cleanupRetention(scope.rootId, scope.dirs, plan.data.policy, deleteItems,
        plan.data.plan_revision, excluded);
    },
    retry: false,
    onSuccess: (result) => {
      const incomplete = result.missing.length + result.blocked.length + Object.keys(result.errors).length;
      const message = result.message || `已清理 ${result.deleted.length} 项`;
      onCompleted(incomplete ? `${message}；另有 ${incomplete} 项未完成，请核对结果` : message);
      if (incomplete) {
        setConfirmed(false);
        setError(`清理未完全完成：缺少 ${result.missing.length} 项，阻止 ${result.blocked.length} 项，失败 ${Object.keys(result.errors).length} 项。请核对后重新预览。`);
        void plan.refetch();
      } else {
        onClose();
      }
    },
    onError: (cause: Error) => setError(`清理失败：${cause.message}`),
  });

  useEffect(() => {
    setConfirmed(false);
  }, [plan.data?.plan_revision]);

  function update(patch: Partial<RetentionPolicy>) {
    // A visible draft must never retain confirmation for the previous plan.
    setConfirmed(false);
    setError("");
    setPolicy((current) => ({ ...current, ...patch }));
  }

  function toggleUnit(relativePath: string, included: boolean) {
    setConfirmed(false);
    setExcluded((current) => included
      ? current.filter((item) => item !== relativePath)
      : [...current, relativePath]);
  }

  // 旧版后端不认识 exclude_units / 缓存规则；提示重启而不是静默失效。
  const staleBackend = Boolean(plan.data) && !("exclude_units" in (plan.data as object));
  const previewReady = Boolean(plan.data) && !plan.error && !plan.isPlaceholderData
    && !plan.isFetching && !plan.isSettling && !staleBackend;
  const ready = previewReady && deleteItems.length > 0 && confirmed && !cleanup.isPending;
  const release = formatBytes(deleteSize);

  return <CommandDialog title="检查点与权重智能清理" size="wide" onClose={onClose} busy={cleanup.isPending}>
    <div className="retention-layout">
      <section className="retention-pane is-form" aria-label="清理策略">
        <div className="retention-scope">
          <SlidersHorizontal size={16} aria-hidden="true" />
          <span>作用范围</span>
          {options.length === 0 ? <strong>没有可清理的目录，请先在下方勾选实验目录</strong> : options.length === 1
            ? <strong>{options[0].label}</strong>
            : <div className="retention-radios" role="group" aria-label="策略作用范围">
              {options.map((option) => <label key={option.id} className="retention-radio">
                <input type="radio" name="retention-scope" checked={scope?.id === option.id}
                  onChange={() => {
                    setScopeId(option.id);
                    setExcluded([]);
                    setConfirmed(false);
                    setError("");
                  }} disabled={cleanup.isPending} />
                <span>{option.label}</span>
              </label>)}
            </div>}
        </div>
        {scope ? <RetentionPolicyForm
          policy={policy}
          patternText={patternText}
          onPatternTextChange={setPatternText}
          onChange={update}
          disabled={cleanup.isPending}
        /> : <p className="retention-note">选择「数据管理」中的训练运行目录后，即可按检查点保留策略批量清理。</p>}
      </section>

      <section className="retention-pane is-preview" aria-label="清理预览">
        <p className="retention-summary" aria-live="polite">
          <strong>{plan.data ? `共匹配 ${plan.data.summary.group_count} 个实验单元` : "等待预览"}</strong>
          {plan.data ? <>
            <span>保留 {keepCount} 项 · 清理 {deleteCount} 项</span>
            <span className="is-delete">预计释放 {release}</span>
          </> : null}
          {excluded.length > 0 ? <span className="retention-excluded-tag">已豁免 {excluded.length} 个实验</span> : null}
          {plan.isFetching || plan.isSettling ? <span className="retention-pending">正在重新计算…</span> : null}
        </p>
        {staleBackend ? <p className="retention-warn" role="status">
          后端仍在运行旧版本，缓存 / 张量规则与“单实验豁免”暂不生效；请重启 WebUI 服务后再使用。
        </p> : null}
        <RetentionPreview
          data={plan.data}
          error={plan.error}
          isPending={plan.isPending}
          excluded={excludedSet}
          onlyMatched={onlyMatched}
          unitSelectionDisabled={staleBackend || cleanup.isPending || plan.isFetching || plan.isSettling}
          onToggleOnlyMatched={setOnlyMatched}
          onToggleUnit={toggleUnit}
        />
        {error ? <p className="retention-error" role="alert">{error}</p> : null}
        <footer className="retention-footer">
          <label className="retention-check retention-confirm">
            <input type="checkbox" checked={confirmed} disabled={!previewReady || deleteItems.length === 0 || cleanup.isPending}
              onChange={(event) => setConfirmed(event.target.checked)} />
            <span>我已核对上方 🟢 保留 / 🔴 清理明细，并确认永久删除</span>
          </label>
          <div className="retention-buttons">
            <button type="button" onClick={onClose} disabled={cleanup.isPending}>取消</button>
            <button type="button" className="retention-danger" onClick={() => cleanup.mutate()} disabled={!ready}>
              <Trash2 size={16} />{cleanup.isPending ? "正在清理" : `确认执行清理（释放 ${release}）`}
            </button>
          </div>
        </footer>
      </section>
    </div>
  </CommandDialog>;
}
