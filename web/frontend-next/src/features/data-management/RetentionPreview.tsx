import { CircleAlert, FolderTree } from "lucide-react";
import { useMemo, useState } from "react";
import { formatBytes } from "./format";
import type { RetentionGroup, RetentionItem, RetentionPreviewResponse } from "./retentionTypes";

function ItemRow({ item }: { item: RetentionItem }) {
  return <li className={`retention-item ${item.action === "delete" ? "is-delete" : "is-keep"}`}>
    <span className="retention-item-dot" aria-hidden="true" />
    <span className="retention-item-main">
      <strong>{item.name}</strong>
      <small>{item.reason}</small>
    </span>
    <span className="retention-item-role">{item.role_label}</span>
    <span className="retention-item-size">{formatBytes(item.size_bytes)}</span>
  </li>;
}

function GroupBlock({
  group,
  excluded,
  selectionDisabled,
  defaultExpanded,
  onToggleUnit,
}: {
  group: RetentionGroup;
  excluded: boolean;
  selectionDisabled: boolean;
  defaultExpanded: boolean;
  onToggleUnit: (relativePath: string, included: boolean) => void;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  return <details
    className={`retention-group${excluded ? " is-excluded" : ""}`}
    open={expanded}
    onToggle={(event) => setExpanded(event.currentTarget.open)}
  >
    <summary>
      <input
        type="checkbox"
        className="retention-group-check"
        checked={!excluded}
        disabled={selectionDisabled || (!excluded && group.delete_count === 0)}
        aria-label={`包含实验 ${group.name}`}
        onClick={(event) => event.stopPropagation()}
        onChange={(event) => onToggleUnit(group.relative_path, event.target.checked)}
      />
      <FolderTree size={14} aria-hidden="true" />
      <span className="retention-group-path">{group.relative_path}</span>
      <span className="retention-group-meta">
        {excluded ? <span className="retention-excluded-tag">已豁免</span> : <>
          <span>{group.checkpoint_count} 个检查点</span>
          <span className="is-delete">{group.delete_count} 项待清理</span>
          <span>{formatBytes(group.delete_size_bytes)}</span>
        </>}
      </span>
    </summary>
    {expanded ? <ul className="retention-items" aria-label={`${group.name} 的保留与清理明细`}>
      {group.items.map((item) => <ItemRow key={item.relative_path} item={item} />)}
    </ul> : null}
    {group.truncated ? <p className="retention-note">该目录条目过多，仅展示部分明细。</p> : null}
  </details>;
}

export function RetentionPreview({
  data,
  error,
  isPending,
  excluded,
  onlyMatched,
  unitSelectionDisabled = false,
  onToggleOnlyMatched,
  onToggleUnit,
}: {
  data?: RetentionPreviewResponse;
  error?: Error | null;
  isPending: boolean;
  excluded: Set<string>;
  onlyMatched: boolean;
  unitSelectionDisabled?: boolean;
  onToggleOnlyMatched: (next: boolean) => void;
  onToggleUnit: (relativePath: string, included: boolean) => void;
}) {
  const groups = data?.groups ?? [];
  // Every candidate that can be submitted must remain inspectable.  Do not
  // slice this list: a virtualized renderer can be introduced later, but a
  // hidden group must never be silently included in a permanent deletion.
  const visibleGroups = useMemo(() => groups
    .filter((group) => !onlyMatched || group.delete_count > 0 || excluded.has(group.relative_path))
    .sort((left, right) => Number(right.delete_count > 0) - Number(left.delete_count > 0)
      || right.delete_size_bytes - left.delete_size_bytes
      || left.relative_path.localeCompare(right.relative_path)), [groups, onlyMatched, excluded]);

  if (error) {
    return <p className="retention-error" role="alert">策略预览失败：{error.message}</p>;
  }
  if (!data) {
    return <p className="retention-note" role="status">{isPending ? "正在扫描所选目录…" : "调整左侧规则即可生成实时预览。"}</p>;
  }

  return <div className="retention-preview">
    <div className="retention-toolbar">
      <label className="retention-check">
        <input type="checkbox" checked={onlyMatched} onChange={(event) => onToggleOnlyMatched(event.target.checked)} />
        <span>仅显示有待清理项的实验</span>
      </label>
      <span className="retention-toolbar-meta">按释放空间从大到小</span>
    </div>
    {data.blocked.length > 0 ? <div className="retention-blocked" role="status">
      <p><CircleAlert size={14} aria-hidden="true" /> {data.blocked.length} 项被阻止，不会删除</p>
      <ul>
        {data.blocked.slice(0, 8).map((item, index) => <li key={`${item.relative_path}-${index}`}>
          <span>{item.relative_path}</span><small>{item.reason}</small>
        </li>)}
      </ul>
    </div> : null}
    <div className="retention-groups">
      {visibleGroups.map((group, index) => <GroupBlock key={group.relative_path} group={group}
        excluded={excluded.has(group.relative_path)} selectionDisabled={unitSelectionDisabled}
        defaultExpanded={index < 3 && group.delete_count > 0 && !excluded.has(group.relative_path)}
        onToggleUnit={onToggleUnit} />)}
      {visibleGroups.length === 0 ? <p className="retention-note">当前过滤条件下没有可清理的实验单元。</p> : null}
    </div>
    <p className="retention-note">明细由服务端清理计划锁定；提交时会重新校验目录状态与活动任务。</p>
  </div>;
}
