import type { useTrainingWorkspace } from "./useTrainingWorkspace";
import { TrainingModelPicker } from "./TrainingExtras";
import { TrainingStageFields } from "./TrainingResourceGroups";
import { GROUPS } from "./useTrainingWorkspace";
import { TrainingDevices } from "./TrainingDevices";

export function TrainingEditor({
  state,
}: {
  state: ReturnType<typeof useTrainingWorkspace>;
}) {
  const {
    context,
    selectedFile,
    rawQuery,
    draft,
    baseline,
    setDraft,
    activeStage,
    setActiveStage,
    fieldSearch,
    setFieldSearch,
    fieldView,
    setFieldView,
    fields,
    ownKeys,
    preview,
    preflight,
    busy,
    visibleFields,
  } = state;
  return (
    <>
      <div className="training-editor-column">
        <TrainingModelPicker
          disabled={busy || context.isPending}
          onSelect={(item) => {
            setDraft((current) => ({
              ...current,
              model_family: item.model_family,
              pretrained_model_name_or_path: item.pretrained_model_name_or_path,
              qwen3: item.qwen3,
              vae: item.vae,
            }));
            preview.reset();
            preflight.reset();
          }}
        />
        <TrainingDevices state={state} />
        <div className="training-field-search">
          <input
            aria-label="搜索参数"
            placeholder="搜索参数名称或配置键"
            value={fieldSearch}
            onChange={(e) => setFieldSearch(e.target.value)}
          />
          <select
            aria-label="参数视图"
            value={fieldView}
            onChange={(e) => setFieldView(e.target.value)}
          >
            <option value="applicable">当前适用</option>
            <option value="changed">已修改</option>
            <option value="all">全部与审计</option>
          </select>
          <span>
            {visibleFields.length} / {fields.length}
          </span>
        </div>
        <nav className="training-stage-nav" role="tablist" aria-label="配置阶段">
          {GROUPS.map(([group, title], index) => (
            <button
              type="button"
              role="tab"
              id={`training-tab-${group}`}
              aria-selected={activeStage === group}
              aria-controls={activeStage === group ? `training-stage-${group}` : undefined}
              tabIndex={activeStage === group ? 0 : -1}
              key={group}
              data-active={activeStage === group}
              onClick={() => setActiveStage(group)}
              onKeyDown={(event) => {
                const target = {
                  ArrowRight: (index + 1) % GROUPS.length,
                  ArrowLeft: (index + GROUPS.length - 1) % GROUPS.length,
                  Home: 0,
                  End: GROUPS.length - 1,
                }[event.key];
                if (target === undefined) return;
                event.preventDefault();
                setActiveStage(GROUPS[target][0]);
                document.getElementById(`training-tab-${GROUPS[target][0]}`)?.focus();
              }}
            >
              {title}
              {(fieldSearch || fieldView !== "applicable") && (
                <small> ({visibleFields.filter((field) => field.group === group).length})</small>
              )}
            </button>
          ))}
        </nav>
        <div className="training-edit-groups" key={activeStage}>
          {GROUPS.filter(([group]) => group === activeStage).map(([group, title]) => (
            <section
              role="tabpanel"
              aria-labelledby={`training-tab-${group}`}
              tabIndex={0}
              className="training-edit-card"
              id={`training-stage-${group}`}
              key={group}
            >
              <header>
                <div>
                  <p className="eyebrow">{String(group).toUpperCase()}</p>
                  <h2>{title}</h2>
                </div>
                <span>
                  {
                    fields.filter(
                      (field) =>
                        field.group === group && ownKeys.has(field.key),
                    ).length
                  }{" "}
                  当前文件
                </span>
              </header>
              {group === "resources" && !fieldSearch && <TrainingDevices state={state} details />}
              {!visibleFields.some((field) => field.group === group) && (
                <p role="status">当前分类没有符合筛选条件的配置项。</p>
              )}
              <TrainingStageFields
                onEditPrompts={() => state.beforeAction("prompts")}
                stage={group}
                search={fieldSearch}
                view={fieldView}
                fields={visibleFields.filter((field) => field.group === group)}
                draft={draft}
                baseline={baseline}
                ownKeys={ownKeys}
                method={selectedFile?.method}
                disabled={
                  !selectedFile ||
                  context.isPending ||
                  rawQuery.isPending ||
                  busy
                }
                onChange={(key, value) => {
                  setDraft((current) => ({ ...current, [key]: value }));
                  preview.reset();
                  preflight.reset();
                }}
              />
            </section>
          ))}
        </div>
      </div>
    </>
  );
}
