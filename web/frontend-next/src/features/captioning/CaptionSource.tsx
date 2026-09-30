import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play, Search, ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { datasetLibraryQuery, datasetPresetQuery } from "../dataset-editor/api";
import {
  captioningKeys,
  fetchCaptionPrompts,
  scanCaptionImages,
  createCaptionJob,
  captionImageUrl,
  type Profiles,
} from "./api";
import { useCaptionDraft } from "./captionDraft";

export function CaptionSource({
  library,
  onCreated,
  onSourceChange,
}: {
  library?: Profiles;
  onCreated: (id: string) => void;
  onSourceChange?: (file: string, index: number, source?: string) => void;
}) {
  const { search } = useLocation();
  const params = new URLSearchParams(search);
  const [file, setFile] = useState(params.get("dataset") || "");
  const [index, setIndex] = useState(() => Math.max(0, Number(params.get("subset")) || 0));
  const [source, setSource] = useState(params.get("source") || "source");
  const { draft, update } = useCaptionDraft({
    dataset: file,
    subset: index,
    source,
  });
  const [offset, setOffset] = useState(0);
  const [scanned, setScanned] = useState(() => draft.selected.length > 0);
  const previousScope = useRef(JSON.stringify([file, index, source]));
  const selected = draft.selected;
  const setSelected = (next: string[] | ((current: string[]) => string[])) =>
    update((current) => ({ selected: typeof next === "function" ? next(current.selected) : next }));
  const profileId = draft.profileId;
  const prompt = draft.prompt;
  const systemPrompt = draft.systemPrompt;
  const datasets = useQuery(datasetLibraryQuery);
  const preset = useQuery({
    ...datasetPresetQuery(file),
    enabled: Boolean(file),
  });
  const prompts = useQuery({
    queryKey: captioningKeys.prompts,
    queryFn: ({ signal }) => fetchCaptionPrompts(signal),
  });
  const images = useQuery({
    queryKey: ["captioning", "images", file, index, source, offset],
    queryFn: ({ signal }) =>
      scanCaptionImages(file, index, source, offset, signal),
    enabled: scanned && Boolean(file),
  });
  const qc = useQueryClient();
  useEffect(() => {
    const nextFile = params.get("dataset") || "";
    const nextIndex = Math.max(0, Number(params.get("subset")) || 0);
    const nextSource = params.get("source") || "source";
    setFile(nextFile);
    setIndex(nextIndex);
    setSource(nextSource);
  }, [search]);
  useEffect(() => {
    const nextScope = JSON.stringify([file, index, source]);
    if (previousScope.current !== nextScope) {
      previousScope.current = nextScope;
      setScanned(draft.selected.length > 0);
      setOffset(0);
    }
  }, [file, index, source, draft.selected.length]);
  const profile = library?.profiles.find(
    (p) => p.id === (profileId === null ? library.active_profile_id : profileId),
  );
  useEffect(() => {
    if (!library || profileId === null || profileId === "") return;
    const restored = library.profiles.find((item) => item.id === profileId);
    if (!restored?.available) update({ profileId: "" });
  }, [library, profileId, update]);
  const create = useMutation({
    mutationFn: async (payload: Parameters<typeof createCaptionJob>[0]) => {
      const data = await createCaptionJob(payload);
      if (!data?.job?.id) {
        throw new Error("打标任务响应缺少任务 ID");
      }
      return data;
    },
    retry: false,
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: captioningKeys.jobs });
      onCreated(data.job.id);
    },
  });
  function submit() {
    if (
      !profile?.available ||
      !selected.length ||
      selected.length > 500 ||
      create.isPending
    )
      return;
    const message =
      profile.kind === "external"
        ? `将 ${selected.length} 张图片及提示词发送至 ${String(profile.config.base_url || profile.name)}。可能产生费用。确认开始打标吗？`
        : `使用本地 ${profile.name} 处理 ${selected.length} 张图片，会占用计算资源。确认开始吗？`;
    if (!window.confirm(message)) return;
    create.mutate({
      dataset_file: file,
      dataset_index: index,
      source,
      profile_id: profile.id,
      user_prompt: prompt,
      system_prompt: systemPrompt,
      items: selected.map((entry) => ({ file: entry })),
    });
  }
  return (
    <section>
      <h2>图片来源</h2>
      {selected.length > 500 && (
        <p role="alert" className="form-error">
          单个任务最多处理 500 张图片，当前已选 {selected.length} 张。
        </p>
      )}
      <fieldset disabled={create.isPending} className="unframed-fieldset">
        <div className="settings-grid">
          <label>
            <span>数据集预设</span>
            <select
              value={file}
              onChange={(e) => {
                const nextFile = e.target.value;
                setFile(nextFile);
                setIndex(0);
                onSourceChange?.(nextFile, 0, source);
              }}
            >
              <option value="">选择数据集</option>
              {datasets.data?.presets.map((p) => (
                <option key={p.path} value={p.path}>
                  {p.label || p.filename || p.path}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>图片组</span>
            <select
              value={index}
              onChange={(e) => {
                const nextIndex = Number(e.target.value);
                setIndex(nextIndex);
                onSourceChange?.(file, nextIndex, source);
              }}
            >
              {preset.data?.datasets.map((row, i) => (
                <option key={i} value={i}>
                  {i + 1} · {row.source_dir || row.image_dir}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>目录类型</span>
            <select
              value={source}
              onChange={(e) => {
                const nextSource = e.target.value;
                setSource(nextSource);
                onSourceChange?.(file, index, nextSource);
              }}
            >
              <option value="source">原始图</option>
              <option value="training">训练图</option>
            </select>
          </label>
          <label>
            <span>接入预设</span>
            <select
              value={profile?.id || ""}
              onChange={(e) => update({ profileId: e.target.value })}
            >
              <option value="">选择接入预设</option>
              {library?.profiles.map((p) => (
                <option value={p.id} key={p.id}>
                  {p.name} · {p.status}
                </option>
              ))}
            </select>
          </label>
          <label className="full-width">
            <span>提示词预设</span>
            <select
              value={prompts.data?.presets.some((p) => p.id === draft.promptPresetId) ? draft.promptPresetId : ""}
              onChange={(e) => {
                const value = prompts.data?.presets.find(
                  (p) => p.id === e.target.value,
                );
                update({ promptPresetId: e.target.value });
                if (value) {
                  update({ prompt: value.user_prompt, systemPrompt: value.system_prompt });
                }
              }}
            >
              <option value="">自定义</option>
              {prompts.data?.presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          {prompts.error && (
            <p className="form-error full-width" role="alert">
              {prompts.error.message}
              <button
                type="button"
                onClick={() => void prompts.refetch()}
                disabled={prompts.isFetching}
              >
                重试提示词
              </button>
            </p>
          )}
          {profile?.kind !== "local" && (
            <>
              <label>
                <span>系统提示词</span>
                <textarea
                  rows={3}
                  value={systemPrompt}
                  onChange={(e) => update({ systemPrompt: e.target.value, promptPresetId: "" })}
                />
              </label>
              <label>
                <span>用户提示词</span>
                <textarea
                  rows={3}
                  value={prompt}
                  onChange={(e) => update({ prompt: e.target.value, promptPresetId: "" })}
                />
              </label>
            </>
          )}
        </div>
        <div className="toolbar">
          <button
            type="button"
            disabled={!file || preset.isPending || images.isFetching}
            onClick={() => {
              setScanned(true);
              if (scanned) images.refetch();
            }}
          >
            <Search size={16} />
            扫描图片
          </button>
          <button
            type="button"
            className="primary-command"
            disabled={
              !selected.length ||
              selected.length > 500 ||
              !profile?.available ||
              create.isPending
            }
            onClick={submit}
          >
            <Play size={16} />
            {create.isPending ? "正在提交" : `开始打标 (${selected.length})`}
          </button>
        </div>
        {images.data && (
          <>
            <div className="toolbar">
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={
                    images.data.images.length > 0 &&
                    images.data.images.every((image) =>
                      selected.includes(image.file),
                    )
                  }
                  onChange={(e) => {
                    const page = images.data!.images.map((image) => image.file);
                    setSelected((ids) =>
                      e.target.checked
                        ? [...new Set([...ids, ...page])]
                        : ids.filter((id) => !page.includes(id)),
                    );
                  }}
                />
                全选本页
              </label>
              <span>
                {images.data.total} 张 · 已选 {selected.length}
              </span>
              <button
                type="button"
                aria-label="上一页图片"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - 60))}
              >
                <ChevronLeft size={16} />
              </button>
              <button
                type="button"
                aria-label="下一页图片"
                disabled={offset + 60 >= images.data.total}
                onClick={() => setOffset(offset + 60)}
              >
                <ChevronRight size={16} />
              </button>
              <button
                type="button"
                disabled={!selected.length}
                onClick={() => setSelected([])}
              >
                清空选择
              </button>
            </div>
            <div className="caption-image-grid">
              {images.data.images.map((image) => (
                <label key={image.file}>
                  <img
                    src={captionImageUrl(image.url)}
                    loading="lazy"
                    alt={image.name}
                  />
                  <span>
                    <input
                      type="checkbox"
                      checked={selected.includes(image.file)}
                      onChange={(e) =>
                        setSelected((ids) =>
                          e.target.checked
                            ? [...ids, image.file]
                            : ids.filter((id) => id !== image.file),
                        )
                      }
                    />
                    {image.name}
                  </span>
                </label>
              ))}
            </div>
            {images.data.images.length === 0 && <p>此目录没有图片</p>}
          </>
        )}
      </fieldset>
      {(datasets.error ||
        preset.error ||
        images.error ||
        create.error) && (
        <p className="form-error" role="alert">
          {
            (
              datasets.error ||
              preset.error ||
              images.error ||
              create.error
            )?.message
          }
        </p>
      )}
    </section>
  );
}
