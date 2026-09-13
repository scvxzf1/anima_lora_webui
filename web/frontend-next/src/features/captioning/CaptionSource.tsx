import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play, Search, ChevronLeft, ChevronRight } from "lucide-react";
import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { datasetLibraryQuery, datasetPresetQuery } from "../dataset-editor/api";
import {
  captioningKeys,
  fetchCaptionPrompts,
  scanCaptionImages,
  createCaptionJob,
  captionImageUrl,
  type Profiles,
} from "./api";

export function CaptionSource({
  library,
  onCreated,
}: {
  library?: Profiles;
  onCreated: (id: string) => void;
}) {
  const [params] = useSearchParams();
  const [file, setFile] = useState(params.get("dataset") || "");
  const [index, setIndex] = useState(0);
  const [source, setSource] = useState("source");
  const [offset, setOffset] = useState(0);
  const [scanned, setScanned] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [profileId, setProfileId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
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
  const profile = library?.profiles.find(
    (p) => p.id === (profileId || library.active_profile_id),
  );
  const create = useMutation({
    mutationFn: createCaptionJob,
    retry: false,
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: captioningKeys.jobs });
      onCreated(data.job.id);
    },
  });
  function resetSource() {
    setScanned(false);
    setSelected([]);
    setOffset(0);
  }
  function submit() {
    if (
      !profile ||
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
                setFile(e.target.value);
                setIndex(0);
                resetSource();
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
                setIndex(Number(e.target.value));
                resetSource();
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
                setSource(e.target.value);
                resetSource();
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
              onChange={(e) => setProfileId(e.target.value)}
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
              defaultValue=""
              onChange={(e) => {
                const value = prompts.data?.presets.find(
                  (p) => p.id === e.target.value,
                );
                if (value) {
                  setPrompt(value.user_prompt);
                  setSystemPrompt(value.system_prompt);
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
          {profile?.kind !== "local" && (
            <>
              <label>
                <span>系统提示词</span>
                <textarea
                  rows={3}
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                />
              </label>
              <label>
                <span>用户提示词</span>
                <textarea
                  rows={3}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
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
        create.error ||
        prompts.error) && (
        <p className="form-error" role="alert">
          {
            (
              datasets.error ||
              preset.error ||
              images.error ||
              create.error ||
              prompts.error
            )?.message
          }
        </p>
      )}
    </section>
  );
}
