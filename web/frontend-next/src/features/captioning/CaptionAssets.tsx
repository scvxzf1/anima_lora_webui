import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, X } from "lucide-react";
import {
  assetKeys,
  cancelDownload,
  downloadActive,
  downloadAsset,
  downloadDictionary,
  fetchCaptionAssets,
  fetchDictionary,
  type DownloadJob,
} from "./assetApi";
import { captioningKeys } from "./api";

function size(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function DownloadProgress({
  job,
  disabled,
  onCancel,
}: {
  job: DownloadJob;
  disabled: boolean;
  onCancel: () => void;
}) {
  return (
    <div className="asset-progress">
      <progress
        aria-label="下载进度"
        max={Math.max(1, job.total_bytes)}
        value={job.bytes_downloaded}
      />
      <span>
        {job.state} · {size(job.bytes_downloaded)} / {size(job.total_bytes)}
      </span>
      {downloadActive(job.state) && (
        <button
          type="button"
          title="取消下载"
          aria-label="取消下载"
          disabled={disabled}
          onClick={onCancel}
        >
          <X size={16} />
        </button>
      )}
      {job.error && (
        <p role="alert" className="form-error">
          {job.error}
        </p>
      )}
    </div>
  );
}

export function CaptionAssets() {
  const qc = useQueryClient();
  const assets = useQuery({
    queryKey: assetKeys.assets,
    queryFn: ({ signal }) => fetchCaptionAssets(signal),
    refetchInterval: (query) =>
      query.state.data?.downloads?.some((job) => downloadActive(job.state))
        ? 2000
        : false,
  });
  const dictionary = useQuery({
    queryKey: assetKeys.dictionary,
    queryFn: ({ signal }) => fetchDictionary(signal),
    refetchInterval: (query) =>
      downloadActive(query.state.data?.download?.state) ? 2000 : false,
  });
  const command = useMutation({
    mutationFn: (operation: () => Promise<unknown>) => operation(),
    retry: false,
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: assetKeys.assets });
      await qc.invalidateQueries({ queryKey: assetKeys.dictionary });
      await qc.invalidateQueries({ queryKey: captioningKeys.profiles });
    },
  });
  function start(id: string, label: string, bytes: number) {
    if (
      command.isPending ||
      !window.confirm(
        `下载 ${label}（约 ${size(bytes)}）到本地模型目录？将访问远程资源并占用磁盘空间。`,
      )
    )
      return;
    command.mutate(() => (id ? downloadAsset(id) : downloadDictionary()));
  }
  const error = assets.error || dictionary.error || command.error;
  return (
    <section className="caption-assets">
      <header className="toolbar">
        <h2>本地资源</h2>
        <button
          type="button"
          onClick={() => {
            assets.refetch();
            dictionary.refetch();
          }}
          disabled={assets.isFetching || dictionary.isFetching}
        >
          刷新状态
        </button>
      </header>
      {error && (
        <p role="alert" className="form-error">
          {error.message}
        </p>
      )}
      {assets.isPending && <p role="status">正在读取模型资源</p>}
      {assets.data?.assets.map((asset) => {
        const job =
          asset.download ||
          assets.data.downloads?.find(
            (entry) =>
              entry.asset_id === asset.id && downloadActive(entry.state),
          );
        return (
          <article className="asset-row" key={asset.id}>
            <div>
              <h3>{asset.label}</h3>
              <p>
                {asset.repo_id} · {asset.license}
              </p>
              <small>
                {asset.id} · {asset.state} · {size(asset.total_size)}
              </small>
            </div>
            <button
              type="button"
              disabled={
                command.isPending ||
                asset.installed ||
                downloadActive(job?.state) ||
                (asset.requires_auth && !asset.auth_configured)
              }
              onClick={() => start(asset.id, asset.label, asset.total_size)}
            >
              <Download size={16} />
              {asset.installed ? "已安装" : "下载模型"}
            </button>
            {asset.requires_auth && !asset.auth_configured && (
              <p className="form-error">{asset.auth_hint}</p>
            )}
            {job && (
              <DownloadProgress
                job={job}
                disabled={command.isPending}
                onCancel={() => command.mutate(() => cancelDownload(job.id))}
              />
            )}
          </article>
        );
      })}
      {dictionary.data && (
        <article className="asset-row">
          <div>
            <h3>标签翻译词典</h3>
            <p>{dictionary.data.source_name}</p>
            <small>
              {dictionary.data.state} · {dictionary.data.entry_count} 条目
            </small>
          </div>
          <button
            type="button"
            disabled={
              command.isPending ||
              downloadActive(dictionary.data.download?.state)
            }
            onClick={() =>
              start("", "标签翻译词典", dictionary.data!.download_size)
            }
          >
            <Download size={16} />
            {dictionary.data.installed ? "重新下载" : "下载词典"}
          </button>
          {dictionary.data.download && (
            <DownloadProgress
              job={dictionary.data.download}
              disabled={command.isPending}
              onCancel={() =>
                command.mutate(() =>
                  cancelDownload(dictionary.data!.download!.id, true),
                )
              }
            />
          )}
        </article>
      )}
    </section>
  );
}
