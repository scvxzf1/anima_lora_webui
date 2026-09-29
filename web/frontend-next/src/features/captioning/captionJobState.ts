type JobProvider = { settings?: { provider?: string } } | null | undefined;

export function captionJobStateLabel(state: string | undefined, job?: JobProvider) {
  if (state === "running") {
    const provider = job?.settings?.provider?.trim().toLowerCase();
    if (provider === "wd14" || provider === "cltagger") return "正在本地打标";
    if (provider === "openai_compatible") return "正在调用外部 API";
    return "正在处理";
  }
  if (state === "queued") {
    const provider = job?.settings?.provider?.trim().toLowerCase();
    return provider === "wd14" || provider === "cltagger" ? "本地任务待处理" : "待处理";
  }
  return (
    {
      completed: "已完成",
      partial: "部分完成",
      failed: "任务失败",
      canceled: "任务已停止",
      cancelled: "任务已停止",
    }[state || ""] || state || "未知状态"
  );
}
