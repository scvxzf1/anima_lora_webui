export async function downloadTrainingGroup(groupId: string): Promise<void> {
  const response = await fetch(`/api/config/file-groups/${encodeURIComponent(groupId)}/export?kind=training`);
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    const error = body && typeof body === "object" && "error" in body ? body.error : null;
    throw new Error(typeof error === "string" && error ? error : `导出分组失败 (${response.status})`);
  }
  const encodedName = response.headers.get("Content-Disposition")?.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  const filename = encodedName
    ? decodeURIComponent(encodedName).split(/[\\/]/).pop() || "training-config-group.zip"
    : "training-config-group.zip";
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
