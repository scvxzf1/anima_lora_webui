export const formatBytes = (value: number | null | undefined) => {
  if (value === null || value === undefined || !Number.isFinite(value)) return "大小未知";
  if (value < 1024) return `${value.toLocaleString("zh-CN")} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = value;
  let unit = -1;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toLocaleString("zh-CN", { maximumFractionDigits: 1 })} ${units[unit]}`;
};

export const formatDate = (value: string | null | undefined) => {
  if (!value) return "修改时间未知";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString("zh-CN");
};
