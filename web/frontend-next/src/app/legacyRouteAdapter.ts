const HASH_ROUTES: Array<[RegExp, string]> = [
  [/^(?:page\/)?live-training(?:\/.*)?$/, "/monitor"],
  [/^(?:page\/)?queue(?:\/.*)?$/, "/queue"],
  [/^dataset-editor(?:\/.*)?$/, "/datasets"],
  [/^model-config(?:\/.*)?$/, "/models"],
  [/^(?:page\/)?(?:captioning|tagging)(?:\/.*)?$/, "/captioning"],
];

export function legacyHashPath(hash: string): string | undefined {
  let value: string;
  try {
    value = decodeURIComponent(hash.replace(/^#/, "")).replace(/^\/+/, "");
  } catch {
    return undefined;
  }
  if (!value || value === "dashboard") return "/training";
  if (value === "config/training-config" || value.startsWith("config/training-config/")) return "/training";
  if (value === "history") return "/history";
  const historyMatch = value.match(/^history\/(.+)$/);
  if (historyMatch) return `/history/${encodeURIComponent(historyMatch[1])}`;
  for (const [pattern, path] of HASH_ROUTES) if (pattern.test(value)) return path;
  return undefined;
}

export function redirectLegacyHash(location: Location = window.location): boolean {
  const path = legacyHashPath(location.hash);
  if (!path || !location.hash) return false;
  window.history.replaceState(null, "", `${path}${location.search}`);
  return true;
}
