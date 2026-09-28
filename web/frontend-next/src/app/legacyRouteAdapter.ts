const HASH_ROUTES: Array<[RegExp, string]> = [
  [/^(?:page\/)?live-training(?:\/.*)?$/, "/monitor"],
  [/^(?:page\/)?queue(?:\/.*)?$/, "/queue"],
  [/^dataset-editor(?:\/.*)?$/, "/datasets"],
  [/^model-config(?:\/.*)?$/, "/models"],
  [/^(?:page\/)?(?:captioning|tagging)(?:\/.*)?$/, "/captioning"],
];

export function legacyHashTarget(hash: string): { path: string; view?: string } | undefined {
  const raw = hash.replace(/^#/, "").replace(/^\/+/, "");
  const rawHistoryMatch = raw.match(/^history\/([^/]+)(?:\/(overview|metrics|artifacts|logs|config))?$/);
  if (rawHistoryMatch) {
    try {
      return { path: `/history/${encodeURIComponent(decodeURIComponent(rawHistoryMatch[1]))}`, view: rawHistoryMatch[2] };
    } catch {
      return undefined;
    }
  }
  let value: string;
  try {
    value = decodeURIComponent(hash.replace(/^#/, "")).replace(/^\/+/, "");
  } catch {
    return undefined;
  }
  if (!value || value === "dashboard") return { path: "/training" };
  if (value === "config/training-config" || value.startsWith("config/training-config/")) return { path: "/training" };
  if (value === "history") return { path: "/history" };
  for (const [pattern, path] of HASH_ROUTES) if (pattern.test(value)) return { path };
  return undefined;
}

export function legacyHashPath(hash: string): string | undefined {
  return legacyHashTarget(hash)?.path;
}

export function redirectLegacyHash(location: Location = window.location): boolean {
  const target = legacyHashTarget(location.hash);
  if (!target || !location.hash) return false;
  const query = new URLSearchParams(location.search);
  if (target.view) query.set("view", target.view);
  const base = location.pathname === "/next" || location.pathname.startsWith("/next/") ? "/next" : "";
  window.history.replaceState(null, "", `${base}${target.path}${query.size ? `?${query}` : ""}`);
  return true;
}
