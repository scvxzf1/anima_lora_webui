export function splitCaptionTags(value: string) {
  return String(value || "")
    .split(/[\n,]+/)
    .map((tag) => tag.trim())
    .filter(Boolean);
}

export function joinCaptionTags(tags: string[]) {
  return tags
    .map((tag) => String(tag || "").trim())
    .filter(Boolean)
    .join(", ");
}

export function updateCaptionTag(tags: string[], index: number, value: string) {
  const next = [...tags];
  const clean = value.trim();
  if (index < 0 || index >= next.length || !clean) return next;
  next[index] = clean;
  return next;
}

export function deleteCaptionTag(tags: string[], index: number) {
  if (index < 0 || index >= tags.length) return [...tags];
  const next = [...tags];
  next.splice(index, 1);
  return next;
}

export function moveCaptionTag(tags: string[], from: number, to: number) {
  if (
    from < 0 ||
    from >= tags.length ||
    to < 0 ||
    to >= tags.length ||
    from === to
  )
    return [...tags];
  const next = [...tags];
  const [tag] = next.splice(from, 1);
  next.splice(to, 0, tag);
  return next;
}

export function appendCaptionTag(tags: string[], value: string) {
  const clean = value.trim().replace(/,+$/g, "").trim();
  return clean ? [...tags, clean] : [...tags];
}
