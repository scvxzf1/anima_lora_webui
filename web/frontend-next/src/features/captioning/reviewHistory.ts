import type { CaptionDelta } from "./captionBatch";
export const REVIEW_HISTORY_BYTES = 2 * 1024 * 1024;
export function boundReviewHistory(history: CaptionDelta[][], budget = REVIEW_HISTORY_BYTES) {
  const encoder = new TextEncoder();
  let bytes = 0;
  const kept: CaptionDelta[][] = [];
  for (const change of history.slice(-30).reverse()) {
    const size = change.reduce((sum, delta) => sum + encoder.encode(delta.before).byteLength + encoder.encode(delta.after).byteLength, 0);
    if (bytes + size > budget) break;
    bytes += size; kept.unshift(change);
  }
  return kept;
}
