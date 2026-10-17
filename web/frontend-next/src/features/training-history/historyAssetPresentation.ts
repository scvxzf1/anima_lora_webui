import { finiteNumber } from "../../components/trainingNumbers";
import { isTrainingMetric } from "../../components/metricSemantics";
import type { HistoryImage, HistoryMetricPoint, HistoryWeight } from "./api";

export function sampleStep(image: HistoryImage) {
  const step = finiteNumber(image.sample?.step);
  return step !== undefined && Number.isSafeInteger(step) && step >= 0 ? step : undefined;
}

export function groupHistorySamples(images: HistoryImage[]) {
  const groups = new Map<string, { key: string; label: string; prompt: string; images: HistoryImage[] }>();
  for (const image of images) {
    const sample = image.sample || {};
    const prompt = typeof sample.prompt === "string" ? sample.prompt : "";
    const index = finiteNumber(sample.prompt_index);
    const seed = typeof sample.seed === "string" || typeof sample.seed === "number" ? String(sample.seed) : "";
    // Reference identity matters for edit samples sharing the same prompt and seed.
    const key = JSON.stringify([index, prompt, seed, sample.reference_files ?? sample.reference_file]);
    if (!groups.has(key)) groups.set(key, {
      key, prompt,
      label: [index !== undefined ? `Prompt ${index + 1}` : prompt ? "Prompt" : "未记录 Prompt", seed ? `Seed ${seed}` : "未记录 Seed"].join(" · "),
      images: [],
    });
    groups.get(key)!.images.push(image);
  }
  return [...groups.values()].map((group) => ({ ...group,
    images: group.images.sort((a, b) => (sampleStep(a) ?? Infinity) - (sampleStep(b) ?? Infinity)),
  }));
}

export function commonWeightDirectory(weights: HistoryWeight[]) {
  if (!weights.length || weights.some((weight) => !weight.abs_path)) return undefined;
  const directories = weights.map((weight) => weight.abs_path!.replace(/\\/g, "/").split("/").slice(0, -1));
  // Only a contiguous prefix is a common directory, never coincidentally matching later segments.
  let length = 0;
  while (length < directories[0].length && directories.every((directory) => directory[length] === directories[0][length])) length++;
  const path = directories[0].slice(0, length).join("/");
  if (!path || /^[A-Za-z]:$/.test(path)) return undefined;
  const windows = weights[0].abs_path!.includes("\\");
  return windows ? path.replace(/\//g, "\\") : path;
}

export function checkpointLosses(metrics: HistoryMetricPoint[]) {
  const losses = new Map<number, number>();
  for (const point of metrics) {
    if (!isTrainingMetric(point)) continue;
    const step = finiteNumber(point.step);
    const loss = finiteNumber(point.loss);
    if (step !== undefined && Number.isSafeInteger(step) && step >= 0 && loss !== undefined) losses.set(step, loss);
  }
  return losses;
}
