import { cleanup, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { TrainingEstimate } from "./TrainingEstimate";
import { collectBuckets, type EstimateDataset } from "./estimateData";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const row: EstimateDataset = {
  index: 1, source_dir: "images/train", image_dir: "resized/train", train_image_count: 10,
  num_repeats: 2, sample_ratio: 1, sampled_image_count: 10, sampled_weighted_image_count: 20,
  trigger_clone_sampled_weighted_image_count: 5, uses_preprocessed_images: true,
  bucket_distribution: { basis: "image_dimensions", status: "ready", image_count: 10, unreadable_count: 0,
    buckets: [{ width: 1024, height: 1024, count: 8 }, { width: 832, height: 1280, count: 2 }] },
};
const reg: EstimateDataset = { ...row, index: 2, is_reg: true, source_dir: "images/reg", image_dir: "resized/reg",
  bucket_distribution: { ...row.bucket_distribution!, image_count: 3, buckets: [{ width: 1280, height: 832, count: 3 }] } };
const data = { total_steps: 200, train_image_count: 20, effective_batch_size: 2, steps_per_epoch: 25,
  repeated_image_count: 50, train_batch_size: 1, gradient_accumulation_steps: 2, duration_mode: "epochs", max_train_epochs: 8, datasets: [row, reg] };

it("renders the saved estimate and filters/sorts exact dimensions without weighting the census", async () => {
  const fetch = vi.fn(async () => jsonResponse(data));
  vi.stubGlobal("fetch", fetch);
  renderInApp(<TrainingEstimate file={{ path: "configs/test.toml" }} preset="default" dirty />);
  const table = await screen.findByRole("table", { name: "分桶尺寸明细" });
  expect(String(fetch.mock.calls[0])).toContain("include_buckets=1");
  expect(screen.getByText(/有未保存修改/)).toBeInTheDocument();
  expect(screen.getByText("正则化")).toBeInTheDocument();
  expect(within(table).getAllByRole("row")[1]).toHaveTextContent("1024 × 1024");
  const user = userEvent.setup();
  await user.selectOptions(screen.getByLabelText("分桶排序"), "aspect");
  expect(within(table).getAllByRole("row")[1]).toHaveTextContent("832 × 1280");
  await user.selectOptions(screen.getByLabelText("分桶数据集"), "2");
  expect(within(table).getAllByRole("row")).toHaveLength(2);
  expect(screen.getByText("共 3 张")).toBeInTheDocument();
  expect(within(table).queryByText("1024 × 1024")).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "重新估算" }));
  expect(fetch).toHaveBeenCalledTimes(2);
});

it("distinguishes unavailable, pending and unreadable distributions", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ ...data, duration_mode: "unset", datasets: [
    { ...row, bucket_distribution: undefined },
    { ...reg, bucket_distribution: { ...reg.bucket_distribution, status: "pending", buckets: [] } },
    { ...row, index: 3, bucket_distribution: { ...row.bucket_distribution, status: "partial", unreadable_count: 2 } },
  ] })));
  renderInApp(<TrainingEstimate file={{ path: "configs/test.toml" }} preset="default" dirty={false} />);
  expect(await screen.findByText(/未返回分桶信息/)).toBeInTheDocument();
  expect(screen.getByText(/源图与训练目录均无匹配图片/)).toBeInTheDocument();
  expect(screen.getByText(/2 张图片无法读取尺寸/)).toBeInTheDocument();
  expect(screen.getByText("未设置")).toBeInTheDocument();
});

it("merges matching sizes but preserves different resolutions at the same aspect ratio", () => {
  expect(collectBuckets([row, row])).toEqual(row.bucket_distribution!.buckets.map((bucket) => ({ ...bucket, count: bucket.count * 2 })));
  expect(collectBuckets([row, { ...reg, bucket_distribution: { ...reg.bucket_distribution!, buckets: [{ width: 512, height: 512, count: 1 }] } }])).toHaveLength(3);
});

it("labels mixed source predictions and measured directories and updates the scope", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ ...data, datasets: [
    { ...row, bucket_distribution: { ...row.bucket_distribution, basis: "source_projection", filtered_count: 2 } }, reg,
  ] })));
  renderInApp(<TrainingEstimate file={{ path: "configs/test.toml" }} preset="default" dirty={false} />);
  expect(await screen.findByText(/源图预测 1 个子集 · 训练目录实测 1 个子集/)).toBeInTheDocument();
  expect(screen.getByText(/排除 2 张源图/)).toBeInTheDocument();
  await userEvent.setup().selectOptions(screen.getByLabelText("分桶数据集"), "2");
  expect(screen.queryByText(/源图预测/)).not.toBeInTheDocument();
  expect(screen.getByText(/image_dir 尺寸实测/)).toBeInTheDocument();
});

it("shows request errors without inventing empty distribution data", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ error: "读取失败" }, 400)));
  renderInApp(<TrainingEstimate file={{ path: "configs/test.toml" }} preset="default" dirty={false} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("读取失败");
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
});
