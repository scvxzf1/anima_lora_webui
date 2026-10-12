import { useEffect, type RefObject } from "react";
import { formatTrainingValue, trainingMetricNames, type TrainingChartPoint } from "./combinedTrainingSeries";

const colors = ["#3b82f6", "#e88723"];
function tooltip(data: TrainingChartPoint[], visible: boolean[], values: boolean[], axis: string, items: { dataIndex?: number }[] | { dataIndex?: number }) {
  const index = (Array.isArray(items) ? items[0] : items)?.dataIndex;
  const point = index === undefined ? undefined : data[index];
  const content = document.createElement("div");
  if (!point) return content;
  const coordinate = document.createElement("div");
  coordinate.textContent = `${axis === "step" ? "STEP" : "采样序号"}: ${point[0]}`;
  content.append(coordinate);
  trainingMetricNames.forEach((name, column) => {
    if (!visible[column] || !values[column]) return;
    const row = document.createElement("div");
    row.textContent = `${name}: ${formatTrainingValue(column, point[column + 1])}`;
    content.append(row);
  });
  return content;
}

export function useCombinedTrainingCanvas(ref: RefObject<HTMLDivElement | null>, data: TrainingChartPoint[], axis: string, visible: boolean[], values: boolean[]) {
  const [loss, lr] = visible, [lossValues, lrValues] = values;
  useEffect(() => {
    const node = ref.current;
    if (!node || !node.clientWidth || !data.length || !visible.some(Boolean)) return;
    let disposed = false;
    let cleanup: (() => void) | undefined;
    void import("./metricsChartRuntime").then(({ init }) => {
      if (disposed) return;
      const chart = init(node);
      const render = () => {
        const style = getComputedStyle(node);
        chart.setOption({
          animation: false,
          grid: { left: loss ? 64 : 20, right: lr ? 78 : 20, bottom: 35, top: 35 },
          tooltip: { trigger: "axis", renderMode: "html", confine: true, formatter: (items: { dataIndex?: number }[] | { dataIndex?: number }) => tooltip(data, visible, values, axis, items) },
          xAxis: { type: "value", axisLabel: { color: style.getPropertyValue("--muted") }, splitLine: { show: false } },
          yAxis: trainingMetricNames.map((name, column) => ({
            type: "value", scale: true, position: column ? "right" : "left", show: visible[column], name: values[column] ? name : "",
            axisLabel: { show: values[column], color: colors[column], formatter: (value: number) => formatTrainingValue(column, value) },
            splitLine: { show: column === (loss ? 0 : 1), lineStyle: { color: style.getPropertyValue("--line") } },
          })),
          series: trainingMetricNames.map((name, column) => ({
            name, type: "line", yAxisIndex: column, data: visible[column] ? data.map((point) => [point[0], point[column + 1]]) : [],
            connectNulls: false, showSymbol: false, lineStyle: { width: 2, color: colors[column] }, itemStyle: { color: colors[column] },
          })),
        });
      };
      render();
      const resize = new ResizeObserver(() => { chart.dispatchAction({ type: "hideTip" }); chart.resize(); });
      resize.observe(node);
      const theme = new MutationObserver(render);
      theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
      cleanup = () => { resize.disconnect(); theme.disconnect(); chart.dispose(); };
    });
    return () => { disposed = true; cleanup?.(); };
  }, [ref, data, axis, loss, lr, lossValues, lrValues]);
}
