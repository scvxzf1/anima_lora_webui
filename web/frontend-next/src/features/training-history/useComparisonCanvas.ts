import { useEffect, type RefObject } from "react";
import type { ComparisonLine } from "./comparisonData";

export const comparisonColors = ["--accent", "--success", "--warning", "--danger"];

export function useComparisonCanvas(ref: RefObject<HTMLDivElement | null>, lines: ComparisonLine[], start?: number, end?: number) {
  useEffect(() => {
    const node = ref.current;
    if (!node || !lines.some(({ data }) => data.length)) return;
    let disposed = false;
    let cleanup: (() => void) | undefined;
    void import("../../components/metricsChartRuntime").then(({ init }) => {
      if (disposed) return;
      const chart = init(node);
      const render = () => {
        const style = getComputedStyle(node);
        const color = (token: string) => style.getPropertyValue(token).trim();
        chart.setOption({
          animation: false,
          grid: { left: 58, right: 20, bottom: 40, top: 20 },
          tooltip: { trigger: "axis", renderMode: "richText" },
          xAxis: { type: "value", name: "step", nameTextStyle: { color: color("--muted") }, min: start === end ? undefined : start, max: start === end ? undefined : end, axisLabel: { color: color("--muted") }, splitLine: { show: false } },
          yAxis: { type: "value", scale: true, axisLabel: { color: color("--muted") }, splitLine: { lineStyle: { color: color("--line") } } },
          series: lines.map((line, index) => ({
            id: line.id, name: `${index + 1}. ${line.name}`, type: "line", data: line.data,
            sampling: "lttb", showSymbol: line.data.length === 1,
            itemStyle: { color: color(comparisonColors[index]) },
            lineStyle: { width: 2, type: index % 2 ? "dashed" : "solid", color: color(comparisonColors[index]) },
          })),
        });
      };
      render();
      const resize = new ResizeObserver(() => chart.resize());
      resize.observe(node);
      const theme = new MutationObserver(render);
      theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
      cleanup = () => { resize.disconnect(); theme.disconnect(); chart.dispose(); };
    });
    return () => { disposed = true; cleanup?.(); };
  }, [ref, lines, start, end]);
}
