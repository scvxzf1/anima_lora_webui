import { useEffect, type RefObject } from "react";

export function useMetricsCanvas(ref: RefObject<HTMLDivElement | null>, data: [number, number][], timeAxis: boolean) {
  useEffect(() => {
    const node = ref.current;
    if (!node || !node.clientWidth || !data.length) return;
    let disposed = false;
    let cleanup: (() => void) | undefined;
    void import("./metricsChartRuntime").then(({ init }) => {
      if (disposed) return;
      const chart = init(node);
      const render = () => {
        const style = getComputedStyle(node);
        chart.setOption({
          animation: false,
          grid: { left: 58, right: 20, bottom: 35, top: 20 },
          tooltip: { trigger: "axis", renderMode: "richText" },
          xAxis: { type: timeAxis ? "time" : "value", axisLabel: { color: style.getPropertyValue("--muted") }, splitLine: { show: false } },
          yAxis: { type: "value", scale: true, axisLabel: { color: style.getPropertyValue("--muted") }, splitLine: { lineStyle: { color: style.getPropertyValue("--line") } } },
          series: [{ type: "line", data, sampling: "lttb", showSymbol: false, lineStyle: { width: 2, color: style.getPropertyValue("--accent") } }],
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
  }, [ref, data, timeAxis]);
}
