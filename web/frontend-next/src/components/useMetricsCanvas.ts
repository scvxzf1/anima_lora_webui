import { useEffect, type RefObject } from "react";

export function useMetricsCanvas(ref: RefObject<HTMLDivElement | null>, data: [number, number][], timeAxis: boolean, metricName: string, axis: string, enabled = true, valueVisible = true) {
  useEffect(() => {
    const node = ref.current;
    if (!enabled || !node || !node.clientWidth || !data.length) return;
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
          tooltip: {
            trigger: "axis",
            renderMode: "html",
            confine: true,
            formatter: (items: { value?: unknown }[] | { value?: unknown }) => {
              const item = Array.isArray(items) ? items[0] : items;
              const value = Array.isArray(item?.value) ? item.value : [];
              const axisValue = value[0];
              const metricValue = value[1];
              const content = document.createElement("div");
              const axisRow = document.createElement("div");
              const metric = document.createElement("div");
              const axisName = axis === "time" ? "时间" : axis === "step" ? "STEP" : "采样序号";
              axisRow.textContent = `${axisName}: ${axis === "time" && typeof axisValue === "number" ? new Date(axisValue).toLocaleString() : String(axisValue ?? "")}`;
              metric.textContent = `${metricName}: ${typeof metricValue === "number" ? metricValue.toPrecision(4) : String(metricValue ?? "")}`;
              content.append(axisRow);
              if (valueVisible) content.append(metric);
              return content;
            },
          },
          xAxis: { type: timeAxis ? "time" : "value", axisLabel: { color: style.getPropertyValue("--muted") }, splitLine: { show: false } },
          yAxis: { type: "value", scale: true, axisLabel: { color: style.getPropertyValue("--muted") }, splitLine: { lineStyle: { color: style.getPropertyValue("--line") } } },
          series: [{ type: "line", data, sampling: "lttb", showSymbol: false, lineStyle: { width: 2, color: style.getPropertyValue("--accent") } }],
        });
      };
      render();
      const resize = new ResizeObserver(() => {
        chart.dispatchAction({ type: "hideTip" });
        chart.resize();
      });
      resize.observe(node);
      const theme = new MutationObserver(render);
      theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
      cleanup = () => { resize.disconnect(); theme.disconnect(); chart.dispose(); };
    });
    return () => { disposed = true; cleanup?.(); };
  }, [ref, data, timeAxis, metricName, enabled, valueVisible]);
}
