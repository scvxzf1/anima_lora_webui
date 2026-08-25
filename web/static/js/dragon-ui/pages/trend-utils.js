/* Small chart primitives shared by live and history training views. */

export function emaValues(values, smoothing = 0) {
    const factor = clamp(Number(smoothing) || 0, 0, .99);
    if (!values.length || factor <= 0) return [...values];
    const alpha = 1 - factor;
    let previous = Number(values[0]);
    return values.map((value, index) => {
        const current = Number(value);
        if (index === 0 || !Number.isFinite(previous)) previous = current;
        else previous = alpha * current + factor * previous;
        return previous;
    });
}

export function smoothSvgPath(points, tension = .2) {
    if (!points.length) return '';
    if (points.length === 1) return `M${pointText(points[0])}`;
    const amount = clamp(Number(tension) || 0, 0, 1);
    if (amount <= 0) return `M${pointText(points[0])}${points.slice(1).map((point) => `L${pointText(point)}`).join('')}`;
    let path = `M${pointText(points[0])}`;
    for (let index = 0; index < points.length - 1; index += 1) {
        const current = points[index];
        const next = points[index + 1];
        const previous = points[index - 1] || current;
        const after = points[index + 2] || next;
        const control1 = [
            current[0] + (next[0] - previous[0]) * amount / 3,
            current[1] + (next[1] - previous[1]) * amount / 3,
        ];
        const control2 = [
            next[0] - (after[0] - current[0]) * amount / 3,
            next[1] - (after[1] - current[1]) * amount / 3,
        ];
        path += `C${pointText(control1)} ${pointText(control2)} ${pointText(next)}`;
    }
    return path;
}

export function areaSvgPath(linePath, points, baseline) {
    if (!linePath || !points.length) return '';
    return `${linePath}L${format(points.at(-1)[0])} ${format(baseline)}L${format(points[0][0])} ${format(baseline)}Z`;
}

function pointText(point) { return `${format(point[0])} ${format(point[1])}`; }
function format(value) { return Number(value).toFixed(2); }
function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
