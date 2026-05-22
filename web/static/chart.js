/**
 * Lightweight Canvas chart for loss curve visualization.
 */
class MetricsChart {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.data = [];
        this.maxPoints = options.maxPoints || 300;
        this.label = options.label || 'Loss';
        this.color = options.color || '#4fc3f7';
        this.gridColor = options.gridColor || options.grid || '#2a3a5e';
        this.textColor = options.textColor || options.text || '#8892a4';
        this.tooltipBg = options.tooltipBg || options.tooltipBackground || '#16213e';
        this.tooltipBorder = options.tooltipBorder || options.border || '#2a3a5e';
        this.tooltipText = options.tooltipText || options.text || '#e0e0e0';
        this.highlightColor = options.highlightColor || options.highlight || '#f0c36a';
        this.crosshairColor = options.crosshairColor || options.crosshair || this.color;
        this.emptyText = options.emptyText || '等待 loss 数据...';
        this.xLabel = options.xLabel || 'step';
        this.hoverIndex = null;
        this.useStepScale = false;
        this.xRange = null;
        this.boundDocumentMouseMove = (event) => this._handleDocumentMouseMove(event);
        this.pixelRatio = 1;
        this.canvas.addEventListener('mousemove', (event) => this._handleMouseMove(event));
        this.canvas.addEventListener('mouseleave', () => this._clearHover());
        document.addEventListener('mousemove', this.boundDocumentMouseMove);
        this.resize();
        window.addEventListener('resize', () => this._resize());
        if (window.ResizeObserver) {
            this.resizeObserver = new ResizeObserver(() => this._resize());
            this.resizeObserver.observe(this.canvas.parentElement);
        }
    }

    setTheme(theme = {}) {
        this.color = theme.color || this.color;
        this.gridColor = theme.grid || theme.gridColor || this.gridColor;
        this.textColor = theme.text || theme.textColor || this.textColor;
        this.tooltipBg = theme.tooltipBg || theme.tooltipBackground || this.tooltipBg;
        this.tooltipBorder = theme.tooltipBorder || theme.border || this.tooltipBorder;
        this.tooltipText = theme.tooltipText || theme.tooltip || theme.textColor || this.tooltipText;
        this.highlightColor = theme.highlight || theme.highlightColor || this.highlightColor;
        this.crosshairColor = theme.crosshair || theme.crosshairColor || this.crosshairColor;
        this.render();
    }

    setXLabel(label) {
        this.xLabel = label || 'step';
        this.render();
    }

    setScaleMode(mode, options = {}) {
        this.useStepScale = mode === 'step';
        this.xRange = this.useStepScale ? this._normalizeRange(options.xRange) : null;
        this.render();
    }

    resize() {
        requestAnimationFrame(() => this._resize());
    }

    _resize() {
        const rect = this.canvas.getBoundingClientRect();
        const cssWidth = Math.max(240, Math.floor(rect.width || this.canvas.parentElement.clientWidth || 600));
        const cssHeight = Math.max(180, Math.floor(rect.height || 200));
        const ratio = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
        this.pixelRatio = ratio;
        this.canvas.width = Math.round(cssWidth * ratio);
        this.canvas.height = Math.round(cssHeight * ratio);
        this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        this.render();
    }

    push(step, value, metadata = {}) {
        const normalized = this._normalizePoint({ ...metadata, step, value });
        if (!normalized) return;
        const last = this.data[this.data.length - 1];
        if (last && last.step === normalized.step) {
            Object.assign(last, normalized);
        } else {
            this.data.push(normalized);
        }
        if (this.data.length > this.maxPoints) {
            this.data.shift();
        }
        this._sanitizeHover();
        this.render();
    }

    clear() {
        this.data = [];
        this.hoverIndex = null;
        this.render();
    }

    setData(points, options = {}) {
        this.data = [];
        for (const point of points || []) {
            const normalized = this._normalizePoint(point);
            if (normalized) this.data.push(normalized);
        }
        if (!options.keepAll && this.data.length > this.maxPoints) {
            this.data = this.data.slice(-this.maxPoints);
        }
        this._sanitizeHover();
        this.render();
    }

    _normalizePoint(point = {}) {
        const step = Number(point.step);
        const value = Number(point.value ?? point.loss);
        if (!Number.isFinite(step) || !Number.isFinite(value)) return null;
        const normalized = { ...point, step, value };
        delete normalized.loss;
        return normalized;
    }

    _padding() {
        return { top: 24, right: 58, bottom: 28, left: 12 };
    }

    _sanitizeHover() {
        if (this.hoverIndex !== null && (this.hoverIndex < 0 || this.hoverIndex >= this.data.length)) {
            this.hoverIndex = null;
        }
    }

    _clearHover() {
        if (this.hoverIndex === null) return;
        this.hoverIndex = null;
        this.render();
    }

    _handleMouseMove(event) {
        if (this.data.length < 2) {
            this._clearHover();
            return;
        }
        const rect = this.canvas.getBoundingClientRect();
        const pad = this._padding();
        const plotW = rect.width - pad.left - pad.right;
        const plotH = rect.height - pad.top - pad.bottom;
        if (plotW <= 0 || plotH <= 0) {
            this._clearHover();
            return;
        }

        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const inPlot = x >= pad.left && x <= rect.width - pad.right && y >= pad.top && y <= rect.height - pad.bottom;
        if (!inPlot) {
            this._clearHover();
            return;
        }

        const ratio = (x - pad.left) / plotW;
        const nextIndex = this.useStepScale
            ? this._nearestIndexByX(x, { pad, plotW })
            : Math.max(0, Math.min(this.data.length - 1, Math.round(ratio * (this.data.length - 1))));
        if (nextIndex === this.hoverIndex) return;
        this.hoverIndex = nextIndex;
        this.render();
    }

    _handleDocumentMouseMove(event) {
        if (this.hoverIndex === null) return;
        if (event.target === this.canvas) return;
        this._clearHover();
    }

    render() {
        const ctx = this.ctx;
        const w = this.canvas.width / this.pixelRatio;
        const h = this.canvas.height / this.pixelRatio;
        const pad = this._padding();

        ctx.clearRect(0, 0, w, h);
        ctx.save();
        ctx.textBaseline = 'middle';

        if (this.data.length < 2) {
            ctx.fillStyle = this.textColor;
            ctx.font = '12px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(this.data.length === 1 ? '已收到 1 个 loss 点，等待更多数据...' : this.emptyText, w / 2, h / 2);
            ctx.restore();
            return;
        }

        const values = this.data.map(d => d.value);
        let minV = Math.min(...values);
        let maxV = Math.max(...values);
        if (minV === maxV) { minV -= 0.01; maxV += 0.01; }
        const rangeV = maxV - minV;

        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;
        const xRange = this._xRange();

        // Grid lines
        ctx.strokeStyle = this.gridColor;
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const y = pad.top + (plotH * i / 4);
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(w - pad.right, y);
            ctx.stroke();
        }

        // Y-axis labels
        ctx.fillStyle = this.textColor;
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
            const val = maxV - (rangeV * i / 4);
            const y = pad.top + (plotH * i / 4);
            ctx.fillText(val.toFixed(4), w - 5, y + 3);
        }

        this._drawStageMarkers(ctx, { w, h, pad, plotW, xRange });

        // Line
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 1.5;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        let hasLine = false;
        for (let i = 0; i < this.data.length; i++) {
            const x = this._xForPoint(this.data[i], i, { pad, plotW, xRange });
            const y = pad.top + ((maxV - this.data[i].value) / rangeV) * plotH;
            if (i === 0 || this.data[i].stageBreakBefore) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            hasLine = true;
        }
        if (hasLine) ctx.stroke();

        // Current value
        const last = this.data[this.data.length - 1];
        ctx.fillStyle = this.color;
        ctx.font = 'bold 11px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(`${this.label}: ${last.value.toFixed(5)}`, pad.left + 5, 14);

        // Step range
        ctx.fillStyle = this.textColor;
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`${this.xLabel} ${this._formatStep(xRange.min)}`, pad.left, h - 5);
        ctx.textAlign = 'right';
        ctx.fillText(`${this.xLabel} ${this._formatStep(xRange.max)}`, w - pad.right, h - 5);

        this._drawHover(ctx, { w, h, pad, plotW, plotH, maxV, rangeV, xRange });
        ctx.restore();
    }

    _pointPosition(index, layout) {
        const point = this.data[index];
        if (!point || this.data.length < 2) return null;
        const { pad, plotW, plotH, maxV, rangeV, xRange } = layout;
        return {
            point,
            x: this._xForPoint(point, index, { pad, plotW, xRange }),
            y: pad.top + ((maxV - point.value) / rangeV) * plotH,
        };
    }

    _xForPoint(point, index, layout) {
        const { pad, plotW, xRange } = layout;
        if (!this.useStepScale || !xRange || xRange.max <= xRange.min) {
            return pad.left + (index / (this.data.length - 1)) * plotW;
        }
        const ratio = (point.step - xRange.min) / (xRange.max - xRange.min);
        return pad.left + Math.max(0, Math.min(1, ratio)) * plotW;
    }

    _nearestIndexByX(x, layout) {
        let bestIndex = 0;
        let bestDistance = Infinity;
        for (let i = 0; i < this.data.length; i++) {
            const pointX = this._xForPoint(this.data[i], i, { ...layout, xRange: this._xRange() });
            const distance = Math.abs(pointX - x);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    _xRange() {
        if (this.xRange) return this.xRange;
        const steps = this.data.map((point) => point.step).filter(Number.isFinite);
        const min = Math.min(...steps);
        const max = Math.max(...steps);
        return { min, max };
    }

    _normalizeRange(range) {
        if (!range) return null;
        const min = Number(range.min ?? range.start);
        const max = Number(range.max ?? range.end);
        if (!Number.isFinite(min) || !Number.isFinite(max) || max < min) return null;
        return { min, max };
    }

    _drawStageMarkers(ctx, layout) {
        const { w, h, pad, plotW, xRange } = layout;
        const markers = this.data.filter((point) => point.stageBreakBefore);
        if (!markers.length) return;
        ctx.save();
        ctx.strokeStyle = this.highlightColor;
        ctx.fillStyle = this.highlightColor;
        ctx.globalAlpha = 0.68;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        for (const point of markers) {
            const x = this._xForPoint(point, this.data.indexOf(point), { pad, plotW, xRange });
            ctx.beginPath();
            ctx.moveTo(x, pad.top);
            ctx.lineTo(x, h - pad.bottom);
            ctx.stroke();
            const label = String(point.stageLabel || `任务${point.sourceTaskIndex || point.source_task_index || ''}`);
            if (label.trim()) {
                ctx.setLineDash([]);
                ctx.fillText(this._shorten(label, 22), Math.min(x + 5, w - pad.right - 60), pad.top + 12);
                ctx.setLineDash([3, 4]);
            }
        }
        ctx.restore();
    }

    _drawHover(ctx, layout) {
        if (this.hoverIndex === null) return;
        const position = this._pointPosition(this.hoverIndex, layout);
        if (!position) return;

        const { w, h, pad } = layout;
        const { point, x, y } = position;

        ctx.save();
        ctx.strokeStyle = this.crosshairColor;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.globalAlpha = 0.72;
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, h - pad.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;

        ctx.fillStyle = this.highlightColor;
        ctx.strokeStyle = this.tooltipBg;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        const lines = this._hoverLines(point, this.hoverIndex);
        const font = '11px monospace';
        const lineHeight = 16;
        const paddingX = 8;
        const paddingY = 7;
        ctx.font = font;
        const maxTextWidth = Math.max(...lines.map((line) => ctx.measureText(line).width));
        const boxW = Math.min(Math.max(150, maxTextWidth + paddingX * 2), Math.max(170, w - 16));
        const boxH = lines.length * lineHeight + paddingY * 2;
        let boxX = x + 12;
        if (boxX + boxW > w - 8) boxX = x - boxW - 12;
        boxX = Math.max(8, Math.min(boxX, w - boxW - 8));
        let boxY = y - boxH - 12;
        if (boxY < 8) boxY = y + 12;
        boxY = Math.max(8, Math.min(boxY, h - boxH - 8));

        ctx.fillStyle = this.tooltipBg;
        ctx.strokeStyle = this.tooltipBorder;
        ctx.lineWidth = 1;
        this._roundedRect(ctx, boxX, boxY, boxW, boxH, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = this.tooltipText;
        ctx.font = font;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        for (let i = 0; i < lines.length; i++) {
            ctx.fillText(lines[i], boxX + paddingX, boxY + paddingY + i * lineHeight);
        }
        ctx.restore();
    }

    _hoverLines(point, index) {
        const lines = [`位置: 第 ${index + 1}/${this.data.length} 个 Loss 点`];
        if (this.xLabel && this.xLabel !== 'step') {
            lines.push(`${this.xLabel}: ${this._formatStep(point.step)}`);
        }
        lines.push(`Loss: ${point.value.toFixed(5)}`);
        const rawStep = this._firstFinite(point.rawStep, point.raw_step, point.originalStep, point.trainStep);
        lines.push(`真实步数: ${this._formatStep(point.step)}`);
        if (rawStep !== null && rawStep !== point.step) {
            lines.push(`阶段内步数: ${this._formatStep(rawStep)}`);
        }
        const offset = this._firstFinite(point.displayStepOffset, point.display_step_offset);
        if (offset) lines.push(`续训偏移: +${this._formatStep(offset)}`);
        const sourceLabel = point.sourceTaskLabel || point.source_task_label;
        if (sourceLabel) lines.push(`来源: ${this._shorten(sourceLabel, 34)}`);
        return lines;
    }

    _firstFinite(...values) {
        for (const value of values) {
            const number = Number(value);
            if (Number.isFinite(number)) return number;
        }
        return null;
    }

    _formatStep(value) {
        const number = Number(value);
        return Number.isFinite(number) ? String(number) : '-';
    }

    _shorten(value, maxLength) {
        const text = String(value || '');
        if (text.length <= maxLength) return text;
        return `${text.slice(0, Math.max(0, maxLength - 1))}…`;
    }

    _roundedRect(ctx, x, y, width, height, radius) {
        const r = Math.min(radius, width / 2, height / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + width - r, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + r);
        ctx.lineTo(x + width, y + height - r);
        ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
        ctx.lineTo(x + r, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }
}
