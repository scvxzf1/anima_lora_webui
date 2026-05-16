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
        this.emptyText = options.emptyText || '等待 loss 数据...';
        this.pixelRatio = 1;
        this.resize();
        window.addEventListener('resize', () => this._resize());
        if (window.ResizeObserver) {
            this.resizeObserver = new ResizeObserver(() => this._resize());
            this.resizeObserver.observe(this.canvas.parentElement);
        }
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

    push(step, value) {
        const nextStep = Number(step);
        const nextValue = Number(value);
        if (!Number.isFinite(nextStep) || !Number.isFinite(nextValue)) return;
        const last = this.data[this.data.length - 1];
        if (last && last.step === nextStep) {
            last.value = nextValue;
        } else {
            this.data.push({ step: nextStep, value: nextValue });
        }
        if (this.data.length > this.maxPoints) {
            this.data.shift();
        }
        this.render();
    }

    clear() {
        this.data = [];
        this.render();
    }

    setData(points) {
        this.data = [];
        for (const point of points || []) {
            const step = Number(point.step);
            const value = Number(point.value ?? point.loss);
            if (Number.isFinite(step) && Number.isFinite(value)) {
                this.data.push({ step, value });
            }
        }
        if (this.data.length > this.maxPoints) {
            this.data = this.data.slice(-this.maxPoints);
        }
        this.render();
    }

    render() {
        const ctx = this.ctx;
        const w = this.canvas.width / this.pixelRatio;
        const h = this.canvas.height / this.pixelRatio;
        const pad = { top: 24, right: 58, bottom: 28, left: 12 };

        ctx.clearRect(0, 0, w, h);
        ctx.save();
        ctx.textBaseline = 'middle';

        if (this.data.length < 2) {
            ctx.fillStyle = '#8892a4';
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

        // Grid lines
        ctx.strokeStyle = '#2a3a5e';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const y = pad.top + (plotH * i / 4);
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(w - pad.right, y);
            ctx.stroke();
        }

        // Y-axis labels
        ctx.fillStyle = '#8892a4';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
            const val = maxV - (rangeV * i / 4);
            const y = pad.top + (plotH * i / 4);
            ctx.fillText(val.toFixed(4), w - 5, y + 3);
        }

        // Line
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 1.5;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < this.data.length; i++) {
            const x = pad.left + (i / (this.data.length - 1)) * plotW;
            const y = pad.top + ((maxV - this.data[i].value) / rangeV) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Current value
        const last = this.data[this.data.length - 1];
        ctx.fillStyle = this.color;
        ctx.font = 'bold 11px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(`${this.label}: ${last.value.toFixed(5)}`, pad.left + 5, 14);

        // Step range
        ctx.fillStyle = '#8892a4';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`step ${this.data[0].step}`, pad.left, h - 5);
        ctx.textAlign = 'right';
        ctx.fillText(`step ${last.step}`, w - pad.right, h - 5);
        ctx.restore();
    }
}
