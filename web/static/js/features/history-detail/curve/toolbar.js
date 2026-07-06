import { optionNode } from '../ui.js?v=module-bootstrap-20260706-1';

export function createHistoryCurveToolbar({ historyCurveState, renderHistoryDetailContent }) {
    function renderHistoryCurveToolbar(allPoints) {
        const toolbar = document.createElement('div');
        toolbar.className = 'history-curve-toolbar';
        toolbar.append(
            historyCurveCheckbox('原始线', 'showRaw'),
            historyCurveCheckbox('平滑线', 'showSmooth'),
            historyCurveSelect('平滑窗口', 'smoothWindow', [
                ['5', '5'],
                ['15', '15'],
                ['31', '31'],
                ['51', '51'],
            ]),
            historyCurveSelect('范围', 'rangeMode', [
                ['all', '全部'],
                ['last100', '最近100点'],
                ['last25', '最近25%'],
                ['custom', '自定义 Step'],
            ]),
        );
        const custom = document.createElement('div');
        custom.className = 'history-curve-custom-range';
        const minStep = allPoints[0]?.step ?? '';
        const maxStep = allPoints[allPoints.length - 1]?.step ?? '';
        custom.append(
            historyCurveNumberInput('起始 Step', 'customStart', minStep),
            historyCurveNumberInput('结束 Step', 'customEnd', maxStep),
        );
        custom.hidden = historyCurveState.rangeMode !== 'custom';
        toolbar.appendChild(custom);
        return toolbar;
    }

    function historyCurveCheckbox(label, key) {
        const wrap = document.createElement('label');
        wrap.className = 'history-curve-toggle';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = Boolean(historyCurveState[key]);
        input.addEventListener('change', () => {
            historyCurveState[key] = input.checked;
            renderHistoryDetailContent();
        });
        const span = document.createElement('span');
        span.textContent = label;
        wrap.append(input, span);
        return wrap;
    }

    function historyCurveSelect(label, key, options) {
        const wrap = document.createElement('label');
        wrap.className = 'history-curve-field';
        const span = document.createElement('span');
        span.textContent = label;
        const select = document.createElement('select');
        select.value = String(historyCurveState[key]);
        for (const [value, text] of options) {
            select.appendChild(optionNode(value, text));
        }
        select.value = String(historyCurveState[key]);
        select.addEventListener('change', () => {
            historyCurveState[key] = key === 'smoothWindow' ? Number(select.value) : select.value;
            historyCurveState.hoverStep = null;
            renderHistoryDetailContent();
        });
        wrap.append(span, select);
        return wrap;
    }

    function historyCurveNumberInput(label, key, fallback) {
        const wrap = document.createElement('label');
        wrap.className = 'history-curve-field';
        const span = document.createElement('span');
        span.textContent = label;
        const input = document.createElement('input');
        input.type = 'number';
        input.placeholder = String(fallback ?? '');
        input.value = historyCurveState[key] || '';
        input.addEventListener('change', () => {
            historyCurveState[key] = input.value;
            historyCurveState.hoverStep = null;
            renderHistoryDetailContent();
        });
        wrap.append(span, input);
        return wrap;
    }

    return { renderHistoryCurveToolbar };
}
