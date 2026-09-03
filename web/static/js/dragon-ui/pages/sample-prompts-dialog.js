import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { confirmDragonDialog } from '../../shared/dialog.js?v=module-bootstrap-20260901-dialog-v1';
import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import {
    blankSamplePromptRow,
    parseSamplePromptRows,
    samplePromptRowFromElement,
    samplePromptsContentNeedsTextMode,
    serializeSamplePromptRow,
} from '../../features/sample-prompts/model.js?v=dragon-ui-20260902-sample-prompts-v2';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

const DIALOG_ID = 'dragon-sample-prompts-dialog';
const UNIFORM_FIELDS = ['width', 'height', 'steps', 'cfg'];

export function renderSamplePromptsFieldControl({ fieldId, name, value, disabled = false }) {
    const path = String(value ?? '');
    const disabledAttribute = disabled ? ' disabled' : '';
    if (path.trim() && !path.trim().toLowerCase().endsWith('.txt')) {
        return `<input class="dragon-input" id="${escapeHtml(fieldId)}" name="${escapeHtml(name)}"
            type="text" autocomplete="off" spellcheck="false" data-key="sample_prompts"${disabledAttribute}
            value="${escapeHtml(path)}" placeholder="样张提示词文件路径"
            title="非 .txt 提示词文件保留为路径编辑。">`;
    }
    return `
        <div class="dragon-sample-prompts-trigger">
            <input class="dragon-input dragon-sample-prompts-path" id="${escapeHtml(fieldId)}"
                   name="${escapeHtml(name)}" type="text" readonly spellcheck="false"
                   data-key="sample_prompts" data-sample-prompts-path data-sample-prompts-open
                   aria-haspopup="dialog" aria-controls="${DIALOG_ID}"${disabledAttribute}
                   value="${escapeHtml(path)}" placeholder="点击管理样张提示词">
            <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button"
                    data-sample-prompts-open aria-haspopup="dialog" aria-controls="${DIALOG_ID}"
                    title="管理样张提示词"${disabledAttribute}>
                ${renderIcon('edit', 'dragon-btn-icon')}<span>管理</span>
            </button>
        </div>`;
}

export function renderSamplePromptsDialog() {
    return `
        <dialog class="dragon-sample-prompts-dialog" id="${DIALOG_ID}"
                data-sample-prompts-dialog aria-labelledby="dragon-sample-prompts-title">
            <div class="dragon-sample-prompts-shell">
                <div class="dragon-sample-prompts-header">
                    <div>
                        <span class="dragon-eyebrow">训练预览</span>
                        <h2 id="dragon-sample-prompts-title">样张提示词</h2>
                        <p data-sample-prompts-file>未选择文件</p>
                    </div>
                    <button class="dragon-icon-button" type="button" data-sample-prompts-action="close"
                            aria-label="关闭" title="关闭">${renderIcon('x')}</button>
                </div>

                <div class="dragon-sample-prompts-toolbar">
                    <div class="dragon-sample-prompts-mode" role="tablist" aria-label="编辑模式">
                        <button id="dragon-sample-prompts-tab-structured" type="button" role="tab"
                                data-sample-prompts-mode="structured" aria-controls="dragon-sample-prompts-panel-structured"
                                aria-selected="true">列表</button>
                        <button id="dragon-sample-prompts-tab-raw" type="button" role="tab"
                                data-sample-prompts-mode="raw" aria-controls="dragon-sample-prompts-panel-raw"
                                aria-selected="false" tabindex="-1">原文</button>
                    </div>
                    <div class="dragon-sample-prompts-state">
                        <output data-sample-prompts-count>0 条</output>
                        <span data-sample-prompts-status role="status" aria-live="polite"></span>
                    </div>
                </div>

                <div class="dragon-sample-prompts-body">
                    <section class="dragon-sample-prompts-uniform" aria-labelledby="dragon-sample-prompts-uniform-title">
                        <h3 id="dragon-sample-prompts-uniform-title">统一参数</h3>
                        <div class="dragon-sample-prompts-uniform-fields">
                            ${uniformField('width', '宽度', 'number', '64', '16')}
                            ${uniformField('height', '高度', 'number', '64', '16')}
                            ${uniformField('steps', '步数', 'number', '1', '1')}
                            ${uniformField('cfg', 'CFG', 'number', '0', '0.1')}
                        </div>
                        <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button"
                                data-sample-prompts-action="apply-uniform">
                            ${renderIcon('settings', 'dragon-btn-icon')}<span>应用到全部</span>
                        </button>
                    </section>

                    <div class="dragon-sample-prompts-panel" id="dragon-sample-prompts-panel-structured"
                         data-sample-prompts-panel="structured" role="tabpanel"
                         aria-labelledby="dragon-sample-prompts-tab-structured">
                        <div class="dragon-sample-prompts-rows" data-sample-prompts-rows></div>
                        <button class="dragon-btn dragon-btn-secondary dragon-sample-prompts-add" type="button"
                                data-sample-prompts-action="add">
                            ${renderIcon('plus', 'dragon-btn-icon')}<span>添加提示词</span>
                        </button>
                    </div>

                    <div class="dragon-sample-prompts-panel" id="dragon-sample-prompts-panel-raw"
                         data-sample-prompts-panel="raw" role="tabpanel"
                         aria-labelledby="dragon-sample-prompts-tab-raw" hidden>
                        <label class="dragon-sample-prompts-raw-field">
                            <span>提示词文件原文</span>
                            <textarea data-sample-prompts-raw spellcheck="false"></textarea>
                        </label>
                    </div>
                </div>

                <div class="dragon-sample-prompts-footer">
                    <button class="dragon-btn dragon-btn-secondary" type="button"
                            data-sample-prompts-action="close">取消</button>
                    <button class="dragon-btn dragon-btn-primary" type="button"
                            data-sample-prompts-action="save" disabled>
                        ${renderIcon('save', 'dragon-btn-icon')}<span>保存提示词</span>
                    </button>
                </div>
            </div>
        </dialog>`;
}

function uniformField(field, label, type, min, step) {
    return `<label><span>${label}</span><input type="${type}" min="${min}" step="${step}"
        inputmode="decimal" data-sample-prompts-uniform="${field}" placeholder="多值"></label>`;
}

export function applyUniformSamplePromptValues(rows, values) {
    const normalized = Object.fromEntries(UNIFORM_FIELDS.map((field) => [field, String(values?.[field] ?? '').trim()]));
    return rows.map((row) => {
        const next = { ...row };
        UNIFORM_FIELDS.forEach((field) => {
            if (normalized[field] !== '') next[field] = normalized[field];
        });
        return next;
    });
}

export function commonSamplePromptValue(rows, field) {
    const values = new Set(rows.map((row) => String(row?.[field] ?? '').trim()));
    return values.size === 1 ? { value: [...values][0], mixed: false } : { value: '', mixed: true };
}

export function serializeStructuredSamplePrompts(rows, originalContent = '') {
    const promptLines = rows.map(serializeSamplePromptRow).filter(Boolean);
    const sourceLines = String(originalContent || '').split(/\r?\n/);
    const hasComments = sourceLines.some((line) => line.trim().startsWith('#'));
    if (!hasComments) return promptLines.join('\n');
    const preserved = sourceLines.filter((line) => !line.trim() || line.trim().startsWith('#'));
    while (preserved.length && !preserved[0].trim()) preserved.shift();
    while (preserved.length && !preserved[preserved.length - 1].trim()) preserved.pop();
    return [...preserved, ...(preserved.length && promptLines.length ? [''] : []), ...promptLines].join('\n');
}

export function validateSamplePromptRows(rows) {
    for (let index = 0; index < rows.length; index += 1) {
        const row = rows[index];
        const hasParameters = Object.entries(row).some(([key, value]) => key !== 'prompt' && String(value || '').trim());
        if (!String(row.prompt || '').trim() && hasParameters) {
            return validationError(index, 'prompt', `第 ${index + 1} 条缺少提示词`);
        }
        for (const field of ['width', 'height']) {
            const error = validateNumber(row[field], { integer: true, min: 64, label: field === 'width' ? '宽度' : '高度' });
            if (error) return validationError(index, field, `第 ${index + 1} 条${error}`);
        }
        const stepsError = validateNumber(row.steps, { integer: true, min: 1, max: 1000, label: '步数' });
        if (stepsError) return validationError(index, 'steps', `第 ${index + 1} 条${stepsError}`);
        const cfgError = validateNumber(row.cfg, { decimalText: true, min: 0, label: 'CFG' });
        if (cfgError) return validationError(index, 'cfg', `第 ${index + 1} 条${cfgError}`);
        const seedError = validateNumber(row.seed, { integer: true, min: 0, label: '种子' });
        if (seedError) return validationError(index, 'seed', `第 ${index + 1} 条${seedError}`);
        const flowShiftError = validateNumber(row.flow_shift, { decimalText: true, min: 0, label: 'Flow Shift' });
        if (flowShiftError) return validationError(index, 'flow_shift', `第 ${index + 1} 条${flowShiftError}`);
    }
    return { ok: true };
}

function validateNumber(value, { decimalText = false, integer = false, min = null, max = null, label }) {
    const text = String(value ?? '').trim();
    if (!text) return '';
    if (decimalText && !/^\d+(?:\.\d+)?$/.test(text)) return `${label}格式不正确`;
    const number = Number(text);
    if (!Number.isFinite(number) || (integer && !Number.isInteger(number))) return `${label}格式不正确`;
    if (min != null && number < min) return `${label}不能小于 ${min}`;
    if (max != null && number > max) return `${label}不能大于 ${max}`;
    return '';
}

function validationError(index, field, message) {
    return { ok: false, index, field, message };
}

export function bindSamplePromptsDialog(root, { trainingContext = {}, apiClient = null } = {}) {
    const dialog = root.querySelector('[data-sample-prompts-dialog]');
    const pathInput = root.querySelector('[data-sample-prompts-path]');
    const triggers = [...root.querySelectorAll('[data-sample-prompts-open]')];
    if (!dialog || !pathInput || !triggers.length) return () => {};
    const controller = new SamplePromptsDialogController({
        api: apiClient || createApiClient(),
        dialog,
        pathInput,
        trainingContext,
        triggers,
    });
    return controller.bind();
}

class SamplePromptsDialogController {
    constructor({ api, dialog, pathInput, trainingContext, triggers }) {
        this.api = api;
        this.dialog = dialog;
        this.pathInput = pathInput;
        this.trainingContext = trainingContext;
        this.triggers = triggers;
        this.elements = collectDialogElements(dialog);
        this.state = createDialogState(pathInput.value);
        this.triggerBindings = [];
        this.disposed = false;
        this.modeSequence = 0;
        this.opener = null;
        this.requestSequence = 0;
        this.handleAction = this.handleAction.bind(this);
        this.handleMode = this.handleMode.bind(this);
        this.handleModeKeydown = this.handleModeKeydown.bind(this);
        this.handleBackdrop = this.handleBackdrop.bind(this);
        this.handleInput = this.handleInput.bind(this);
        this.handleCancel = this.handleCancel.bind(this);
    }

    bind() {
        if (document.body && this.dialog.parentElement !== document.body) {
            document.body.appendChild(this.dialog);
        }
        this.triggerBindings = this.triggers.map((trigger) => this.bindTrigger(trigger));
        this.dialog.addEventListener('click', this.handleAction);
        this.dialog.addEventListener('click', this.handleMode);
        this.dialog.addEventListener('keydown', this.handleModeKeydown);
        this.dialog.addEventListener('click', this.handleBackdrop);
        this.dialog.addEventListener('input', this.handleInput);
        this.dialog.addEventListener('cancel', this.handleCancel);
        return () => this.destroy();
    }

    bindTrigger(trigger) {
        const click = (event) => { event.preventDefault(); this.open(trigger); };
        const keydown = (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            this.open(trigger);
        };
        trigger.addEventListener('click', click);
        if (trigger.matches('input')) trigger.addEventListener('keydown', keydown);
        return { trigger, click, keydown };
    }

    setStatus(message, tone = '') {
        this.elements.status.textContent = message || '';
        this.elements.status.dataset.tone = tone;
    }

    setBusy(busy, operation = '') {
        this.state.busy = busy;
        this.state.operation = busy ? operation : '';
        this.dialog.setAttribute('aria-busy', String(busy));
        this.elements.modes.forEach((button) => { button.disabled = busy; });
        this.elements.closeButtons.forEach((button) => { button.disabled = busy && operation === 'save'; });
        this.syncSaveState();
    }

    markDirty() {
        this.state.dirty = true;
        this.syncSaveState();
        this.setStatus('未保存', 'warning');
    }

    syncSaveState() {
        this.elements.save.disabled = this.state.busy || !this.state.loaded || !this.state.dirty;
    }

    renderRows(rows, { syncUniform = false } = {}) {
        const normalized = rows.length ? rows : [blankSamplePromptRow()];
        this.elements.rows.innerHTML = normalized.map(renderPromptRow).join('');
        this.syncCount(normalized);
        if (syncUniform) syncUniformFields(this.elements.uniform, normalized);
    }

    syncCount(rows = null) {
        const prompts = rows || collectPromptRows(this.elements.rows);
        this.elements.count.textContent = `${prompts.filter((row) => String(row.prompt || '').trim()).length} 条`;
    }

    async setMode(mode) {
        const sequence = ++this.modeSequence;
        if (mode === this.state.mode) return true;
        if (mode === 'raw') {
            const rows = collectPromptRows(this.elements.rows);
            this.elements.raw.value = this.state.dirty
                ? serializeStructuredSamplePrompts(rows, this.state.sourceContent)
                : this.state.sourceContent;
        } else if (!(await this.allowStructuredMode())) {
            return false;
        }
        if (this.disposed || sequence !== this.modeSequence) return false;
        if (mode === 'structured') {
            this.renderRows(parseSamplePromptRows(this.elements.raw.value), { syncUniform: true });
        }
        this.state.mode = mode;
        syncModeUi(this.elements, mode);
        return true;
    }

    async allowStructuredMode() {
        if (!samplePromptsContentNeedsTextMode(this.elements.raw.value) || this.state.rawWarningAccepted) return true;
        const confirmed = await confirmDragonDialog({
            eyebrow: '切换编辑模式',
            title: '转为列表编辑？',
            message: '原文中包含注释、空行或非标准格式。',
            description: '修改列表后会整理行格式，原文模式可保留完整排版。',
            tone: 'warning',
            icon: 'edit',
            confirmText: '继续转换',
        });
        if (!confirmed || this.disposed || !this.dialog.isConnected) return false;
        this.state.rawWarningAccepted = true;
        return true;
    }

    async load() {
        const request = ++this.requestSequence;
        this.state.loaded = false;
        this.state.dirty = false;
        this.setBusy(true, 'load');
        this.setStatus('正在读取…');
        try {
            const file = String(this.pathInput.value || '').trim();
            const query = file ? `?file=${encodeURIComponent(file)}` : '';
            const response = await this.api(`/api/config/sample-prompts${query}`);
            if (!this.requestIsCurrent(request)) return false;
            if (!response || response.ok === false) throw new Error(response?.error || '读取提示词失败');
            this.applyLoadedResponse(response, file);
            this.setStatus('已加载', 'success');
            return true;
        } catch (error) {
            if (!this.requestIsCurrent(request)) return false;
            this.state.loaded = false;
            this.setStatus(error.message || String(error), 'error');
            return false;
        } finally {
            if (this.requestIsCurrent(request)) this.setBusy(false);
        }
    }

    requestIsCurrent(request) {
        return !this.disposed && request === this.requestSequence;
    }

    applyLoadedResponse(response, fallbackFile) {
        this.state.currentFile = String(response.file || fallbackFile);
        this.state.sourceContent = String(response.content || '');
        this.state.mode = samplePromptsContentNeedsTextMode(this.state.sourceContent) ? 'raw' : 'structured';
        this.state.loaded = true;
        this.state.rawWarningAccepted = false;
        this.elements.file.textContent = this.state.currentFile || '未选择文件';
        this.elements.file.title = this.state.currentFile;
        this.elements.raw.value = this.state.sourceContent;
        this.renderRows(parseSamplePromptRows(this.state.sourceContent), { syncUniform: true });
        syncModeUi(this.elements, this.state.mode);
    }

    async open(opener = null) {
        if (this.dialog.open || this.disposed) return;
        this.opener = opener || document.activeElement;
        if (typeof this.dialog.showModal === 'function') this.dialog.showModal();
        else this.dialog.setAttribute('open', '');
        const loaded = await this.load();
        if (!loaded || this.disposed || !this.dialog.open) return;
        const target = this.state.mode === 'raw'
            ? this.elements.raw
            : this.elements.rows.querySelector('[data-sample-prompt-field="prompt"]');
        target?.focus({ preventScroll: true });
    }

    async requestClose() {
        if (this.state.operation === 'save') {
            this.setStatus('正在保存，完成后可关闭', 'warning');
            return;
        }
        if (this.state.dirty && !(await this.confirmDiscard())) return;
        if (this.disposed || !this.dialog.isConnected) return;
        this.requestSequence += 1;
        this.setBusy(false);
        if (typeof this.dialog.close === 'function') this.dialog.close('cancel');
        else this.dialog.removeAttribute('open');
        if (this.opener?.isConnected) this.opener.focus({ preventScroll: true });
    }

    async confirmDiscard() {
        return confirmDragonDialog({
            eyebrow: '未保存提示词',
            title: '放弃本次修改？',
            message: '当前提示词文件尚未保存。',
            description: '关闭后本次添加、删除和参数调整都会丢失。',
            tone: 'warning',
            icon: 'x',
            confirmText: '放弃修改',
        });
    }

    structuredContent() {
        const rows = collectPromptRows(this.elements.rows);
        const validation = validateSamplePromptRows(rows);
        if (!validation.ok) {
            this.setStatus(validation.message, 'error');
            this.elements.rows.querySelectorAll('[data-sample-prompt-row]')[validation.index]
                ?.querySelector(`[data-sample-prompt-field="${validation.field}"]`)?.focus();
            return null;
        }
        return this.state.dirty
            ? serializeStructuredSamplePrompts(rows, this.state.sourceContent)
            : this.state.sourceContent;
    }

    async save() {
        const content = this.state.mode === 'structured' ? this.structuredContent() : this.elements.raw.value;
        if (content == null) return;
        const request = ++this.requestSequence;
        this.setBusy(true, 'save');
        this.setStatus('保存中…');
        try {
            const response = await this.api('/api/config/sample-prompts', {
                method: 'PUT',
                body: JSON.stringify(this.savePayload(content)),
            });
            if (!this.requestIsCurrent(request)) return;
            if (!response || response.ok === false) throw new Error(response?.error || '保存提示词失败');
            this.applySavedResponse(response, content);
            this.setStatus(response.message || '已保存', 'success');
        } catch (error) {
            if (!this.requestIsCurrent(request)) return;
            this.setStatus(error.message || String(error), 'error');
        } finally {
            if (this.requestIsCurrent(request)) this.setBusy(false);
        }
    }

    savePayload(content) {
        return {
            file: this.state.currentFile || String(this.pathInput.value || '').trim(),
            train_config_file: this.trainingContext.configFile || null,
            content,
        };
    }

    applySavedResponse(response, content) {
        this.state.currentFile = String(response.file || this.state.currentFile);
        this.state.sourceContent = String(response.content ?? content);
        this.state.dirty = false;
        this.elements.raw.value = this.state.sourceContent;
        this.elements.file.textContent = this.state.currentFile || '未选择文件';
        this.elements.file.title = this.state.currentFile;
        this.pathInput.value = this.state.currentFile;
        this.pathInput.dispatchEvent(new Event('input', { bubbles: true }));
        const pathTooltip = this.pathInput.closest('[data-config-field-key]')?.querySelector('.dragon-config-path-tooltip');
        if (pathTooltip) pathTooltip.textContent = this.state.currentFile;
    }

    addRow() {
        const rows = collectPromptRows(this.elements.rows);
        const defaults = readUniformFields(this.elements.uniform);
        this.renderRows([...rows, ...applyUniformSamplePromptValues([blankSamplePromptRow()], defaults)]);
        this.markDirty();
        this.elements.rows.querySelector('[data-sample-prompt-row]:last-child [data-sample-prompt-field="prompt"]')?.focus();
    }

    removeRow(event) {
        const rows = collectPromptRows(this.elements.rows);
        const index = Number(event.target.closest('[data-sample-prompt-row]')?.dataset.samplePromptIndex);
        if (Number.isInteger(index)) rows.splice(index, 1);
        this.renderRows(rows);
        this.markDirty();
    }

    applyUniform() {
        const values = readUniformFields(this.elements.uniform);
        if (!UNIFORM_FIELDS.some((field) => values[field] !== '')) {
            this.setStatus('请先填写至少一项统一参数', 'error');
            return;
        }
        this.renderRows(applyUniformSamplePromptValues(collectPromptRows(this.elements.rows), values));
        this.markDirty();
    }

    async handleAction(event) {
        const action = event.target.closest('[data-sample-prompts-action]')?.dataset.samplePromptsAction;
        if (action === 'close') await this.requestClose();
        if (action === 'save') await this.save();
        if (action === 'add') this.addRow();
        if (action === 'remove') this.removeRow(event);
        if (action === 'apply-uniform') this.applyUniform();
    }

    async handleMode(event) {
        const mode = event.target.closest('[data-sample-prompts-mode]')?.dataset.samplePromptsMode;
        if (!mode) return;
        const changed = await this.setMode(mode);
        if (!changed) this.elements.modes.find((button) => button.getAttribute('aria-selected') === 'true')?.focus();
    }

    async handleModeKeydown(event) {
        const current = event.target.closest('[data-sample-prompts-mode]');
        if (!current || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const modes = this.elements.modes;
        const currentIndex = modes.indexOf(current);
        const targetIndex = event.key === 'Home' ? 0
            : event.key === 'End' ? modes.length - 1
                : (currentIndex + (event.key === 'ArrowRight' ? 1 : -1) + modes.length) % modes.length;
        const changed = await this.setMode(modes[targetIndex]?.dataset.samplePromptsMode);
        const focusTarget = changed
            ? modes[targetIndex]
            : modes.find((button) => button.getAttribute('aria-selected') === 'true');
        focusTarget?.focus();
    }

    handleInput(event) {
        if (!event.target.matches('[data-sample-prompts-raw], [data-sample-prompt-field]')) return;
        this.markDirty();
        if (event.target.matches('[data-sample-prompts-raw]')) {
            this.syncCount(parseSamplePromptRows(event.target.value));
        } else if (event.target.dataset.samplePromptField === 'prompt') {
            this.syncCount();
        }
    }

    handleCancel(event) {
        event.preventDefault();
        this.requestClose();
    }

    handleBackdrop(event) {
        if (event.target === this.dialog) this.requestClose();
    }

    destroy() {
        this.disposed = true;
        this.modeSequence += 1;
        this.requestSequence += 1;
        this.triggerBindings.forEach(({ trigger, click, keydown }) => {
            trigger.removeEventListener('click', click);
            trigger.removeEventListener('keydown', keydown);
        });
        this.dialog.removeEventListener('click', this.handleAction);
        this.dialog.removeEventListener('click', this.handleMode);
        this.dialog.removeEventListener('keydown', this.handleModeKeydown);
        this.dialog.removeEventListener('click', this.handleBackdrop);
        this.dialog.removeEventListener('input', this.handleInput);
        this.dialog.removeEventListener('cancel', this.handleCancel);
        if (this.dialog.open && typeof this.dialog.close === 'function') this.dialog.close();
        this.dialog.remove();
    }
}

function createDialogState(file) {
    return {
        mode: 'structured',
        sourceContent: '',
        currentFile: String(file || ''),
        loaded: false,
        dirty: false,
        busy: false,
        operation: '',
        rawWarningAccepted: false,
    };
}

function collectDialogElements(dialog) {
    return {
        file: dialog.querySelector('[data-sample-prompts-file]'),
        status: dialog.querySelector('[data-sample-prompts-status]'),
        count: dialog.querySelector('[data-sample-prompts-count]'),
        rows: dialog.querySelector('[data-sample-prompts-rows]'),
        raw: dialog.querySelector('[data-sample-prompts-raw]'),
        save: dialog.querySelector('[data-sample-prompts-action="save"]'),
        closeButtons: [...dialog.querySelectorAll('[data-sample-prompts-action="close"]')],
        uniform: [...dialog.querySelectorAll('[data-sample-prompts-uniform]')],
        uniformSection: dialog.querySelector('.dragon-sample-prompts-uniform'),
        panels: [...dialog.querySelectorAll('[data-sample-prompts-panel]')],
        modes: [...dialog.querySelectorAll('[data-sample-prompts-mode]')],
    };
}

function syncModeUi(elements, mode) {
    if (elements.uniformSection) elements.uniformSection.hidden = mode !== 'structured';
    elements.panels.forEach((panel) => { panel.hidden = panel.dataset.samplePromptsPanel !== mode; });
    elements.modes.forEach((button) => {
        const selected = button.dataset.samplePromptsMode === mode;
        button.setAttribute('aria-selected', String(selected));
        button.tabIndex = selected ? 0 : -1;
        button.dataset.active = String(selected);
    });
}

function syncUniformFields(inputs, rows) {
    inputs.forEach((input) => {
        const common = commonSamplePromptValue(rows, input.dataset.samplePromptsUniform);
        input.value = common.value;
        input.placeholder = common.mixed ? '多值' : '默认';
    });
}

function readUniformFields(inputs) {
    return Object.fromEntries(inputs.map((input) => [input.dataset.samplePromptsUniform, input.value.trim()]));
}

function collectPromptRows(rowsElement) {
    return [...rowsElement.querySelectorAll('[data-sample-prompt-row]')].map(samplePromptRowFromElement);
}

function renderPromptRow(row, index) {
    return `
        <article class="dragon-sample-prompt-row" data-sample-prompt-row data-sample-prompt-index="${index}">
            <div class="dragon-sample-prompt-row-head">
                <span>提示词 ${index + 1}</span>
                <button class="dragon-icon-button" type="button" data-sample-prompts-action="remove"
                        aria-label="删除提示词 ${index + 1}" title="删除">${renderIcon('trash')}</button>
            </div>
            ${textAreaField('提示词', 'prompt', row.prompt, 'dragon-sample-prompt-main')}
            <div class="dragon-sample-prompt-params">
                ${inputField('宽度', 'width', row.width, 'number', '64', '16')}
                ${inputField('高度', 'height', row.height, 'number', '64', '16')}
                ${inputField('步数', 'steps', row.steps, 'number', '1', '1')}
                ${inputField('CFG', 'cfg', row.cfg, 'number', '0', '0.1')}
            </div>
            <details class="dragon-sample-prompt-more">
                <summary>更多参数</summary>
                <div class="dragon-sample-prompt-more-grid">
                    ${textAreaField('负面提示词', 'negative_prompt', row.negative_prompt, 'dragon-sample-prompt-negative')}
                    ${inputField('种子', 'seed', row.seed, 'number', '0', '1')}
                    ${inputField('Flow Shift', 'flow_shift', row.flow_shift, 'number', '0', '0.1')}
                    ${samplerField(row.sample_sampler)}
                    ${inputField('额外参数', 'extra', row.extra, 'text', '', '', 'dragon-sample-prompt-extra')}
                </div>
            </details>
        </article>`;
}

function textAreaField(label, field, value, className = '') {
    return `<label class="dragon-sample-prompt-field ${className}"><span>${label}</span>
        <textarea data-sample-prompt-field="${field}" spellcheck="false">${escapeHtml(value || '')}</textarea></label>`;
}

function inputField(label, field, value, type = 'text', min = '', step = '', className = '') {
    const numeric = type === 'number';
    return `<label class="dragon-sample-prompt-field ${className}"><span>${label}</span>
        <input type="${type}" data-sample-prompt-field="${field}" value="${escapeHtml(value || '')}"
               ${numeric ? 'inputmode="decimal"' : 'spellcheck="false"'}${min ? ` min="${min}"` : ''}${step ? ` step="${step}"` : ''}></label>`;
}

function samplerField(value) {
    const options = ['', 'euler', 'er_sde', 'lcm'];
    const current = String(value || '');
    if (current && !options.includes(current)) options.splice(1, 0, current);
    return `<label class="dragon-sample-prompt-field"><span>采样器</span>
        <select data-sample-prompt-field="sample_sampler">${options.map((option) =>
            `<option value="${escapeHtml(option)}"${option === current ? ' selected' : ''}>${escapeHtml(option || '默认')}</option>`
        ).join('')}</select></label>`;
}
