import {
    IMAGE_TEST_SELECTIVE_LORA_BLOCKS,
    IMAGE_TEST_SELECTIVE_LORA_GROUPS,
    IMAGE_TEST_SELECTIVE_LORA_PRESET_OPTIONS,
    IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MAX,
    IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MIN,
    IMAGE_TEST_SELECTIVE_LORA_STRENGTH_STEP,
    blockStrengthsForImageTestSelectiveLoraPreset,
    clampImageTestSelectiveLoraStrength,
    enabledBlocksForImageTestSelectiveLoraStrengths,
    normalizeImageTestSelectiveLoraBlockStrengths,
    normalizeImageTestSelectiveLoraPreset,
} from './state.js?v=module-bootstrap-20260703-8';

const DEFAULT_LAYER_STRENGTH = 1.0;
const LAYER_ROW_SELECTOR = '[data-image-test-layer-row]';
const LAYER_LAYOUT_SINGLE = 'single';
const LAYER_LAYOUT_DOUBLE = 'double';
const IMAGE_TEST_LAYER_DIALOG_STORAGE_KEY = 'anima.imageTest.layerDialog';
const IMAGE_TEST_LAYER_DIALOG_STORAGE_VERSION = 1;

export function createImageTestSelectiveLoraController({
    storageKey = IMAGE_TEST_LAYER_DIALOG_STORAGE_KEY,
    storage = window.localStorage,
} = {}) {
    let initialized = false;
    let restoring = false;

    function init() {
        if (initialized) return;
        initialized = true;
        restoring = true;
        populatePresetOptions();
        renderLayerRows();
        bindEvents();
        restorePersistedDialogState();
        restoring = false;
        syncMasterState();
        updateSummary();
        setIoStatus('导出会同时写入这里，方便手动微调后再导入。');
        persistDialogState();
    }

    function collectPayload() {
        const blockStrengths = collectBlockStrengths();
        return {
            anima_selective_lora: isEnabled(),
            anima_selective_preset: normalizeImageTestSelectiveLoraPreset(
                document.getElementById('image-test-layer-preset')?.value,
            ),
            anima_selective_strength: '1.0',
            anima_selective_blocks: enabledBlocksForImageTestSelectiveLoraStrengths(blockStrengths),
            anima_selective_block_strengths: blockStrengths,
        };
    }

    function validate(payload = {}) {
        if (!isEnabled()) {
            return '';
        }
        if (String(payload.weight_path || '').trim()) {
            return '';
        }
        return '启用 LoRA 分层加载时，需要先选择一个 LoRA 权重。';
    }

    function bindEvents() {
        document.getElementById('btn-open-image-test-layer-dialog')?.addEventListener('click', openDialog);
        document.getElementById('image-test-layer-enable')?.addEventListener('change', () => {
            syncMasterState();
            updateSummary();
        });
        document.getElementById('image-test-layer-preset')?.addEventListener('change', (event) => {
            applyPreset(event.target.value || 'default');
        });
        document.getElementById('btn-image-test-layer-enable-all')?.addEventListener('click', () => {
            setStrengthMap(
                Object.fromEntries(
                    IMAGE_TEST_SELECTIVE_LORA_BLOCKS.map((blockId) => [blockId, DEFAULT_LAYER_STRENGTH]),
                ),
                { preset: 'custom' },
            );
        });
        document.getElementById('btn-image-test-layer-disable-all')?.addEventListener('click', () => {
            setStrengthMap(
                Object.fromEntries(
                    IMAGE_TEST_SELECTIVE_LORA_BLOCKS.map((blockId) => [blockId, 0]),
                ),
                { preset: 'all_off' },
            );
        });
        document.getElementById('btn-image-test-layer-reset-default')?.addEventListener('click', () => {
            applyPreset('default');
        });
        document.getElementById('btn-image-test-layer-layout-toggle')?.addEventListener('click', toggleLayoutMode);
        document.getElementById('btn-image-test-layer-export')?.addEventListener('click', () => {
            void exportLayerConfig();
        });
        document.getElementById('btn-image-test-layer-import')?.addEventListener('click', () => {
            importLayerConfig();
        });
        document.getElementById('image-test-layer-io-text')?.addEventListener('input', () => {
            persistDialogState();
        });
        document.getElementById('image-test-layer-selection')?.addEventListener('change', handleLayerSelectionChange);
        document.getElementById('image-test-layer-selection')?.addEventListener('input', handleLayerSelectionInput);

        const dialog = document.getElementById('image-test-layer-dialog');
        dialog?.addEventListener('click', (event) => {
            if (event.target === dialog) {
                dialog.close();
            }
        });
    }

    function openDialog() {
        const dialog = document.getElementById('image-test-layer-dialog');
        if (!dialog) return;
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
        document.getElementById('image-test-layer-enable')?.focus({ preventScroll: true });
    }

    function populatePresetOptions() {
        const select = document.getElementById('image-test-layer-preset');
        if (!select) return;
        select.innerHTML = '';
        IMAGE_TEST_SELECTIVE_LORA_PRESET_OPTIONS.forEach((preset) => {
            const option = document.createElement('option');
            option.value = preset.value;
            option.textContent = preset.label;
            select.appendChild(option);
        });
    }

    function restorePersistedDialogState() {
        const enabledInput = document.getElementById('image-test-layer-enable');
        if (enabledInput instanceof HTMLInputElement) {
            enabledInput.checked = false;
        }
        setLayoutMode(LAYER_LAYOUT_SINGLE, { persist: false });
        applyPreset('default');
        setIoText('', { persist: false });

        const stored = readStoredDialogState();
        if (!hasStoredDialogState(stored)) {
            return;
        }

        setLayoutMode(stored.layout, { persist: false });
        const preset = normalizeImageTestSelectiveLoraPreset(stored.preset, 'default');
        if (stored.block_strengths && typeof stored.block_strengths === 'object' && !Array.isArray(stored.block_strengths)) {
            setStrengthMap(stored.block_strengths, { preset });
        } else {
            applyPreset(preset);
        }
        if (enabledInput instanceof HTMLInputElement && Object.hasOwn(stored, 'enabled')) {
            enabledInput.checked = Boolean(stored.enabled);
        }
        if (typeof stored.io_text === 'string') {
            setIoText(stored.io_text, { persist: false });
        }
    }

    function hasStoredDialogState(value) {
        return Boolean(
            value
            && typeof value === 'object'
            && !Array.isArray(value)
            && (
                Object.hasOwn(value, 'enabled')
                || Object.hasOwn(value, 'preset')
                || Object.hasOwn(value, 'block_strengths')
                || Object.hasOwn(value, 'layout')
                || Object.hasOwn(value, 'io_text')
            )
        );
    }

    function readStoredDialogState() {
        try {
            const parsed = JSON.parse(storage.getItem(storageKey) || '{}');
            if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
                return {};
            }
            return parsed;
        } catch (_) {
            return {};
        }
    }

    function persistDialogState() {
        if (!initialized || restoring) return;
        try {
            storage.setItem(storageKey, JSON.stringify({
                version: IMAGE_TEST_LAYER_DIALOG_STORAGE_VERSION,
                ...exportableLayerConfig(),
                layout: currentLayoutMode(),
                io_text: currentIoText(),
            }));
        } catch (_) {
            // 忽略浏览器禁用 localStorage 的情况，当前页面内状态仍然可继续使用。
        }
    }

    function renderLayerRows() {
        IMAGE_TEST_SELECTIVE_LORA_GROUPS.forEach((group) => {
            const container = document.getElementById(group.containerId);
            if (!container) return;
            container.innerHTML = '';
            group.blocks.forEach((blockId, index) => {
                container.appendChild(createLayerRow(blockId, group, index));
            });
        });
    }

    function createLayerRow(blockId, group, index) {
        const row = document.createElement('div');
        row.className = 'image-test-layer-row';
        row.dataset.imageTestLayerRow = blockId;

        const toggle = document.createElement('label');
        toggle.className = 'image-test-layer-row-toggle';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.dataset.role = 'enabled';
        checkbox.value = blockId;
        checkbox.checked = true;
        const toggleLabel = document.createElement('span');
        toggleLabel.textContent = '启用';
        toggle.append(checkbox, toggleLabel);

        const name = document.createElement('div');
        name.className = 'image-test-layer-row-name';
        const strong = document.createElement('strong');
        strong.textContent = blockId;
        const hint = document.createElement('span');
        hint.textContent = `${group.label} · ${indexLabel(blockId, index)}`;
        name.append(strong, hint);

        const sliderWrap = document.createElement('div');
        sliderWrap.className = 'image-test-layer-row-slider';
        const range = document.createElement('input');
        range.type = 'range';
        range.min = String(IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MIN);
        range.max = String(IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MAX);
        range.step = String(IMAGE_TEST_SELECTIVE_LORA_STRENGTH_STEP);
        range.value = String(DEFAULT_LAYER_STRENGTH);
        range.dataset.role = 'range';
        sliderWrap.appendChild(range);

        const numberWrap = document.createElement('label');
        numberWrap.className = 'image-test-layer-row-number';
        const number = document.createElement('input');
        number.type = 'number';
        number.min = String(IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MIN);
        number.max = String(IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MAX);
        number.step = String(IMAGE_TEST_SELECTIVE_LORA_STRENGTH_STEP);
        number.value = formatStrength(DEFAULT_LAYER_STRENGTH);
        number.dataset.role = 'number';
        numberWrap.appendChild(number);

        row.append(toggle, name, sliderWrap, numberWrap);
        return row;
    }

    function indexLabel(blockId, index) {
        if (blockId.startsWith('block_')) return `主干 ${index}`;
        if (blockId.startsWith('llm_adapter_') && blockId !== 'llm_adapter_io') return `Adapter ${index}`;
        if (blockId === 'llm_adapter_io') return '输入 / 输出';
        if (blockId === 'final_layer') return '最终层';
        if (blockId === 't_embedder') return '时间嵌入';
        if (blockId === 'x_embedder') return '图像嵌入';
        return '剩余权重';
    }

    function handleLayerSelectionChange(event) {
        const input = event.target;
        if (!(input instanceof HTMLInputElement)) return;
        const row = input.closest(LAYER_ROW_SELECTOR);
        if (!row) return;
        if (input.dataset.role === 'enabled') {
            handleEnabledToggle(row, input.checked);
            syncPresetFromSelection();
            updateSummary();
            return;
        }
        if (input.dataset.role === 'number') {
            commitRowStrength(row, input.value);
            syncPresetFromSelection();
            updateSummary();
        }
    }

    function handleLayerSelectionInput(event) {
        const input = event.target;
        if (!(input instanceof HTMLInputElement)) return;
        const row = input.closest(LAYER_ROW_SELECTOR);
        if (!row) return;
        if (input.dataset.role === 'range') {
            setRowStrength(row, input.value, { autoEnable: true });
            syncPresetFromSelection();
            updateSummary();
            return;
        }
        if (input.dataset.role === 'number') {
            const raw = String(input.value ?? '').trim();
            if (!raw || raw.endsWith('.')) {
                return;
            }
            setRowStrength(row, raw, { autoEnable: true, syncNumber: false });
            syncPresetFromSelection();
            updateSummary();
        }
    }

    function handleEnabledToggle(row, enabled) {
        const checkbox = row.querySelector('[data-role="enabled"]');
        const lastStrength = clampImageTestSelectiveLoraStrength(row.dataset.lastStrength || DEFAULT_LAYER_STRENGTH, DEFAULT_LAYER_STRENGTH);
        if (enabled && currentRowStrength(row) <= 0) {
            setRowStrength(row, lastStrength > 0 ? lastStrength : DEFAULT_LAYER_STRENGTH, {
                forceEnabled: true,
            });
        }
        if (!enabled && currentRowStrength(row) > 0) {
            row.dataset.lastStrength = formatStrength(currentRowStrength(row));
        }
        if (checkbox) checkbox.checked = enabled;
        updateRowEnabledState(row);
    }

    function setStrengthMap(nextMap, options = {}) {
        const normalized = normalizeImageTestSelectiveLoraBlockStrengths(nextMap, options.preset || 'default');
        queryLayerRows().forEach((row) => {
            const blockId = row.dataset.imageTestLayerRow;
            const value = normalized[blockId] ?? 0;
            setRowStrength(row, value, {
                forceEnabled: value > 0,
            });
        });
        const select = document.getElementById('image-test-layer-preset');
        if (select) {
            select.value = normalizeImageTestSelectiveLoraPreset(options.preset || 'custom', 'custom');
        }
        updateSummary();
    }

    function applyPreset(value) {
        const preset = normalizeImageTestSelectiveLoraPreset(value);
        setStrengthMap(blockStrengthsForImageTestSelectiveLoraPreset(preset), { preset });
    }

    function setRowStrength(row, value, options = {}) {
        const normalized = clampImageTestSelectiveLoraStrength(value, currentRowStrength(row));
        const checkbox = row.querySelector('[data-role="enabled"]');
        const range = row.querySelector('[data-role="range"]');
        const number = row.querySelector('[data-role="number"]');
        if (range) {
            range.value = String(normalized);
        }
        if (number) {
            number.value = options.syncNumber === false
                ? String(value ?? '').trim()
                : formatStrength(normalized);
        }
        if (normalized > 0) {
            row.dataset.lastStrength = formatStrength(normalized);
        }
        const shouldEnable = options.forceEnabled === true
            ? true
            : options.forceEnabled === false
                ? false
                : options.autoEnable === true
                    ? normalized > 0
                    : Boolean(checkbox?.checked);
        if (checkbox) {
            checkbox.checked = shouldEnable;
        }
        updateRowEnabledState(row);
    }

    function commitRowStrength(row, value) {
        const normalized = clampImageTestSelectiveLoraStrength(value, currentRowStrength(row));
        setRowStrength(row, normalized, {
            forceEnabled: normalized > 0,
        });
    }

    function currentRowStrength(row) {
        const number = row.querySelector('[data-role="number"]');
        return clampImageTestSelectiveLoraStrength(number?.value, DEFAULT_LAYER_STRENGTH);
    }

    function updateRowEnabledState(row) {
        const enabled = Boolean(row.querySelector('[data-role="enabled"]')?.checked);
        const range = row.querySelector('[data-role="range"]');
        const number = row.querySelector('[data-role="number"]');
        row.classList.toggle('is-disabled', !enabled);
        if (range) {
            range.disabled = !enabled;
            syncRangeVisual(range, currentRowStrength(row), enabled);
        }
        if (number) number.disabled = !enabled;
    }

    function collectBlockStrengths() {
        return Object.fromEntries(
            queryLayerRows().map((row) => {
                const blockId = row.dataset.imageTestLayerRow;
                const enabled = Boolean(row.querySelector('[data-role="enabled"]')?.checked);
                return [blockId, enabled ? currentRowStrength(row) : 0];
            }),
        );
    }

    function syncPresetFromSelection() {
        const current = collectBlockStrengths();
        const matched = IMAGE_TEST_SELECTIVE_LORA_PRESET_OPTIONS.find((preset) => {
            if (preset.value === 'custom') return false;
            const expected = blockStrengthsForImageTestSelectiveLoraPreset(preset.value);
            return IMAGE_TEST_SELECTIVE_LORA_BLOCKS.every((blockId) => expected[blockId] === current[blockId]);
        });
        const select = document.getElementById('image-test-layer-preset');
        if (select) {
            select.value = matched?.value || 'custom';
        }
    }

    function toggleLayoutMode() {
        setLayoutMode(currentLayoutMode() === LAYER_LAYOUT_DOUBLE ? LAYER_LAYOUT_SINGLE : LAYER_LAYOUT_DOUBLE);
    }

    function setLayoutMode(layout, options = {}) {
        const normalized = layout === LAYER_LAYOUT_DOUBLE ? LAYER_LAYOUT_DOUBLE : LAYER_LAYOUT_SINGLE;
        const body = document.getElementById('image-test-layer-selection');
        const button = document.getElementById('btn-image-test-layer-layout-toggle');
        if (body) {
            body.dataset.layout = normalized;
        }
        if (button) {
            button.dataset.layout = normalized;
            button.setAttribute('aria-pressed', normalized === LAYER_LAYOUT_DOUBLE ? 'true' : 'false');
            button.title = normalized === LAYER_LAYOUT_DOUBLE ? '当前是一行二列，点击切回一行一列。' : '当前是一行一列，点击切到一行二列。';
        }
        setText('image-test-layer-layout-label', normalized === LAYER_LAYOUT_DOUBLE ? '一行二列' : '一行一列');
        if (options.persist !== false) {
            persistDialogState();
        }
    }

    function currentLayoutMode() {
        const body = document.getElementById('image-test-layer-selection');
        return body?.dataset.layout === LAYER_LAYOUT_DOUBLE ? LAYER_LAYOUT_DOUBLE : LAYER_LAYOUT_SINGLE;
    }

    function syncMasterState() {
        const enabled = isEnabled();
        const dialog = document.getElementById('image-test-layer-dialog');
        const body = document.getElementById('image-test-layer-selection');
        document.getElementById('image-test-layer-dialog-tools')?.classList.toggle('is-disabled', !enabled);
        if (body) {
            body.classList.toggle('is-disabled', !enabled);
        }
        if (dialog) {
            dialog.dataset.enabled = enabled ? 'true' : 'false';
        }
    }

    function queryLayerRows() {
        return Array.from(document.querySelectorAll(LAYER_ROW_SELECTOR));
    }

    function isEnabled() {
        return Boolean(document.getElementById('image-test-layer-enable')?.checked);
    }

    function updateSummary() {
        const strengths = collectBlockStrengths();
        const enabledBlocks = enabledBlocksForImageTestSelectiveLoraStrengths(strengths);
        const enabled = isEnabled();
        const preset = currentPresetValue();
        const presetLabel = IMAGE_TEST_SELECTIVE_LORA_PRESET_OPTIONS.find((item) => item.value === preset)?.label || 'Custom';
        const average = enabledBlocks.length
            ? enabledBlocks.reduce((sum, blockId) => sum + strengths[blockId], 0) / enabledBlocks.length
            : 0;
        const summaryText = enabled
            ? `${presetLabel} · ${enabledBlocks.length}/${IMAGE_TEST_SELECTIVE_LORA_BLOCKS.length} 层启用 · 平均 ${formatStrength(average)}x`
            : `已关闭 · 保留 ${enabledBlocks.length} 层配置`;
        const detailText = enabled
            ? (
                enabledBlocks.length
                    ? enabledBlocks.slice(0, 5).map((blockId) => `${blockId} ${formatStrength(strengths[blockId])}x`).join(' · ')
                    : '当前没有启用任何层位。'
            )
            : `未启用，但已保留 ${presetLabel} 的层配置。`;

        setText('image-test-layer-summary', summaryText);
        setText('image-test-layer-inline-summary', detailText);
        setText('image-test-layer-dialog-summary', summaryText);
        setText('image-test-layer-count', `${enabledBlocks.length}/${IMAGE_TEST_SELECTIVE_LORA_BLOCKS.length}`);
        setText('image-test-layer-dialog-count', `${enabledBlocks.length}/${IMAGE_TEST_SELECTIVE_LORA_BLOCKS.length}`);
        setText('image-test-layer-status', enabled ? '分层已启用' : '分层未启用');
        persistDialogState();
    }

    async function exportLayerConfig() {
        const serialized = JSON.stringify(exportableLayerConfig(), null, 2);
        const textarea = document.getElementById('image-test-layer-io-text');
        if (textarea instanceof HTMLTextAreaElement) {
            setIoText(serialized);
            textarea.focus({ preventScroll: true });
            textarea.select();
        }

        let copied = false;
        if (navigator.clipboard?.writeText) {
            try {
                await navigator.clipboard.writeText(serialized);
                copied = true;
            } catch {
                copied = false;
            }
        }
        if (!copied && textarea instanceof HTMLTextAreaElement && document.queryCommandSupported?.('copy')) {
            copied = document.execCommand('copy');
        }
        setIoStatus(
            copied
                ? '当前分层参数已复制到剪切板，并同步写入下方文本框。'
                : '已写入下方文本框；当前环境未允许直接写入剪切板。',
            copied ? 'success' : 'warning',
        );
    }

    function importLayerConfig() {
        const textarea = document.getElementById('image-test-layer-io-text');
        if (!(textarea instanceof HTMLTextAreaElement)) return;
        const raw = textarea.value.trim();
        if (!raw) {
            setIoStatus('请先粘贴要导入的 JSON 参数。', 'error');
            return;
        }

        let parsed;
        try {
            parsed = JSON.parse(raw);
        } catch {
            setIoStatus('导入失败：JSON 格式不合法。', 'error');
            return;
        }

        try {
            applyImportedConfig(parsed);
            setIoText(JSON.stringify(exportableLayerConfig(), null, 2));
            setIoStatus('参数已导入并应用到当前层编辑器。', 'success');
        } catch (error) {
            setIoStatus(`导入失败：${error?.message || '内容不合法。'}`, 'error');
        }
    }

    function applyImportedConfig(parsed) {
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            throw new Error('导入内容必须是一个 JSON 对象。');
        }
        const preset = normalizeImageTestSelectiveLoraPreset(
            parsed.preset ?? parsed.anima_selective_preset,
            'custom',
        );
        const importedStrengths = importedBlockStrengthsFromConfig(parsed);
        if (importedStrengths) {
            setStrengthMap(importedStrengths, { preset });
            syncPresetFromSelection();
        } else if (Object.hasOwn(parsed, 'preset') || Object.hasOwn(parsed, 'anima_selective_preset')) {
            applyPreset(preset);
        } else {
            throw new Error('缺少 preset 或 block_strengths 字段。');
        }

        const enabledInput = document.getElementById('image-test-layer-enable');
        if (enabledInput instanceof HTMLInputElement) {
            enabledInput.checked = importedEnabledFromConfig(parsed, importedStrengths);
        }
        syncMasterState();
        updateSummary();
    }

    function importedBlockStrengthsFromConfig(parsed) {
        const candidate = (
            parsed.block_strengths
            ?? parsed.anima_selective_block_strengths
            ?? plainBlockStrengthMap(parsed)
        );
        if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
            return null;
        }
        return normalizeImageTestSelectiveLoraBlockStrengths(candidate, 'custom');
    }

    function plainBlockStrengthMap(parsed) {
        const entries = IMAGE_TEST_SELECTIVE_LORA_BLOCKS
            .filter((blockId) => Object.hasOwn(parsed, blockId))
            .map((blockId) => [blockId, parsed[blockId]]);
        return entries.length ? Object.fromEntries(entries) : null;
    }

    function importedEnabledFromConfig(parsed, strengths) {
        if (Object.hasOwn(parsed, 'enabled')) {
            return Boolean(parsed.enabled);
        }
        if (Object.hasOwn(parsed, 'anima_selective_lora')) {
            return Boolean(parsed.anima_selective_lora);
        }
        if (strengths) {
            return enabledBlocksForImageTestSelectiveLoraStrengths(strengths).length > 0;
        }
        return isEnabled();
    }

    function exportableLayerConfig() {
        return {
            enabled: isEnabled(),
            preset: currentPresetValue(),
            block_strengths: collectBlockStrengths(),
        };
    }

    function currentIoText() {
        return String(document.getElementById('image-test-layer-io-text')?.value || '');
    }

    function setIoText(value, options = {}) {
        const textarea = document.getElementById('image-test-layer-io-text');
        if (textarea instanceof HTMLTextAreaElement) {
            textarea.value = String(value || '');
        }
        if (options.persist !== false) {
            persistDialogState();
        }
    }

    function currentPresetValue() {
        return normalizeImageTestSelectiveLoraPreset(
            document.getElementById('image-test-layer-preset')?.value,
            'custom',
        );
    }

    function syncRangeVisual(range, value, enabled) {
        const normalized = clampImageTestSelectiveLoraStrength(value, DEFAULT_LAYER_STRENGTH);
        const total = IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MAX - IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MIN;
        const ratio = total > 0
            ? (normalized - IMAGE_TEST_SELECTIVE_LORA_STRENGTH_MIN) / total
            : 0;
        range.style.setProperty('--image-test-layer-slider-fill', `${(ratio * 100).toFixed(2)}%`);
        range.dataset.enabled = enabled ? 'true' : 'false';
        range.setAttribute('aria-valuetext', `${formatStrength(normalized)}x`);
    }

    function setIoStatus(message, tone = '') {
        const el = document.getElementById('image-test-layer-io-status');
        if (!el) return;
        el.textContent = message;
        el.classList.remove('is-success', 'is-warning', 'is-error');
        if (tone) {
            el.classList.add(`is-${tone}`);
        }
    }

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = text;
        }
    }

    function formatStrength(value) {
        return clampImageTestSelectiveLoraStrength(value, 0).toFixed(2).replace(/\.00$/, '');
    }

    return {
        init,
        collectPayload,
        validate,
    };
}
