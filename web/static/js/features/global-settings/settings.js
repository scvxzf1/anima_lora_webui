/**
 * Global settings load/save/UI scale helpers.
 * Moved out of anima-app mechanical chunks.
 */
import {
    GLOBAL_MODEL_PATH_FIELDS,
    GLOBAL_SETTING_INPUTS,
    GLOBAL_UI_OVERRIDE_FIELDS,
    help,
} from '../../config/catalog.js?v=module-bootstrap-20260711-ir6';
import { loadOutputRuns } from '../anima-app/helpers/toml-manager-bridge.js?v=module-bootstrap-20260711-ir6';
import { updateChoiceGuide } from '../config-form/choice-guide-ui.js?v=module-bootstrap-20260711-ir6';
import { getUiScaleController } from '../anima-app/helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260711-ir6';
import { getAppShellState } from '../anima-app/helpers/app-shell-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getHistoryDetailFeature } from '../anima-app/helpers/history-detail-bridge.js?v=module-bootstrap-20260711-ir6';
import { api } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260711-ir6';

const appShellState = getAppShellState();
const tomlState = getTomlState();

export function resolveGlobalUIScaleDefaultValue(snapshot = appShellState.globalSettings || {}) {
        const uiScaleController = getUiScaleController();
        const fallback = uiScaleController?.DEFAULT_SCALE || 100;
        const raw = snapshot?.ui_scale ?? snapshot?.defaults?.ui_scale ?? document.getElementById('global-ui-scale')?.value;
        const scale = Number.parseInt(String(raw ?? '').trim(), 10);
        if (!Number.isFinite(scale)) return fallback;
        return uiScaleController?.clampScale?.(scale) ?? fallback;
    }

export function syncGlobalUIScaleOverrideField(field, options = {}) {
        const uiScaleController = getUiScaleController();
        if (!field?.inputId || !field?.followDefaultId) return;
        const input = document.getElementById(field.inputId);
        const followToggle = document.getElementById(field.followDefaultId);
        if (!input || !followToggle) return;
        const snapshot = options.snapshot ?? null;
        const defaultScale = resolveGlobalUIScaleDefaultValue(snapshot || appShellState.globalSettings || {});
        const row = input.closest('.global-ui-scale-row');
        let followDefault = Boolean(followToggle.checked);
        let nextValue = input.value;

        if (snapshot) {
            const raw = snapshot?.[field.key] ?? snapshot?.defaults?.[field.key] ?? '';
            followDefault = String(raw ?? '').trim() === '';
            nextValue = followDefault
                ? defaultScale
                : (uiScaleController?.clampScale?.(raw) ?? defaultScale);
        } else if (followDefault || !options.preserveCustom || !String(nextValue || '').trim()) {
            nextValue = defaultScale;
        } else {
            nextValue = uiScaleController?.clampScale?.(nextValue) ?? defaultScale;
        }

        followToggle.checked = followDefault;
        input.disabled = followDefault;
        input.placeholder = String(defaultScale);
        input.value = String(nextValue);
        row?.classList.toggle('is-follow-default', followDefault);
    }

export function syncAllGlobalUIScaleOverrideFields(options = {}) {
        const snapshot = options.snapshot ?? null;
        const defaultScale = resolveGlobalUIScaleDefaultValue(snapshot || appShellState.globalSettings || {});
        for (const field of GLOBAL_UI_OVERRIDE_FIELDS) {
            syncGlobalUIScaleOverrideField(field, {
                ...options,
                snapshot,
                defaultScale,
            });
        }
    }

export function applyGlobalUIScaleOverrideInputs(snapshot = appShellState.globalSettings || {}) {
        syncAllGlobalUIScaleOverrideFields({ snapshot });
    }

export function collectGlobalUIScaleOverridePayload(payload) {
        const uiScaleController = getUiScaleController();
        const defaultScale = resolveGlobalUIScaleDefaultValue();
        for (const field of GLOBAL_UI_OVERRIDE_FIELDS) {
            const input = document.getElementById(field.inputId);
            const followToggle = document.getElementById(field.followDefaultId);
            if (!input || !followToggle) {
                payload[field.key] = appShellState.globalSettings?.[field.key] ?? '';
                continue;
            }
            if (followToggle.checked) {
                payload[field.key] = '';
                continue;
            }
            payload[field.key] = String(uiScaleController?.clampScale?.(input.value) ?? defaultScale);
        }
        return payload;
    }

export async function loadGlobalSettings() {
        const uiScaleController = getUiScaleController();
        const historyDetailFeature = getHistoryDetailFeature();
        if (location.protocol === 'file:') return;
        try {
            const data = await api('/api/settings/global');
            if (!data.ok) throw new Error(data.error || '读取全局设置失败');
            appShellState.globalSettings = data;
            applyGlobalSettingsToInputs(data);
            uiScaleController?.applyScaleFromSettings?.(data, {
                activeHistoryDetailTab: historyDetailFeature?.getActiveTab?.(),
            });
            updateChoiceGuide();
            setGlobalSettingsStatus('', '');
            if (tomlState.tomlManagerMode === 'output') {
                await loadOutputRuns({ keepSelection: true });
            }
        } catch (e) {
            setGlobalSettingsStatus('读取全局设置失败: ' + e.message, 'error');
        }
    }

export async function saveGlobalSettings() {
        const uiScaleController = getUiScaleController();
        const historyDetailFeature = getHistoryDetailFeature();
        try {
            const payload = collectGlobalSettingsPayload();
            const res = await api('/api/settings/global', {
                method: 'PUT',
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                setGlobalSettingsStatus(res.error || '保存失败', 'error');
                return;
            }
            appShellState.globalSettings = {
                ...(appShellState.globalSettings || {}),
                ...res,
            };
            applyGlobalSettingsToInputs(appShellState.globalSettings);
            uiScaleController?.applyScaleFromSettings?.(appShellState.globalSettings, {
                activeHistoryDetailTab: historyDetailFeature?.getActiveTab?.(),
            });
            updateChoiceGuide();
            if (res.requires_reload) {
                setGlobalSettingsStatus('全局设置已保存，正在切换新的配置根目录...', 'ok');
                window.setTimeout(() => location.reload(), 250);
                return;
            }
            setGlobalSettingsStatus(res.message || '全局设置已保存', 'ok');
        } catch (e) {
            setGlobalSettingsStatus('保存失败: ' + e.message, 'error');
        }
    }

export async function resetGlobalSettings() {
        const defaults = appShellState.globalSettings?.defaults || {};
        applyGlobalSettingsToInputs({
            defaults,
            ...Object.fromEntries(GLOBAL_SETTING_INPUTS.map(([key]) => [key, defaults[key] ?? ''])),
            ...Object.fromEntries(GLOBAL_UI_OVERRIDE_FIELDS.map(({ key }) => [key, defaults[key] ?? ''])),
        });
        await saveGlobalSettings();
    }

export function setGlobalSettingsStatus(text, state = '') {
        const el = document.getElementById('global-settings-status');
        if (!el) return;
        el.textContent = text;
        el.className = `preview-status ${state}`.trim();
    }

export function applyGlobalSettingsToInputs(data) {
        const snapshot = data || appShellState.globalSettings || {};
        for (const [key, id] of GLOBAL_SETTING_INPUTS) {
            const input = document.getElementById(id);
            if (!input) continue;
            const fallback = snapshot?.defaults?.[key] || '';
            input.value = snapshot?.[key] ?? fallback;
        }
        applyGlobalUIScaleOverrideInputs(snapshot);
    }

export function collectGlobalSettingsPayload() {
        const payload = {};
        for (const [key, id] of GLOBAL_SETTING_INPUTS) {
            const input = document.getElementById(id);
            payload[key] = input ? input.value : (appShellState.globalSettings?.[key] || '');
        }
        return collectGlobalUIScaleOverridePayload(payload);
    }

export function getGlobalModelPathOverrides() {
        const overrides = {};
        const source = appShellState.globalSettings || {};
        for (const [key] of GLOBAL_MODEL_PATH_FIELDS) {
            const value = source[key] ?? source.defaults?.[key] ?? '';
            if (String(value || '').trim()) {
                overrides[key] = String(value).trim();
            }
        }
        return overrides;
    }

export function toggleGlobalSettingHelp(button) {
        if (!button) return;
        const helpId = button.getAttribute('aria-controls');
        const help = helpId ? document.getElementById(helpId) : null;
        if (!help) return;
        const visible = help.classList.toggle('visible');
        button.classList.toggle('active', visible);
        button.setAttribute('aria-expanded', visible ? 'true' : 'false');
    }

    // ── 预览图 ──

