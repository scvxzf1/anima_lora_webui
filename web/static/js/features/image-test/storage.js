import { normalizeImageTestHistoryRange } from './state.js?v=module-bootstrap-20260703-7';

export const IMAGE_TEST_UI_STORAGE_KEY = 'anima.imageTest.ui';
export const IMAGE_TEST_UI_STORAGE_VERSION = 1;

export const IMAGE_TEST_PERSISTED_FIELD_IDS = Object.freeze([
    'image-test-prompt',
    'image-test-negative-prompt',
    'image-test-width',
    'image-test-height',
    'image-test-infer-steps',
    'image-test-guidance-scale',
    'image-test-flow-shift',
    'image-test-seed',
    'image-test-sampler',
    'image-test-attn-mode',
    'image-test-runtime-dtype',
    'image-test-text-encoder-dtype',
    'image-test-gpu-index',
    'image-test-weight-select',
    'image-test-weight-path',
    'image-test-lora-multiplier',
]);

const DEFAULT_HISTORY_RANGE = '7';

export function createImageTestUiStorage({
    storageKey = IMAGE_TEST_UI_STORAGE_KEY,
    storage = window.localStorage,
} = {}) {
    let bound = false;
    let snapshot = readStoredSnapshot();

    function bind(onFieldChange) {
        if (bound) return;
        bound = true;
        IMAGE_TEST_PERSISTED_FIELD_IDS.forEach((id) => {
            const field = document.getElementById(id);
            if (!field) return;
            const handler = () => {
                persistFromDom();
                onFieldChange?.(id);
            };
            field.addEventListener('input', handler);
            field.addEventListener('change', handler);
        });
    }

    function readStoredSnapshot() {
        try {
            const parsed = JSON.parse(storage.getItem(storageKey) || '{}');
            return normalizeStoredSnapshot(parsed);
        } catch (_) {
            return emptySnapshot();
        }
    }

    function normalizeStoredSnapshot(value) {
        const fields = normalizeStoredFields(value?.fields);
        const historyRange = Object.prototype.hasOwnProperty.call(value || {}, 'history_range')
            ? normalizeImageTestHistoryRange(value.history_range, DEFAULT_HISTORY_RANGE)
            : DEFAULT_HISTORY_RANGE;
        return {
            version: IMAGE_TEST_UI_STORAGE_VERSION,
            hasStoredDraft: Boolean(
                Object.keys(fields).length
                || Object.prototype.hasOwnProperty.call(value || {}, 'history_range'),
            ),
            history_range: historyRange,
            fields,
        };
    }

    function normalizeStoredFields(value) {
        if (!value || typeof value !== 'object' || Array.isArray(value)) {
            return {};
        }
        return Object.fromEntries(
            IMAGE_TEST_PERSISTED_FIELD_IDS
                .filter((id) => Object.prototype.hasOwnProperty.call(value, id))
                .map((id) => [id, stringifyFieldValue(value[id])]),
        );
    }

    function emptySnapshot() {
        return {
            version: IMAGE_TEST_UI_STORAGE_VERSION,
            hasStoredDraft: false,
            history_range: DEFAULT_HISTORY_RANGE,
            fields: {},
        };
    }

    function hasStoredDraft() {
        return snapshot.hasStoredDraft;
    }

    function storedHistoryRange() {
        return snapshot.history_range || DEFAULT_HISTORY_RANGE;
    }

    function restoreToDom() {
        const restoredFieldIds = new Set();
        IMAGE_TEST_PERSISTED_FIELD_IDS.forEach((id) => {
            if (!Object.prototype.hasOwnProperty.call(snapshot.fields, id)) {
                return;
            }
            const field = document.getElementById(id);
            if (!field) return;
            field.value = snapshot.fields[id];
            restoredFieldIds.add(id);
        });
        return restoredFieldIds;
    }

    function restoreDeferredField(id) {
        if (!Object.prototype.hasOwnProperty.call(snapshot.fields, id)) {
            return false;
        }
        const field = document.getElementById(id);
        if (!field) return false;
        field.value = snapshot.fields[id];
        return field.value === snapshot.fields[id];
    }

    function persistFromDom(extra = {}) {
        snapshot = {
            version: IMAGE_TEST_UI_STORAGE_VERSION,
            hasStoredDraft: true,
            history_range: normalizeImageTestHistoryRange(
                extra.history_range ?? snapshot.history_range,
                DEFAULT_HISTORY_RANGE,
            ),
            fields: collectFieldValues(),
        };
        try {
            storage.setItem(storageKey, JSON.stringify({
                version: snapshot.version,
                history_range: snapshot.history_range,
                fields: snapshot.fields,
            }));
        } catch (_) {
            // 浏览器禁用 localStorage 时，当前页面内草稿依然可继续使用。
        }
        return snapshot;
    }

    function collectFieldValues() {
        return Object.fromEntries(
            IMAGE_TEST_PERSISTED_FIELD_IDS
                .map((id) => {
                    const field = document.getElementById(id);
                    return field ? [id, stringifyFieldValue(field.value)] : null;
                })
                .filter(Boolean),
        );
    }

    function stringifyFieldValue(value) {
        return value == null ? '' : String(value);
    }

    return {
        bind,
        hasStoredDraft,
        persistFromDom,
        restoreDeferredField,
        restoreToDom,
        storedHistoryRange,
    };
}
