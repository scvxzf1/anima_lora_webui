import {
    FIELD_LABEL_ZH,
    FIELD_OPTIONS,
} from '../../../config/catalog.js?v=module-bootstrap-20260711-ir6';
import { isTruthy } from './config-values.js?v=module-bootstrap-20260711-ir6';

export function compactList(items) {
    return items.filter((item) => item !== undefined && item !== null && String(item).trim() !== '');
}

export function formatChoiceValue(value) {
    if (Array.isArray(value)) return value.join(', ');
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    return String(value);
}

export function valueDetail(key, value) {
    if (value === undefined || value === null || value === '') return '';
    return `${FIELD_LABEL_ZH[key] || key}: ${formatChoiceValue(value)}`;
}

export function flagDetail(key, label, value) {
    if (value === undefined || value === null || value === '') return '';
    return `${label}: ${isTruthy(value) ? '开启' : '关闭'}`;
}

export function formatFieldName(key) {
    const label = FIELD_LABEL_ZH[key];
    return label ? `${label} / ${key}` : key;
}

export function shouldRenderSelectInput(key, value) {
    return Boolean(FIELD_OPTIONS[key]) && !Array.isArray(value);
}
