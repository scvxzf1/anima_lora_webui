import {
    IMAGE_TEST_HISTORY_RANGE_OPTIONS,
    daysForImageTestHistoryRange,
} from '../state.js?v=module-bootstrap-20260809-nf4-v2';

export function emptyMessageForCurrentFilter(payload, filterValue) {
    if (Array.isArray(payload.images) && payload.images.length) {
        return `当前筛选范围“${historyRangeLabel(filterValue)}”内没有图片，试试放宽时间范围。`;
    }
    return payload.message || '还没有生图结果。';
}

export function historyRangeLabel(value) {
    return IMAGE_TEST_HISTORY_RANGE_OPTIONS.find((item) => item.value === value)?.label || '近 7 天';
}

export function cutoffTimestampMs(filterValue) {
    const days = daysForImageTestHistoryRange(filterValue);
    return typeof days === 'number'
        ? Date.now() - days * 24 * 60 * 60 * 1000
        : null;
}

export function mergedFileName() {
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+$/, '').replace('T', '-');
    return `anima-image-test-merged-${stamp}.png`;
}

export function originalsZipFileName() {
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+$/, '').replace('T', '-');
    return `anima-image-test-originals-${stamp}.zip`;
}

export function imageDownloadName(image) {
    return String(image?.name || image?.file || 'image-test-result.png')
        .split('/')
        .filter(Boolean)
        .pop() || 'image-test-result.png';
}

export function normalizeZipEntryName(name, usedNames = new Set()) {
    const base = String(name || 'image-test-result.png')
        .replace(/[\\/:*?"<>|\r\n\t]+/g, '_')
        .replace(/\s+/g, '_')
        .replace(/^[._]+|[._]+$/g, '') || 'image-test-result.png';
    if (!usedNames.has(base)) {
        return base;
    }
    const dotIndex = base.lastIndexOf('.');
    const stem = dotIndex > 0 ? base.slice(0, dotIndex) : base;
    const ext = dotIndex > 0 ? base.slice(dotIndex) : '';
    let index = 2;
    let candidate = `${stem}_${index}${ext}`;
    while (usedNames.has(candidate)) {
        index += 1;
        candidate = `${stem}_${index}${ext}`;
    }
    return candidate;
}

export function imageKey(image) {
    return String(image?.file || image?.url || image?.name || '').trim();
}

export function imageTimestampMs(image) {
    const generated = Number((image?.sample || {}).generated_at);
    const modified = Number(image?.mtime);
    if (Number.isFinite(generated) && generated > 0) return generated * 1000;
    if (Number.isFinite(modified) && modified > 0) return modified * 1000;
    return 0;
}

export function imageTimestampText(image) {
    const sampleText = String((image?.sample || {}).generated_at_text || '').trim();
    if (sampleText) return sampleText;
    const modifiedText = String(image?.mtime_text || '').trim();
    return modifiedText || '时间未知';
}

export function imageCardMetaText(image, formatBytesFn) {
    const dims = image?.width && image?.height ? `${image.width}x${image.height}` : '尺寸未知';
    const parts = [
        dims,
        image?.sample?.parameters?.sample_steps ? `${image.sample.parameters.sample_steps} steps` : '',
        image?.sample?.sampler || image?.sample?.parameters?.sample_sampler || '',
        formatBytesFn(Number(image?.size_bytes || 0)),
    ].filter(Boolean);
    return parts.join(' · ');
}

export function dateKeyFromTimestamp(timestampMs) {
    const date = timestampMs > 0 ? new Date(timestampMs) : new Date();
    return [
        date.getFullYear(),
        pad2(date.getMonth() + 1),
        pad2(date.getDate()),
    ].join('-');
}

export function dateStartMs(dateKey) {
    return new Date(`${dateKey}T00:00:00`).getTime();
}

export function historyGroupMetaLabel(dateKey) {
    const diffDays = Math.max(0, Math.floor((startOfTodayMs() - dateStartMs(dateKey)) / (24 * 60 * 60 * 1000)));
    if (diffDays === 0) return '今天';
    if (diffDays === 1) return '昨天';
    if (diffDays < 7) return `${diffDays} 天前`;
    if (diffDays < 30) return '近 30 天';
    return '更早记录';
}

export function startOfTodayMs() {
    const now = new Date();
    return new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
}

export function trimExportLabel(value, maxLength = 34) {
    const text = String(value || '').trim();
    if (text.length <= maxLength) return text;
    return `${text.slice(0, maxLength - 1)}…`;
}

export function pad2(value) {
    return String(value).padStart(2, '0');
}
