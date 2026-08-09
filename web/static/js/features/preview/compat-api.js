/**
 * Preview settings/view compatibility API for older bridge consumers.
 * Thin wrappers around the real preview feature.
 */
import { ensurePreviewFeature } from '../anima-app/helpers/feature-ensurers.js?v=module-bootstrap-20260809-nf4-v2';
import { getAppContext } from '../anima-app/helpers/app-context-bridge.js?v=module-bootstrap-20260809-nf4-v2';

const ctx = getAppContext();

export async function loadPreviewSettings() {
    return ensurePreviewFeature().loadPreviewSettings();
}

export async function savePreviewSettings() {
    return ensurePreviewFeature().savePreviewSettings();
}

export async function resetPreviewSettings() {
    return ensurePreviewFeature().resetPreviewSettings();
}

export async function loadPreviewImages() {
    return ensurePreviewFeature().loadPreviewImages();
}

export async function loadPreviewWeights() {
    return ensurePreviewFeature().loadPreviewWeights();
}

export function setPreviewSource(source) {
    return ensurePreviewFeature().setPreviewSource(source);
}

export async function openTrainingPreview(options = {}) {
    return ensurePreviewFeature().openTrainingPreview(options);
}

export function openCurrentTrainingPreview(event) {
    return ensurePreviewFeature().openCurrentTrainingPreview(event);
}

export function openLiveSamplingPreview(event) {
    return ensurePreviewFeature().openLiveSamplingPreview(event);
}

export async function openHistoryConfigGroupPreview(group) {
    return ensurePreviewFeature().openHistoryConfigGroupPreview(group);
}

export function normalizePreviewGroup(group) {
    return ensurePreviewFeature().normalizePreviewGroup(group);
}

export function renderPreviewTaskSelect() {
    return ensurePreviewFeature().renderPreviewTaskSelect();
}

export async function changePreviewTask(taskId) {
    return ensurePreviewFeature().changePreviewTask(taskId);
}

export function togglePreviewWeightSort() {
    return ensurePreviewFeature().togglePreviewWeightSort();
}

export function openPreviewDialog(...args) {
    return ensurePreviewFeature().openPreviewDialog(...args);
}

export function closePreviewImageDialog() {
    return ensurePreviewFeature().closePreviewImageDialog();
}

export function openPreviewPanel() {
    return ensurePreviewFeature().openPreviewPanel();
}

export function closePreviewPanel() {
    return ensurePreviewFeature().closePreviewPanel();
}

export function restorePreviewWorkspaceAfterPanelClose() {
    return ensurePreviewFeature().restorePreviewWorkspaceAfterPanelClose();
}

export function setPreviewStatus(text, state = '') {
    return ensurePreviewFeature().setPreviewStatus(text, state);
}

export function createPreviewDetailRow(label, value) {
    return ensurePreviewFeature().createPreviewDetailRow(label, value);
}

export function createPreviewDetailBlock(label, value, preformatted = false) {
    return ensurePreviewFeature().createPreviewDetailBlock(label, value, preformatted);
}

export function renderDatasetImageDialogDetails(box, image, dims) {
    const caption = image.caption || {};
    const rows = [
        ['文件时间', image.mtime_text || '-'],
        ['尺寸', dims],
        ['长', image.height ? `${image.height} px` : '-'],
        ['宽', image.width ? `${image.width} px` : '-'],
        ['总像素', formatTotalPixels(image.total_pixels)],
        ['文件大小', formatBytes(image.size_bytes)],
    ];
    for (const [label, value] of rows) {
        box.appendChild(createPreviewDetailRow(label, value));
    }
    box.appendChild(createPreviewDetailBlock('文件路径', image.file || '-'));
    box.appendChild(createPreviewDetailBlock('标注文件', caption.file || '未找到同名标注文件'));
    const captionText = caption.ok ? (caption.text || '(空标注)') : '未找到同名 caption 文件';
    box.appendChild(createPreviewDetailBlock('标注内容', captionText, true));
}

export function formatTotalPixels(totalPixels) {
    const count = Number(totalPixels);
    if (!Number.isFinite(count) || count <= 0) return '-';
    return `${count.toLocaleString('zh-CN')} px (${(count / 1000000).toFixed(2)} MP)`;
}

export async function copyText(text) {
    return ctx.dom.copyText(text);
}

export function formatBytes(bytes) {
    return ctx.format.formatBytes(bytes);
}
