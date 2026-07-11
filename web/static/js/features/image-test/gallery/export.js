import { createZipDataBlob, downloadBlob as triggerBlobDownload } from '../../../shared/download.js?v=module-bootstrap-20260711-ir6';
import {
    EXPORT_BACKGROUND,
    EXPORT_CELL_BACKGROUND,
    EXPORT_CELL_MAX,
    EXPORT_CELL_MIN,
    EXPORT_MAX_EDGE,
    EXPORT_META,
    EXPORT_TEXT,
} from './constants.js?v=module-bootstrap-20260711-ir6';
import {
    imageDownloadName,
    imageKey,
    imageTimestampText,
    mergedFileName,
    normalizeZipEntryName,
    originalsZipFileName,
    trimExportLabel,
} from './image-meta.js?v=module-bootstrap-20260711-ir6';

export async function fetchImageBytes(image) {
    const response = await fetch(image.url, { credentials: 'same-origin' });
    if (!response.ok) {
        throw new Error(`读取 ${image.name || image.file || '图片'} 失败`);
    }
    return new Uint8Array(await response.arrayBuffer());
}

export function loadImageElement(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.decoding = 'async';
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('有图片无法读取，无法完成合并导出。'));
        img.src = url;
    });
}

export async function createMergedImageBlob(images) {
    const loaded = await Promise.all(images.map(async (image) => ({
        image,
        bitmap: await loadImageElement(image.url),
    })));
    const count = loaded.length;
    const columns = Math.max(1, Math.ceil(Math.sqrt(count)));
    const rows = Math.max(1, Math.ceil(count / columns));
    const naturalCellWidth = Math.max(...loaded.map((item) => item.bitmap.naturalWidth || item.bitmap.width || EXPORT_CELL_MIN));
    const naturalCellHeight = Math.max(...loaded.map((item) => item.bitmap.naturalHeight || item.bitmap.height || EXPORT_CELL_MIN));

    let cellWidth = Math.min(EXPORT_CELL_MAX, Math.max(EXPORT_CELL_MIN, naturalCellWidth));
    let cellHeight = Math.min(EXPORT_CELL_MAX, Math.max(EXPORT_CELL_MIN, naturalCellHeight));
    const gap = 16;
    const padding = 20;
    const labelHeight = 34;
    const labelGap = 10;

    let width = padding * 2 + columns * cellWidth + (columns - 1) * gap;
    let height = padding * 2 + rows * (cellHeight + labelHeight) + (rows - 1) * gap;
    const maxEdge = Math.max(width, height);
    if (maxEdge > EXPORT_MAX_EDGE) {
        const scale = EXPORT_MAX_EDGE / maxEdge;
        cellWidth = Math.max(160, Math.floor(cellWidth * scale));
        cellHeight = Math.max(160, Math.floor(cellHeight * scale));
        width = padding * 2 + columns * cellWidth + (columns - 1) * gap;
        height = padding * 2 + rows * (cellHeight + labelHeight) + (rows - 1) * gap;
    }

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error('当前浏览器不支持 Canvas。');
    }

    ctx.fillStyle = EXPORT_BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.textBaseline = 'top';
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    loaded.forEach(({ image, bitmap }, index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        const originX = padding + column * (cellWidth + gap);
        const originY = padding + row * (cellHeight + labelHeight + gap);

        ctx.fillStyle = EXPORT_CELL_BACKGROUND;
        ctx.fillRect(originX, originY, cellWidth, cellHeight);

        const drawWidth = bitmap.naturalWidth || bitmap.width || cellWidth;
        const drawHeight = bitmap.naturalHeight || bitmap.height || cellHeight;
        const scale = Math.min(cellWidth / drawWidth, cellHeight / drawHeight, 1);
        const fittedWidth = Math.max(1, Math.round(drawWidth * scale));
        const fittedHeight = Math.max(1, Math.round(drawHeight * scale));
        const imageX = originX + Math.round((cellWidth - fittedWidth) / 2);
        const imageY = originY + Math.round((cellHeight - fittedHeight) / 2);
        ctx.drawImage(bitmap, imageX, imageY, fittedWidth, fittedHeight);

        ctx.fillStyle = EXPORT_TEXT;
        ctx.font = '600 16px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif';
        ctx.fillText(trimExportLabel(image.name || image.file || `image-${index + 1}`), originX, originY + cellHeight + labelGap);

        ctx.fillStyle = EXPORT_META;
        ctx.font = '12px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif';
        ctx.fillText(
            trimExportLabel(imageTimestampText(image), 48),
            originX,
            originY + cellHeight + labelGap + 18,
        );
    });

    return await new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob) {
                resolve(blob);
                return;
            }
            reject(new Error('浏览器未返回导出数据。'));
        }, 'image/png');
    });
}

export function createGalleryExport({
    state,
    setSelectionStatus,
    syncSelectionToolbar,
    formatBytes: _formatBytes,
} = {}) {
    async function exportMergedSelection() {
        const selectedImages = selectedImagesInDisplayOrder();
        if (!selectedImages.length || state.exportPending || state.rawExportPending || state.deletePending) {
            return;
        }

        state.exportPending = true;
        setSelectionStatus(`正在合并 ${selectedImages.length} 张图片...`, 'warning');
        syncSelectionToolbar();
        try {
            const blob = await createMergedImageBlob(selectedImages);
            triggerBlobDownload(blob, mergedFileName());
            setSelectionStatus(`已导出 ${selectedImages.length} 张图片的合并图。`, 'success');
        } catch (error) {
            setSelectionStatus(`导出失败：${error?.message || '无法生成合并图。'}`, 'error');
        } finally {
            state.exportPending = false;
            syncSelectionToolbar();
        }
    }

    async function exportOriginalZipSelection() {
        const selectedImages = selectedImagesInDisplayOrder();
        if (!selectedImages.length || state.exportPending || state.rawExportPending || state.deletePending) {
            return;
        }

        state.rawExportPending = true;
        setSelectionStatus(`正在打包 ${selectedImages.length} 张原图...`, 'warning');
        syncSelectionToolbar();
        try {
            const entries = await Promise.all(selectedImages.map(async (image) => ({
                name: imageDownloadName(image),
                data: await fetchImageBytes(image),
            })));
            const blob = createZipDataBlob(entries, normalizeZipEntryName);
            triggerBlobDownload(blob, originalsZipFileName());
            setSelectionStatus(`已导出 ${selectedImages.length} 张原图 zip。`, 'success');
        } catch (error) {
            setSelectionStatus(`原图打包失败：${error?.message || '无法读取图片。'}`, 'error');
        } finally {
            state.rawExportPending = false;
            syncSelectionToolbar();
        }
    }

    function selectedImagesInDisplayOrder() {
        return state.filteredOrderedKeys
            .filter((key) => state.selectedKeys.has(key))
            .map((key) => state.imageMap.get(key))
            .filter(Boolean);
    }

    function dedupeImages(images) {
        const seen = new Set();
        return (Array.isArray(images) ? images : []).filter((image) => {
            const key = imageKey(image);
            if (!key || seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }

    return {
        exportMergedSelection,
        exportOriginalZipSelection,
        selectedImagesInDisplayOrder,
        dedupeImages,
        createMergedImageBlob,
        fetchImageBytes,
        loadImageElement,
        // 文件名 helpers 已收口到 image-meta.js；这里再透出，方便 gallery facade 渐进接线
        mergedFileName,
        originalsZipFileName,
        imageDownloadName,
        normalizeZipEntryName,
    };
}
