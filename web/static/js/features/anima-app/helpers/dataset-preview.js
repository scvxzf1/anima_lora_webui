export function datasetPreviewValidationText(settings) {
    const validationNum = Number(settings.validation_split_num || 0);
    if (validationNum > 0) return `固定 ${validationNum} 张`;
    const validationSplit = Number(settings.validation_split ?? 0);
    if (validationSplit > 0) return `比例 ${validationSplit}`;
    return '关闭';
}

export function datasetPreviewImageToPreviewImage(image) {
    return {
        ...image,
        detailContext: 'dataset',
        sample: {},
        source_task: null,
    };
}
