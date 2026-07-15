import { FIELD_LABEL_ZH } from '../../../config/catalog.js?v=module-bootstrap-20260714-stage-dataset5';

export function datasetConfigLabel(key) {
    const labels = {
        resolution: '分辨率',
        enable_bucket: '启用长宽比分桶',
        min_bucket_reso: '最小桶边长',
        max_bucket_reso: '最大桶边长',
        bucket_reso_steps: '桶尺寸步长',
        bucket_no_upscale: '禁止放大图片',
        validation_split: '验证集比例',
        validation_split_num: '固定验证数量',
        validation_seed: '验证随机种子',
        caption_extension: '文本标注扩展名',
        keep_tokens: '保留前置 token',
        prefer_json_caption: '优先 JSON 标注',
        caption_source_mode: '标注来源',
    };
    return `${labels[key] || FIELD_LABEL_ZH[key] || key} / ${key}`;
}

export function datasetConfigValue(key, defaults) {
    return defaults[key] ?? '';
}
