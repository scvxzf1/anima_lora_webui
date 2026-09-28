import { BLANK_PRESET_TEMPLATE_FILE, GLOBAL_MODEL_PATH_FIELDS } from '../../config/catalog/defaults.js?v=qwen-cache-policy-20260928';

export async function loadBlankTrainingPreset(api, file) {
    const template = await api(`/api/config/raw?file=${encodeURIComponent(BLANK_PRESET_TEMPLATE_FILE)}`);
    if (template.ok === false || !String(template.content || '').trim()) {
        throw new Error(template.error || '默认空白预设不存在或内容为空');
    }
    const settings = await api('/api/settings/global');
    if (settings.ok === false) throw new Error(settings.error || '读取全局模型配置失败');
    const values = { model_family: String(settings.model_family || 'anima').trim() || 'anima' };
    for (const [key] of GLOBAL_MODEL_PATH_FIELDS) {
        const value = String(settings[key] ?? settings.defaults?.[key] ?? '').trim();
        if (value) values[key] = value;
    }
    const patched = await api('/api/config/raw/patch-preview', {
        method: 'POST',
        body: JSON.stringify({ file, content: template.content, values }),
    });
    if (patched.ok === false || typeof patched.content !== 'string') {
        throw new Error(patched.error || '应用全局模型配置失败');
    }
    return patched.content;
}
