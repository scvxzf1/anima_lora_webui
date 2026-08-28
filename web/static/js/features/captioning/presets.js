const FALLBACK_PRESETS = {
    detailed: {
        label: 'Flux / SD3 长描述', mode: 'natural',
        prompt: 'Write a detailed English training caption. Describe the primary subject, pose, expression, composition, camera angle, lighting, color palette, materials, textures, spatial relationships and background. Be visually factual, avoid quality claims and return only one fluent paragraph.',
    },
    danbooru: {
        label: 'Danbooru / Anime 标签', mode: 'tags',
        prompt: 'Analyze this image and return only precise standard Danbooru-style English tags separated by comma and space. Include subject count, identity traits, clothing, pose, expression, framing and background. Do not use Markdown, sentences or quality boilerplate.',
    },
    character_action: {
        label: '主体解耦 / 姿态动作', mode: 'tags',
        prompt: 'Return only comma-separated English tags useful for character LoRA training. Focus on pose, action, expression, framing, camera angle and environment. Exclude intrinsic identity traits such as hair color, eye color, permanent clothing design, artist name and character name. No Markdown.',
    },
    anima_three_format: {
        label: 'Anima 三格式训练 Caption', mode: 'three_format', output_variant: 'tag',
        prompt: 'Return exactly three standalone JSON objects with type tag, mixed_70tag_30nl and pure_nl. Each object must contain one non-empty caption string. Do not use Markdown or extra text.',
    },
    anima_style_overfit: {
        label: 'Anima 画风过拟合纯 Tag', mode: 'tags', output_variant: 'tag',
        prompt: 'Return a short comma-separated English tag caption. Preserve the supplied style trigger first, tag variable visible content, and omit repeated fixed style traits and quality boilerplate.',
    },
    anima_style_trigger_json: {
        label: 'Anima 固定触发串全量 Tag', mode: 'style_trigger_json', output_variant: 'full_tag_with_style_trigger',
        prompt: 'Return exactly one JSON object with type full_tag_with_style_trigger and a non-empty caption. Preserve the supplied fixed trigger first and append complete visible content tags.',
    },
};

export function normalizePresets(payload) {
    const items = Array.isArray(payload?.presets) ? payload.presets : [];
    if (!items.length) return Object.entries(FALLBACK_PRESETS).map(([id, preset]) => ({id, output_mode: preset.mode, output_variant: preset.output_variant || preset.mode, ...preset}));
    return items.map((item) => ({...item, mode: item.output_mode}));
}

export function findPreset(presets, id) {
    return presets.find((item) => item.id === id) || presets[0];
}

export function presetOptions(presets, selected = 'danbooru') {
    return presets.map((preset) => `<option value="${preset.id}" ${preset.id === selected ? 'selected' : ''}>${preset.label}</option>`).join('');
}
