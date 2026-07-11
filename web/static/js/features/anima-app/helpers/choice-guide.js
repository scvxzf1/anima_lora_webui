import { choiceHelp } from '../../../config/catalog.js?v=module-bootstrap-20260711-ir6';

export function choiceLine(label, text, extraClass = '') {
    const line = document.createElement('p');
    line.className = extraClass;
    const strong = document.createElement('strong');
    strong.textContent = `${label}: `;
    line.appendChild(strong);
    line.appendChild(document.createTextNode(text));
    return line;
}

export function defaultMethodGuide() {
    return choiceHelp(
        '自定义方法',
        '当前方法没有专门说明，通常表示它来自后端方法列表。',
        '请结合变体 TOML 判断实际训练行为。',
        '不确定时使用 lora。'
    );
}

export function defaultVariantGuide() {
    return choiceHelp(
        '自定义变体',
        '当前变体对应一个 gui-methods TOML 文件，里面才是实际训练参数。',
        '自定义变体灵活，但需要自行确认字段组合是否合理。',
        '不确定时从内置 lora 变体复制再改。'
    );
}

export function defaultPresetGuide() {
    return choiceHelp(
        '自定义预设',
        '当前预设来自 presets.toml 或自定义配置。',
        '它会覆盖部分硬件、采样或性能参数。',
        '不确定时使用 default。'
    );
}
