import { SECTION_GROUPS } from './section-groups.js?v=dragon-ui-20260902-krea2-pp-v1';
import { isBooleanConfigField } from './config-field-types.js?v=dragon-ui-20260902-lokr-backend-v4';
import { configFieldAvailability } from './config-field-availability.js?v=dragon-ui-20260903-pp-multimodel-v1';

const TAG_META = Object.freeze({
    '基础模型路径': { id: 'models', label: '模型', tone: 'required' },
    '数据集设置': { id: 'dataset', label: '数据', tone: 'required' },
    '常用训练设置': { id: 'core', label: '核心训练' },
    '步数与训练量': { id: 'volume', label: '训练量' },
    'LoKr 专用优化': { id: 'lokr', label: 'LoKr', tone: 'experimental' },
    '训练中预览图': { id: 'preview', label: '预览' },
    '显存与速度优化': { id: 'memory-speed', label: '显存与速度' },
    '数据加载与 VAE 资源': { id: 'data-vae', label: '数据与 VAE' },
    '实验性功能': { id: 'experimental', label: '实验性', tone: 'experimental' },
    '无数据集正则化': { id: 'regularization', label: '正则化' },
    '缓存与预处理': { id: 'cache', label: '缓存与预处理' },
    '更多数据集配置': { id: 'dataset-advanced', label: '数据高级' },
    'SPD CLI 实验': { id: 'spd', label: 'SPD', tone: 'experimental' },
    '输出格式与训练范围': { id: 'output', label: '输出' },
    '方法内部与实验架构': { id: 'architecture', label: '方法架构', tone: 'experimental' },
});

const CHAPTER_META = Object.freeze([
    { id: 'foundation', label: '模型与数据', color: 'amber', accent: '#d99114', tags: ['models', 'dataset'] },
    { id: 'training', label: '核心训练', color: 'blue', accent: '#3182ce', tags: ['core', 'volume', 'output'] },
    { id: 'preview', label: '训练预览', color: 'cyan', accent: '#0891b2', tags: ['preview'] },
    { id: 'performance', label: '显存与速度', color: 'green', accent: '#3a9b58', tags: ['memory-speed'] },
    { id: 'data-runtime', label: '数据与缓存', color: 'teal', accent: '#0f8f83', tags: ['data-vae', 'cache', 'dataset-advanced'] },
    { id: 'regularization', label: '正则化', color: 'rose', accent: '#c45a6a', tags: ['regularization'] },
    { id: 'methods', label: '方法架构', color: 'indigo', accent: '#6366b5', tags: ['architecture', 'lokr'] },
    { id: 'experimental', label: '实验与高级', color: 'purple', accent: '#8b5cf6', tags: ['experimental', 'spd', 'advanced', 'snr-weighting'] },
]);

const CHAPTER_BY_TAG = new Map(CHAPTER_META.flatMap((chapter) => chapter.tags.map((tag) => [tag, chapter])));
const REQUIRED_KEYS = new Set(['pretrained_model_name_or_path', 'qwen3', 'vae', 'dataset_config']);
const EXPERIMENTAL_KEYS = new Set(['weighting_scheme', 'min_snr_gamma', 'p2_gamma', 'p2_k', 'sigmoid_scale', 'sigmoid_bias']);

function sectionForField(entry, key) {
    return (SECTION_GROUPS[entry.sub.id] || []).find((section) => section.keys.includes(key));
}

function fallbackTag(entry) {
    return {
        id: String(entry.sub.id || 'other').replace(/[^A-Za-z0-9_-]+/g, '-'),
        label: entry.sub.label || '其他',
        tone: entry.sub.id === 'advanced' ? 'experimental' : 'neutral',
    };
}

function spanForField(key, value, options) {
    if (key.includes('prompt') || key === 'optimizer_args' || key === 'network_args') return 2;
    if (/(?:^|_)(?:path|dir|file|jsonl)$/.test(key) || key.endsWith('_path')) return 2;
    if (typeof value === 'string' && /[\\/]/.test(value)) return 2;
    return 1;
}

function controlKind(value, options, key) {
    if (isBooleanConfigField(key, value, options)) return 'toggle';
    if (options) return 'select';
    if (typeof value === 'number') return 'number';
    if (key.includes('prompt') || key === 'optimizer_args' || key === 'network_args') return 'textarea';
    return 'text';
}

function chapterForField(tag, entry, experimental) {
    const known = CHAPTER_BY_TAG.get(tag.id);
    if (known) return known;
    if (experimental || entry.sub.id === 'advanced') return CHAPTER_BY_TAG.get('experimental');
    return CHAPTER_BY_TAG.get('architecture');
}

function isPathField(key, value) {
    return /(?:^|_)(?:path|dir|file|jsonl)$/.test(key)
        || key.endsWith('_path')
        || (typeof value === 'string' && /[\\/]/.test(value));
}

export function buildConfigBlocks(entries, values, optionsByKey, defaults, availabilityContext = null) {
    const rawBlocks = entries.flatMap((entry) => entry.keys.map((key) => {
        const section = sectionForField(entry, key);
        const tag = TAG_META[section?.title] || fallbackTag(entry);
        const value = values[key];
        const experimental = ['experimental', 'spd'].includes(tag.id) || EXPERIMENTAL_KEYS.has(key);
        const chapter = chapterForField(tag, entry, experimental);
        const metadata = {
            key,
            entryId: entry.sub.id,
            span: spanForField(key, value, optionsByKey[key]),
            tagId: tag.id,
            tagLabel: tag.label,
            chapterId: chapter.id,
            chapterLabel: chapter.label,
            chapterColor: chapter.color,
            required: REQUIRED_KEYS.has(key),
            experimental,
            pathField: isPathField(key, value),
            tone: experimental ? 'experimental' : (REQUIRED_KEYS.has(key) ? 'required' : 'neutral'),
            control: controlKind(value, optionsByKey[key], key),
            defaultValue: Object.prototype.hasOwnProperty.call(defaults, key) ? defaults[key] : undefined,
            availability: availabilityContext ? configFieldAvailability(key, availabilityContext) : null,
        };
        return metadata;
    }));

    const chapters = CHAPTER_META.map((chapter) => {
        const blocks = rawBlocks.filter((block) => block.chapterId === chapter.id);
        return { id: chapter.id, label: chapter.label, color: chapter.color, accent: chapter.accent, count: blocks.length, blocks };
    }).filter((chapter) => chapter.count > 0);
    return { blocks: chapters.flatMap((chapter) => chapter.blocks), chapters };
}
