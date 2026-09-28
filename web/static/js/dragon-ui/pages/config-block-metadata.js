import { SECTION_GROUPS } from './section-groups.js?v=auto-block-swap-20260908-v3';
import { isBooleanConfigField } from './config-field-types.js?v=auto-block-swap-20260908-v3';
import { configFieldAvailability } from './config-field-availability.js?v=auto-block-swap-20260908-v3';
import { configFieldIsAdvanced, configFieldVisibilityLevel } from './config-field-tiers.js?v=auto-block-swap-20260908-v3';
import {
    CONFIG_STAGE_META,
    configFieldCatalogEntry,
    sortConfigCatalogItems,
} from './config-field-catalog.js?v=qwen-cache-policy-20260928';
import { configFieldDisclosure } from './config-field-disclosure-rules.js?v=qwen-cache-policy-20260928';

const STAGE_BY_ID = new Map(CONFIG_STAGE_META.map((stage) => [stage.id, stage]));
const CLUSTER_BY_ID = new Map(CONFIG_STAGE_META.flatMap((stage) =>
    stage.clusters.map((cluster) => [`${stage.id}/${cluster.id}`, cluster])
));
const REQUIRED_KEYS = new Set(['pretrained_model_name_or_path', 'qwen3', 'vae', 'dataset_config']);
const EXPERIMENTAL_KEYS = new Set(['weighting_scheme', 'min_snr_gamma', 'p2_gamma', 'p2_k', 'sigmoid_scale', 'sigmoid_bias']);
const EXPERIMENTAL_SECTION_TITLES = new Set([
    '实验性功能',
    'SPD CLI 实验',
    'Soft Tokens 参数',
    'IP-Adapter 高级参数',
    'EasyControl 高级参数',
]);

function sectionForField(entry, key) {
    return (SECTION_GROUPS[entry.sub.id] || []).find((section) => section.keys.includes(key));
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

function isPathField(key, value) {
    return /(?:^|_)(?:path|dir|file|jsonl)$/.test(key)
        || key.endsWith('_path')
        || (typeof value === 'string' && /[\\/]/.test(value));
}

export function buildConfigBlocks(entries, values, optionsByKey, defaults, availabilityContext = null) {
    const rawBlocks = entries.flatMap((entry) => entry.keys.map((key) => {
        const section = sectionForField(entry, key);
        const catalog = configFieldCatalogEntry(key);
        const stage = STAGE_BY_ID.get(catalog.stage);
        const cluster = CLUSTER_BY_ID.get(`${catalog.stage}/${catalog.cluster}`);
        const value = values[key];
        const availability = availabilityContext ? configFieldAvailability(key, availabilityContext) : null;
        const presentation = configFieldDisclosure(key, availabilityContext || {});
        const experimental = EXPERIMENTAL_KEYS.has(key)
            || EXPERIMENTAL_SECTION_TITLES.has(section?.title)
            || catalog.location === 'audit_only';
        const metadata = {
            key,
            entryId: entry.sub.id,
            span: spanForField(key, value, optionsByKey[key]),
            tagId: cluster.id,
            tagLabel: cluster.label,
            chapterId: stage.id,
            chapterLabel: stage.label,
            chapterColor: stage.color,
            location: catalog.location,
            stage: catalog.stage,
            cluster: catalog.cluster,
            policy: catalog.policy,
            controlledBy: catalog.controlledBy,
            dependsOn: catalog.dependsOn,
            siblingOrder: catalog.siblingOrder,
            required: REQUIRED_KEYS.has(key),
            experimental,
            advanced: configFieldIsAdvanced(key),
            visibilityLevel: configFieldVisibilityLevel(key),
            pathField: isPathField(key, value),
            tone: experimental ? 'experimental' : (REQUIRED_KEYS.has(key) ? 'required' : 'neutral'),
            control: controlKind(value, optionsByKey[key], key),
            defaultValue: Object.prototype.hasOwnProperty.call(defaults, key) ? defaults[key] : undefined,
            availability,
            presentation,
        };
        return metadata;
    }));

    const sortedBlocks = sortConfigCatalogItems(rawBlocks);
    const chapters = CONFIG_STAGE_META.map((chapter) => {
        const blocks = sortedBlocks.filter((block) => block.chapterId === chapter.id);
        const clusters = chapter.clusters.map((cluster) => {
            const clusterBlocks = blocks.filter((block) => block.cluster === cluster.id);
            return { ...cluster, count: clusterBlocks.length, blocks: clusterBlocks };
        }).filter((cluster) => cluster.count > 0);
        return {
            id: chapter.id,
            label: chapter.label,
            shortLabel: chapter.shortLabel,
            color: chapter.color,
            accent: chapter.accent,
            count: blocks.length,
            blocks,
            clusters,
        };
    });
    return { blocks: chapters.flatMap((chapter) => chapter.blocks), chapters };
}
