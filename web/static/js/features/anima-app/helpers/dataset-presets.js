import { getConfigState } from './config-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getDatasetState } from './dataset-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';

export function selectedDatasetConfigOverride() {
    const configState = getConfigState();
    const datasetState = getDatasetState();
    const currentDataset = configState.currentConfig?.dataset_config || '';
    return datasetState.selectedConfigDatasetFile === currentDataset ? null : (datasetState.selectedConfigDatasetFile || '');
}

export function datasetPresetByFile(file) {
    return (getDatasetState().datasetPresetState.presets || []).find((item) => item.path === file) || null;
}

export function datasetPresetSummaryByFile(file) {
    return datasetPresetByFile(file)?.summary || null;
}

export function datasetPresetGroupsForDisplay() {
    const datasetState = getDatasetState();
    const datasetPresetState = datasetState.datasetPresetState;
    const keyword = datasetPresetState.search.trim().toLowerCase();
    const presetMap = new Map((datasetPresetState.presets || []).map((preset) => [preset.path, preset]));
    const sourceGroups = (datasetPresetState.groups || []).length
        ? datasetPresetState.groups
        : [{
            id: 'datasets',
            label: '数据集配置',
            open: false,
            kind: 'dataset',
            files: datasetPresetState.presets || [],
            movable: true,
        }];
    const covered = new Set();
    const groups = [];

    for (const rawGroup of sourceGroups) {
        const files = (rawGroup.files || [])
            .map((item) => presetMap.get(item.path) || item)
            .filter((item) => item?.path && presetMap.has(item.path))
            .filter((item) => datasetPresetMatchesSearch(item, keyword));
        (rawGroup.files || []).forEach((item) => {
            if (item?.path && presetMap.has(item.path)) covered.add(item.path);
        });
        if (keyword && !files.length) continue;
        if (!files.length && rawGroup.kind !== 'dataset' && rawGroup.id !== 'datasets' && rawGroup.id !== 'unfiled_datasets') continue;
        groups.push({ ...rawGroup, files });
    }

    const ungrouped = (datasetPresetState.presets || [])
        .filter((preset) => !covered.has(preset.path))
        .filter((preset) => datasetPresetMatchesSearch(preset, keyword));
    if (ungrouped.length) {
        groups.push({
            id: 'unfiled_datasets',
            label: '未分组数据集配置',
            open: true,
            kind: 'dataset',
            movable: true,
            files: ungrouped,
        });
    }
    return sortDatasetPresetGroups(groups);
}

export function isUnfiledDatasetGroup(group) {
    return group?.id === 'unfiled_datasets';
}

export function sortDatasetPresetGroups(groups) {
    return [...groups].sort((a, b) => {
        if (isUnfiledDatasetGroup(a)) return -1;
        if (isUnfiledDatasetGroup(b)) return 1;
        return 0;
    });
}

export function orderDatasetPresetsForGroups(presets, groups) {
    const presetMap = new Map((presets || []).map((preset) => [preset.path, preset]));
    const ordered = [];
    const seen = new Set();
    for (const group of sortDatasetPresetGroups(groups || [])) {
        for (const item of group.files || []) {
            if (!item?.path || seen.has(item.path) || !presetMap.has(item.path)) continue;
            ordered.push(presetMap.get(item.path));
            seen.add(item.path);
        }
    }
    for (const preset of presets || []) {
        if (!preset?.path || seen.has(preset.path)) continue;
        ordered.push(preset);
    }
    return ordered;
}

export function datasetPresetMatchesSearch(preset, keyword) {
    if (!keyword) return true;
    const summary = preset?.summary || {};
    return [
        preset?.label,
        preset?.filename,
        preset?.path,
        summary.source_dir,
        summary.image_dir,
        summary.cache_dir,
    ].some((value) => String(value || '').toLowerCase().includes(keyword));
}
