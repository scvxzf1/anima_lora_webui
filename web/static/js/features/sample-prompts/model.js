export function blankSamplePromptRow() {
    return {
        prompt: '',
        negative_prompt: '',
        height: '',
        width: '',
        cfg: '',
        steps: '',
        seed: '',
        flow_shift: '',
        sample_sampler: '',
        extra: '',
        sample_task: 't2i',
        reference_image: '',
        reference_images: [],
    };
}

export function samplePromptsContentNeedsTextMode(content) {
    const text = String(content || '');
    if (!text) return false;
    return text.split(/\r?\n/).some((line) => {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) return true;
        if (trimmed.startsWith('{')) return true;
        return serializeSamplePromptRow(parseSamplePromptLine(line)) !== trimmed;
    });
}

export function parseSamplePromptRows(content) {
    const rows = String(content || '')
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line && !line.startsWith('#'))
        .map(parseSamplePromptLine);
    return rows.length ? rows : [blankSamplePromptRow()];
}

export function parseSamplePromptLine(line) {
    const text = String(line || '').trim();
    if (/^\{\s*"[^"\n]+"\s*:/.test(text)) return parseJsonSamplePrompt(text);
    const parts = String(line || '').trim().split(/\s+--/);
    const row = blankSamplePromptRow();
    row.prompt = (parts.shift() || '').trim();
    const extras = [];

    for (const rawPart of parts) {
        const part = rawPart.trim();
        let match = part.match(/^h\s+(\d+)$/i);
        if (match) {
            row.height = match[1];
            continue;
        }
        match = part.match(/^w\s+(\d+)$/i);
        if (match) {
            row.width = match[1];
            continue;
        }
        match = part.match(/^g\s+([\d.]+)$/i);
        if (match) {
            row.cfg = match[1];
            continue;
        }
        match = part.match(/^l\s+([\d.]+)$/i);
        if (match) {
            row.cfg = match[1];
            continue;
        }
        match = part.match(/^s\s+(\d+)$/i);
        if (match) {
            row.steps = match[1];
            continue;
        }
        match = part.match(/^d\s+(\d+)$/i);
        if (match) {
            row.seed = match[1];
            continue;
        }
        match = part.match(/^n\s+(.+)$/i);
        if (match) {
            row.negative_prompt = match[1].trim();
            continue;
        }
        match = part.match(/^ss\s+(.+)$/i);
        if (match) {
            row.sample_sampler = match[1].trim();
            continue;
        }
        match = part.match(/^fs\s+([\d.]+)$/i);
        if (match) {
            row.flow_shift = match[1];
            continue;
        }
        if (part) extras.push(`--${part}`);
    }
    row.extra = extras.join(' ');
    return row;
}

export function serializeSamplePromptsEditor(editor) {
    if (editor.dataset.mode === 'text') {
        return editor.querySelector('.sample-prompts-textarea')?.value || '';
    }
    const rows = [];
    for (const rowEl of editor.querySelectorAll('.sample-prompt-row')) {
        const row = samplePromptRowFromElement(rowEl);
        const line = serializeSamplePromptRow(row);
        if (line) rows.push(line);
    }
    return rows.join('\n');
}

export function samplePromptRowFromElement(rowEl) {
    const value = (field) => rowEl.querySelector(`[data-sample-prompt-field="${field}"]`)?.value?.trim() || '';
    return {
        prompt: value('prompt'),
        height: value('height'),
        width: value('width'),
        cfg: value('cfg'),
        steps: value('steps'),
        seed: value('seed'),
        negative_prompt: value('negative_prompt'),
        flow_shift: value('flow_shift'),
        sample_sampler: value('sample_sampler'),
        extra: value('extra'),
    };
}

export function serializeSamplePromptRow(row) {
    if (row.json || row.sample_task === 'edit' || row.reference_image || row.reference_images?.length) return serializeJsonSamplePrompt(row);
    if (!row.prompt) return '';
    const args = [];
    if (row.negative_prompt) args.push(`--n ${row.negative_prompt.trim()}`);
    if (row.width) args.push(`--w ${positiveIntegerText(row.width)}`);
    if (row.height) args.push(`--h ${positiveIntegerText(row.height)}`);
    if (row.steps) args.push(`--s ${positiveIntegerText(row.steps)}`);
    if (row.cfg) args.push(`--g ${positiveNumberText(row.cfg)}`);
    if (row.seed) args.push(`--d ${positiveIntegerText(row.seed)}`);
    if (row.flow_shift) args.push(`--fs ${positiveNumberText(row.flow_shift)}`);
    if (row.sample_sampler) args.push(`--ss ${row.sample_sampler.trim()}`);
    if (row.extra) args.push(row.extra.trim());
    return [row.prompt.trim(), ...args.filter(Boolean)].join(' ');
}

const JSON_FIELDS = {
    prompt: 'prompt', negative_prompt: 'negative_prompt', width: 'width', height: 'height',
    steps: 'sample_steps', cfg: 'guidance_scale', seed: 'seed', flow_shift: 'flow_shift',
    sample_sampler: 'sample_sampler', sample_task: 'sample_task', reference_image: 'reference_image',
};
const NUMERIC_FIELDS = new Set(['width', 'height', 'steps', 'cfg', 'seed', 'flow_shift']);

function parseJsonSamplePrompt(text) {
    const row = blankSamplePromptRow();
    try {
        const data = JSON.parse(text);
        if (!data || Array.isArray(data) || typeof data !== 'object' || typeof data.prompt !== 'string') throw new Error('prompt required');
        row.json = data;
        row.sample_task = data.sample_task || (data.reference_image || data.reference_images?.length ? 'edit' : 't2i');
        row.reference_images = Array.isArray(data.reference_images) ? data.reference_images.filter((path) => typeof path === 'string') :
            (typeof data.reference_image === 'string' && data.reference_image ? [data.reference_image] : []);
        if (data.guidance_scale === undefined && data.scale !== undefined) row.cfg = String(data.scale);
        for (const [field, key] of Object.entries(JSON_FIELDS)) {
            if (data[key] !== undefined && data[key] !== null) row[field] = String(data[key]);
        }
    } catch {
        row.prompt = text;
        row.json_error = 'JSON 样张格式无效，请在原文模式修正。';
    }
    return row;
}

function serializeJsonSamplePrompt(row) {
    const data = { ...row.json };
    const originalReferences = Array.isArray(data.reference_images) ? data.reference_images :
        (typeof data.reference_image === 'string' && data.reference_image ? [data.reference_image] : []);
    const references = Array.isArray(row.reference_images) ? row.reference_images : [];
    const referencesChanged = JSON.stringify(references) !== JSON.stringify(originalReferences);
    for (const [field, key] of Object.entries(JSON_FIELDS)) {
        if (field === 'reference_image' && (referencesChanged || row.sample_task === 't2i' || Array.isArray(data.reference_images))) continue;
        const value = row[field] ?? '';
        if (String(data[key] ?? (field === 'sample_task' ? 't2i' : '')) === value) continue;
        if (value === '') delete data[key];
        else data[key] = NUMERIC_FIELDS.has(field) ? Number(value) : value;
    }
    if (referencesChanged || row.sample_task === 't2i' || Array.isArray(data.reference_images)) {
        delete data.reference_image;
        if (row.sample_task === 'edit' && references.length) data.reference_images = [...references];
        else delete data.reference_images;
    }
    if (row.cfg === '' && data.guidance_scale === undefined) delete data.scale;
    if (row.cfg !== '' && data.guidance_scale === undefined && data.scale !== undefined) delete data.scale;
    return JSON.stringify(data);
}

export function positiveIntegerText(value) {
    const n = Math.max(0, Math.floor(Number(value)));
    return Number.isFinite(n) ? String(n) : '';
}

export function positiveNumberText(value) {
    const text = String(value ?? '').trim();
    if (!/^\d+(?:\.\d+)?$/.test(text)) return '';
    return text;
}
