export function blankSamplePromptRow() {
    return { prompt: '', height: '', width: '', cfg: '', steps: '', seed: '', extra: '' };
}

export function samplePromptsContentNeedsTextMode(content) {
    const text = String(content || '');
    if (!text) return false;
    return text.split(/\r?\n/).some((line) => {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) return true;
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
        extra: value('extra'),
    };
}

export function serializeSamplePromptRow(row) {
    if (!row.prompt) return '';
    const args = [];
    if (row.width) args.push(`--w ${positiveIntegerText(row.width)}`);
    if (row.height) args.push(`--h ${positiveIntegerText(row.height)}`);
    if (row.steps) args.push(`--s ${positiveIntegerText(row.steps)}`);
    if (row.cfg) args.push(`--g ${positiveNumberText(row.cfg)}`);
    if (row.seed) args.push(`--d ${positiveIntegerText(row.seed)}`);
    if (row.extra) args.push(row.extra.trim());
    return [row.prompt.trim(), ...args.filter(Boolean)].join(' ');
}

export function positiveIntegerText(value) {
    const n = Math.max(0, Math.floor(Number(value)));
    return Number.isFinite(n) ? String(n) : '';
}

export function positiveNumberText(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n < 0) return '';
    return String(n);
}
