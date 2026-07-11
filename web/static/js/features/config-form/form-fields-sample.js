/**
 * Sample prompts editor controls used by config form field rows.
 */
import {
    appendSamplePromptRow,
    updateSamplePromptRemoveButtons,
} from '../anima-app/helpers/config-field-ui-bridge.js?v=module-bootstrap-20260711-ir2';
import {
    blankSamplePromptRow,
    parseSamplePromptRows,
    samplePromptsContentNeedsTextMode,
    serializeSamplePromptsEditor,
} from '../sample-prompts/model.js?v=module-bootstrap-20260711-ir2';

function createSamplePromptsPathInput(value) {
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'field-input';
    input.value = value ?? '';
    input.title = '当前 sample_prompts 指向非 .txt 文件，保留为文件路径。';
    return input;
}

function createSamplePromptsEditor(value, originalValue = value, touched = false) {
    const editor = document.createElement('div');
    editor.className = 'field-input sample-prompts-editor';
    editor.dataset.originalContent = originalValue ?? '';
    editor.dataset.touched = touched ? '1' : '0';

    const rows = document.createElement('div');
    rows.className = 'sample-prompts-rows';

    editor.appendChild(rows);

    editor.addEventListener('input', (event) => {
        if (event.target?.closest?.('.sample-prompt-row')) {
            markSamplePromptsEditorTouched(editor);
        }
    });
    editor.addEventListener('change', (event) => {
        if (event.target?.closest?.('.sample-prompt-row')) {
            markSamplePromptsEditorTouched(editor);
        }
    });

    renderSamplePromptRows(editor, value ?? '');
    return editor;
}

function createSamplePromptAddButton(rowsWrap) {
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'btn btn-small sample-prompts-add-btn';
    addBtn.textContent = '添加行';
    addBtn.addEventListener('click', () => {
        const editor = rowsWrap.closest('.sample-prompts-editor');
        if (editor?.dataset.mode === 'text') {
            const textarea = editor.querySelector('.sample-prompts-textarea');
            if (textarea) {
                if (textarea.value && !textarea.value.endsWith('\n')) textarea.value += '\n';
                textarea.focus();
                textarea.setSelectionRange(textarea.value.length, textarea.value.length);
                markSamplePromptsEditorTouched(editor);
                handleFormFieldChange();
                return;
            }
        }
        appendSamplePromptRow(rowsWrap, blankSamplePromptRow());
        markSamplePromptsEditorTouched(editor);
        handleFormFieldChange();
    });
    return addBtn;
}

function createSamplePromptTextModeButton(editor) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-small sample-prompts-add-btn sample-prompts-mode-btn';
    btn.dataset.samplePromptsModeToggle = '1';
    updateSamplePromptModeButtonState(btn, editor);
    btn.addEventListener('click', () => {
        if (editor.dataset.mode === 'text') {
            switchSamplePromptsEditorToTableMode(editor);
        } else {
            switchSamplePromptsEditorToTextMode(editor);
        }
        updateSamplePromptModeButtonState(btn, editor);
        markSamplePromptsEditorTouched(editor);
        handleFormFieldChange();
    });
    return btn;
}

function updateSamplePromptModeButtonState(btn, editor) {
    if (!btn || !editor) return;
    const textMode = editor.dataset.mode === 'text';
    btn.textContent = textMode ? '表格模式' : '文本模式';
    btn.title = textMode ? '切回按列编辑提示词' : '保留注释、空行和原始参数格式';
    btn.setAttribute('aria-pressed', String(textMode));
}

export function setSamplePromptsEditorContent(editor, content) {
    if (!editor) return;
    editor.dataset.originalContent = content || '';
    editor.dataset.touched = '0';
    renderSamplePromptRows(editor, content || '');
    updateSamplePromptModeButtonState(editor.closest('.field-row')?.querySelector('[data-sample-prompts-mode-toggle]'), editor);
}

export function markSamplePromptsEditorTouched(editor) {
    if (editor) editor.dataset.touched = '1';
}

function renderSamplePromptRows(editor, content) {
    const rowsWrap = editor.querySelector('.sample-prompts-rows');
    if (!rowsWrap) return;
    rowsWrap.innerHTML = '';
    editor.dataset.mode = samplePromptsContentNeedsTextMode(content) ? 'text' : 'table';
    if (editor.dataset.mode === 'text') {
        const textarea = document.createElement('textarea');
        textarea.className = 'sample-prompts-textarea';
        textarea.value = content || '';
        textarea.spellcheck = false;
        textarea.addEventListener('input', () => markSamplePromptsEditorTouched(editor));
        rowsWrap.appendChild(textarea);
        return;
    }
    const rows = parseSamplePromptRows(content);
    for (const row of rows) {
        appendSamplePromptRow(rowsWrap, row);
    }
    updateSamplePromptRemoveButtons(rowsWrap);
}

function switchSamplePromptsEditorToTextMode(editor) {
    if (!editor || editor.dataset.mode === 'text') return;
    const rowsWrap = editor.querySelector('.sample-prompts-rows');
    if (!rowsWrap) return;
    const text = serializeSamplePromptsEditor(editor);
    rowsWrap.innerHTML = '';
    editor.dataset.mode = 'text';
    const textarea = document.createElement('textarea');
    textarea.className = 'sample-prompts-textarea';
    textarea.value = text;
    textarea.spellcheck = false;
    textarea.addEventListener('input', () => markSamplePromptsEditorTouched(editor));
    rowsWrap.appendChild(textarea);
    textarea.focus();
}

function switchSamplePromptsEditorToTableMode(editor) {
    if (!editor || editor.dataset.mode !== 'text') return;
    const rowsWrap = editor.querySelector('.sample-prompts-rows');
    if (!rowsWrap) return;
    const text = serializeSamplePromptsEditor(editor);
    rowsWrap.innerHTML = '';
    editor.dataset.mode = 'table';
    for (const row of parseSamplePromptRows(text)) {
        appendSamplePromptRow(rowsWrap, row);
    }
    updateSamplePromptRemoveButtons(rowsWrap);
    rowsWrap.querySelector('[data-sample-prompt-field="prompt"]')?.focus();
}

export {
    createSamplePromptsPathInput,
    createSamplePromptsEditor,
    createSamplePromptAddButton,
    createSamplePromptTextModeButton,
};
