/**
 * Compact config field row layout helpers.
 * Extracted from anima-app chunk 06.
 */
import {
    CONFIG_COMPACT_FIELD_GROUPS,
} from '../../config/catalog.js?v=module-bootstrap-20260714-stage-dataset5';
import { createFieldRow } from './form-fields.js?v=module-bootstrap-20260714-stage-dataset5';

export function appendFieldRows(content, fields, groupClass) {
    const compactGroups = CONFIG_COMPACT_FIELD_GROUPS[groupClass] || [];
    const usedLayouts = new Set();
    let index = 0;

    while (index < fields.length) {
        const [key] = fields[index];
        const compactLayout = compactGroups.find((layout) => {
            if (usedLayouts.has(layout)) return false;
            return layout.keys.includes(key);
        });

        if (!compactLayout) {
            content.appendChild(createFieldRow(fields[index][0], fields[index][1]));
            index += 1;
            continue;
        }

        usedLayouts.add(compactLayout);
        const compactKeys = new Set(compactLayout.keys);
        const grid = document.createElement('div');
        grid.className = ['config-field-grid', compactLayout.className].filter(Boolean).join(' ');

        while (index < fields.length && compactKeys.has(fields[index][0])) {
            const [compactKey, compactValue] = fields[index];
            const row = createFieldRow(compactKey, compactValue);
            row.classList.add('field-row-compact');
            grid.appendChild(row);
            index += 1;
        }

        if (grid.childElementCount <= 1) {
            const onlyRow = grid.firstElementChild;
            if (onlyRow) content.appendChild(onlyRow);
        } else {
            normalizeCompactGridColumns(grid);
            appendCompactGridFillers(grid);
            content.appendChild(grid);
        }
    }
}

function appendCompactGridFillers(grid) {
    grid.querySelectorAll('.field-row-filler').forEach((node) => node.remove());
    const columnCount = compactGridColumnCount(grid);
    if (columnCount <= 1) return;
    const remainder = grid.childElementCount % columnCount;
    if (remainder === 0) return;
    const fillerCount = columnCount - remainder;
    for (let index = 0; index < fillerCount; index += 1) {
        grid.appendChild(createCompactGridFiller());
    }
}

function createCompactGridFiller() {
    const filler = document.createElement('div');
    filler.className = 'field-row field-row-compact field-row-filler';
    filler.setAttribute('aria-hidden', 'true');
    filler.setAttribute('role', 'presentation');
    return filler;
}

function compactGridColumnCount(grid) {
    if (!grid) return 0;
    if (grid.classList.contains('config-field-grid-5col')) return 5;
    if (grid.classList.contains('config-field-grid-4col')) return 4;
    if (grid.classList.contains('config-field-grid-3col')) return 3;
    if (grid.classList.contains('config-field-grid-2col')) return 2;
    return 2;
}

function preferredCompactGridColumns(grid) {
    if (!grid) return 0;
    if (grid.classList.contains('config-field-grid-5col')) return 5;
    if (grid.classList.contains('config-field-grid-4col')) return 4;
    if (grid.classList.contains('config-field-grid-3col')) return 3;
    if (grid.classList.contains('config-field-grid-2col')) return 2;
    return 0;
}

function normalizeCompactGridColumns(grid) {
    const count = grid.childElementCount;
    // Keep an explicit layout preference from CONFIG_COMPACT_FIELD_GROUPS
    // (e.g. four boolean flags as 2x2, not auto-upgraded to 4 columns).
    const preferred = preferredCompactGridColumns(grid);
    grid.classList.remove('config-field-grid-2col', 'config-field-grid-3col', 'config-field-grid-4col', 'config-field-grid-5col');
    if (preferred === 2) {
        grid.classList.add('config-field-grid-2col');
        return;
    }
    if (preferred === 3) {
        grid.classList.add('config-field-grid-3col');
        return;
    }
    if (preferred === 4) {
        grid.classList.add('config-field-grid-4col');
        return;
    }
    if (preferred === 5) {
        grid.classList.add('config-field-grid-5col');
        return;
    }
    if (count >= 4) {
        grid.classList.add('config-field-grid-4col');
    } else if (count === 3) {
        grid.classList.add('config-field-grid-3col');
    } else if (count === 2) {
        grid.classList.add('config-field-grid-2col');
    }
}
