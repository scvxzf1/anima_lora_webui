/* Shared page primitives for Dragon's monitoring and system workspaces. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

export function renderToolHero({ eyebrow, title, description, badge = '', actions = '' }) {
    return `
        <header class="dragon-tool-hero dragon-reveal">
            <div class="dragon-tool-hero-copy">
                ${eyebrow ? `<span class="dragon-eyebrow">${eyebrow}</span>` : ''}
                <h1>${title}</h1>
                <p>${description}</p>
            </div>
            ${(badge || actions) ? `<div class="dragon-tool-hero-side">${badge}${actions ? `<div class="dragon-tool-actions">${actions}</div>` : ''}</div>` : ''}
        </header>
    `;
}

export function renderToolButton(icon, label, action, className = 'dragon-btn-secondary', attributes = '') {
    return `<button class="dragon-btn ${className}" type="button" data-tool-action="${action}" ${attributes}>${renderIcon(icon, 'dragon-btn-icon')}<span>${label}</span></button>`;
}

export function renderStatusRegion(attribute, message = '', tone = '') {
    return `<p class="dragon-config-feedback dragon-status-region${message ? ' dragon-config-feedback-visible' : ''}" ${attribute} data-tone="${tone}" role="status" aria-live="polite">${message}</p>`;
}

export function formatToolPath(value, fallback = '未设置') {
    return String(value || '').trim() || fallback;
}
