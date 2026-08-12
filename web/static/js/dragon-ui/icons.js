/* Small Lucide icon renderer shared by the Dragon trainer UI. */

const ICON_PATHS = {
    activity: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    chart: '<path d="M3 3v18h18"/><path d="m7 16 4-5 4 3 4-6"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    cpu: '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    folder: '<path d="M3 7.5V6a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>',
    gauge: '<path d="m12 14 4-4"/><path d="M3.3 18a10 10 0 1 1 17.4 0"/><path d="M6.8 18h10.4"/>',
    hash: '<path d="M4 9h16M4 15h16M10 3 8 21M16 3l-2 18"/>',
    history: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/>',
    layers: '<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 17l9 5 9-5"/>',
    list: '<path d="M8 6h13M8 12h13M8 18h13"/><path d="M3 6h.01M3 12h.01M3 18h.01"/>',
    memory: '<path d="M6 19v2M10 19v2M14 19v2M18 19v2M6 3v2M10 3v2M14 3v2M18 3v2"/><rect width="18" height="14" x="3" y="5" rx="2"/><path d="M7 9h10v6H7z"/>',
    moon: '<path d="M20.8 15.2A9 9 0 0 1 8.8 3.2 9 9 0 1 0 20.8 15.2Z"/>',
    panels: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18M9 9v12"/>',
    settings: '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    stop: '<rect width="14" height="14" x="5" y="5" rx="2"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.42"/>',
    terminal: '<path d="m4 17 6-5-6-5M12 19h8"/>',
    thermometer: '<path d="M14 4a2 2 0 0 0-4 0v9.5a4 4 0 1 0 4 0Z"/><path d="M12 9v7"/>',
    trendDown: '<path d="m3 7 6 6 4-4 8 8"/><path d="M21 10v7h-7"/>',
    trendUp: '<path d="m3 17 6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
    zap: '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8Z"/>',
};

export function renderIcon(name, className = 'dragon-icon') {
    const paths = ICON_PATHS[name] || ICON_PATHS.activity;
    return `<svg class="${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;
}
