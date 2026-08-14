import { DRAGON_NAV_CATEGORIES } from './category-map.js?v=dragon-ui-20260814v43';
import { getThemePreference, setThemePreference } from './theme.js?v=dragon-ui-20260814v45';
import { renderIcon } from './icons.js?v=dragon-ui-20260812v35';
import { switchToClassicUI } from '../shared/ui-mode.js?v=dragon-ui-20260814v45';

let openCategoryId = null;
let mobileMenuOpen = false;
let settingsOpen = false;
let navRootElement = null;
let onNavigate = null;
let globalAbortController = null;

const NAV_SHORTCUTS = [
    { id: 'home', label: '首页', compactLabel: '首页', icon: 'home', hash: '#dashboard', iconOnly: true },
    { id: 'configs', label: '配置文件', compactLabel: '配置', icon: 'folder', hash: '#config/training-config', iconOnly: true },
    { id: 'datasets', label: '数据集', compactLabel: '数据集', icon: 'database', hash: '#dataset-editor', iconOnly: true },
    { id: 'history', label: '训练历史', compactLabel: '历史', icon: 'history', hash: '#history', iconOnly: true },
];

export function initNav(callback) {
    destroyNav();
    globalAbortController = new AbortController();
    onNavigate = callback;
    navRootElement = document.getElementById('dragon-nav');
    if (!navRootElement) return;
    renderNav();
    bindGlobalEvents();
}

function renderNav() {
    navRootElement.innerHTML = `
        <nav class="dragon-nav" id="dragon-nav-bar">
            <div class="dragon-nav-content">
                <div class="dragon-nav-list">
                    <div class="dragon-nav-brand-wrap">
                        <button class="dragon-nav-link dragon-nav-brand" id="dragon-nav-brand" type="button" aria-label="打开界面设置" aria-expanded="false" aria-controls="dragon-nav-settings">
                            ${renderTrainerLogo()}
                            <span class="dragon-nav-brand-name">Dragon trainer</span>
                        </button>
                        ${renderSettingsPanel()}
                    </div>
                    <ul class="dragon-nav-categories" aria-label="主导航">
                        ${DRAGON_NAV_CATEGORIES.map(renderCategoryItem).join('')}
                    </ul>
                    <div class="dragon-nav-utilities" aria-label="快捷入口">
                        ${NAV_SHORTCUTS.map(renderShortcutButton).join('')}
                    </div>
                    <button class="dragon-nav-link dragon-nav-mobile-menu-btn" id="dragon-mobile-menu-toggle" type="button" aria-label="打开导航菜单" aria-expanded="false" aria-controls="dragon-nav-mobile-panel">
                        <span class="dragon-nav-mobile-menu-icon" aria-hidden="true"></span>
                    </button>
                </div>
                ${renderMobilePanel()}
            </div>
        </nav>
        <div class="dragon-nav-scrim" id="dragon-nav-scrim"></div>
    `;
    bindNavEvents();
    bindScrollListener();
}

function renderShortcutButton(shortcut) {
    const iconOnlyClass = shortcut.iconOnly ? ' dragon-nav-utility-button--icon' : '';
    const label = shortcut.iconOnly ? '' : `<span class="dragon-nav-utility-label">${shortcut.label}</span>`;
    return `<button class="dragon-nav-utility-button${iconOnlyClass}" type="button" data-nav-shortcut="${shortcut.id}" data-target-hash="${shortcut.hash}" data-tooltip="${shortcut.label}" aria-label="${shortcut.label}">${renderIcon(shortcut.icon, 'dragon-nav-utility-icon')}${label}</button>`;
}

function renderTrainerLogo() {
    return `<span class="dragon-trainer-logo" aria-hidden="true"><i></i><i></i><i></i></span>`;
}

function renderSettingsPanel() {
    const preference = getThemePreference();
    return `
        <div class="dragon-nav-settings" id="dragon-nav-settings" aria-hidden="true">
            <div class="dragon-nav-settings-head">
                <span class="dragon-nav-settings-logo">${renderTrainerLogo()}</span>
                <div><strong>Dragon trainer</strong><span>显示与界面模式</span></div>
            </div>
            <div class="dragon-nav-settings-group">
                <span class="dragon-nav-settings-label">外观</span>
                <div class="dragon-theme-segment" role="radiogroup" aria-label="界面外观">
                    ${themeOption('system', '跟随系统', 'settings', preference)}
                    ${themeOption('light', '浅色', 'sun', preference)}
                    ${themeOption('dark', '深色', 'moon', preference)}
                </div>
            </div>
            <button class="dragon-nav-settings-command" id="dragon-ui-toggle" type="button">
                ${renderIcon('panels', 'dragon-nav-settings-icon')}<span><strong>经典界面</strong><small>返回基础工作台</small></span>
            </button>
        </div>
    `;
}

function themeOption(value, label, icon, current) {
    return `<button type="button" role="radio" aria-checked="${value === current}" data-theme-choice="${value}">${renderIcon(icon, 'dragon-theme-choice-icon')}<span>${label}</span></button>`;
}

function renderCategoryItem(category) {
    const layout = category.layout || 'config';
    const groupsHtml = (category.groups || []).map((group, index) => {
        const elevated = group.elevated || index === 0 ? ' dragon-nav-flyout-group-elevated' : '';
        return `
            <div class="dragon-nav-flyout-group${elevated}">
                ${group.header ? `<div class="dragon-nav-flyout-group-header">${group.header}</div>` : ''}
                ${group.items.map((item) => `
                    <button class="dragon-nav-flyout-link" type="button" data-sub-id="${item.id}">
                        <span class="dragon-nav-flyout-link-label">${item.label}</span>
                        ${item.desc ? `<span class="dragon-nav-flyout-link-desc">${item.desc}</span>` : ''}
                    </button>
                `).join('')}
            </div>
        `;
    }).join('');
    return `
        <li class="dragon-nav-item dragon-nav-item-category" data-category-id="${category.id}" data-flyout-layout="${layout}">
            <button class="dragon-nav-link dragon-nav-category-btn" type="button">${category.label}</button>
            <div class="dragon-nav-flyout dragon-nav-flyout--${layout}"><div class="dragon-nav-flyout-inner">${groupsHtml}</div></div>
        </li>
    `;
}

function renderMobilePanel() {
    const sections = DRAGON_NAV_CATEGORIES.map((category) => `
        <section class="dragon-nav-mobile-section">
            <h2>${category.label}</h2>
            <div class="dragon-nav-mobile-links">
                ${category.groups.flatMap((group) => group.items).map((item) => `
                    <button class="dragon-nav-mobile-link" type="button" data-category-id="${category.id}" data-sub-id="${item.id}">
                        <span>${item.label}</span>${item.desc ? `<small>${item.desc}</small>` : ''}
                    </button>
                `).join('')}
            </div>
        </section>
    `).join('');
    const shortcuts = NAV_SHORTCUTS.map((shortcut) => `
        <button class="dragon-nav-mobile-shortcut" type="button" data-nav-shortcut="${shortcut.id}" data-target-hash="${shortcut.hash}">
            ${renderIcon(shortcut.icon, 'dragon-nav-mobile-shortcut-icon')}<span>${shortcut.compactLabel}</span>
        </button>
    `).join('');
    return `<div class="dragon-nav-mobile-panel" id="dragon-nav-mobile-panel" aria-hidden="true"><div class="dragon-nav-mobile-panel-inner"><div class="dragon-nav-mobile-shortcuts" aria-label="快捷入口">${shortcuts}</div>${sections}</div></div>`;
}

function bindNavEvents() {
    document.getElementById('dragon-nav-brand')?.addEventListener('click', (event) => {
        event.stopPropagation();
        toggleSettings();
    });
    document.querySelectorAll('[data-theme-choice]').forEach((button) => {
        button.addEventListener('click', (event) => {
            event.stopPropagation();
            setThemePreference(button.dataset.themeChoice);
            document.querySelectorAll('[data-theme-choice]').forEach((item) => item.setAttribute('aria-checked', String(item === button)));
        });
    });
    document.getElementById('dragon-ui-toggle')?.addEventListener('click', async () => {
        const allowed = await onNavigate?.({ type: 'external', target: 'classic-ui' });
        if (allowed !== false) switchToClassicUI();
    });
    document.querySelectorAll('.dragon-nav-item-category').forEach(bindCategoryEvents);
    document.querySelectorAll('.dragon-nav-flyout-link').forEach((link) => {
        link.addEventListener('click', (event) => {
            event.stopPropagation();
            navigateToSubItem(link.closest('[data-category-id]')?.dataset.categoryId, link.dataset.subId);
        });
    });
    document.querySelectorAll('.dragon-nav-mobile-link').forEach((link) => {
        link.addEventListener('click', () => navigateToSubItem(link.dataset.categoryId, link.dataset.subId));
    });
    document.querySelectorAll('[data-nav-shortcut]').forEach((button) => {
        button.addEventListener('click', (event) => {
            event.stopPropagation();
            navigateToShortcut(button.dataset.targetHash);
        });
    });
    document.getElementById('dragon-nav-scrim')?.addEventListener('click', closeAllMenus);
    document.getElementById('dragon-mobile-menu-toggle')?.addEventListener('click', (event) => {
        event.stopPropagation();
        mobileMenuOpen ? closeMobileMenu() : openMobileMenu();
    });
    updateShortcutState();
}

function bindCategoryEvents(item) {
    const categoryId = item.dataset.categoryId;
    let hoverTimer = null;
    item.addEventListener('mouseenter', () => { hoverTimer = setTimeout(() => openDropdown(categoryId), 120); });
    item.addEventListener('mouseleave', () => { if (hoverTimer) clearTimeout(hoverTimer); });
    item.querySelector('.dragon-nav-category-btn')?.addEventListener('click', (event) => {
        event.stopPropagation();
        if (event.detail > 0) event.currentTarget.blur();
        openCategoryId === categoryId ? closeDropdown() : openDropdown(categoryId);
    });
}

function navigateToSubItem(categoryId, subId) {
    closeAllMenus();
    const sub = DRAGON_NAV_CATEGORIES.flatMap((category) => category.groups.flatMap((group) => group.items)).find((item) => item.id === subId);
    const targetHash = sub?.isPage ? `#${subId}` : `#config/${categoryId}/${subId}`;
    if (window.location.hash !== targetHash) window.location.hash = targetHash;
    else onNavigate?.({ type: sub?.isPage ? 'page' : 'category', page: sub?.isPage, categoryId, subId });
}

function navigateToShortcut(targetHash) {
    closeAllMenus();
    if (!targetHash) return;
    if (targetHash === '#dashboard') {
        if (window.location.hash === '' || window.location.hash === '#dashboard') {
            window.scrollTo({ top: 0, behavior: 'smooth' });
            onNavigate?.({ type: 'page', page: 'dashboard' });
            return;
        }
        window.location.hash = targetHash;
        return;
    }
    if (window.location.hash !== targetHash) {
        window.location.hash = targetHash;
        return;
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateShortcutState() {
    const hash = window.location.hash;
    const activeById = {
        home: hash === '' || hash === '#dashboard',
        configs: hash.startsWith('#config/'),
        datasets: hash.startsWith('#dataset-editor'),
        history: hash.startsWith('#history'),
    };
    document.querySelectorAll('[data-nav-shortcut]').forEach((button) => {
        const active = Boolean(activeById[button.dataset.navShortcut]);
        button.dataset.active = String(active);
        if (active) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
    });
}

function toggleSettings() {
    settingsOpen ? closeSettings() : openSettings();
}

function openSettings() {
    closeDropdown();
    closeMobileMenu();
    settingsOpen = true;
    document.getElementById('dragon-nav-settings')?.setAttribute('aria-hidden', 'false');
    document.getElementById('dragon-nav-brand')?.setAttribute('aria-expanded', 'true');
}

function closeSettings() {
    settingsOpen = false;
    document.getElementById('dragon-nav-settings')?.setAttribute('aria-hidden', 'true');
    document.getElementById('dragon-nav-brand')?.setAttribute('aria-expanded', 'false');
}

function openDropdown(categoryId) {
    if (openCategoryId === categoryId) return;
    closeDropdown();
    closeSettings();
    openCategoryId = categoryId;
    document.getElementById('dragon-nav-bar')?.setAttribute('data-flyout-open', 'true');
    document.querySelector(`.dragon-nav-item[data-category-id="${categoryId}"]`)?.setAttribute('data-open', 'true');
    document.getElementById('dragon-nav-scrim')?.setAttribute('data-visible', 'true');
}

function closeDropdown() {
    if (!openCategoryId) return;
    document.getElementById('dragon-nav-bar')?.removeAttribute('data-flyout-open');
    document.querySelector(`.dragon-nav-item[data-category-id="${openCategoryId}"]`)?.removeAttribute('data-open');
    document.getElementById('dragon-nav-scrim')?.setAttribute('data-visible', 'false');
    openCategoryId = null;
}

function openMobileMenu() {
    closeDropdown();
    closeSettings();
    mobileMenuOpen = true;
    document.getElementById('dragon-nav-bar')?.setAttribute('data-mobile-menu-open', 'true');
    document.getElementById('dragon-nav-mobile-panel')?.setAttribute('aria-hidden', 'false');
    const button = document.getElementById('dragon-mobile-menu-toggle');
    button?.setAttribute('aria-expanded', 'true');
    button?.setAttribute('aria-label', '关闭导航菜单');
    document.body.dataset.dragonMobileMenuOpen = '';
}

function closeMobileMenu() {
    if (!mobileMenuOpen) return;
    mobileMenuOpen = false;
    document.getElementById('dragon-nav-bar')?.removeAttribute('data-mobile-menu-open');
    document.getElementById('dragon-nav-mobile-panel')?.setAttribute('aria-hidden', 'true');
    const button = document.getElementById('dragon-mobile-menu-toggle');
    button?.setAttribute('aria-expanded', 'false');
    button?.setAttribute('aria-label', '打开导航菜单');
    delete document.body.dataset.dragonMobileMenuOpen;
}

function closeAllMenus() {
    closeDropdown();
    closeMobileMenu();
    closeSettings();
}

function bindScrollListener() {
    const nav = document.getElementById('dragon-nav-bar');
    const update = () => { if (nav) nav.dataset.scrolled = window.scrollY > 0 ? 'true' : 'false'; };
    update();
    window.addEventListener('scroll', update, {
        passive: true,
        signal: globalAbortController?.signal,
    });
}

function bindGlobalEvents() {
    const signal = globalAbortController?.signal;
    document.addEventListener('click', (event) => {
        if (!event.target.closest('.dragon-nav-brand-wrap') && !event.target.closest('.dragon-nav-item')) closeAllMenus();
    }, { signal });
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeAllMenus(); }, { signal });
    window.addEventListener('hashchange', updateShortcutState, { signal });
    window.addEventListener('dragon-route-restored', updateShortcutState, { signal });
    window.addEventListener('resize', () => { if (window.innerWidth > 833) closeMobileMenu(); }, { passive: true, signal });
}

export function destroyNav() {
    globalAbortController?.abort();
    globalAbortController = null;
    closeAllMenus();
    navRootElement?.replaceChildren();
    navRootElement = null;
    onNavigate = null;
    openCategoryId = null;
    mobileMenuOpen = false;
    settingsOpen = false;
    delete document.body.dataset.dragonMobileMenuOpen;
}
