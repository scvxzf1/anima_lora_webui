import { APPLE_NAV_CATEGORIES } from './category-map.js?v=apple-ui-20260812v33';
import { getThemePreference, setThemePreference } from './theme.js?v=apple-ui-20260812v33';
import { renderIcon } from './icons.js?v=apple-ui-20260812v33';
import { switchToClassicUI } from '../shared/ui-mode.js?v=apple-ui-20260812v33';

let openCategoryId = null;
let mobileMenuOpen = false;
let settingsOpen = false;
let navRootElement = null;
let onNavigate = null;

const NAV_SHORTCUTS = [
    { id: 'configs', label: '配置文件', compactLabel: '配置', icon: 'folder', hash: '#config/training-config' },
    { id: 'datasets', label: '数据集', compactLabel: '数据集', icon: 'database', hash: '#dataset-editor' },
    { id: 'history', label: '训练历史', compactLabel: '历史', icon: 'history', hash: '#history' },
];

export function initNav(callback) {
    onNavigate = callback;
    navRootElement = document.getElementById('apple-nav');
    if (!navRootElement) return;
    renderNav();
    bindGlobalEvents();
}

function renderNav() {
    navRootElement.innerHTML = `
        <nav class="apple-nav" id="apple-nav-bar">
            <div class="apple-nav-content">
                <div class="apple-nav-list">
                    <div class="apple-nav-brand-wrap">
                        <button class="apple-nav-link apple-nav-brand" id="apple-nav-brand" type="button" aria-label="打开界面设置" aria-expanded="false" aria-controls="apple-nav-settings">
                            ${renderTrainerLogo()}
                            <span class="apple-nav-brand-name">Anima LoRA</span>
                        </button>
                        ${renderSettingsPanel()}
                    </div>
                    <ul class="apple-nav-categories" aria-label="主导航">
                        ${APPLE_NAV_CATEGORIES.map(renderCategoryItem).join('')}
                    </ul>
                    <div class="apple-nav-utilities" aria-label="存储快捷入口">
                        ${NAV_SHORTCUTS.map(renderShortcutButton).join('')}
                    </div>
                    <button class="apple-nav-link apple-nav-mobile-menu-btn" id="apple-mobile-menu-toggle" type="button" aria-label="打开导航菜单" aria-expanded="false" aria-controls="apple-nav-mobile-panel">
                        <span class="apple-nav-mobile-menu-icon" aria-hidden="true"></span>
                    </button>
                </div>
                ${renderMobilePanel()}
            </div>
        </nav>
        <div class="apple-nav-scrim" id="apple-nav-scrim"></div>
    `;
    bindNavEvents();
    bindScrollListener();
}

function renderShortcutButton(shortcut) {
    return `<button class="apple-nav-utility-button" type="button" data-nav-shortcut="${shortcut.id}" data-target-hash="${shortcut.hash}" data-tooltip="${shortcut.label}" aria-label="${shortcut.label}" title="${shortcut.label}">${renderIcon(shortcut.icon, 'apple-nav-utility-icon')}</button>`;
}

function renderTrainerLogo() {
    return `<span class="apple-trainer-logo" aria-hidden="true"><i></i><i></i><i></i></span>`;
}

function renderSettingsPanel() {
    const preference = getThemePreference();
    return `
        <div class="apple-nav-settings" id="apple-nav-settings" aria-hidden="true">
            <div class="apple-nav-settings-head">
                <span class="apple-nav-settings-logo">${renderTrainerLogo()}</span>
                <div><strong>Anima LoRA</strong><span>显示与界面模式</span></div>
            </div>
            <div class="apple-nav-settings-group">
                <span class="apple-nav-settings-label">外观</span>
                <div class="apple-theme-segment" role="radiogroup" aria-label="界面外观">
                    ${themeOption('system', '跟随系统', 'settings', preference)}
                    ${themeOption('light', '浅色', 'sun', preference)}
                    ${themeOption('dark', '深色', 'moon', preference)}
                </div>
            </div>
            <button class="apple-nav-settings-command" id="apple-ui-toggle" type="button">
                ${renderIcon('panels', 'apple-nav-settings-icon')}<span><strong>经典界面</strong><small>返回基础工作台</small></span>
            </button>
        </div>
    `;
}

function themeOption(value, label, icon, current) {
    return `<button type="button" role="radio" aria-checked="${value === current}" data-theme-choice="${value}">${renderIcon(icon, 'apple-theme-choice-icon')}<span>${label}</span></button>`;
}

function renderCategoryItem(category) {
    const layout = category.layout || 'config';
    const groupsHtml = (category.groups || []).map((group, index) => {
        const elevated = group.elevated || index === 0 ? ' apple-nav-flyout-group-elevated' : '';
        return `
            <div class="apple-nav-flyout-group${elevated}">
                ${group.header ? `<div class="apple-nav-flyout-group-header">${group.header}</div>` : ''}
                ${group.items.map((item) => `
                    <button class="apple-nav-flyout-link" type="button" data-sub-id="${item.id}">
                        <span class="apple-nav-flyout-link-label">${item.label}</span>
                        ${item.desc ? `<span class="apple-nav-flyout-link-desc">${item.desc}</span>` : ''}
                    </button>
                `).join('')}
            </div>
        `;
    }).join('');
    return `
        <li class="apple-nav-item apple-nav-item-category" data-category-id="${category.id}" data-flyout-layout="${layout}">
            <button class="apple-nav-link apple-nav-category-btn" type="button">${category.label}</button>
            <div class="apple-nav-flyout apple-nav-flyout--${layout}"><div class="apple-nav-flyout-inner">${groupsHtml}</div></div>
        </li>
    `;
}

function renderMobilePanel() {
    const sections = APPLE_NAV_CATEGORIES.map((category) => `
        <section class="apple-nav-mobile-section">
            <h2>${category.label}</h2>
            <div class="apple-nav-mobile-links">
                ${category.groups.flatMap((group) => group.items).map((item) => `
                    <button class="apple-nav-mobile-link" type="button" data-category-id="${category.id}" data-sub-id="${item.id}">
                        <span>${item.label}</span>${item.desc ? `<small>${item.desc}</small>` : ''}
                    </button>
                `).join('')}
            </div>
        </section>
    `).join('');
    const shortcuts = NAV_SHORTCUTS.map((shortcut) => `
        <button class="apple-nav-mobile-shortcut" type="button" data-nav-shortcut="${shortcut.id}" data-target-hash="${shortcut.hash}">
            ${renderIcon(shortcut.icon, 'apple-nav-mobile-shortcut-icon')}<span>${shortcut.compactLabel}</span>
        </button>
    `).join('');
    return `<div class="apple-nav-mobile-panel" id="apple-nav-mobile-panel" aria-hidden="true"><div class="apple-nav-mobile-panel-inner"><div class="apple-nav-mobile-shortcuts" aria-label="存储快捷入口">${shortcuts}</div>${sections}</div></div>`;
}

function bindNavEvents() {
    document.getElementById('apple-nav-brand')?.addEventListener('click', (event) => {
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
    document.getElementById('apple-ui-toggle')?.addEventListener('click', () => switchToClassicUI());
    document.querySelectorAll('.apple-nav-item-category').forEach(bindCategoryEvents);
    document.querySelectorAll('.apple-nav-flyout-link').forEach((link) => {
        link.addEventListener('click', (event) => {
            event.stopPropagation();
            navigateToSubItem(link.closest('[data-category-id]')?.dataset.categoryId, link.dataset.subId);
        });
    });
    document.querySelectorAll('.apple-nav-mobile-link').forEach((link) => {
        link.addEventListener('click', () => navigateToSubItem(link.dataset.categoryId, link.dataset.subId));
    });
    document.querySelectorAll('[data-nav-shortcut]').forEach((button) => {
        button.addEventListener('click', (event) => {
            event.stopPropagation();
            navigateToShortcut(button.dataset.targetHash);
        });
    });
    document.getElementById('apple-nav-scrim')?.addEventListener('click', closeAllMenus);
    document.getElementById('apple-mobile-menu-toggle')?.addEventListener('click', (event) => {
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
    item.querySelector('.apple-nav-category-btn')?.addEventListener('click', (event) => {
        event.stopPropagation();
        if (event.detail > 0) event.currentTarget.blur();
        openCategoryId === categoryId ? closeDropdown() : openDropdown(categoryId);
    });
}

function navigateToSubItem(categoryId, subId) {
    closeAllMenus();
    const sub = APPLE_NAV_CATEGORIES.flatMap((category) => category.groups.flatMap((group) => group.items)).find((item) => item.id === subId);
    const targetHash = sub?.isPage ? `#${subId}` : `#config/${categoryId}/${subId}`;
    if (window.location.hash !== targetHash) window.location.hash = targetHash;
    else onNavigate?.({ type: sub?.isPage ? 'page' : 'category', page: sub?.isPage, categoryId, subId });
}

function navigateToShortcut(targetHash) {
    closeAllMenus();
    if (!targetHash) return;
    if (window.location.hash !== targetHash) {
        window.location.hash = targetHash;
        return;
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateShortcutState() {
    const hash = window.location.hash;
    const activeById = {
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
    document.getElementById('apple-nav-settings')?.setAttribute('aria-hidden', 'false');
    document.getElementById('apple-nav-brand')?.setAttribute('aria-expanded', 'true');
}

function closeSettings() {
    settingsOpen = false;
    document.getElementById('apple-nav-settings')?.setAttribute('aria-hidden', 'true');
    document.getElementById('apple-nav-brand')?.setAttribute('aria-expanded', 'false');
}

function openDropdown(categoryId) {
    if (openCategoryId === categoryId) return;
    closeDropdown();
    closeSettings();
    openCategoryId = categoryId;
    document.getElementById('apple-nav-bar')?.setAttribute('data-flyout-open', 'true');
    document.querySelector(`.apple-nav-item[data-category-id="${categoryId}"]`)?.setAttribute('data-open', 'true');
    document.getElementById('apple-nav-scrim')?.setAttribute('data-visible', 'true');
}

function closeDropdown() {
    if (!openCategoryId) return;
    document.getElementById('apple-nav-bar')?.removeAttribute('data-flyout-open');
    document.querySelector(`.apple-nav-item[data-category-id="${openCategoryId}"]`)?.removeAttribute('data-open');
    document.getElementById('apple-nav-scrim')?.setAttribute('data-visible', 'false');
    openCategoryId = null;
}

function openMobileMenu() {
    closeDropdown();
    closeSettings();
    mobileMenuOpen = true;
    document.getElementById('apple-nav-bar')?.setAttribute('data-mobile-menu-open', 'true');
    document.getElementById('apple-nav-mobile-panel')?.setAttribute('aria-hidden', 'false');
    const button = document.getElementById('apple-mobile-menu-toggle');
    button?.setAttribute('aria-expanded', 'true');
    button?.setAttribute('aria-label', '关闭导航菜单');
    document.body.dataset.appleMobileMenuOpen = '';
}

function closeMobileMenu() {
    if (!mobileMenuOpen) return;
    mobileMenuOpen = false;
    document.getElementById('apple-nav-bar')?.removeAttribute('data-mobile-menu-open');
    document.getElementById('apple-nav-mobile-panel')?.setAttribute('aria-hidden', 'true');
    const button = document.getElementById('apple-mobile-menu-toggle');
    button?.setAttribute('aria-expanded', 'false');
    button?.setAttribute('aria-label', '打开导航菜单');
    delete document.body.dataset.appleMobileMenuOpen;
}

function closeAllMenus() {
    closeDropdown();
    closeMobileMenu();
    closeSettings();
}

function bindScrollListener() {
    const nav = document.getElementById('apple-nav-bar');
    const update = () => { if (nav) nav.dataset.scrolled = window.scrollY > 0 ? 'true' : 'false'; };
    update();
    window.addEventListener('scroll', update, { passive: true });
}

function bindGlobalEvents() {
    document.addEventListener('click', (event) => {
        if (!event.target.closest('.apple-nav-brand-wrap') && !event.target.closest('.apple-nav-item')) closeAllMenus();
    });
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeAllMenus(); });
    window.addEventListener('hashchange', updateShortcutState);
    window.addEventListener('resize', () => { if (window.innerWidth > 833) closeMobileMenu(); }, { passive: true });
}
