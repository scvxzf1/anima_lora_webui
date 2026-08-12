/* Page router for Apple-style UI.
 * Handles full-page transitions between config sub-pages, dashboard, etc.
 * Uses fade in/out animation between page mounts.
 * Supports three return types from page loaders:
 *   1. string (HTML)
 *   2. Node (DOM element)
 *   3. { html: string, onMount: (wrapper) => void }
 */

import { findSubItem, isConfigCategory } from './category-map.js?v=apple-ui-20260812v33';
import { scanForReveal } from './animations.js?v=apple-ui-20260812v33';

let mountElement = null;
let currentPage = null;
let pageLoaders = {};

export function initRouter(mountEl, loaders) {
    mountElement = mountEl;
    pageLoaders = loaders || {};
}

export async function navigate(route) {
    if (route.type === 'page') {
        await renderPage(route.page, {});
        return;
    }

    if (route.type === 'category') {
        if (!isConfigCategory(route.categoryId)) return;
        if (!route.force && focusMountedConfigCategory(route.categoryId, route.subId)) return;
        await renderPage('config', {
            categoryId: route.categoryId,
            subId: route.subId || null,
        });
        return;
    }

    if (route.type === 'sub') {
        const sub = findSubItem(route.subId);
        if (!sub) return;

        if (sub.isPage) {
            await renderPage(sub.isPage, { sub });
        } else if (isConfigCategory(sub.categoryId)) {
            if (!route.force && focusMountedConfigCategory(sub.categoryId, sub.id)) return;
            await renderPage('config', { categoryId: sub.categoryId, subId: sub.id });
        } else {
            await renderPage('config', { sub });
        }
    }
}

function focusMountedConfigCategory(categoryId, subId) {
    if (currentPage?.pageType !== 'config' || currentPage.context?.categoryId !== categoryId) {
        return false;
    }

    const page = mountElement?.querySelector(`[data-config-category="${categoryId}"]`);
    if (!page) return false;

    const target = subId
        ? page.querySelector(`[data-config-entry="${subId}"]`)
        : page;
    if (!target) return false;

    const navHeight = Number.parseFloat(
        getComputedStyle(document.documentElement).getPropertyValue('--apple-nav-height')
    ) || 44;
    const top = Math.max(0, window.scrollY + target.getBoundingClientRect().top - navHeight - 16);
    window.scrollTo({ top, behavior: 'smooth' });

    page.querySelectorAll('.apple-config-index-link').forEach((link) => {
        link.dataset.active = String(Boolean(subId) && link.dataset.configTarget === subId);
    });
    currentPage.context = { categoryId, subId: subId || null };
    return true;
}

async function renderPage(pageType, context) {
    const loader = pageLoaders[pageType];
    if (!loader) {
        mountElement.innerHTML = '<div class="apple-empty-state"><p>页面暂未实现</p></div>';
        return;
    }

    // Fade out current content
    if (currentPage && mountElement.firstElementChild) {
        mountElement.firstElementChild.classList.add('apple-page-leave');
        await new Promise((r) => setTimeout(r, 200));
    }

    // Render new page
    const content = await loader(context);
    mountElement.innerHTML = '';
    if (content) {
        const wrapper = document.createElement('div');
        wrapper.className = 'apple-page-wrapper apple-page-enter';

        let onMount = null;
        if (typeof content === 'string') {
            wrapper.innerHTML = content;
        } else if (content instanceof Node) {
            wrapper.appendChild(content);
        } else if (content && typeof content === 'object' && content.html != null) {
            wrapper.innerHTML = content.html;
            if (typeof content.onMount === 'function') {
                onMount = () => content.onMount(wrapper);
            }
        }

        mountElement.appendChild(wrapper);
        currentPage = { pageType, context };

        requestAnimationFrame(() => {
            scanForReveal();
            if (onMount) onMount();
        });
    }
}

export async function refreshCurrentRoute() {
    if (!currentPage) return;
    await renderPage(currentPage.pageType, { ...currentPage.context, force: true });
}

export function getCurrentPage() {
    return currentPage;
}
