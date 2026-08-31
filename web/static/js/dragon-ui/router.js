/* Page router for the Dragon trainer UI.
 * Handles full-page transitions between config sub-pages, dashboard, etc.
 * Uses fade in/out animation between page mounts.
 * Supports three return types from page loaders:
 *   1. string (HTML)
 *   2. Node (DOM element)
 *   3. { html: string, onMount: (wrapper) => void }
 */

import { findSubItem, isConfigCategory } from './category-map.js?v=dragon-ui-20260826v45';
import { scanForReveal } from './animations.js?v=dragon-ui-20260824v69';
import { dragonScrollBehavior, isDragonMotionEnabled } from './motion.js?v=dragon-ui-20260824v1';
import { renderTrainingWorkspaceNav } from './training-workspace-nav.js?v=dragon-ui-20260828v1';

let mountElement = null;
let currentPage = null;
let pageLoaders = {};
let navigationSequence = 0;
let mountedRouteUpdateSequence = 0;

export function initRouter(mountEl, loaders) {
    mountElement = mountEl;
    pageLoaders = loaders || {};
}

export async function navigate(route) {
    if (route.type === 'page') {
        const context = {
            taskId: route.taskId || null,
            sub: route.sub || null,
        };
        if (!route.force && await updateMountedPage(route.page, context)) return true;
        return renderPage(route.page, context);
    }

    if (route.type === 'category') {
        if (!isConfigCategory(route.categoryId)) return;
        const matchedSub = route.subId ? findSubItem(route.subId) : null;
        const subId = matchedSub?.categoryId === route.categoryId ? matchedSub.id : route.subId;
        if (!route.force && await updateMountedConfigCategory(route.categoryId, subId)) return true;
        if (!route.force && focusMountedConfigCategory(route.categoryId, subId)) return true;
        return renderPage('config', {
            categoryId: route.categoryId,
            subId: subId || null,
        });
    }

    if (route.type === 'sub') {
        const sub = findSubItem(route.subId);
        if (!sub) return;

        if (sub.isPage) {
            return renderPage(sub.isPage, { sub });
        } else if (isConfigCategory(sub.categoryId)) {
            if (!route.force && await updateMountedConfigCategory(sub.categoryId, sub.id)) return true;
            if (!route.force && focusMountedConfigCategory(sub.categoryId, sub.id)) return true;
            return renderPage('config', { categoryId: sub.categoryId, subId: sub.id });
        } else {
            return renderPage('config', { sub });
        }
    }
}

export function canLeaveCurrentPage() {
    if (typeof currentPage?.beforeLeave !== 'function') return true;
    return currentPage.beforeLeave() !== false;
}

export function isCurrentPage(pageType) {
    return currentPage?.pageType === pageType;
}

async function updateMountedPage(pageType, context) {
    if (currentPage?.pageType !== pageType || typeof currentPage.onRouteUpdate !== 'function') return false;
    const sequence = ++mountedRouteUpdateSequence;
    const updated = await currentPage.onRouteUpdate(context);
    if (sequence !== mountedRouteUpdateSequence) return true;
    if (updated === false) return false;
    currentPage.context = context;
    return true;
}

async function updateMountedConfigCategory(categoryId, subId) {
    if (currentPage?.pageType !== 'config' || currentPage.context?.categoryId !== categoryId) return false;
    if (typeof currentPage.onRouteUpdate !== 'function') return false;
    const sequence = ++mountedRouteUpdateSequence;
    const updated = await currentPage.onRouteUpdate({ categoryId, subId: subId || null });
    if (sequence !== mountedRouteUpdateSequence) return true;
    if (updated === false) return false;
    currentPage.context = { categoryId, subId: subId || null };
    return true;
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
        getComputedStyle(document.documentElement).getPropertyValue('--dragon-nav-height')
    ) || 44;
    const top = Math.max(0, window.scrollY + target.getBoundingClientRect().top - navHeight - 16);
    window.scrollTo({ top, behavior: dragonScrollBehavior() });

    page.querySelectorAll('.dragon-config-index-link').forEach((link) => {
        link.dataset.active = String(Boolean(subId) && link.dataset.configTarget === subId);
    });
    currentPage.context = { categoryId, subId: subId || null };
    return true;
}

async function renderPage(pageType, context) {
    const loader = pageLoaders[pageType];
    if (!loader) {
        mountElement.innerHTML = `${renderTrainingWorkspaceNav(pageType)}<div class="dragon-empty-state"><p>页面暂未实现</p></div>`;
        return;
    }

    const sequence = ++navigationSequence;
    mountedRouteUpdateSequence += 1;
    // End the previous page lifecycle before waiting for the leave animation or
    // the next loader. Otherwise an in-flight partial config transition can
    // commit stale context while the destination page is already loading.
    if (currentPage) {
        currentPage.onUnmount?.();
        currentPage = {
            ...currentPage,
            beforeLeave: null,
            onUnmount: null,
            onRouteUpdate: null,
        };
    }

    // Fade out current content
    const currentWrapper = mountElement.querySelector('.dragon-page-wrapper');
    if (currentPage && currentWrapper && isDragonMotionEnabled()) {
        currentWrapper.classList.add('dragon-page-leave');
        await new Promise((r) => setTimeout(r, 200));
        if (sequence !== navigationSequence) return;
    }

    // Keep route changes legible while slower APIs resolve. The previous page
    // has already left, so replace it with a stable status surface instead of
    // leaving an empty mount for several seconds.
    renderLoadingState(pageType);

    let content;
    try {
        content = await loader(context);
    } catch (error) {
        if (sequence !== navigationSequence) return;
        currentPage = { pageType, context, beforeLeave: null, onUnmount: null };
        renderLoadError(pageType, error);
        console.error(`[dragon-ui] ${pageLabel(pageType)}加载失败`, error);
        return false;
    }
    if (sequence !== navigationSequence) {
        content?.onUnmount?.();
        return;
    }
    mountElement.innerHTML = renderTrainingWorkspaceNav(pageType);
    mountElement.removeAttribute('aria-busy');
    if (content) {
        const wrapper = document.createElement('div');
        wrapper.className = 'dragon-page-wrapper dragon-page-enter';

        let onMount = null;
        let beforeLeave = null;
        let onUnmount = null;
        let onRouteUpdate = null;
        if (typeof content === 'string') {
            wrapper.innerHTML = content;
        } else if (content instanceof Node) {
            wrapper.appendChild(content);
        } else if (content && typeof content === 'object' && content.html != null) {
            wrapper.innerHTML = content.html;
            if (typeof content.onMount === 'function') {
                onMount = () => content.onMount(wrapper);
            }
            if (typeof content.beforeLeave === 'function') beforeLeave = content.beforeLeave;
            if (typeof content.onUnmount === 'function') onUnmount = content.onUnmount;
            if (typeof content.onRouteUpdate === 'function') onRouteUpdate = content.onRouteUpdate;
        }

        mountElement.appendChild(wrapper);
        currentPage = { pageType, context, beforeLeave, onUnmount, onRouteUpdate };

        requestAnimationFrame(() => {
            wrapper.classList.remove('dragon-page-enter');
            scanForReveal();
            if (onMount) onMount();
        });
    }
}

function renderLoadingState(pageType) {
    if (!mountElement) return;
    mountElement.setAttribute('aria-busy', 'true');
    mountElement.innerHTML = `
        ${renderTrainingWorkspaceNav(pageType)}
        <div class="dragon-route-loading" role="status" aria-live="polite">
            <span class="dragon-spinner" aria-hidden="true"></span>
            <div><strong>正在打开${pageLabel(pageType)}…</strong><span>正在读取最新内容</span></div>
        </div>
    `;
}

function renderLoadError(pageType, error) {
    if (!mountElement) return;
    mountElement.removeAttribute('aria-busy');
    const message = escapeHtml(error?.message || '页面数据读取失败');
    mountElement.innerHTML = `${renderTrainingWorkspaceNav(pageType)}<div class="dragon-empty-state" role="alert"><p>${message}</p><p>请确认 WebUI 服务仍在运行，然后刷新重试。</p></div>`;
}

function pageLabel(pageType) {
    const labels = {
        dashboard: '首页', config: '配置页面', 'live-training': '训练监控',
        history: '训练历史', queue: '训练队列', 'weight-analysis': '权重分析',
        'image-test': '生图测试', environment: '环境检测',
        'dataset-editor': '数据集配置', 'model-config': '模型配置',
        'global-settings': '全局设置', 'preview-workspace': '预览工作区',
        captioning: '外部 API 打标', tagging: '外部 API 打标',
        'captioning-prompts': '提示词预设',
        'captioning-results': '最终打标结果',
        'captioning-logs': '打标日志',
    };
    return labels[pageType] || '页面';
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[character]));
}

export async function refreshCurrentRoute() {
    if (!currentPage) return;
    await renderPage(currentPage.pageType, { ...currentPage.context, force: true });
}

export function getCurrentPage() {
    return currentPage;
}

export function destroyRouter() {
    navigationSequence += 1;
    mountedRouteUpdateSequence += 1;
    currentPage?.onUnmount?.();
    currentPage = null;
    pageLoaders = {};
    if (mountElement) {
        mountElement.removeAttribute('aria-busy');
        mountElement.replaceChildren();
    }
    mountElement = null;
}
