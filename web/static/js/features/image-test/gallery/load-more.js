import {
    GROUP_INITIAL_RENDER_COUNT,
    GROUP_LOAD_MORE_ROOT_MARGIN,
    GROUP_RENDER_INCREMENT,
} from './constants.js?v=module-bootstrap-20260714-stage-dataset5';

/**
 * 纯函数：在网格中定位某个 history group section。
 */
export function findGroupSection(groupKey) {
    return [...document.querySelectorAll('#image-test-grid .image-test-history-group[data-group-key]')]
        .find((element) => element.getAttribute('data-group-key') === groupKey);
}

/**
 * 工厂：注入 state / 渲染 / 虚拟窗口回调，避免循环依赖。
 *
 * @param {object} deps
 * @param {object} deps.state gallery 内部状态（需含 groupsByKey / renderWindowByGroup / loadMoreObserver）
 * @param {(body: Element, group: object, visibleCount: number) => void} deps.renderGroupBody
 * @param {() => void} deps.refreshVisibleOrderedKeys
 * @param {() => void} deps.scheduleVirtualWindowRefresh
 * @param {(groupKey: string) => Element|undefined|null} [deps.findGroupSectionFn]
 */
export function createLoadMoreApi({
    state,
    renderGroupBody,
    refreshVisibleOrderedKeys,
    scheduleVirtualWindowRefresh,
    findGroupSectionFn = findGroupSection,
}) {
    function visibleCountForGroup(groupKey, totalCount) {
        const current = Number(state.renderWindowByGroup.get(groupKey) || GROUP_INITIAL_RENDER_COUNT);
        return Math.max(0, Math.min(totalCount, current));
    }

    function createLoadMoreFooter(group, visibleCount) {
        const footer = document.createElement('div');
        footer.className = 'image-test-history-load-more';
        footer.dataset.imageTestLoadMoreFooter = group.key;
        footer.dataset.groupKey = group.key;

        const meta = document.createElement('span');
        meta.className = 'image-test-history-load-more-meta';
        meta.textContent = `已渲染 ${visibleCount} / ${group.items.length} 张`;

        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-small image-test-history-load-more-btn';
        button.textContent = `继续加载 ${Math.min(GROUP_RENDER_INCREMENT, group.items.length - visibleCount)} 张`;
        button.addEventListener('click', () => {
            expandGroupRenderWindow(group.key);
        });

        footer.append(meta, button);
        return footer;
    }

    function installLoadMoreObserver() {
        if (!('IntersectionObserver' in window)) return;
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                const groupKey = String(entry.target?.getAttribute?.('data-group-key') || '').trim();
                if (!groupKey) return;
                observer.unobserve(entry.target);
                expandGroupRenderWindow(groupKey);
            });
        }, {
            root: null,
            rootMargin: GROUP_LOAD_MORE_ROOT_MARGIN,
            threshold: 0,
        });
        document.querySelectorAll('[data-image-test-load-more-footer]').forEach((footer) => {
            observer.observe(footer);
        });
        state.loadMoreObserver = observer;
    }

    function disconnectLoadMoreObserver() {
        state.loadMoreObserver?.disconnect?.();
        state.loadMoreObserver = null;
    }

    function expandGroupRenderWindow(groupKey) {
        const group = state.groupsByKey.get(groupKey);
        if (!group) return;
        const previousCount = visibleCountForGroup(groupKey, group.items.length);
        const nextCount = Math.min(group.items.length, previousCount + GROUP_RENDER_INCREMENT);
        if (nextCount <= previousCount) return;
        state.renderWindowByGroup.set(groupKey, nextCount);
        refreshVisibleOrderedKeys();
        const section = findGroupSectionFn(groupKey);
        const body = section?.querySelector('.image-test-history-group-grid');
        if (body && !body.hidden) {
            renderGroupBody(body, group, nextCount);
        }
        updateLoadMoreFooter(section, group, nextCount);
        scheduleVirtualWindowRefresh();
    }

    function updateLoadMoreFooter(section, group, visibleCount) {
        if (!section) return;
        const existing = section.querySelector('[data-image-test-load-more-footer]');
        if (visibleCount >= group.items.length) {
            existing?.remove();
            return;
        }
        const nextFooter = createLoadMoreFooter(group, visibleCount);
        if (existing) {
            existing.replaceWith(nextFooter);
        } else {
            section.appendChild(nextFooter);
        }
        if (state.loadMoreObserver) {
            state.loadMoreObserver.observe(nextFooter);
        }
    }

    return {
        createLoadMoreFooter,
        installLoadMoreObserver,
        disconnectLoadMoreObserver,
        expandGroupRenderWindow,
        updateLoadMoreFooter,
        findGroupSection: findGroupSectionFn,
        visibleCountForGroup,
    };
}
