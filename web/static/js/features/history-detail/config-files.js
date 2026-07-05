import {
    createHistoryDetailCopyButton,
    historyDetailEmptyText,
    historyDetailIcon,
    historyDetailIconButton,
    historyDetailIconLink,
    historyDetailFlashToolButton,
    historyDetailRow as createHistoryDetailRow,
    historyDetailRunRoot,
    historyDetailSection,
    selectAllTextOnDoubleClick,
} from './ui.js?v=module-bootstrap-20260705-3';

export function createHistoryConfigFilesRenderer({ ctx, deps }) {
    const historyDetailCopyButton = (value, label) => createHistoryDetailCopyButton(ctx.dom.copyText, value, label);
    const historyDetailRow = (label, value, options = {}) => createHistoryDetailRow(label, value, options, {
        copyButton: historyDetailCopyButton,
    });

    function renderHistoryDetailConfig(payload) {
        const isGroup = payload.mode === 'config_group';
        const content = isGroup
            ? deps.configGroupTimelineSummary(payload)
            : (payload.config_toml || '# 无配置快照');
        const filename = historyConfigSnapshotFilename(payload, isGroup);
        const viewer = document.createElement('div');
        viewer.className = 'history-config-viewer';

        const toolbar = document.createElement('div');
        toolbar.className = 'history-config-toolbar';

        const searchWrap = document.createElement('label');
        searchWrap.className = 'history-config-search';
        searchWrap.appendChild(historyDetailIcon('search'));
        const searchInput = document.createElement('input');
        searchInput.type = 'search';
        searchInput.placeholder = '搜索参数名或值';
        searchInput.setAttribute('aria-label', '搜索配置快照');
        searchWrap.appendChild(searchInput);

        const matchStatus = document.createElement('span');
        matchStatus.className = 'history-config-match-status';
        matchStatus.textContent = '未搜索';

        const nav = document.createElement('div');
        nav.className = 'history-config-nav';
        const prevBtn = historyDetailIconButton('上一个匹配项', 'chevron-up', () => {
            if (!matchTotal) return;
            currentMatch = (currentMatch - 1 + matchTotal) % matchTotal;
            renderCode({ scrollToCurrent: true });
        });
        const nextBtn = historyDetailIconButton('下一个匹配项', 'chevron-down', () => {
            if (!matchTotal) return;
            currentMatch = (currentMatch + 1) % matchTotal;
            renderCode({ scrollToCurrent: true });
        });
        nav.append(prevBtn, nextBtn);

        const actions = document.createElement('div');
        actions.className = 'history-config-actions';
        const copyBtn = historyDetailIconButton('复制全部配置', 'copy', async () => {
            try {
                await deps.copyText(content);
                historyDetailFlashToolButton(copyBtn, '已复制全部配置', '复制全部配置');
            } catch (e) {
                alert('复制配置失败: ' + e.message);
            }
        });
        const downloadBtn = historyDetailIconButton('下载配置快照', 'download', () => {
            deps.downloadBlob(new Blob([content], { type: 'text/plain;charset=utf-8' }), filename);
            historyDetailFlashToolButton(downloadBtn, `已下载 ${filename}`, '下载配置快照');
        });
        actions.append(copyBtn, downloadBtn);

        toolbar.append(searchWrap, matchStatus, nav, actions);

        const pre = document.createElement('pre');
        pre.className = 'history-detail-pre history-config-code';
        pre.setAttribute('aria-label', '配置快照代码');

        let searchText = '';
        let currentMatch = 0;
        let matchTotal = 0;
        const renderCode = (options = {}) => {
            matchTotal = historyConfigMatchCount(content, searchText);
            if (!matchTotal) currentMatch = 0;
            else if (currentMatch >= matchTotal) currentMatch = matchTotal - 1;
            renderHistoryConfigCode(pre, content, searchText, currentMatch);
            matchStatus.textContent = searchText
                ? (matchTotal ? `${currentMatch + 1} / ${matchTotal}` : '无匹配')
                : '未搜索';
            prevBtn.disabled = !matchTotal;
            nextBtn.disabled = !matchTotal;
            if (options.scrollToCurrent && matchTotal) {
                requestAnimationFrame(() => {
                    pre.querySelector('.history-config-search-hit.current')?.scrollIntoView({ block: 'center' });
                });
            }
        };
        searchInput.addEventListener('input', () => {
            searchText = searchInput.value || '';
            currentMatch = 0;
            renderCode({ scrollToCurrent: true });
        });
        searchInput.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter') return;
            event.preventDefault();
            if (event.shiftKey) prevBtn.click();
            else nextBtn.click();
        });
        renderCode();

        viewer.append(toolbar, pre);
        return viewer;
    }

    function historyConfigSnapshotFilename(payload, isGroup) {
        if (isGroup) {
            const group = payload.group || {};
            return `${safeDownloadName(deps.configGroupLabel(group) || 'history-config-group')}.summary.txt`;
        }
        const task = payload.task || {};
        return `${safeDownloadName(task.id || task.name || 'history-task')}.config.snapshot.toml`;
    }

    function safeDownloadName(value) {
        return String(value || 'download')
            .replace(/[\\/:*?"<>|\s]+/g, '-')
            .replace(/-+/g, '-')
            .replace(/^-|-$/g, '')
            .slice(0, 120) || 'download';
    }

    function renderHistoryConfigCode(pre, text, searchText, currentMatch) {
        pre.innerHTML = '';
        const code = document.createElement('code');
        code.className = 'history-config-code-lines';
        const query = String(searchText || '').trim();
        const state = {
            query,
            queryLower: query.toLowerCase(),
            index: 0,
            currentMatch,
        };
        const lines = String(text || '').split(/\r?\n/);
        lines.forEach((line, index) => {
            const row = document.createElement('span');
            row.className = 'history-config-line';
            const gutter = document.createElement('span');
            gutter.className = 'history-config-line-no';
            gutter.textContent = String(index + 1);
            const body = document.createElement('span');
            body.className = 'history-config-line-body';
            renderHistoryConfigLine(body, line, state);
            row.append(gutter, body);
            code.appendChild(row);
        });
        pre.appendChild(code);
    }

    function renderHistoryConfigLine(parent, line, state) {
        if (line === '') {
            parent.appendChild(document.createTextNode(' '));
            return;
        }
        const commentIndex = historyConfigCommentIndex(line);
        const codePart = commentIndex >= 0 ? line.slice(0, commentIndex) : line;
        const commentPart = commentIndex >= 0 ? line.slice(commentIndex) : '';
        const section = codePart.match(/^(\s*)(\[\[?.+\]\]?)(\s*)$/);
        if (section) {
            appendSearchableConfigText(parent, section[1], '', state);
            appendSearchableConfigText(parent, section[2], 'history-config-token-section', state);
            appendSearchableConfigText(parent, section[3], '', state);
        } else {
            const assignment = codePart.match(/^(\s*)([^=:#]+?)(\s*[=:]\s*)(.*)$/);
            if (assignment && assignment[2].trim()) {
                appendSearchableConfigText(parent, assignment[1], '', state);
                appendSearchableConfigText(parent, assignment[2], 'history-config-token-key', state);
                appendSearchableConfigText(parent, assignment[3], 'history-config-token-operator', state);
                renderHistoryConfigValueTokens(parent, assignment[4], state);
            } else {
                renderHistoryConfigValueTokens(parent, codePart, state);
            }
        }
        if (commentPart) {
            appendSearchableConfigText(parent, commentPart, 'history-config-token-comment', state);
        }
    }

    function historyConfigCommentIndex(line) {
        let quote = '';
        let escaped = false;
        for (let i = 0; i < line.length; i += 1) {
            const ch = line[i];
            if (escaped) {
                escaped = false;
                continue;
            }
            if (quote && ch === '\\') {
                escaped = true;
                continue;
            }
            if (ch === '"' || ch === "'") {
                if (!quote) quote = ch;
                else if (quote === ch) quote = '';
                continue;
            }
            if (!quote && ch === '#') return i;
        }
        return -1;
    }

    function renderHistoryConfigValueTokens(parent, text, state) {
        let i = 0;
        while (i < text.length) {
            const ch = text[i];
            if (ch === '"' || ch === "'") {
                const end = historyConfigStringEnd(text, i, ch);
                const token = text.slice(i, end);
                appendSearchableConfigText(parent, token, historyConfigStringClass(token), state);
                i = end;
                continue;
            }
            const rest = text.slice(i);
            const word = rest.match(/^[A-Za-z0-9_.+\-/\\]+/);
            if (word) {
                const token = word[0];
                appendSearchableConfigText(parent, token, historyConfigBareTokenClass(token), state);
                i += token.length;
                continue;
            }
            appendSearchableConfigText(parent, ch, /[\[\]{},]/.test(ch) ? 'history-config-token-operator' : '', state);
            i += 1;
        }
    }

    function historyConfigStringEnd(text, start, quote) {
        let escaped = false;
        for (let i = start + 1; i < text.length; i += 1) {
            const ch = text[i];
            if (escaped) {
                escaped = false;
                continue;
            }
            if (ch === '\\') {
                escaped = true;
                continue;
            }
            if (ch === quote) return i + 1;
        }
        return text.length;
    }

    function historyConfigStringClass(token) {
        return historyConfigLooksLikePath(token) ? 'history-config-token-path' : 'history-config-token-string';
    }

    function historyConfigBareTokenClass(token) {
        if (/^(true|false|null)$/i.test(token)) return 'history-config-token-bool';
        if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(token)) return 'history-config-token-number';
        if (historyConfigLooksLikePath(token)) return 'history-config-token-path';
        return '';
    }

    function historyConfigLooksLikePath(token) {
        return /(?:\/|\\|\.(?:toml|json|jsonl|txt|safetensors|pt|pth|ckpt)\b)/i.test(String(token || ''));
    }

    function appendSearchableConfigText(parent, text, className, state) {
        const value = String(text || '');
        if (!value) return;
        if (!state.query) {
            parent.appendChild(historyConfigTokenSpan(value, className));
            return;
        }
        const lower = value.toLowerCase();
        let cursor = 0;
        while (cursor < value.length) {
            const hit = lower.indexOf(state.queryLower, cursor);
            if (hit < 0) break;
            if (hit > cursor) {
                parent.appendChild(historyConfigTokenSpan(value.slice(cursor, hit), className));
            }
            const mark = document.createElement('mark');
            mark.className = 'history-config-search-hit';
            mark.classList.toggle('current', state.index === state.currentMatch);
            mark.textContent = value.slice(hit, hit + state.query.length);
            parent.appendChild(mark);
            state.index += 1;
            cursor = hit + state.query.length;
        }
        if (cursor < value.length) {
            parent.appendChild(historyConfigTokenSpan(value.slice(cursor), className));
        }
    }

    function historyConfigTokenSpan(text, className) {
        if (!className) return document.createTextNode(text);
        const span = document.createElement('span');
        span.className = className;
        span.textContent = text;
        return span;
    }

    function historyConfigMatchCount(text, searchText) {
        const needle = String(searchText || '').trim().toLowerCase();
        if (!needle) return 0;
        const haystack = String(text || '').toLowerCase();
        let count = 0;
        let index = 0;
        while (index < haystack.length) {
            const found = haystack.indexOf(needle, index);
            if (found < 0) break;
            count += 1;
            index = found + needle.length;
        }
        return count;
    }

    function renderHistoryDetailPaths(payload) {
        if (payload.mode === 'config_group') {
            const box = document.createElement('div');
            box.className = 'history-detail-kv';
            const group = payload.group || {};
            const summary = payload.summary || {};
            [
                ['配置文件', deps.configGroupLabel(group)],
                ['源配置', group.history_source_config_file || '-'],
                ['合并训练数', `${summary.task_count || 0}`],
                ['时间范围', `${summary.started_at_text || '-'} → ${summary.finished_at_text || '未结束'}`],
            ].forEach(([label, value]) => box.appendChild(historyDetailRow(label, value)));
            return box;
        }

        const task = payload.task || {};
        const rootPath = historyDetailRunRoot(task);
        const browser = document.createElement('div');
        browser.className = 'history-detail-file-browser';
        if (rootPath) {
            browser.appendChild(historyDetailFileRoot(rootPath));
        }
        const list = document.createElement('div');
        list.className = 'history-detail-kv history-file-list';
        for (const [label, value, artifactKey] of deps.runtimePathItems(task)) {
            list.appendChild(historyDetailFileRow(task, label, value, artifactKey));
        }
        if (!list.childElementCount) {
            list.appendChild(historyDetailEmptyText('这个任务没有记录可展示的文件路径。'));
        }
        browser.appendChild(list);
        return browser;
    }

    function historyDetailFileRoot(rootPath) {
        const root = document.createElement('div');
        root.className = 'history-file-root';
        const label = document.createElement('span');
        label.textContent = '基础目录';
        const code = document.createElement('code');
        code.textContent = rootPath;
        code.title = rootPath;
        selectAllTextOnDoubleClick(code);
        root.append(label, code, historyDetailCopyButton(rootPath, '基础目录'));
        return root;
    }

    function historyDetailFileRow(task, label, value, artifactKey) {
        const rawValue = String(value || '-');
        const row = document.createElement('div');
        row.className = 'history-file-row has-file-actions';
        const key = document.createElement('span');
        key.textContent = label;
        const val = document.createElement('code');
        val.textContent = rawValue;
        val.title = rawValue;
        selectAllTextOnDoubleClick(val);
        const actions = document.createElement('div');
        actions.className = 'history-file-actions';
        actions.appendChild(historyDetailCopyButton(rawValue, `${label}完整路径`));
        if (artifactKey && task.id) {
            actions.append(
                historyDetailIconLink('查看文件', 'eye', deps.historyArtifactUrl(task, artifactKey)),
                historyDetailIconLink('下载文件', 'download', deps.historyArtifactUrl(task, artifactKey, { download: true }), {
                    download: true,
                }),
            );
        }
        row.append(key, val, actions);
        return row;
    }

    function renderHistoryDetailConfigFiles(payload) {
        const box = document.createElement('div');
        box.className = 'history-detail-config-files';
        box.append(
            historyDetailSection(
                payload.mode === 'config_group' ? '配置分组摘要' : '配置快照',
                renderHistoryDetailConfig(payload),
                'history-detail-section config-snapshot',
            ),
            historyDetailSection(
                payload.mode === 'config_group' ? '合并范围' : '文件路径',
                renderHistoryDetailPaths(payload),
                'history-detail-section file-paths',
            ),
        );
        return box;
    }


    return { renderHistoryDetailConfigFiles, renderHistoryDetailConfig, renderHistoryDetailPaths };
}
