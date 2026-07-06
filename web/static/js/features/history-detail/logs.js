import { historyDetailLimitNotice } from './system.js?v=module-bootstrap-20260706-1';

const HISTORY_LOG_RENDER_BATCH_SIZE = 200;

export function createHistoryLogsRenderer({ state, deps }) {
    let consoleRenderToken = 0;

    function renderHistoryDetailLogs(payload) {
        const box = document.createElement('div');
        box.className = 'history-detail-log-block';
        syncHistoryLogConsoleState(payload);
        const logs = payload.logs || [];
        const commands = historyLogCommandEntries(payload, logs);
        box.classList.toggle('has-command', commands.length > 0);
        if (commands.length) box.appendChild(renderHistoryLogCommandCard(commands));

        const consoleShell = document.createElement('section');
        consoleShell.className = 'history-log-console';

        const toolbar = document.createElement('div');
        toolbar.className = 'history-log-toolbar';

        const searchWrap = document.createElement('label');
        searchWrap.className = 'history-log-search';
        searchWrap.setAttribute('aria-label', '搜索日志');
        const search = document.createElement('input');
        search.type = 'search';
        search.placeholder = '搜索 Error、Epoch...';
        search.value = state.logs.query;
        searchWrap.appendChild(search);

        const matchStatus = document.createElement('span');
        matchStatus.className = 'history-log-match-status';

        const previous = createHistoryLogNavButton('↑', '上一个匹配项');
        const next = createHistoryLogNavButton('↓', '下一个匹配项');
        const nav = document.createElement('div');
        nav.className = 'history-log-search-nav';
        nav.append(previous, next);

        const levelFilters = document.createElement('div');
        levelFilters.className = 'history-log-level-filters';
        levelFilters.setAttribute('role', 'tablist');
        levelFilters.setAttribute('aria-label', '日志级别筛选');
        for (const [level, label] of [
            ['all', '全部'],
            ['info', 'INFO'],
            ['warning', 'WARNING'],
            ['error', 'ERROR'],
        ]) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'history-log-level-filter';
            btn.dataset.level = level;
            btn.textContent = label;
            btn.classList.toggle('active', state.logs.level === level);
            btn.setAttribute('role', 'tab');
            btn.setAttribute('aria-selected', state.logs.level === level ? 'true' : 'false');
            btn.addEventListener('click', () => {
                state.logs.level = level;
                state.logs.matchIndex = 0;
                levelFilters.querySelectorAll('.history-log-level-filter').forEach((item) => {
                    const active = item.dataset.level === level;
                    item.classList.toggle('active', active);
                    item.setAttribute('aria-selected', active ? 'true' : 'false');
                });
                renderConsole();
            });
            levelFilters.appendChild(btn);
        }

        const download = renderHistoryLogDownloadControl(payload, logs);
        toolbar.append(searchWrap, matchStatus, nav, levelFilters, download);

        const notice = historyDetailLimitNotice(payload, 'logs', '日志');
        const pre = document.createElement('pre');
        pre.className = 'history-detail-pre history-log-output';
        const summary = document.createElement('div');
        summary.className = 'history-log-summary';

        const visibleLogs = logs
            .filter((record) => !historyLogCommandMatch(record.line || ''))
            .map((record) => {
                const text = historyLogRecordText(payload, record);
                return { text, tone: deps.logLineTone(stripAnsiCodes(text)) };
            });

        function renderConsole(options = {}) {
            const token = ++consoleRenderToken;
            const query = state.logs.query.trim();
            const matches = [];
            const filtered = visibleLogs.filter((item) => historyLogMatchesLevel(item.tone, state.logs.level));
            pre.dataset.rendering = 'true';
            pre.textContent = '';
            previous.disabled = true;
            next.disabled = true;
            matchStatus.textContent = query ? '搜索中...' : '';
            summary.textContent = `显示 ${filtered.length} / ${visibleLogs.length} 行`;
            if (!filtered.length) {
                const empty = document.createElement('span');
                empty.className = 'history-log-empty';
                empty.textContent = visibleLogs.length ? '当前筛选条件下没有日志。' : '无日志。';
                pre.appendChild(empty);
                finishConsoleRender(token, matches, query, options);
                return;
            }

            const appendBatch = (start = 0) => {
                if (token !== consoleRenderToken) return;
                const end = Math.min(start + HISTORY_LOG_RENDER_BATCH_SIZE, filtered.length);
                const fragment = document.createDocumentFragment();
                for (let index = start; index < end; index += 1) {
                    const item = filtered[index];
                    const line = document.createElement('span');
                    line.className = `history-log-line ${item.tone}`;
                    appendAnsiLogText(line, item.text, query, matches);
                    fragment.appendChild(line);
                }
                pre.appendChild(fragment);
                if (end < filtered.length) {
                    scheduleHistoryLogRenderBatch(() => appendBatch(end));
                    return;
                }
                finishConsoleRender(token, matches, query, options);
            };

            appendBatch();
        }

        function finishConsoleRender(token, matches, query, options = {}) {
            if (token !== consoleRenderToken) return;
            delete pre.dataset.rendering;
            if (!matches.length) {
                state.logs.matchIndex = 0;
            } else {
                state.logs.matchIndex = Math.min(state.logs.matchIndex, matches.length - 1);
            }
            matches.forEach((item, index) => item.classList.toggle('current', index === state.logs.matchIndex));
            matchStatus.textContent = query
                ? (matches.length ? `${state.logs.matchIndex + 1} / ${matches.length}` : '0 / 0')
                : '';
            previous.disabled = !matches.length;
            next.disabled = !matches.length;
            if (options.focusMatch && matches.length) {
                matches[state.logs.matchIndex].scrollIntoView({ block: 'center' });
            }
        }

        function moveMatch(offset) {
            if (pre.dataset.rendering === 'true') return;
            const count = pre.querySelectorAll('.history-log-match').length;
            if (!count) return;
            state.logs.matchIndex = (state.logs.matchIndex + offset + count) % count;
            renderConsole({ focusMatch: true });
        }

        search.addEventListener('input', (event) => {
            state.logs.query = event.target.value || '';
            state.logs.matchIndex = 0;
            renderConsole();
        });
        search.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter') return;
            event.preventDefault();
            moveMatch(event.shiftKey ? -1 : 1);
        });
        previous.addEventListener('click', () => moveMatch(-1));
        next.addEventListener('click', () => moveMatch(1));

        consoleShell.append(toolbar);
        if (notice) consoleShell.appendChild(notice);
        consoleShell.append(pre, summary);
        box.appendChild(consoleShell);
        renderConsole();
        return box;
    }

    function syncHistoryLogConsoleState(payload) {
        const key = payload?.task?.id
            || (payload?.summary?.selected_task_ids || []).join(',')
            || payload?.group?.key
            || payload?.mode
            || 'history';
        if (state.logs.payloadKey === key) return;
        state.logs = {
            payloadKey: key,
            query: '',
            level: 'all',
            matchIndex: 0,
        };
    }

    function createHistoryLogNavButton(label, title) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'history-log-nav-btn';
        btn.textContent = label;
        btn.title = title;
        btn.setAttribute('aria-label', title);
        return btn;
    }

    function scheduleHistoryLogRenderBatch(callback) {
        const schedule = window.requestAnimationFrame
            ? (fn) => window.requestAnimationFrame(fn)
            : (fn) => window.setTimeout(fn, 16);
        schedule(callback);
    }

    function historyLogMatchesLevel(tone, level) {
        if (level === 'all') return true;
        if (level === 'info') return !['warning', 'error'].includes(tone);
        return tone === level;
    }

    function historyLogRecordText(payload, record) {
        if (payload.mode === 'config_group') return deps.formatGroupTimelineLogRecord(record);
        return `${record.kind === 'progress' ? '[进度] ' : ''}${record.line || ''}`;
    }

    function historyLogCommandMatch(line) {
        const text = stripAnsiCodes(line);
        return text.match(/^(训练命令|预处理命令|续训命令|队列训练命令)\s*:\s*(.+)$/);
    }

    function historyLogCommandEntries(payload, logs) {
        const entries = [];
        const seen = new Set();
        const append = (label, command) => {
            const text = stripAnsiCodes(command).trim();
            if (!text || seen.has(text)) return;
            entries.push({ label, command: text });
            seen.add(text);
        };
        if (Array.isArray(payload?.task?.command)) {
            append(payload.task.job === 'preprocess' ? '预处理命令' : '训练命令', payload.task.command.join(' '));
        }
        for (const record of logs || []) {
            const match = historyLogCommandMatch(record.line || '');
            if (match) append(match[1], match[2]);
        }
        return entries;
    }

    function renderHistoryLogCommandCard(entries) {
        const box = document.createElement('section');
        box.className = 'history-command-card';
        for (const entry of entries) {
            const row = document.createElement('div');
            row.className = 'history-command-row';
            const details = document.createElement('details');
            const summary = document.createElement('summary');
            const primary = document.createElement('strong');
            primary.textContent = `${entry.label}: ${historyCommandScriptName(entry.command)}`;
            const hint = document.createElement('span');
            hint.textContent = '点击展开查看';
            summary.append(primary, hint);
            const command = document.createElement('code');
            command.textContent = entry.command;
            details.append(summary, command);

            const copy = document.createElement('button');
            copy.type = 'button';
            copy.className = 'btn btn-small history-command-copy';
            copy.textContent = '复制完整命令';
            copy.addEventListener('click', async () => {
                try {
                    await deps.copyText(entry.command);
                    copy.textContent = '已复制';
                    window.setTimeout(() => { copy.textContent = '复制完整命令'; }, 1200);
                } catch (e) {
                    alert('复制命令失败: ' + e.message);
                }
            });
            row.append(details, copy);
            box.appendChild(row);
        }
        return box;
    }

    function historyCommandScriptName(command) {
        const matches = String(command || '').match(/[^\s/\\]+\.py(?=\s|$)/g);
        return matches?.[matches.length - 1] || '完整参数';
    }

    function renderHistoryLogDownloadControl(payload, logs) {
        if (payload?.task?.id) {
            const link = document.createElement('a');
            link.className = 'btn btn-small history-log-download';
            link.href = `/api/training/history/${encodeURIComponent(payload.task.id)}/logs/download`;
            link.download = `${payload.task.id}.logs.jsonl`;
            link.textContent = '下载完整日志';
            link.title = '下载历史目录中的完整 logs.jsonl 文件';
            return link;
        }
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-small history-log-download';
        btn.textContent = '导出当前视图';
        btn.title = '导出当前合并视图中的日志';
        btn.addEventListener('click', () => {
            const text = (logs || []).map((record) => historyLogRecordText(payload, record)).join('\n');
            deps.downloadBlob(new Blob([text ? `${text}\n` : ''], { type: 'text/plain;charset=utf-8' }), 'history-group.log');
        });
        return btn;
    }

    function appendAnsiLogText(parent, text, query, matches) {
        for (const segment of ansiLogSegments(text)) {
            const node = segment.className ? document.createElement('span') : parent;
            if (segment.className) node.className = `history-log-${segment.className}`;
            appendHighlightedText(node, segment.text, query, matches);
            if (segment.className) parent.appendChild(node);
        }
    }

    function appendHighlightedText(parent, text, query, matches) {
        if (!query) {
            parent.appendChild(document.createTextNode(text));
            return;
        }
        const lowerText = text.toLowerCase();
        const lowerQuery = query.toLowerCase();
        let offset = 0;
        while (offset < text.length) {
            const index = lowerText.indexOf(lowerQuery, offset);
            if (index < 0) break;
            if (index > offset) parent.appendChild(document.createTextNode(text.slice(offset, index)));
            const mark = document.createElement('mark');
            mark.className = 'history-log-match';
            mark.textContent = text.slice(index, index + query.length);
            matches.push(mark);
            parent.appendChild(mark);
            offset = index + query.length;
        }
        if (offset < text.length) parent.appendChild(document.createTextNode(text.slice(offset)));
    }

    function ansiLogSegments(text) {
        const source = String(text || '');
        const segments = [];
        const regex = /\x1b\[([0-9;]*)m/g;
        let offset = 0;
        let className = '';
        let match;
        while ((match = regex.exec(source))) {
            if (match.index > offset) {
                segments.push({ text: stripAnsiCodes(source.slice(offset, match.index)), className });
            }
            className = ansiLogClassName(className, match[1]);
            offset = regex.lastIndex;
        }
        if (offset < source.length) {
            segments.push({ text: stripAnsiCodes(source.slice(offset)), className });
        }
        return segments.length ? segments : [{ text: stripAnsiCodes(source), className: '' }];
    }

    function ansiLogClassName(current, value) {
        let next = current;
        const codes = String(value || '0').split(';').map((item) => Number(item || 0));
        for (const code of codes) {
            if (code === 0 || code === 39) next = '';
            else if ([31, 91].includes(code)) next = 'ansi-red';
            else if ([32, 92].includes(code)) next = 'ansi-green';
            else if ([33, 93].includes(code)) next = 'ansi-yellow';
            else if ([34, 94].includes(code)) next = 'ansi-blue';
            else if ([35, 95].includes(code)) next = 'ansi-magenta';
            else if ([36, 96].includes(code)) next = 'ansi-cyan';
            else if ([37, 97].includes(code)) next = 'ansi-white';
        }
        return next;
    }

    function stripAnsiCodes(text) {
        return String(text || '').replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');
    }

    return { renderHistoryDetailLogs, stripAnsiCodes };
}
