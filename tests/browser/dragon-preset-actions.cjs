/* Real UI, in-memory config mutations. No user files are modified. */
const assert = require('node:assert/strict');
const { mkdir } = require('node:fs/promises');
const { chromium } = require('playwright');
const base = process.env.DRAGON_QA_URL || 'http://127.0.0.1:20204';
const output = process.env.DRAGON_QA_OUTPUT || '/tmp/dragon-preset-actions';

async function scenario(browser, width, theme) {
    const groups = await (await fetch(`${base}/api/config/file-groups?kind=training`)).json();
    const source = groups.flatMap(g => g.files || []).find(f => f.trainable && !f.locked && !f.readonly);
    assert(source, 'requires an editable training config');
    const records = new Map();
    const errors = [];
    const context = await browser.newContext({ viewport: { width, height: 900 }, colorScheme: theme, reducedMotion: 'reduce' });
    await context.addInitScript(({ source, theme }) => {
        localStorage.setItem('anima_dragon_training_context', JSON.stringify({ configFile: source.path }));
        localStorage.setItem('anima_dragon_theme', theme);
        localStorage.setItem('anima_dragon_config_ui', JSON.stringify({ workspaceVersion: 1, presetCollapsed: false }));
    }, { source, theme });
    await context.route('**/api/**', async route => {
        const request = route.request();
        const url = new URL(request.url());
        const json = payload => route.fulfill({ json: payload });
        if (url.pathname === '/api/config/file-groups') return json(groups);
        if (url.pathname === '/api/config/raw' && records.has(url.searchParams.get('file'))) {
            return json({ ok: true, content: records.get(url.searchParams.get('file')) });
        }
        if (url.pathname === '/api/config/merged' && records.has(url.searchParams.get('config_file'))) {
            url.searchParams.set('config_file', source.path);
            url.searchParams.set('variant', source.method);
            url.searchParams.set('methods_subdir', source.methods_subdir);
            return route.fulfill({ response: await route.fetch({ url: url.href }) });
        }
        if (request.method() === 'GET') return route.continue();
        const data = request.postDataJSON();
        if (url.pathname === '/api/config/raw/patch-preview') return route.continue();
        if (url.pathname === '/api/config/raw/save-as') {
            if (records.has(data.file)) return json({ ok: false, error: '配置文件已存在' });
            records.set(data.file, data.content);
            groups.find(g => g.id === 'imported').files.push({ ...source, path: data.file,
                filename: data.file.split('/').pop(), label: data.file.split('/').pop() });
            return json({ ok: true });
        }
        if (url.pathname === '/api/config/raw/rename') {
            assert(records.has(data.source));
            records.set(data.target, records.get(data.source));
            records.delete(data.source);
            const file = groups.flatMap(g => g.files || []).find(f => f.path === data.source);
            Object.assign(file, { path: data.target, filename: data.target.split('/').pop(), label: data.target.split('/').pop() });
            return json({ ok: true });
        }
        throw new Error(`Unexpected mutation: ${request.method()} ${url.pathname}`);
    });
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('dialog', dialog => { errors.push(`native dialog: ${dialog.type()}`); dialog.dismiss(); });
    page.setDefaultTimeout(15000);
    try {
        await page.goto(`${base}/?ui=dragon#config/training-config/all`);
        const library = page.locator('[data-training-preset-library]');
        await library.waitFor({ state: 'visible' });
        await page.evaluate(() => document.fonts.ready);
        const bounds = await library.locator('.dragon-training-preset-pill').evaluateAll(buttons => buttons.map(button => {
            const rect = button.getBoundingClientRect();
            const label = button.querySelector('span').getBoundingClientRect();
            const parent = button.closest('[data-training-preset-library]').getBoundingClientRect();
            return { text: button.textContent.trim(), radius: getComputedStyle(button).borderRadius,
                fits: rect.left >= parent.left && rect.right <= parent.right && label.width > 0 && label.right <= rect.right };
        }));
        assert(bounds.length === 6 && bounds.every(b => b.fits), JSON.stringify(bounds));
        await page.screenshot({ path: `${output}/${theme}-${width}.png` });
        const action = name => library.locator(`[data-training-preset-action="${name}"]`);
        const dialog = page.locator('[data-dragon-dialog-host][open]');
        const input = dialog.locator('#dragon-dialog-input');
        async function nameAction(actionName, name) {
            await action(actionName).click();
            await dialog.waitFor();
            await input.fill(name);
            await dialog.locator('[data-dragon-dialog-confirm]').click();
            await page.waitForFunction(name => document.querySelector('.dragon-training-current-preset strong')?.textContent === `${name}.toml`, name);
        }
        await action('new').click();
        await dialog.waitFor();
        assert(await input.evaluate(el => document.activeElement === el));
        const dialogBounds = await dialog.evaluate(el => {
            const box = el.getBoundingClientRect();
            const field = el.querySelector('input').getBoundingClientRect();
            return { fits: box.left >= 0 && box.right <= innerWidth && box.top >= 0 && box.bottom <= innerHeight,
                fieldFits: field.left >= box.left && field.right <= box.right, overflow: el.scrollWidth - el.clientWidth };
        });
        assert(dialogBounds.fits && dialogBounds.fieldFits && dialogBounds.overflow <= 1, JSON.stringify(dialogBounds));
        await page.screenshot({ path: `${output}/${theme}-${width}-dialog.png` });
        await page.keyboard.press('Escape');
        assert(await action('new').evaluate(el => document.activeElement === el));
        await nameAction('new', 'qa-new');
        assert(records.get('configs/imported/qa-new.toml').includes('model_family'));
        await nameAction('rename', 'qa-renamed');
        assert(!records.has('configs/imported/qa-new.toml'));
        await nameAction('save-as', 'qa-copy');
        assert.equal(records.get('configs/imported/qa-copy.toml'), records.get('configs/imported/qa-renamed.toml'));
        await library.locator('[data-training-preset-import-file]').setInputFiles({ name: 'qa-import.toml', mimeType: 'text/plain', buffer: Buffer.from('output_name = "qa-import"\n') });
        await dialog.waitFor();
        await dialog.locator('[data-dragon-dialog-confirm]').click();
        await page.waitForFunction(() => document.querySelector('.dragon-training-current-preset strong')?.textContent === 'qa-import.toml');
        const download = page.waitForEvent('download');
        await action('export').click();
        assert.equal((await download).suggestedFilename(), 'qa-import.toml');
        assert.deepEqual(errors, []);
        console.log(`PASS ${theme} ${width}: text pills, built-in dialogs, create/rename/save-as/import/export`);
    } finally { await context.close(); }
}

(async () => {
    await mkdir(output, { recursive: true });
    const browser = await chromium.launch({ headless: true });
    try {
        for (const theme of ['light', 'dark']) for (const width of [1440, 390]) await scenario(browser, width, theme);
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
