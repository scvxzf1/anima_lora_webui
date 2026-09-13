/* Read-only resource-control QA; all API writes are blocked. */
const assert = require('node:assert/strict');
const { mkdir, writeFile } = require('node:fs/promises');
const { chromium } = require('playwright');

const baseURL = process.env.DRAGON_QA_URL || 'http://127.0.0.1:20207/';
const output = process.env.DRAGON_QA_OUTPUT || '/tmp/auto-swap-resource-qa';

async function scenario(browser, width, theme) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce' });
    const errors = [], writes = [];
    await context.route('**/api/**', route => {
        if (!['GET', 'HEAD', 'OPTIONS'].includes(route.request().method())) {
            writes.push(route.request().method());
            return route.abort('blockedbyclient');
        }
        return route.continue();
    });
    await context.addInitScript(theme => {
        localStorage.setItem('anima_dragon_theme', theme);
        localStorage.setItem('anima_dragon_config_ui', JSON.stringify({ workspaceVersion: 1, presetCollapsed: true }));
    }, theme);
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.setDefaultTimeout(20000);
    try {
        await page.goto(new URL('?ui=dragon#config/training-config/all', baseURL).href);
        await page.locator('.dragon-config-all-toolbar').waitFor();
        await page.evaluate(() => document.fonts.ready);
        const field = key => page.locator(`#dragon-config-fields [data-key="${key}"]`);
        await field('model_family').fill('krea2_raw');
        await field('model_family').blur();
        await page.locator('[data-config-field-search]').fill('auto_block_swap');
        const auto = field('auto_block_swap');
        if (await auto.getAttribute('data-checked') === 'true') await auto.click();
        const reserve = field('auto_block_swap_vram_reserve_percent');
        const preference = field('auto_block_swap_preference');
        assert(await reserve.isDisabled());
        assert(await preference.isDisabled());
        await auto.click();
        assert(await reserve.isEnabled());
        assert(await preference.isEnabled());
        await field('auto_block_swap_mode').selectOption('dynamic');
        await reserve.fill('25.5');
        assert(await reserve.evaluate(input => input.checkValidity()));
        await reserve.fill('91');
        assert.equal(await reserve.evaluate(input => input.checkValidity()), false);
        await reserve.fill('25.5');
        await preference.selectOption('ram');
        assert.deepEqual(await preference.locator('option').allTextContents(), ['均衡', '优先节省显存', '优先节省内存']);
        const patch = await page.evaluate(async () => {
            const values = await import('/static/js/dragon-ui/pages/config-values.js?v=auto-block-swap-20260908-v3');
            const changed = {};
            for (const key of ['auto_block_swap_vram_reserve_percent', 'auto_block_swap_preference']) {
                const input = document.querySelector(`#dragon-config-fields [data-key="${key}"]`);
                changed[key] = values.serializeConfigValue(input, values.displayConfigValue(key, {}));
            }
            return values.prepareConfigPatch(changed, {});
        });
        assert.deepEqual(patch, { auto_block_swap_vram_reserve_percent: 25.5, auto_block_swap_preference: 'ram' });
        await reserve.scrollIntoViewIfNeeded();
        await page.screenshot({ path: `${output}/${theme}-${width}-reserve.png` });
        await preference.scrollIntoViewIfNeeded();
        await page.screenshot({ path: `${output}/${theme}-${width}-preference.png` });
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
        const bounds = await preference.evaluate(input => {
            const r = input.getBoundingClientRect();
            return { left: r.left, right: r.right, width: innerWidth };
        });
        assert(bounds.left >= 0 && bounds.right <= bounds.width);
        assert.deepEqual(errors, []);
        assert.deepEqual(writes, []);
        return { width, theme, patch, errors, writes };
    } finally {
        await context.close();
    }
}

(async () => {
    await mkdir(output, { recursive: true });
    const browser = await chromium.launch({ headless: true });
    try {
        const results = [];
        for (const theme of ['light', 'dark']) {
            for (const width of [1440, 390]) results.push(await scenario(browser, width, theme));
        }
        await writeFile(`${output}/results.json`, JSON.stringify(results, null, 2));
        console.log(JSON.stringify(results));
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
