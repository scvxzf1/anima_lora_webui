/* Read-only browser QA against a running WebUI. Requires Playwright on NODE_PATH. */
const assert = require('node:assert/strict');
const { mkdir, writeFile } = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');

const baseURL = process.env.DRAGON_QA_URL || 'http://127.0.0.1:20203/';
const output = process.env.DRAGON_QA_OUTPUT || '/tmp/dragon-config-visual-qa';
const route = '?ui=dragon#config/training-config/all';
const viewports = [[2499, 1447], [1920, 1080], [1440, 900], [1280, 720], [390, 844]];
const results = [];

async function settle(page) {
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function snapshot(page, name) {
    await settle(page);
    await page.screenshot({ path: path.join(output, `${name}.png`) });
}

async function checkBounds(page, selector, label) {
    const bounds = await page.locator(selector).evaluate(element => {
        const rect = element.getBoundingClientRect();
        return { x: rect.x, y: rect.y, right: rect.right, bottom: rect.bottom, width: innerWidth, height: innerHeight };
    });
    assert(bounds.x >= -1 && bounds.right <= bounds.width + 1, `${label}: horizontal clipping ${JSON.stringify(bounds)}`);
    assert(bounds.y >= -1 && bounds.bottom <= bounds.height + 1, `${label}: vertical clipping ${JSON.stringify(bounds)}`);
}

async function checkSurface(page) {
    const state = await page.evaluate(() => {
        const root = document.querySelector('.dragon-config-category-page');
        const style = getComputedStyle(root);
        const rgb = value => value.match(/[\d.]+/g).slice(0, 3).map(Number);
        const luminance = color => rgb(color).map(v => {
            v /= 255;
            return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
        }).reduce((sum, v, i) => sum + v * [.2126, .7152, .0722][i], 0);
        const contrast = (a, b) => (Math.max(luminance(a), luminance(b)) + .05) / (Math.min(luminance(a), luminance(b)) + .05);
        const button = getComputedStyle(root.querySelector('[data-training-action="start"]'));
        const field = getComputedStyle(root.querySelector('.dragon-field-label-text'));
        return {
            overflow: document.documentElement.scrollWidth - innerWidth,
            canvas: style.backgroundColor,
            contrast: contrast(field.color, style.backgroundColor),
            buttonContrast: contrast(button.color, button.backgroundColor),
            labelSize: parseFloat(field.fontSize),
        };
    });
    assert(state.overflow <= 1, `page overflow: ${state.overflow}`);
    assert(state.contrast >= 4.5, `label contrast: ${state.contrast}`);
    assert(state.buttonContrast >= 4.5, `primary contrast: ${state.buttonContrast}`);
    assert(state.labelSize >= 14, `label size: ${state.labelSize}`);
    return state;
}

async function checkHelp(page, name) {
    const trigger = page.locator('[data-help-key="model_family"]');
    await trigger.click();
    const dialog = page.locator('#dragon-config-help-dialog[open]');
    await dialog.waitFor();
    assert.equal(await dialog.locator('[data-config-help-title]').textContent(), await trigger.getAttribute('data-help-label'));
    await checkBounds(page, '#dragon-config-help-dialog', name);
    assert.equal(await dialog.evaluate(e => getComputedStyle(e, '::backdrop').backdropFilter), 'none');
    await snapshot(page, `${name}-help`);
    await page.keyboard.press('Escape');
    assert.equal(await trigger.evaluate(e => e === document.activeElement), true, 'help focus restoration');
}

async function checkDrawer(page, name) {
    const toolbarToggle = page.locator('.dragon-config-all-toolbar [data-config-preset-toggle]');
    if (await toolbarToggle.getAttribute('aria-expanded') === 'false') await toolbarToggle.click();
    await checkBounds(page, '.dragon-training-preset-library', name);
    const buttons = page.locator('.dragon-training-preset-toolbar button:visible');
    for (const button of await buttons.all()) {
        assert(await button.getAttribute('aria-label'), 'unnamed preset icon');
        assert(await button.getAttribute('title'), 'missing preset tooltip');
    }
    await snapshot(page, `${name}-drawer`);
    await page.locator('.dragon-training-preset-library [data-config-preset-toggle]').click();
    assert.equal(await toolbarToggle.getAttribute('aria-expanded'), 'false');
}

async function checkDraftAndSearch(page, name) {
    const input = page.locator('.dragon-config-category-page [data-key="pretrained_model_name_or_path"]');
    const field = page.locator('.dragon-config-block[data-config-field-key="pretrained_model_name_or_path"]');
    const original = await input.inputValue();
    await input.fill(`${original}.qa`);
    await input.blur();
    assert.equal(await field.getAttribute('data-dirty'), 'true');
    await snapshot(page, `${name}-dirty`);
    await field.locator('[data-config-reset-field]').click();
    assert.equal(await input.inputValue(), original);
    const search = page.locator('[data-config-field-search]');
    await search.fill('pretrained_model_name_or_path');
    await page.waitForFunction(() => document.querySelector('#dragon-config-fields').dataset.searching === 'true');
    assert.equal(await field.isVisible(), true);
    assert.equal(await page.locator('.dragon-config-block[hidden]:visible').count(), 0, 'responsive layout overrides hidden fields');
    await search.fill('no_such_parameter_visual_qa');
    await page.locator('[data-config-filter-empty]').waitFor({ state: 'visible' });
    assert.equal(await page.locator('.dragon-config-block:visible').count(), 0);
    await snapshot(page, `${name}-empty`);
    await search.fill('');
    await page.waitForFunction(() => document.querySelector('#dragon-config-fields').dataset.searching === 'false');
}

async function checkPickers(page, name) {
    for (const [trigger, dialog, close] of [
        ['[data-model-quick-action="open"]', '.dragon-model-quick-dialog', '[data-model-quick-action="close"]'],
        ['[data-config-dataset-action="open"]', '.dragon-dataset-picker-dialog', '[data-config-dataset-action="close"]'],
    ]) {
        await page.locator(trigger).click();
        await page.locator(`${dialog}[open]`).waitFor();
        await page.locator(`${dialog} [data-active="true"]`).first().waitFor();
        if (dialog.includes('dataset')) {
            await page.locator(`${dialog} img`).first().waitFor();
            await page.waitForFunction(() => [...document.querySelectorAll('.dragon-dataset-picker-dialog img')].every(img => img.complete && img.naturalWidth > 0));
        }
        await checkBounds(page, dialog, name);
        assert.equal(await page.locator(dialog).evaluate(e => getComputedStyle(e, '::backdrop').backdropFilter), 'none');
        await snapshot(page, `${name}-${dialog.slice(1)}`);
        await page.locator(`${dialog} ${close}`).click();
    }
}

async function checkGroupedAndScale(page, name) {
    await page.goto(new URL('?ui=dragon#config/training-config/common', baseURL).href);
    await page.locator('.dragon-config-detail:not(.dragon-config-all-detail)').waitFor();
    await page.evaluate(() => window.scrollTo(0, 0));
    const groupedToggle = page.locator('.dragon-field:has(> .dragon-field-floating-actions)').first();
    assert(await groupedToggle.evaluate(field => {
        const toggle = field.querySelector('.dragon-toggle').getBoundingClientRect();
        const help = field.querySelector('.dragon-field-help-btn').getBoundingClientRect();
        return toggle.right <= help.left;
    }), 'grouped toggle overlaps help');
    await snapshot(page, `${name}-grouped`);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth), 0);
    await page.goto(new URL(route, baseURL).href);
    await page.locator('.dragon-config-all-toolbar').waitFor();
    const toggle = page.locator('.dragon-config-all-toolbar [data-config-preset-toggle]');
    if (await toggle.getAttribute('aria-expanded') === 'false') await toggle.click();
    for (const scale of [80, 125]) {
        await page.evaluate(async value => {
            const { applyDragonUIScale } = await import('/static/js/dragon-ui/ui-scale.js');
            applyDragonUIScale({ ui_scale: 100, ui_scale_config: value }, 'config');
            window.dispatchEvent(new Event('resize'));
        }, scale);
        await snapshot(page, `${name}-scale-${scale}`);
        await checkSurface(page);
        await checkBounds(page, '.dragon-config-all-footer', `scale ${scale}`);
        await checkBounds(page, '.dragon-training-preset-library', `scale ${scale} sidebar`);
    }
}

async function checkMobileFields(page, name) {
    const input = page.locator('.dragon-config-category-page [data-key="pretrained_model_name_or_path"]');
    await input.focus();
    const tooltip = page.locator('[data-config-field-key="pretrained_model_name_or_path"] .dragon-config-path-tooltip');
    await tooltip.waitFor({ state: 'visible' });
    assert.equal(await tooltip.textContent(), await input.inputValue());
    await checkBounds(page, '[data-config-field-key="pretrained_model_name_or_path"] .dragon-config-path-tooltip', 'path tooltip');
    await snapshot(page, `${name}-path-focus`);
    await input.blur();
    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
    await settle(page);
    assert(await page.evaluate(() => {
        const fields = [...document.querySelectorAll('.dragon-config-block')].filter(e => e.getClientRects().length);
        const footer = document.querySelector('.dragon-config-all-footer').getBoundingClientRect();
        return fields.at(-1).getBoundingClientRect().bottom <= footer.top - 16;
    }), 'last field hidden under mobile footer');
    assert(await page.evaluate(() => {
        const toolbar = document.querySelector('.dragon-config-all-toolbar');
        const rect = toolbar.getBoundingClientRect();
        for (let x = rect.left + 8; x < rect.right - 8; x += 16) {
            for (let y = rect.top + 4; y < rect.bottom - 4; y += 10) {
                if (!toolbar.contains(document.elementFromPoint(x, y))) return false;
            }
        }
        return true;
    }), 'field layer paints over the sticky toolbar');
    await snapshot(page, `${name}-last-field`);
}

async function scenario(browser, theme, width, height) {
    const name = `${theme}-${width}`;
    const context = await browser.newContext({ viewport: { width, height }, colorScheme: theme, reducedMotion: 'reduce' });
    const errors = [];
    const writes = [];
    await context.route('**/api/**', async request => {
        if (!['GET', 'HEAD', 'OPTIONS'].includes(request.request().method())) {
            writes.push(`${request.request().method()} ${new URL(request.request().url()).pathname}`);
            return request.abort('blockedbyclient');
        }
        return request.continue();
    });
    await context.addInitScript(({ theme, collapsed }) => {
        localStorage.setItem('anima_dragon_theme', theme);
        localStorage.setItem('anima_dragon_config_ui', JSON.stringify({ workspaceVersion: 1, presetCollapsed: collapsed }));
    }, { theme, collapsed: width <= 1000 });
    const page = await context.newPage();
    page.setDefaultTimeout(20000);
    page.on('pageerror', error => errors.push(error.message));
    try {
        await page.goto(new URL(route, baseURL).href);
        await page.locator('.dragon-config-all-toolbar').waitFor();
        await page.evaluate(() => document.fonts.ready);
        const state = await checkSurface(page);
        await snapshot(page, name);
        if (width === 1440 || width === 390) {
            await checkHelp(page, name);
            await checkPickers(page, name);
            await checkDrawer(page, name);
            await checkDraftAndSearch(page, name);
        }
        if (width === 1440) await checkGroupedAndScale(page, name);
        if (width === 390) await checkMobileFields(page, name);
        assert.deepEqual(errors, [], 'browser errors');
        assert.deepEqual(writes, [], 'unexpected mutation request');
        results.push({ name, ...state, errors, writes });
        console.log(`PASS ${name}`);
    } finally {
        await context.close();
    }
}

async function main() {
    await mkdir(output, { recursive: true });
    const browser = await chromium.launch({
        headless: true,
        ...(process.env.DRAGON_QA_BROWSER ? { executablePath: process.env.DRAGON_QA_BROWSER } : {}),
    });
    try {
        for (const theme of ['light', 'dark']) {
            for (const [width, height] of viewports) await scenario(browser, theme, width, height);
        }
        await writeFile(path.join(output, 'results.json'), JSON.stringify(results, null, 2));
    } finally {
        await browser.close();
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
