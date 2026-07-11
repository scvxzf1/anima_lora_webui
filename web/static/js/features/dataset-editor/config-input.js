/**
 * Dataset config field inputs (defaults editor).
 * Extracted from former chunk 10.
 */
import { datasetConfigLabel, datasetConfigValue } from '../anima-app/helpers/dataset-config-fields.js?v=module-bootstrap-20260711-ir6';
import { updateDatasetDefault } from './row-fields.js?v=module-bootstrap-20260711-ir6';

export function createDatasetConfigInput(key, type, defaults) {
        if (type === 'switch') {
            return createDatasetConfigSwitch(key, defaults);
        }

        let input;
        if (type === 'select') {
            input = document.createElement('select');
            const options = key === 'enable_bucket'
                ? [[true, '启用'], [false, '关闭']]
                : [[false, '允许放大'], [true, '不放大小图']];
            const current = Boolean(defaults[key]);
            for (const [value, label] of options) {
                const opt = document.createElement('option');
                opt.value = value ? 'true' : 'false';
                opt.textContent = label;
                opt.selected = value === current;
                input.appendChild(opt);
            }
            input.dataset.valueType = 'boolean';
        } else {
            input = document.createElement('input');
            input.type = type;
            input.dataset.valueType = type === 'number' ? 'number' : 'string';
            input.value = datasetConfigValue(key, defaults);
            if (type === 'number') {
                input.min = '0';
                input.step = key === 'validation_split' ? '0.001' : (key === 'resolution' || key.endsWith('_reso') || key === 'bucket_reso_steps' ? '16' : '1');
            }
        }
        input.className = 'field-input dataset-config-input';
        input.dataset.key = key;
        input.addEventListener('input', () => updateDatasetConfigValue(key, input));
        input.addEventListener('change', () => updateDatasetConfigValue(key, input));
        return input;
    }

    function createDatasetConfigSwitch(key, defaults) {
        const checked = Boolean(defaults[key]);
        const wrap = document.createElement('label');
        wrap.className = ['dataset-json-switch', checked ? 'enabled' : ''].filter(Boolean).join(' ');

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.className = 'dataset-json-switch-input';
        input.dataset.key = key;
        input.checked = checked;
        input.setAttribute('aria-label', datasetConfigLabel(key));

        const copy = document.createElement('span');
        copy.className = 'dataset-json-switch-copy';
        const title = document.createElement('span');
        title.className = 'dataset-json-switch-title';
        title.textContent = 'JSON 标注';
        const desc = document.createElement('span');
        desc.className = 'dataset-json-switch-desc';
        desc.textContent = '.json 优先，失败回退 .txt';
        copy.append(title, desc);

        const state = document.createElement('span');
        state.className = 'dataset-json-switch-state';
        state.textContent = checked ? '已启用' : '已关闭';

        const track = document.createElement('span');
        track.className = 'dataset-json-switch-track';
        track.setAttribute('aria-hidden', 'true');
        const thumb = document.createElement('span');
        thumb.className = 'dataset-json-switch-thumb';
        track.appendChild(thumb);

        input.addEventListener('change', () => {
            wrap.classList.toggle('enabled', input.checked);
            state.textContent = input.checked ? '已启用' : '已关闭';
            updateDatasetConfigValue(key, input);
        });

        wrap.append(input, copy, state, track);
        return wrap;
    }

	    function updateDatasetConfigValue(key, input) {
	        updateDatasetDefault(key, input);
	    }
