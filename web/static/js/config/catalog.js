import * as defaults from './catalog/defaults.js?v=auto-block-swap-20260908-v3';
import * as extraFieldHelp from './catalog/extra-field-help.js?v=module-bootstrap-20260831-release-v1';
import * as formLayout from './catalog/form-layout.js?v=auto-block-swap-20260908-v3';
import { FIELD_HELP_ZH } from './catalog/field-help.js?v=auto-block-swap-20260908-v3';
import * as guides from './catalog/guides.js?v=module-bootstrap-20260831-release-v1';
import { choiceHelp, help } from './catalog/help-builder.js?v=module-bootstrap-20260831-release-v1';
import * as labelsOptions from './catalog/labels-options.js?v=auto-block-swap-20260908-v3';

export * from './catalog/defaults.js?v=auto-block-swap-20260908-v3';
export * from './catalog/extra-field-help.js?v=module-bootstrap-20260831-release-v1';
export * from './catalog/form-layout.js?v=auto-block-swap-20260908-v3';
export { FIELD_HELP_ZH } from './catalog/field-help.js?v=auto-block-swap-20260908-v3';
export * from './catalog/guides.js?v=module-bootstrap-20260831-release-v1';
export { choiceHelp, help } from './catalog/help-builder.js?v=module-bootstrap-20260831-release-v1';
export * from './catalog/labels-options.js?v=auto-block-swap-20260908-v3';

export function createCatalog() {
    return Object.freeze({
        ...defaults,
        ...formLayout,
        ...extraFieldHelp,
        ...labelsOptions,
        ...guides,
        FIELD_HELP_ZH,
        choiceHelp,
        help,
    });
}
