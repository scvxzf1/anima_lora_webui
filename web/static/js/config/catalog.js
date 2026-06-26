import * as defaults from './catalog/defaults.js?v=module-bootstrap-20260625-9';
import * as extraFieldHelp from './catalog/extra-field-help.js?v=module-bootstrap-20260625-9';
import * as formLayout from './catalog/form-layout.js?v=module-bootstrap-20260625-9';
import { FIELD_HELP_ZH } from './catalog/field-help.js?v=module-bootstrap-20260625-9';
import * as guides from './catalog/guides.js?v=module-bootstrap-20260625-9';
import { choiceHelp, help } from './catalog/help-builder.js?v=module-bootstrap-20260625-9';
import * as labelsOptions from './catalog/labels-options.js?v=module-bootstrap-20260625-9';

export * from './catalog/defaults.js?v=module-bootstrap-20260625-9';
export * from './catalog/extra-field-help.js?v=module-bootstrap-20260625-9';
export * from './catalog/form-layout.js?v=module-bootstrap-20260625-9';
export { FIELD_HELP_ZH } from './catalog/field-help.js?v=module-bootstrap-20260625-9';
export * from './catalog/guides.js?v=module-bootstrap-20260625-9';
export { choiceHelp, help } from './catalog/help-builder.js?v=module-bootstrap-20260625-9';
export * from './catalog/labels-options.js?v=module-bootstrap-20260625-9';

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
