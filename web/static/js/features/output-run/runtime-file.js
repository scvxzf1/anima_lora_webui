/**
 * Output-run runtime file helpers.
 * Extracted from anima-app chunk 13.
 */
import { selectedOutputRun } from '../anima-app/helpers/output-run-bridge.js?v=module-bootstrap-20260711-ir1';

export function outputRunRuntimeFile(run = selectedOutputRun()) {
    const runtime = (run?.files || []).find((item) => item.kind === 'runtime');
    return runtime?.file || '';
}
