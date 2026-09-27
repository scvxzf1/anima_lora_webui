import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const repoRoot = path.resolve(import.meta.dirname, "../../..");
const matrixPath = path.resolve(
  repoRoot,
  process.argv[2] || "docs/findings/dragon_next_s1_state_matrix_r93.json",
);
const matrix = JSON.parse(fs.readFileSync(matrixPath, "utf8"));
const states = new Set(["loading", "error", "empty", "busy", "success", "conflict_409", "offline_unknown"]);
const statuses = new Set(["PASS_EVIDENCE", "NOT_RUN", "N/A", "MANUAL_REQUIRED"]);
const scopes = new Set(["mock-api-ws", "vitest", "real-service", "manual", "n/a"]);
const errors = [];

if (matrix.schema_version !== 1) errors.push("schema_version must be 1");
if (matrix.overall_state !== "NOT_RUN") errors.push("overall_state must remain NOT_RUN until S1 exits");
if (matrix.evidence_mode !== "mock_api_ws") errors.push("evidence_mode must be mock_api_ws");

const expectedWorkspaces = new Set(matrix.workspaces);
const expectedStates = new Set(matrix.required_states);
const ids = new Set();
for (const row of matrix.rows || []) {
  if (ids.has(row.id)) errors.push(`duplicate row id: ${row.id}`);
  ids.add(row.id);
  if (!expectedWorkspaces.has(row.workspace)) errors.push(`${row.id}: unknown workspace`);
  if (!expectedStates.has(row.state) || !states.has(row.state)) errors.push(`${row.id}: unknown state`);
  if (!statuses.has(row.status)) errors.push(`${row.id}: unknown status`);
  if (!scopes.has(row.scope)) errors.push(`${row.id}: unknown scope`);

  if (row.status === "PASS_EVIDENCE") {
    if (!row.test_file || !row.test_title) errors.push(`${row.id}: PASS_EVIDENCE needs test_file and test_title`);
    const testPath = path.resolve(repoRoot, row.test_file || "");
    if (!testPath.startsWith(`${repoRoot}${path.sep}`) || !fs.existsSync(testPath)) {
      errors.push(`${row.id}: test_file does not exist: ${row.test_file}`);
    } else if (!fs.readFileSync(testPath, "utf8").includes(row.test_title)) {
      errors.push(`${row.id}: test_title not found in ${row.test_file}: ${row.test_title}`);
    }
  } else {
    if (!row.reason) errors.push(`${row.id}: non-pass row needs reason`);
    if (row.status === "MANUAL_REQUIRED" && !row.manual_check) {
      errors.push(`${row.id}: MANUAL_REQUIRED needs manual_check`);
    }
    if (row.test_file || row.test_title) {
      errors.push(`${row.id}: non-pass row must not claim a test title`);
    }
  }
}

for (const workspace of expectedWorkspaces) {
  for (const state of expectedStates) {
    const id = `${workspace}.${state}`;
    if (!ids.has(id)) errors.push(`missing required row: ${id}`);
  }
}

if (errors.length) {
  console.error(errors.map((error) => `- ${error}`).join("\n"));
  process.exitCode = 1;
} else {
  console.log(`S1 state matrix valid: ${matrix.rows.length} rows, overall_state=${matrix.overall_state}`);
}
