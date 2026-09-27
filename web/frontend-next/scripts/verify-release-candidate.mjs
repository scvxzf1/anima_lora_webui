import { createHash } from "node:crypto";
import { createServer } from "node:http";
import {
  cp,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";

const candidate = resolve(
  process.env.DRAGON_S5_CANDIDATE || process.argv[2] || "",
);
const reportPath = resolve(
  process.env.DRAGON_S5_REPORT || join(candidate, "s5-report.json"),
);
if (!candidate || candidate === resolve(".")) {
  throw new Error("Set DRAGON_S5_CANDIDATE or pass the candidate directory");
}

async function listFiles(root, prefix = "") {
  const entries = await readdir(join(root, prefix), { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const relative = join(prefix, entry.name);
    if (entry.isDirectory()) files.push(...(await listFiles(root, relative)));
    else files.push(relative);
  }
  return files;
}

function sha256(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

async function publishAndRollback(entry, assets) {
  const publishRoot = await mkdtemp(join(tmpdir(), "dragon-next-s5-publish-"));
  const destination = join(publishRoot, "dragon-next");
  await mkdir(join(destination, "assets"), { recursive: true });
  const legacyAsset = join(destination, "assets", "legacy-s5-sentinel.js");
  const oldIndex = "<!doctype html><title>previous S5 candidate</title>\n";
  await writeFile(legacyAsset, "legacy chunk retained\n", "utf8");
  await writeFile(join(destination, "index.html"), oldIndex, "utf8");
  await cp(assets, join(destination, "assets"), { recursive: true });

  await cp(join(destination, "index.html"), join(destination, "previous-index.html"));
  const publishTemp = join(destination, `.index-${process.pid}.html`);
  await writeFile(publishTemp, entry);
  await rename(publishTemp, join(destination, "index.html"));
  if ((await readFile(join(destination, "previous-index.html"), "utf8")) !== oldIndex)
    throw new Error("previous-index.html did not preserve the prior entry");
  if ((await readFile(join(destination, "index.html"), "utf8")) !== entry)
    throw new Error("atomic candidate entry replacement failed");
  await stat(legacyAsset);

  const rollbackTemp = join(destination, `.rollback-${process.pid}.html`);
  await cp(join(destination, "previous-index.html"), rollbackTemp);
  await rename(rollbackTemp, join(destination, "index.html"));
  if ((await readFile(join(destination, "index.html"), "utf8")) !== oldIndex)
    throw new Error("rollback did not restore the prior entry");
  const leftovers = (await readdir(destination)).filter((name) => name.startsWith("."));
  if (leftovers.length) throw new Error(`temporary publish files remain: ${leftovers.join(", ")}`);
  await rm(publishRoot, { recursive: true, force: true });
  return { previousIndex: true, atomicReplace: true, rollback: true, oldChunkRetained: true };
}

async function checkDeepLinks(entryPath, assetsRoot) {
  const entry = await readFile(entryPath);
  const assetRoot = resolve(assetsRoot);
  const server = createServer(async (request, response) => {
    const pathname = new URL(request.url || "/", "http://127.0.0.1").pathname;
    if (pathname === "/next" || pathname.startsWith("/next/")) {
      response.writeHead(200, { "content-type": "text/html" });
      response.end(entry);
      return;
    }
    if (pathname.startsWith("/static/dragon-next/assets/")) {
      const relative = decodeURIComponent(pathname.slice("/static/dragon-next/assets/".length));
      const target = resolve(assetRoot, relative);
      if (target !== assetRoot && !target.startsWith(`${assetRoot}/`)) {
        response.writeHead(403);
        response.end();
        return;
      }
      try {
        const body = await readFile(target);
        response.writeHead(200);
        response.end(body);
      } catch {
        response.writeHead(404);
        response.end();
      }
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await new Promise((resolveServer) => server.listen(0, "127.0.0.1", resolveServer));
  const address = server.address();
  const origin = `http://127.0.0.1:${address.port}`;
  const paths = [
    "/next",
    "/next/training",
    "/next/history/example-task",
    "/next/datasets/workspace/preview",
  ];
  const checks = await Promise.all(
    paths.map(async (path) => ({ path, status: (await fetch(`${origin}${path}`)).status })),
  );
  await new Promise((resolveServer) => server.close(resolveServer));
  if (checks.some(({ status }) => status !== 200))
    throw new Error(`deep-link refresh failed: ${JSON.stringify(checks)}`);
  return checks;
}

const entry = await readFile(join(candidate, "index.html"), "utf8");
const assetsRoot = join(candidate, "assets");
const files = await listFiles(assetsRoot);
const assetRefs = [...entry.matchAll(/(?:src|href)="([^\"]+\/assets\/[^\"]+)"/g)].map(
  (match) => match[1],
);
const referencedFiles = assetRefs.map((ref) => ref.split("/assets/").at(-1));
for (const file of referencedFiles) {
  if (!files.includes(file)) throw new Error(`entry references missing asset: ${file}`);
}
const records = await Promise.all(
  files.sort().map(async (file) => {
    const body = await readFile(join(assetsRoot, file));
    return { path: file, bytes: body.length, sha256: sha256(body) };
  }),
);
if (records.some(({ path }) => path.endsWith(".map")))
  throw new Error("release candidate unexpectedly contains source maps");

const publish = await publishAndRollback(entry, assetsRoot);
const deepLinkChecks = await checkDeepLinks(join(candidate, "index.html"), assetsRoot);
const report = {
  candidate,
  entrySha256: sha256(Buffer.from(entry)),
  assetCount: records.length,
  referencedAssetCount: referencedFiles.length,
  assets: records,
  deepLinkChecks,
  ...publish,
};
await writeFile(reportPath, JSON.stringify(report, null, 2), "utf8");
console.log(JSON.stringify({ reportPath, ...publish, assetCount: records.length }));
