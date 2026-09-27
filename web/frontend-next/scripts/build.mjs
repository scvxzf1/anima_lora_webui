import {
  copyFile,
  cp,
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const configuredDestination = process.env.DRAGON_NEXT_DESTINATION?.trim();
const destination = resolve(
  configuredDestination || join(root, "../static/dragon-next"),
);
const staging = await mkdtemp(join(tmpdir(), "dragon-next-build-"));
try {
  await build({ root, build: { outDir: staging, emptyOutDir: true } });
  const html = await readFile(join(staging, "index.html"), "utf8");
  if (!html.includes("/static/dragon-next/assets/"))
    throw new Error("Unexpected asset base");
  await mkdir(destination, { recursive: true });
  // Publish chunks first; older HTML and lazy imports remain valid during rollout.
  await cp(join(staging, "assets"), join(destination, "assets"), {
    recursive: true,
  });
  try {
    await copyFile(
      join(destination, "index.html"),
      join(destination, "previous-index.html"),
    );
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  const entry = join(destination, `.index-${process.pid}.html`);
  await copyFile(join(staging, "index.html"), entry);
  await rename(entry, join(destination, "index.html"));
  console.log("Published /next; previous entry and hashed chunks retained.");
} finally {
  await rm(staging, { recursive: true, force: true });
}
