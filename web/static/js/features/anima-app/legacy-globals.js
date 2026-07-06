export function installLegacyGlobals(runtime) {
    globalThis.__animaRuntime = runtime;
    globalThis.ctx = runtime.ctx;
    globalThis.__animaAppContext = runtime.ctx;
}
