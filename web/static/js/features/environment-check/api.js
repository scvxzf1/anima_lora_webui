export async function fetchEnvironmentCheck(ctx) {
    return ctx.api('/api/environment/check');
}
