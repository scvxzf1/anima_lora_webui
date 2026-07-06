export function createAnimaRuntime(ctx) {
    return {
        ctx,
        app: {},
        state: {
            config: {},
            training: {},
            toml: {},
            dataset: {},
            history: {},
        },
        features: {},
        timers: {},
        dom: {
            byId(id) {
                return document.getElementById(id);
            },
        },
    };
}
