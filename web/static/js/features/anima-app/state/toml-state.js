export function createTomlState() {
    return {
        tomlStatusTimer: null,
        tomlFiles: [],
        tomlFileGroups: [],
        tomlFileMeta: {},
        currentTomlFile: '',
        tomlSavedContent: '',
        tomlDeleteConfirmFile: '',
        tomlDeleteConfirmTimer: null,
        tomlSaveConfirmFile: '',
        tomlSaveConfirmTimer: null,
        tomlManagerMode: 'project',
        configSwitchToastTimer: null,
        sharedDialogBusy: false,
        tomlGroupActionBusy: false,
    };
}
