export function createConfigState() {
    return {
        configLoadSeq: 0,
        stepEstimateSeq: 0,
        samplePromptsLoadSeq: 0,
        configGroupHintSeq: 0,
        choiceGuideHintSeq: 0,
        fieldHelp: {},
        currentConfig: {},
        configFormState: {
            activeCategory: 'required',
            showAdvanced: false,
            search: '',
            expandedGroups: new Set(),
            collapsedGroups: new Set(),
            draftValues: new Map(),
        },
        selectionSnapshot: {
            method: '',
            variant: '',
            preset: '',
        },
        currentStepEstimate: null,
        stepEstimateStatus: { loading: false, error: '' },
        samplePromptsPath: 'configs/sample_prompts.txt',
        samplePromptsContent: '',
        samplePromptsMode: 'editor-inline',
        stageResolutionState: {
            enabled: false,
            selectedIndex: 0,
            stages: [
                { name: 'EP1', epochs: 1, maxSide: 1024, downRange: 256, manualRepeats: false, repeats: 1 },
                { name: 'EP2', epochs: 1, maxSide: 1536, downRange: 512, manualRepeats: false, repeats: 1 },
            ],
        },
    };
}
