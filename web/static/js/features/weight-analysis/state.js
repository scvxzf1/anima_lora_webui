export function createWeightAnalysisState() {
    return {
        loadingWeights: false,
        analyzing: false,
        uploading: false,
        weights: [],
        selectedPath: '',
        primaryFile: null,
        compareEnabled: false,
        comparePath: '',
        compareFile: null,
        result: null,
        compareResult: null,
        activeCandidateKind: 'style',
        candidateExpanded: {
            style: false,
            character: false,
        },
        activeComponent: '',
        activeBlock: '',
        requestSeq: 0,
    };
}
