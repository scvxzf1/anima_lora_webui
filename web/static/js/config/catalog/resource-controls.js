/* Shared presentation metadata; stored values remain the backend enums. */
export const RESOURCE_NUMBER_CONSTRAINTS = Object.freeze({
    auto_block_swap_vram_reserve_percent: Object.freeze({ min: '0', max: '90', step: '0.1' }),
    auto_block_swap_swap_io_limit_mb: Object.freeze({ min: '0', max: '1048576', step: '1' }),
});

const SWAP_PREFERENCE_LABELS = Object.freeze({
    balanced: '均衡',
    vram: '优先节省显存',
    ram: '优先节省内存',
});

export function resourceOptionLabel(key, value) {
    return key === 'auto_block_swap_preference' ? SWAP_PREFERENCE_LABELS[value] : undefined;
}
