/**
 * Live (non-blocking) compatibility warnings for the config form.
 * Mirrors a small subset of library/training/compat_matrix.py codes for UX only.
 * Does NOT replace server preflight.
 */
import { isKrea2ModelFamily, normalizeModelFamily } from './model-family.js?v=module-bootstrap-20260903-pp-audit-v2';

/**
 * @typedef {Object} LiveCompatIssue
 * @property {string} code
 * @property {string} key
 * @property {'error'|'warning'} severity
 * @property {string} message
 */

function boolValue(value, fallback = false) {
    if (typeof value === 'boolean') return value;
    if (value == null) return fallback;
    if (typeof value === 'number') return Boolean(value);
    const text = String(value).trim().toLowerCase();
    if (['1', 'true', 'yes', 'on'].includes(text)) return true;
    if (['0', 'false', 'no', 'off', 'none', ''].includes(text)) return false;
    return fallback;
}

function intValue(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.trunc(n) : fallback;
}

/**
 * @param {Record<string, unknown>} config
 * @returns {LiveCompatIssue[]}
 */
export function collectLiveCompatIssues(config = {}) {
    const issues = [];
    const selective = String(config.selective_checkpoint ?? 'off').trim().toLowerCase() || 'off';
    const gradientCheckpointing = boolValue(config.gradient_checkpointing, false);
    const blocksToSwap = intValue(config.blocks_to_swap, 0);
    const networkModule = String(config.network_module ?? '');
    const selectiveEnabled = selective !== 'off' && selective !== '';
    const blockSwapEnabled = blocksToSwap > 0;
    const softTokens = /soft_tokens/i.test(networkModule);
    const modelFamily = normalizeModelFamily(config.model_family);
    const krea2Family = isKrea2ModelFamily(modelFamily);

    if (krea2Family) {
        const attnMode = String(config.attn_mode ?? 'torch').trim().toLowerCase() || 'torch';
        if (!['torch', 'flash', 'sdpa'].includes(attnMode)) {
            issues.push({
                code: 'krea2_invalid_attn_mode',
                key: 'attn_mode',
                severity: 'error',
                message: 'live 兼容：Krea-2 注意力后端仅支持 torch 或 flash（sdpa 是 torch 别名）。',
            });
        }
        if (boolValue(config.compile_dynamic_seq, false)) {
            issues.push({
                code: 'krea2_compile_dynamic_seq',
                key: 'compile_dynamic_seq',
                severity: 'warning',
                message: 'live 兼容：Krea-2 使用固定 token-family 编译图，训练启动时会自动关闭 compile_dynamic_seq。',
            });
        }
        if (boolValue(config.compile_seq_bands, false)) {
            issues.push({
                code: 'krea2_compile_seq_bands',
                key: 'compile_seq_bands',
                severity: 'warning',
                message: 'live 兼容：Krea-2 使用固定 token-family 编译图，训练启动时会自动关闭 compile_seq_bands。',
            });
        }
        const compileMode = String(config.compile_inductor_mode ?? 'default').trim().toLowerCase() || 'default';
        if (compileMode !== 'default') {
            issues.push({
                code: 'krea2_compile_inductor_mode',
                key: 'compile_inductor_mode',
                severity: 'error',
                message: 'live 兼容：Krea-2 仅支持 compile_inductor_mode=default。',
            });
        }
        if (!['off', 'every_other'].includes(selective)) {
            issues.push({
                code: 'krea2_selective_checkpoint',
                key: 'selective_checkpoint',
                severity: 'error',
                message: 'live 兼容：Krea-2 选择性检查点仅支持 off 或 every_other。',
            });
        }
        const v100Mode = String(config.v100_flash_stability ?? 'off').trim().toLowerCase() || 'off';
        if (v100Mode !== 'off') {
            issues.push({
                code: 'krea2_v100_flash_stability',
                key: 'v100_flash_stability',
                severity: 'error',
                message: 'live 兼容：v100_flash_stability 是 Anima 专用项，Krea-2 下必须关闭。',
            });
        }
    }

    if (modelFamily === 'z_image' && boolValue(config.compile_seq_bands, false)) {
        issues.push({
            code: 'z_image_compile_seq_bands',
            key: 'compile_seq_bands',
            severity: 'warning',
            message: 'live 兼容：分带动态序列编译仅适用于 Anima，Z-Image 启动时会自动关闭。',
        });
    }

    if (selectiveEnabled && gradientCheckpointing) {
        issues.push({
            code: 'selective_full_gradient_checkpointing',
            key: 'gradient_checkpointing',
            severity: 'error',
            message:
                'live 兼容：selective_checkpoint 不能与完整 gradient_checkpointing 同时开启（保存/训练前 preflight 也会拦截）。',
        });
    }

    if (blockSwapEnabled && softTokens) {
        issues.push({
            code: 'block_swap_soft_tokens',
            key: 'blocks_to_swap',
            severity: 'error',
            message:
                'live 兼容：blocks_to_swap 不支持 Soft Tokens；请保持 blocks_to_swap=0（preflight 仍会正式校验）。',
        });
    }

    if (blockSwapEnabled && boolValue(config.cpu_offload_checkpointing, false)) {
        issues.push({
            code: 'block_swap_cpu_offload',
            key: 'cpu_offload_checkpointing',
            severity: 'error',
            message:
                'live 兼容：blocks_to_swap 不能与 cpu_offload_checkpointing 同时开启。',
        });
    }

    const baseCompute = String(config.base_compute ?? 'bf16').trim().toLowerCase() || 'bf16';
    const convrotActive = baseCompute === 'w8a16_convrot' || baseCompute === 'w8a8_convrot';
    const nf4Active = baseCompute === 'nf4';
    const transferDtype = String(config.block_swap_transfer_dtype ?? 'bf16').trim().toLowerCase() || 'bf16';
    if (convrotActive && (transferDtype === 'int8' || transferDtype === 'int8_linear' || transferDtype === 'i8')) {
        issues.push({
            code: 'convrot_block_swap_int8_mutex',
            key: 'base_compute',
            severity: 'error',
            message:
                'live 兼容：base_compute 的 ConvRot 路径与 block_swap_transfer_dtype=int8 互斥（preflight 仍会正式校验）。',
        });
    }
    // NF4 × block_swap 已验证通过 (方向 A: deepcopy master + Params4bit.to()
    // 整体搬运, offloading.py isinstance 分流不碰 bf16/int8/fp8 路径). 端到端
    // 探针绿 (PG199, 1024, swap=4, 30 步): host RAM 18.18GB, GPU 10.26GB, loss
    // 单调下降. 主战场是 host RAM (pinned NF4 master ~5.7GB vs bf16 22.64GB),
    // 与后端 compat_matrix 的 nf4_block_swap_host_ram warning 对齐, 不再硬拒.
    if (nf4Active && blockSwapEnabled) {
        issues.push({
            code: 'nf4_block_swap_host_ram',
            key: 'base_compute',
            severity: 'warning',
            message:
                'live 兼容：base_compute=nf4 + blocks_to_swap 已验证通过（offloader NF4 路径整体搬运，端到端探针绿）；主约束是 host RAM，pinned NF4 master ~5.7GB（bf16 路径 22.64GB），低内存主机请关注 pinned master 预算。',
        });
    }

    return issues;
}

/**
 * @param {LiveCompatIssue[]} issues
 * @returns {string}
 */
export function formatLiveCompatStatus(issues) {
    if (!issues?.length) return '';
    const head = issues[0];
    const more = issues.length > 1 ? `（另有 ${issues.length - 1} 条）` : '';
    return `${head.message}${more}`;
}
