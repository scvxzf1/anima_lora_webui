export const FIELD_PLACEHOLDER_ZH = Object.freeze({
    network_args: '例如：dropout=0.1（每行一项），随机丢弃 10% LoRA 神经元',
    alpha_rank_scale: '例如：1.0，秩按时间步线性变化',
    layer_start: '例如：0，从首个模型块开始应用',
    layer_end: '例如：8，只应用前 8 个模型块',
    balance_loss_weight: '例如：1e-7，抑制专家路由坍缩',
    balance_loss_warmup_ratio: '例如：0.4，前 40% 步关闭均衡损失',
    network_router_lr_scale: '例如：10，路由学习率放大 10 倍',
    router_targets: '例如：mlp\\.layer[12]$，仅匹配 FFN',
    sigma_feature_dim: '例如：16，用 16 维时间步特征路由',
    per_bucket_balance_weight: '例如：0.3，加强桶内专家均衡',
    sigma_bucket_boundaries: '例如：[0,0.5,0.8,1]，分 3 个桶',
    ip_image_drop_p: '例如：0.1，丢弃 10% IP 图像条件',
    easycontrol_drop_p: '例如：0.1，丢弃 10% 控制条件',
    easycontrol_cond_noise_max: '例如：0.1，加入轻度条件噪声',
    fei_feature_dim: '例如：2，用低/高频双特征路由',
    fei_sigma_low_div: '例如：4.0，低频尺度为最小边长 1/4',
    router_hidden_dim: '例如：64，用 64 维路由隐藏层',
    router_tau: '例如：0.7，越低专家选择越集中',
    fera_fecl_weight: '例如：0.0，0 表示关闭 FECL 损失',
    fera_num_bands: '例如：3，将 FEI 划分为 3 个频带',
});

export function configFieldPlaceholder(key, label) {
    return Object.prototype.hasOwnProperty.call(FIELD_PLACEHOLDER_ZH, key)
        ? FIELD_PLACEHOLDER_ZH[key]
        : `例如：${label}…`;
}
