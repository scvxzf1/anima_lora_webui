export function help(summary, fill, benefit, cost, risk, recommend, ps) {
    return { summary, fill, benefit, cost, risk, recommend, ps };
}

export function choiceHelp(title, summary, tradeoff, recommend) {
    return { title, summary, tradeoff, recommend };
}

// 中文字段说明。每项按固定栏目渲染，避免用户只看到一句模糊解释。
