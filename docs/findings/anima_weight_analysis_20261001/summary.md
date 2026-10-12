# Anima 人物与画风调查汇总（2026-10-01）

状态：历史调查快照
适用范围：2026-10-01 本批静态权重分析

> 历史调查快照；描述本批静态有效更新，不是当前训练行为或因果敏感性的重新验收。

[分类导航](README.md) · [画风完整报告](style/report.md) · [人物完整报告及两组对照](character/report.md)

共15份权重、5,700模块：画风10份、4,132模块（9个LoRA、1个LoKr）；人物5份、1,568模块（均LoRA）。无非有限值、遗漏或不支持模块。比较统一到全部15文件相同的280个主DiT attention/MLP targets；先按各文件在此交集的能量归一化，再等权求均值。

|共同目标统计|n|MLP均值%|self-attn均值%|cross-attn均值%|
|---|---:|---:|---:|---:|
|人物|5|40.156864|31.036860|28.806275|
|全画风|10|43.982850|28.778856|27.238294|
|画风暂排除ba_nai|9|40.604847|30.475747|28.919405|

均值会受例外影响：ba_nai的早层MLP集中更新拉高全画风MLP均值；暂排除后，人物与画风三组均值差均小于0.6个百分点。这是保留原始全体结果的描述性敏感性检查，不是删除样本，也不同于排除LoKr的普通LoRA子集。不能由此认定人物训练必然更依赖attention。

稳定热点按每文件共同目标的top5出现次数观察（block从0计数）：人物block17、18均为4/5；画风block17、16均为9/10，18为8/10。人物Sakura偏9/8/11，画风ba_nai偏4/5/3，Turbo和lenovo另有尾层偏重，不能把均值热点当作所有文件的共同分布。

这两组本地选择样本不代表普遍规律。作者、训练条件、rank、覆盖、保存约定及基座来源不完全一致；缺失target不等于低敏感。分析使用multiplier=1的静态等价ΔW，没有基座范数、激活或图像验收，不能解释为质量、语义还原、量化敏感性或因果重要性。

已有原理和历史研究只链接、不重复复制；其结论与本次统计口径应分别阅读：

- [LoRA结构原理](../../structure/lora.md)
- [OrthoLoRA结构原理](../../structure/ortholora.md)
- [Channel Scaling方法](../../methods/channel_scaling.md)
- [Channel统计与内容独立性历史调查](../channel_stats_content_independence.md)
- [Channel dominance历史分析](../../../bench/channel_stats/channel_dominance_analysis.md)
