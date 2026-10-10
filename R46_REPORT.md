# R46：数值与重放通过，延长训练仍未联合达标

两组冻结模型均通过预登记64步候选/128步参考检查，完成11968条生成、40个HGB和32个密度拟合，以及独立CPU进程8000条主输出全链逐位重放。LONG50000是唯一主候选，六门结果为 **通过/失败/失败/失败/失败/通过**，按原协议关闭；SHORT2950只作对照，不改选为候选。项目模型目标尚未完成，A/B失败，C的全部阶段门未通过，D/E未启动。

| 同为64步推理 | SHORT 2950步训练 | LONG 50000步训练 |
|---|---:|---:|
| shape493 BA | 0.880 | 0.780 |
| u480 BA | 0.892 | 0.738 |
| V480 BA | 0.894 | 0.824 |
| Q–PSD二维 BA | 0.714 | 0.704 |
| baseline BA | 0.662 | 0.570 |
| 总体 signed-PSD W1 | 0.04618358 | 0.02418821 |
| TRAIN64/128 RMS中位，LSB | 0.047928 | 0.105813 |
| TRAIN64/128 RMS p95，LSB | 0.141050 | 0.213042 |

LONG−SHORT的shape493配对差为−0.100，95%区间[−0.141007,−0.062014]；u480为−0.154，区间[−0.196392,−0.112860]。总体PSD W1降低47.63%，固定Q1–7平均降低58.30%，但baseline RMS W1恶化17.20%，超过10%限制。固定Q2/Q4两密度拟合不收敛/不满足分离要求，因此Q2–7峰位及raw宽度整体验收为未定义/失败，不能删去这两区后计算通过。预登记“完整响应改善”联合描述也不成立。

原213/181条实际输入pure-had源组，LONG输出落在冻结谷区的比例为44.60%/44.75%，SHORT为0/0；这是源条件保真风险，不是粒子误分类率。LONG的两个时间边界、同处理REAL控制也失败。密度拟合失败不证明物理粒子峰消失；同处理REAL的绝对BA很高，仍须完整报告。

最终A要求五个SOURCE及同处理REAL视图的95%区间均包含在[0.40,0.60]，LONG没有任何一个视图通过该区间要求。B的总体/条件PSD仍超过0.010，噪声和原必需峰/宽度条件也未通过。数值成功和逐位重放不能替代这些科学指标。缺失RAW REAL–REAL二维对照仍为unknown；源与实测电荷人口差异仍按原样保留。

R46新增响应训练为0。历史SHORT/LONG分别2950/50000次更新；LONG的源/实测唯一父人口没有增加，增加的是同分布Monte Carlo训练抽样。本轮是同一继承优化轨迹上的训练预算比较，不是等训练预算的结构优势或独立seed证据。R45固定32步数值失败保留，不追改历史。

科学总耗时 **1696.427724秒**，服务器CPU12完成数值、生成、评价及重放，本机CPU4绘图；五个主要科学进程和一次1.230288秒图题修正均native0，修正耗时也计入总额。实际worker RSS峰1,933,824,000B，RSS查询缺测1次，缺测原因保留unknown。科学失败与进程失败分开，预评价固定分箱修正及普通控制错误均留档。

最终封存377文件、1,358,692,408B，根与执行器实际打开全部13张图。阶段只关闭当前固定长训练/64步方案；当前总预算仍有余量，不把预算余量当作继续盲扫的理由。下一机制判断与总体目标范围澄清分开记录，未答复的范围选择不视为同意修改验收。

[完整报告](work/R46_FIXED_INTEGRATION_BUDGET_001/REPORT.md) · [长短预算图](work/R46_FIXED_INTEGRATION_BUDGET_001/results/figures/long_short_budget_comparison.png) · [联合Q–PSD](work/R46_FIXED_INTEGRATION_BUDGET_001/results/figures/charge_psd_full_range.png) · [源保真](work/R46_FIXED_INTEGRATION_BUDGET_001/results/figures/source_preservation.png) · [重放](work/R46_FIXED_INTEGRATION_BUDGET_001/REPLAY_RECEIPT.json) · [验收观察](reviews/R46_ACCEPTANCE_OBSERVATION_001.json) · [复现入口](work/R46_FIXED_INTEGRATION_BUDGET_001/REPRODUCTION.md)。私有Git与D盘副本只以实际R46_PUBLICATION_RECEIPT.json及R46_BACKUP_RECEIPT.json为准。
