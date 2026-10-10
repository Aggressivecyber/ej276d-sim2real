# R47：完成新计划首轮诊断，定位已有噪声和学习路径的共同失真

已完成三个电荷一维分类器和两个初态基线分类器，均使用原固定训练/留出行及 500 次区组重抽。响应训练、ODE、密度拟合和新增响应生成均为零。原 A–F 不变，本轮没有产生新的模型候选。

## 电荷人口差异：分布不同，固定判别结论仍为不确定

| 固定 charge-only 诊断 | 留出 BA | 95% 区间 | 预定规则下的判断 |
|---|---:|---:|---|
| 原 SOURCE 对 REAL DEV | 0.586 | [0.540830, 0.626028] | uncertain |
| R46 LONG 对 REAL DEV | 0.590 | [0.543917, 0.626282] | uncertain |
| RAW TRAIN reference 对 REAL DEV | 0.478 | [0.423344, 0.525233] | 此诊断未检出超限差异 |

全体 2000 条 SOURCE 的平均电荷为 2807.42 mV·ns，REAL DEV 为 4649.83 mV·ns；SOURCE 对 REAL 的 KS 为 0.2685，W1 为 1842.56 mV·ns。原 SOURCE、一次 ADC 和 LONG 的完整电荷人口、原八区及支持外全部保留。SOURCE 与 LONG 的判别区间跨过 0.60，因此不能按本轮预定规则称为强人口判别证据，更不能当作原 Q–PSD 二维评价器必失败的证明。

每域仍为 600 行训练、250 行留出。这里的区间条件于已拟合的分类器及既有分组，不包含重训或独立采集不确定性。保存字段中的 `fixed_R26_phase` 原名实际指域列；纯元数据补充已明确训练 phase0、留出 phase1 和 SOURCE / REAL / reference 域列，实际选择未改变、没有重拟合。

## 噪声：初始化已扩大实测幅度，学习路径又继续改变它

| 基线 RMS 均值，mV | 原始输入 | 初态，一次 ADC | SHORT | LONG |
|---|---:|---:|---:|---:|
| SOURCE | 0 | 0.998213 | 0.685231 | 0.697719 |
| REAL TRAIN reference | 0.645992 | 1.193383 | 1.234196 | 1.537006 |

REAL DEV 的基线 RMS 均值为 0.646423 mV。REAL reference 对它的 RMS W1 从 RAW 的 0.003115 mV，升至初态的 0.546959 mV、SHORT 的 0.587773 mV和 LONG 的 0.890583 mV。ACF1 W1 依次为 0.005861、0.026574、0.213110、0.407618。问题既出现在初态，也出现在后续学习路径；本次描述不能支持“只减小 Gaussian 就足够”或“删除保存的某分量便能修好模型”。

REAL 初态的 baseline-local BA 为 0.572；相对复用的 RAW REAL–REAL 0.522，配对差为 +0.050，95% 区间 [−0.019827, +0.110045]。该视图对局部波形做幅度归一化，BA 区间跨零并不否定直接观察到的 RMS 幅度变化。配对 AUC 差为 +0.132528，95% 区间 [+0.066678, +0.196228]。旧 SHORT / LONG REAL 控制 baseline BA 为 0.826 / 0.922，保留原结果、未重拟合。

四组保存输出均满足 S+G+F+R 的代数还原，最大波形残差约 2.78e−17 V；四方差与全部六个 2Cov 项之和的最大残差约 1.02e−20 V²。SOURCE 的 G/F 交叉项为负，REAL LONG 的相应平均交叉项为正；不能把这些功能项当成独立物理信号/噪声或因果贡献百分比。SHORT 与 LONG 使用的原输入和初始 Gaussian 在各自域内逐值相同。

## 下一步与交付范围

[执行计划](PROJECT_EXECUTION_PLAN_20261010_002.md)的第一阶段已完成。下一方法的设计要求见[后续方向](reviews/R47_ROUTE_DECISION_001.md)：完整源驱动、明确处理输入已有噪声、同一路径检验 REAL 稳定性，一个固定主机制配等数据/更新对照。主验收仍沿用原 A–F；可选口径问题尚无答复，不自动改变标准。本轮尚未登记 R48 的完整算子、网络或训练配置，没有新响应训练在后台运行。

累计科学时间、封存文件数、实际私有 Git 与 D 盘增量副本以最终资源和交付回执为准。六张最终图已实际打开；两张图例遮挡修复前的图片和代码保留，修复没有新增拟合或统计。根读取关键结果时一次只读摘要脚本因 REAL_DEV 使用不同嵌套结构而退出 1，修正读取后退出 0，未产生或改变科学结果。

[完整执行报告](work/R47_POPULATION_NOISE_DIAGNOSTIC_001/REPORT.md) · [电荷人口图](work/R47_POPULATION_NOISE_DIAGNOSTIC_001/results/figures/charge_population.png) · [噪声路径图](work/R47_POPULATION_NOISE_DIAGNOSTIC_001/results/figures/baseline_paths.png) · [方差与交叉项](work/R47_POPULATION_NOISE_DIAGNOSTIC_001/results/figures/baseline_variance_components.png) · [复现说明](work/R47_POPULATION_NOISE_DIAGNOSTIC_001/REPRODUCTION.md)
