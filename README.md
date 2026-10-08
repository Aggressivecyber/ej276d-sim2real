# EJ-276D：G4 到实测响应变换

R1–R6 已完成并封存，支持局部响应改善；完整响应相似性仍未通过，合成数据效用与连续电荷区间推广也有明确限制。持续目标下，R7 残差诊断、R8 配对实验及 R8B 整体路线复审已封存。R8 改善峰后局部，完整 shape BA 仍 0.996；现执行 R8C 整段随机映射及唯一配对消融，之后按已登记的 R9 补充协议统一冻结验证。研究范围仍为同一次 AmBe 采集，旧 EVAL 已查看。

- [R7 残差诊断](work/R7_RESPONSE_RESIDUAL_DIAGNOSIS_001/REPORT.md)、[R8 配对结果](work/R8_LOCAL_JOINT_FLUCTUATION_001/REPORT.md)、[R8B 整体路线依据](work/R8B_FULL_RESPONSE_ROUTE_REVIEW_001/REPORT.md)。
- [R8C 有界完整映射实验](tasks/R8C_WHOLE_RESPONSE_MAPPING_001.md)、[原 R9 验证口径](tasks/R9_FROZEN_RETROSPECTIVE_VALIDATION_001.md)及[新增候选补充](tasks/R9_CANDIDATE_SCOPE_ADDENDUM_001.md)。
- [本轮方法登记](CONTINUATION_METHOD_PLAN.json)与[实时阶段状态](STATUS.md)。本轮最终候选、验证与交付尚未完成。

- [R1–R6 封存研究报告](FINAL_REPORT.md)：数据、方法、留出响应、峰宽、任务效用、区间排除与结论边界。
- [R1–R6 本机复现入口](REPRODUCTION.md)：固定权重生成、R5 保存估计器预测、R6 配对模型重放。
- [R1–R6 封存交付索引](DELIVERY_INDEX.json)：需求对应证据、关键文件摘要与交付回执入口。
- [计划与开发记录](PLAN.md)、[当前状态](STATUS.md)、[机器可读状态](PROGRAM.json)。
- [冻结方法和任务](FROZEN_METHOD.json)、[数据角色](CURRENT_ROLE_INDEX.json)：当前首次 EVAL 科学读取前登记。

## 主要结果

| 科学问题 | 已执行结果 |
| --- | --- |
| 留出响应是否更接近实测 | R3B PSD W1 为 0.013729，原 SIM 为 0.104569；二维距离 0.199078，等样本量 REAL–REAL 中位 0.036285。两视图 C2ST BA 均 0.988，仍有完整响应失配。 |
| 两峰是否随电荷变窄 | 实测的总体收窄趋势得到支持。R3B 改善了部分中间区间，最低区与高端低 PSD 成分仍偏离，拟合无支持处保留 undefined。 |
| 合成数据是否有任务价值 | 固定 44 个拟合完成。256 条 Ridge 的 +R3B PSD MAE 从 0.069210 降至 0.061745，但原 SIM 增强为 0.057754；1024 条时 +R3B 与 REAL-only 几乎相同，部分电荷目标恶化。 |
| 留出连续区间能否推广 | FULL/GAP 各完成 1600→2000→2000 更新。输出 q3 的 PSD W1 为 0.005036/0.008257，上升时间 W1 为 0.934/1.965 ns；两臂 q3 C2ST BA 均 1.000。覆盖与源→输出迁移一起报告，不支持可靠完整响应推广。 |
| 保存结果能否复现 | R3/R3B 各 2000×480 波形、R5 的 2816 个预测值、R6 两臂合计 1920000 个生成值的独立进程重放均逐位通过；R2 原始初始化记录缺失单独保留。 |

[留出电荷–PSD](work/R4_FROZEN_HELDOUT_RESPONSE_001/results/figures/charge_psd.png) · [留出峰宽](work/R4_FROZEN_HELDOUT_RESPONSE_001/results/figures/energy_peak_width.png) · [任务误差–容差](work/R5_PREFIX_RECONSTRUCTION_UTILITY_001/results/figures/tolerance_coverage.png) · [区间排除迁移](work/R6_CHARGE_GAP_GENERALIZATION_001/results/figures/charge_migration.png)

## 分阶段材料

| 阶段 | 报告 |
| --- | --- |
| R1 共同接口与完整基线 | [REPORT](work/R1_RESPONSE_SIMILARITY_BASELINE_001/REPORT.md) |
| R2 平滑条件随机响应 | [REPORT](work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/REPORT.md) |
| R3 连续时间形状自由度 | [REPORT](work/R3_FLEXIBLE_TEMPORAL_RESPONSE_001/REPORT.md) |
| R3B 电荷条件 PSD 峰宽 | [REPORT](work/R3B_ENERGY_CONDITIONAL_WIDTH_001/REPORT.md) |
| R4 冻结候选的实测留出评价 | [REPORT](work/R4_FROZEN_HELDOUT_RESPONSE_001/REPORT.md) |
| R5 固定前缀重建任务 | [REPORT](work/R5_PREFIX_RECONSTRUCTION_UTILITY_001/REPORT.md) |
| R6 连续电荷区间排除 | [REPORT](work/R6_CHARGE_GAP_GENERALIZATION_001/REPORT.md) · [紧凑指标](work/R6_CHARGE_GAP_GENERALIZATION_001/METRICS_COMPACT.json) |

## 数据与版本

本地研究根目录为 `E:/GAN/research/sim2real_two_papers_20261008_0305`。GitHub 私有仓库为 [Aggressivecyber/ej276d-sim2real](https://github.com/Aggressivecyber/ej276d-sim2real)，默认分支 `main`。源码、定义、配置、报告、汇总结果及图按阶段提交；逐事件数据、模型、随机状态、日志和缓存保存于 E/D，克隆 GitHub 不会取得这些载荷。

2026-10-08 11:41 UTC 用户要求后续改为本机执行，之后无新 SSH 或服务器依赖。R4 已成功取得的 EVAL 缓存可供本机固定验证使用。输入来源、哈希和完整本地清单保留在阶段文件中；复现时使用新目录，不覆盖封存结果。

所有完整分布包含 QC 有效的 unknown/重叠事件。电荷代理不等于入射中子能量；人工组不是独立采集批次；CCM 伪标签不是粒子真值；任务小样本预算不是全部实测成本。未建立跨真实采集条件、唯一物理响应映射或总采集节省的证据。
