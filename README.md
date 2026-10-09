# 当前入口：R34保护方案因边界失败关闭，R35检验训练中的PSD顺序

R34恢复BASE基线和电荷/PSD代理，Independent仍有完整波形局部收益，但两个保护端点均出现固定边界跳变。R35只训练一个Q内PSD同序配对CFM端点；这是当前冻结BASE流分支最后一次配对检验。完整响应仍未通过，项目active。

[R34根报告](R34_REPORT.md) · [交付索引](R34_DELIVERY_INDEX.json) · [R35任务](tasks/R35_PSD_ORDERED_FLOW_001.md) · [项目目标](PROJECT_OBJECTIVE.md)。以下保留历史入口。

# 当前入口：R33流修正有局部收益，R34检验源信息保护

R33两臂完整波形可分性降低，但源PSD、条件分布和噪声未共同通过，固定OT配方关闭。R34对两份冻结流施加同一固定投影，保护BASE基线与两段积分，检查形状收益和边界伪差。完整响应仍未通过，项目active。

[R33根报告](R33_REPORT.md) · [交付索引](R33_DELIVERY_INDEX.json) · [R34任务](tasks/R34_PROTECTED_FLOW_CORRECTION_001.md) · [项目目标](PROJECT_OBJECTIVE.md)。以下保留历史入口。

# 当前入口：R32均值配方关闭，R33完整波形流修正

R32均值MSE降低48.07%，完整波形无可信改善，PSD/高侧宽度恶化；固定均值配方关闭。R33冻结源驱动响应为起点，比较等预算独立CFM与区内OT-CFM的完整480维修正。完整响应仍未通过，项目active。

[R32根报告](R32_REPORT.md) · [交付索引](R32_DELIVERY_INDEX.json) · [R33任务](tasks/R33_RESIDUAL_OT_FLOW_001.md) · [项目目标](PROJECT_OBJECTIVE.md)。以下保留历史入口。

# 当前入口：R31归因路线关闭，R32条件均值训练

R31确认分域中心化后的峰归一化约束可完全恢复域身份；固定prompt排除后REAL–REAL仍未满足控制规则，关闭此矩+C2ST归因路线。当前R32直接检验TRAIN条件均值约束，配恢复服务器上的同平台原方案对照。完整响应仍未通过。

[R31根报告](R31_REPORT.md) · [交付索引](R31_DELIVERY_INDEX.json) · [R32任务](tasks/R32_CONDITIONAL_MEAN_TRANSLATION_001.md) · [项目目标](PROJECT_OBJECTIVE.md)。以下保留历史入口。

# 当前入口：R30矩诊断及R31固定负对照

R30显示域别中心/尺度处理会使REAL–REAL也变得可分，故不能据高残余BA定位唯一物理原因。R28仍是开发比较基线，完整响应未通过。当前推进R31峰归一化约束负对照，方案见tasks/R31_NORMALIZATION_ANCHOR_DIAGNOSTIC_001.md。

[R30根报告](R30_REPORT.md) · [交付索引](R30_DELIVERY_INDEX.json) · [项目目标](PROJECT_OBJECTIVE.md)。以下保留历史入口。

# EJ-276D：G4 到实测响应变换

项目目标active。R29 energy U替换改善部分PSD宽度，但完整波形与开发投影距离退化，关闭该单次配方。4000条独立回放通过。R30只读保存输出，诊断全局/条件均值及逐点波动后的剩余可分性。见[R29结论](R29_REPORT.md)、[R30任务](tasks/R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001.md)、[项目进度](PROJECT_PROGRESS.json)。

R1–R9已完成有界研究。R7定位总体波形残差，R8局部修正及R8C整段随机映射均已执行，R9保留五个冻结候选完成回顾性验证。JOINT_SIGNAL的峰后局部BA由0.922降至0.810，但完整shape BA仍0.994；全部候选为0.988–1.000，真实对真实为0.504。条件峰宽也未进一步改善，本轮不采用新模型作为已验证的完整响应转换器。

- [R7–R9最终报告](CONTINUATION_REPORT.md)、[复现说明](CONTINUATION_REPRODUCTION.md)、[交付索引](CONTINUATION_DELIVERY_INDEX.json)。
- [R9固定评价详情](work/R9_FROZEN_RETROSPECTIVE_VALIDATION_001/REPORT.md)、[五候选冻结绑定](CONTINUATION_FROZEN_METHOD.json)。
- [R7残差诊断](work/R7_RESPONSE_RESIDUAL_DIAGNOSIS_001/REPORT.md)、[R8局部配对实验](work/R8_LOCAL_JOINT_FLUCTUATION_001/REPORT.md)、[R8B整体路线依据](work/R8B_FULL_RESPONSE_ROUTE_REVIEW_001/REPORT.md)、[R8C整段映射](work/R8C_WHOLE_RESPONSE_MAPPING_001/REPORT.md)。
- [R1–R6原报告](FINAL_REPORT.md)、[原复现入口](REPRODUCTION.md)、[原交付索引](DELIVERY_INDEX.json)保留。
- [计划记录](PLAN.md)、[当前状态](STATUS.md)、[机器可读状态](PROGRAM.json)。

结论限于同一次AmBe采集和登记的固定配方、种子及预算。旧EVAL已查看，本轮属于冻结后回顾性验证；完整物理响应、独立采集推广和唯一失配根因仍未建立。

## R1–R6已封存结果

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

2026-10-09用户要求后续科学计算使用服务器；早先本机限定已被覆盖。R4 已成功取得的 EVAL 缓存可供本机固定验证使用。输入来源、哈希和完整本地清单保留在阶段文件中；复现时使用新目录，不覆盖封存结果。

所有完整分布包含 QC 有效的 unknown/重叠事件。电荷代理不等于入射中子能量；人工组不是独立采集批次；CCM 伪标签不是粒子真值；任务小样本预算不是全部实测成本。未建立跨真实采集条件、唯一物理响应映射或总采集节省的证据。
