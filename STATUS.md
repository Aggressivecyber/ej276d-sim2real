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

# 当前研究状态

项目active。R29已封存：相同94400行、刷新方向和2950更新，energy U替换后J20选2750；预定末点fullshape BA .970→.992，开发投影距离全部退化，PSD宽度有部分收益。4000独立主回放通过。R30冻结输出的条件矩诊断已登记，0响应拟合/生成、最多15HGB。实际科学作业状态见PROGRAM.json。

R29入口为R29_REPORT.md与R29_DELIVERY_INDEX.json，Git/D副本读取真实回执。
