# R44：关闭平均 TRAIN PSD 速度残差解释

冻结 R43 在原 94400 个实际 TRAIN 插值状态上只前向一次，2950 个 chunk32调用。预登记方向规则要求原 bins2–7 中至少4区达到条件，实际只有bin7，共1/6，按规则关闭该平均残差解释。没有训练、新ODE、HGB/EM或新候选，不能据此自动加补偿或扫描PSD损失权重。

整体 PSD 速度有符号残差均值 0.0097688100，RMSE 0.0294167428；全480维速度 MSE 0.8690044797。PSD 对应的一维切空间方向占误差能量 1.3700%。这个比例只描述当前归一化损失几何，不说明PSD物理不重要，也不能把TRAIN路径残差当作实际ODE轨迹误差或完整输出偏移的因果解释。

实际消费目标、唯一目标、全量REAL TRAIN、RAW DEV和已有R43输出的人口统计分别保留。总体PSD均值依次为0.247304、0.247627、0.247884、0.247327及R43 preADC 0.293287；这些相近的目标均值不能消除条件人口差异。Q区外TRAIN与417条源输出仍单列，没有重新加权或删区。统计耦合不构成逐事件真值。

PSD速度与端点差恒等式最大误差4.16e-16，目标/预测切向电荷残差不超过2.22e-16，逐行能量分解闭合误差不超过9.10e-13；原模型权重未改。载入SOURCE/REAL为6593/21047，实际唯一为3752/18739；94400是重复行出现次数。

服务器RTX5070/CUDA执行16.233382秒，本机人口统计和绘图12.680087秒，科学累计28.913469/3600秒，两个实际科学进程均native0。服务器CPU12、作业树RSS48GiB、GPU10GiB预算生效；实际GPU allocated/reserved峰59,164,672/67,108,864B，进程显存采样峰308,281,344B，worker RSS高水位1,656,750,080B。两次RSS查询缺测原因unknown。没有为了占满资源而重算。

三个准备/读取控制错误及一次纯封存tar成员名错误完整保留；修复后seal工具610cdf/PID25704/native0，无科学重复。实际封存79文件/38,367,583字节，最后阶段写入2026-10-10 00:57:22.556782 UTC。四张最终图已由执行器和根实际打开检查；根没有重算科学结果。

下一问题是原2950更新是否不足：R45固定结构、损失、源/目标人口、耦合和32步求解器，仅延长同一优化器轨迹至总50000更新，并固定新的TRAIN路径探针。新增白噪声和时间是同一目标的额外Monte Carlo抽样，不称重复原94400个数值状态。通过原TRAIN数值门后才执行完整分布评价；训练误差降低不能替代联合验收。R43和本R44结论保持关闭，完整模型目标尚未完成。

[完整报告](work/R44_PSD_VELOCITY_FIT_DIAGNOSTIC_001/REPORT.md) · [残差图](work/R44_PSD_VELOCITY_FIT_DIAGNOSTIC_001/results/figures/PSD_velocity_Q_time.png) · [方向规则](work/R44_PSD_VELOCITY_FIT_DIAGNOSTIC_001/results/figures/population_means_direction_markers.png) · [复现入口](work/R44_PSD_VELOCITY_FIT_DIAGNOSTIC_001/REPRODUCTION.md) · [R45协议](tasks/R45_EXTENDED_BUDGET_NEURAL_FLOW_001.md)。私有Git与正常D副本以实际R44_PUBLICATION_RECEIPT.json、R44_BACKUP_RECEIPT.json为准。
