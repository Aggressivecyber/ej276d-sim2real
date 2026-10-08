# 当前研究状态

2026-10-08 UTC 修订016。现行目标与简化执行要求以 PLAN.md、AGENTS.md、FROZEN_METHOD.json 为准。GitHub 私有仓库为 https://github.com/Aggressivecyber/ej276d-sim2real，默认 main；完成阶段单独提交，原始/逐事件数据、模型和缓存留 E/D。

R1 已完成完整 2000 父事件共同评价。原 SIM / 历史 GEN_W0 的 PSD W1 为 0.104627 / 0.041618，形状 C2ST 均为 0.996；完整响应分布仍存在明显偏差。

R2_SMOOTH_STOCHASTIC_RESPONSE_001 已完成并固定。单作业75.4096秒、4线程、峰值RSS828317696字节；1600更新，以TRAIN内部组验证选第1200步。R2 PSD W1=0.021781、电荷W1=868.8mV ns，固定时间纹理消退，噪声谱与RMS明显改善。但独立C2ST两视图BA均0.992；q1只有130/2000事件，实测467/2000；q1/q2上升时间W1为2.9067/1.4600ns。报告：work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/REPORT.md。

R2 从头训练初始化存在已披露缺口：隐藏层构造前未设置主Torch种子且未保存初始RNG。当前保存权重的16条固定输入复载逐位通过，原执行源码与模型完整保留；不追溯补写、不为补种子重训。该限制与响应分布改善同时记录。

R3_FLEXIBLE_TEMPORAL_RESPONSE_001已封存。PSD W1=0.008904、上升时间W1=0.35074ns，完整C2ST BA=0.994；q3/q4条件距离恶化，局部肩部/双峰保留。2000更新，数值累计163.6576秒，峰RSS935022592字节；初始状态已保存，从实际权重/RNG的独立2000×480推理逐位一致。ROOT已审阅最终报告/回执与5张关键图。报告：work/R3_FLEXIBLE_TEMPORAL_RESPONSE_001/REPORT.md。

R3B_ENERGY_CONDITIONAL_WIDTH_001已封存。实测TRAIN/DEV支持当前电荷范围内两成分总体收窄；共同合格bin2–7的低/高sigma MAE从R3的0.007507/0.013381降为0.005017/0.004390。最低区偏差、第7区低侧过宽、第8区不收敛仍未解决，C2ST BA=0.996。2000更新、TRAIN验证选1700；实际保存权重/RNG完整960000值逐位重放通过，数值累计257.481秒，4线程、峰RSS1036435456字节。报告work/R3B_ENERGY_CONDITIONAL_WIDTH_001/REPORT.md，ROOT已审阅关键证据及4图。

实测来自同一次历史曝光采集：P1K QC有效DEV2000、TRAIN3191、CAL1594，人工组不是独立采集。CAL电荷边界与参考冻结；unknown/重叠事件保留，q4未收敛和局部双带支持不足不通过补标签解决。11:15:15.7983273 UTC在EVAL科学解码仍0时生成FROZEN_METHOD.json，SHA b4e1f86807dba40f6004bda48a40d0644fe5039535a3c3eb2e402fc213065186。

当前已授权R4留出响应→R5前缀重建任务→R6完整电荷区间留出的连续执行；R4需在提取EVAL前保存实现READINESS。模型候选、44个任务小拟合、q3间隔与FULL/GAP相同初始化三阶段课程已预先固定，不依据EVAL改变。完整目标仍未完成，不以开发集局部改善替代后续证据。
