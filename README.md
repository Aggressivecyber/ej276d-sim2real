# EJ-276D G4-to-real response research

以 Geant4 沉积及派生波形为输入，学习到 EJ-276D/AmBe 实测响应的变换。当前工作限于同一次采集内的回顾性留出验证；不将分布相似、FoM 接近或 C2ST 接近随机当作物理等效证明。

[GitHub 私有仓库](https://github.com/Aggressivecyber/ej276d-sim2real)，默认分支 `main`。

## 当前入口

- [研究计划](PLAN.md) 与 [当前状态](STATUS.md)
- [执行约定](AGENTS.md) 与 [机器可读状态](PROGRAM.json)
- [实测用途分区](CURRENT_ROLE_INDEX.json)：TRAIN/CAL/DEV 分开使用，EVAL 留作方法选择后的最终评价
- [R1 完整基线报告](work/R1_RESPONSE_SIMILARITY_BASELINE_001/REPORT.md)
- [R2 平滑随机响应报告](work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/REPORT.md)：已完成
- [R3 平滑时间形状报告](work/R3_FLEXIBLE_TEMPORAL_RESPONSE_001/REPORT.md)：已完成
- [R3B 能量条件峰宽报告](work/R3B_ENERGY_CONDITIONAL_WIDTH_001/REPORT.md)：已完成，部分条件改善及边界失败均保留
- [冻结方法与后续任务](FROZEN_METHOD.json)：EVAL解码前固定，R4留出响应验证进行中

R1 使用相同的 2000 个 G4 父事件比较原 SIM、历史转换与实测。历史 GEN_W0 的 PSD Wasserstein 距离由 0.104627 降到 0.041618，但形状 C2ST 平衡准确率仍为 0.996，实测子集之间为 0.490；条件带宽、固定时间纹理和噪声相关性仍有明显偏差。该结果支持继续改进响应模型，不能写成已经复现实测。

[电荷—PSD 图](work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/results/figures/charge_psd.png) · [条件波形图](work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/results/figures/conditional_waveforms.png) · [噪声与涨落图](work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/results/figures/fluctuations.png)

R2 使用 8192 个 G4 TRAIN 父事件和独立实测 TRAIN，学习平滑随机响应。PSD W1 进一步降到 0.021781，电荷 W1 为 868.8 mV ns，旧固定纹理消失、噪声显著改善；但独立形状 C2ST 仍为 0.992，低电荷覆盖和前沿范围不足，不能声称完整响应已经匹配。1600 次更新后只按 TRAIN 组验证选择第 1200 步，没有 DEV 选点。R2 初始隐藏层 Torch 种子未记录，不能保证从头训练逐位复现；已保存权重的固定输入复载通过。当前权重和输入作为外部文件保存在 E 盘，详见报告和清单。

[R2 电荷—PSD 图](work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/results/figures/charge_psd.png) · [R2 条件波形图](work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/results/figures/conditional_waveforms.png) · [R2 噪声图](work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/results/figures/fluctuations.png)

## 版本与数据

R3 的总体PSD W1降到0.008904、上升时间W1降到0.35074 ns，但q3/q4条件覆盖变差，完整C2ST BA仍0.994。固定权重与RNG的2000×480输出已在独立进程逐位重放。按用户指出的峰宽随能量变化问题，R3B随后显式检查了窄电荷区间内的峰位和宽度，并增加条件PSD分布约束；不把全局相似视为条件响应成功。

[R3 电荷—PSD 图](work/R3_FLEXIBLE_TEMPORAL_RESPONSE_001/results/figures/charge_psd.png) · [R3 条件波形图](work/R3_FLEXIBLE_TEMPORAL_RESPONSE_001/results/figures/conditional_waveforms.png) · [R3 固定条件对](work/R3_FLEXIBLE_TEMPORAL_RESPONSE_001/results/figures/fixed_latent_condition_pairs.png)

R3B在相同六个可比窄电荷区间内，将低/高PSD成分的宽度MAE降至0.005017/0.004390，实测本身的收窄趋势也已用组bootstrap与去趋势诊断确认。最低区、第7区低侧和最高区仍不匹配，C2ST BA为0.996；不将这些局部改进写成完整响应成功。最终三个响应候选及R4/R5/R6方案已在EVAL科学值解码前固定，继续完成留出验证、合成任务价值和连续电荷区间排除测试。

[峰位与宽度随电荷](work/R3B_ENERGY_CONDITIONAL_WIDTH_001/results/figures/energy_peak_width.png) · [八个窄区间PSD截面](work/R3B_ENERGY_CONDITIONAL_WIDTH_001/results/figures/energy_psd_slices.png) · [实测宽度不确定性](work/R3B_ENERGY_CONDITIONAL_WIDTH_001/results/figures/REAL_width_uncertainty.png)

本地仓库位于 `E:/GAN/research/sim2real_two_papers_20261008_0305`。Git 保存代码、配置、研究计划、阶段报告、紧凑指标和关键图；原始/派生逐事件数据、模型权重、日志、缓存和备份留在 E/D。输入路径及已有校验值见各阶段的 `INPUT_MANIFEST.json`，克隆仓库不会自动取得这些外部输入。

当前提交历史从 2026-10-08 的已完成 R1 开始。更早的本地材料及负结果继续保留，现行计划已经覆盖旧逐步骤审批流程。每个完成阶段单独提交；提交前按明确路径选择文件，不使用无范围的 `git add .`，不提交仍在写入的运行结果。GitHub 仓库默认保持私有。

R1 的依赖版本见 [ENVIRONMENT.json](work/R1_RESPONSE_SIMILARITY_BASELINE_001/ENVIRONMENT.json)。源代码、当次配置和运行回执已保留。历史配置含原工作站路径、时间预算和独立运行编号；复现时需具备清单中的输入，并在新目录记录运行环境调整，不能覆盖已完成结果。设置临时目录和缓存到 E/D 后再执行。历史数值结果不因建仓而重新计算。
