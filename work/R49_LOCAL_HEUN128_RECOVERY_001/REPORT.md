# R49：冻结两组权重的 Heun128 完整响应评价

本轮没有新增训练或噪声校准。R48 两组 50000 更新末点采用共同 Heun128，256 步仅作原128条 TRAIN检查的参考。原阈值、源人口与全部权重保持。本次是用户授权的本机恢复；本机数组、评价器和重放单独执行，远端旧指标不绑定为本机结果。

两臂数值门均通过，8000主输出独立重放 通过。指定主候选的 A/B/C 为 False/False/False；决定：close_fixed_R49_Heun128_recipe。项目模型目标尚未完成，D/E未执行。对照不能替代主候选成为赢家。

## 数值与复现

| 冻结臂 | TRAIN RMS中位/LSB | p95/LSB | 最大相对q误差 |
|---|---:|---:|---:|
| GUIDED_FIXED_NOISE_CONTROL | 0.0384442 | 0.357123 | 1.11934e-08 |
| GUIDED_NOISE_COMPLETION | 0.0288385 | 0.375439 | 9.34925e-09 |

原限值为0.25/1 LSB和1e-5；没有调整检查人口或尝试第三种步数。所有主生成和重放使用本机同一CPU后端、4线程、chunk64，float64坐标/积分、float32网络与末次cast、一次ADC。256步不是精确真解。

## 完整响应与直接两臂比较

| 人口/冻结臂 | 视图 | BA | 95%分组区间 | AUC |
|---|---|---:|---|---:|
| GUIDED_FIXED_NOISE_CONTROL | full_historical_shape | 0.762 | [0.724532, 0.796433] | 0.858464 |
| GUIDED_FIXED_NOISE_CONTROL | normalized_u480 | 0.736 | [0.699208, 0.772559] | 0.844816 |
| GUIDED_FIXED_NOISE_CONTROL | full_voltage | 0.832 | [0.798577, 0.86313] | 0.911056 |
| GUIDED_FIXED_NOISE_CONTROL | Q_PSD_2D | 0.7 | [0.660055, 0.740324] | 0.784096 |
| GUIDED_FIXED_NOISE_CONTROL | baseline_local | 0.574 | [0.527452, 0.618289] | 0.597248 |
| GUIDED_NOISE_COMPLETION | full_historical_shape | 0.78 | [0.744886, 0.817317] | 0.859152 |
| GUIDED_NOISE_COMPLETION | normalized_u480 | 0.76 | [0.724708, 0.794672] | 0.838688 |
| GUIDED_NOISE_COMPLETION | full_voltage | 0.816 | [0.782691, 0.848071] | 0.900512 |
| GUIDED_NOISE_COMPLETION | Q_PSD_2D | 0.688 | [0.643866, 0.72476] | 0.772752 |
| GUIDED_NOISE_COMPLETION | baseline_local | 0.604 | [0.557737, 0.649192] | 0.63024 |
| GUIDED_FIXED_NOISE_REAL_CONTROL | full_historical_shape | 0.516 | [0.463429, 0.561082] | 0.538288 |
| GUIDED_FIXED_NOISE_REAL_CONTROL | normalized_u480 | 0.512 | [0.469205, 0.551883] | 0.557216 |
| GUIDED_FIXED_NOISE_REAL_CONTROL | full_voltage | 0.83 | [0.800295, 0.858589] | 0.923488 |
| GUIDED_FIXED_NOISE_REAL_CONTROL | Q_PSD_2D | 0.5 | [0.462029, 0.540916] | 0.49608 |
| GUIDED_FIXED_NOISE_REAL_CONTROL | baseline_local | 0.52 | [0.485247, 0.561957] | 0.511504 |
| GUIDED_COMPLETION_REAL_CONTROL | full_historical_shape | 0.49 | [0.451106, 0.530038] | 0.505904 |
| GUIDED_COMPLETION_REAL_CONTROL | normalized_u480 | 0.542 | [0.495387, 0.582965] | 0.544896 |
| GUIDED_COMPLETION_REAL_CONTROL | full_voltage | 0.828 | [0.795632, 0.857996] | 0.902272 |
| GUIDED_COMPLETION_REAL_CONTROL | Q_PSD_2D | 0.49 | [0.44808, 0.530398] | 0.488616 |
| GUIDED_COMPLETION_REAL_CONTROL | baseline_local | 0.52 | [0.481843, 0.563716] | 0.522496 |

A要求同一个主候选的SOURCE及同处理REAL五个视图95%区间均包含在[0.40,0.60]内；不是仅凭点值接近0.5。以下差值为主候选减固定噪声对照，复用已拟合预测和固定500次配对区组bootstrap；没有追加分类器。
baseline_local采用100点基线逐事件去均值、除RMS后的z及ACF/归一化谱，检验标准化基线局部形态与相对噪声特征，没有附加log(std)。该视图通过不能独立证明绝对基线或噪声幅度匹配；完整电压视图与B中的绝对baseline RMS距离另行保留。

| 人口 | 视图 | ΔBA | 95%配对区间 |
|---|---|---:|---|
| SOURCE | full_historical_shape | 0.018 | [-0.00806577, 0.044514] |
| SOURCE | normalized_u480 | 0.024 | [-0.00563183, 0.0557092] |
| SOURCE | full_voltage | -0.016 | [-0.0443364, 0.0179308] |
| SOURCE | Q_PSD_2D | -0.012 | [-0.0471801, 0.0219514] |
| SOURCE | baseline_local | 0.03 | [-0.00472061, 0.0669332] |
| REAL | full_historical_shape | -0.026 | [-0.0691498, 0.0155892] |
| REAL | normalized_u480 | 0.03 | [-0.0096065, 0.0653679] |
| REAL | full_voltage | -0.002 | [-0.0376635, 0.0310567] |
| REAL | Q_PSD_2D | -0.01 | [-0.0435394, 0.0173463] |
| REAL | baseline_local | 0 | [-0.0438432, 0.0392333] |

SOURCE与同处理REAL原五个视图的全部两臂配对区间都包含零。本轮固定分类器结果尚不能确认噪声扣除规则相对固定噪声带来改善；这些区间仍允许两臂差异，不能据此认定等效。

同处理REAL同时报告相对RAW的绝对改变和相对旧BASE_REAL的预登记差；不预设随机响应对已含噪REAL恒等。RAW二维参照缺失继续unknown，不补拟合。分组区间条件于当前已拟合评价器和采集，不包含重新训练或独立采集不确定性。

## PSD、噪声、源保真和门槛

| 臂 | 总体PSD W1 | 固定Q1–7等权W1 | 低/高峰位MAE(Q2–7) | 原六门 |
|---|---:|---:|---|---|
| GUIDED_FIXED_NOISE_CONTROL | 0.0418382 | 0.0504213 | undefined/undefined | [True, False, False, False, True, True] |
| GUIDED_NOISE_COMPLETION | 0.0288961 | 0.0393207 | undefined/undefined | [True, False, False, False, True, True] |

GUIDED_FIXED_NOISE_CONTROL raw-width相对误差（固定Q2–7）：component0: undefined, failed bins [3]；component1: undefined, failed bins [3]


GUIDED_NOISE_COMPLETION raw-width相对误差（固定Q2–7）：component0: undefined, failed bins [3]；component1: undefined, failed bins [3]

固定必需区失败或缺失不删后平均；undefined不是0，也不能从密度拟合失败断言真实粒子峰消失。低/高拟合成分不是粒子真值。

### GUIDED_FIXED_NOISE_CONTROL

- baseline_RMS_V_overall_W1: 3.94781e-05，BASE 1.93584e-05，相对变化 1.03933；pass=False。
- ACF1_overall_W1: 0.0585626，BASE 0.064235，相对变化 -0.0883059；pass=True。
- source_Q_to_output_Q: 0.999999，BASE 0.999726；pass=True。
- source_PSD_to_output_PSD: 0.441126，BASE 0.418992；pass=True。
- 实际未量化source-bin3 pure-had N=213，谷比例 0.230047，BASE 0.056338，差 0.173709；pass=False。
- 实际未量化source-bin4 pure-had N=181，谷比例 0.0552486，BASE 0.00552486，差 0.0497238；pass=False。
- 失败时间边界：['second_center_150']。
- 同处理REAL失败门视图：[]。

### GUIDED_NOISE_COMPLETION

- baseline_RMS_V_overall_W1: 3.80563e-05，BASE 1.93584e-05，相对变化 0.96588；pass=False。
- ACF1_overall_W1: 0.0578283，BASE 0.064235，相对变化 -0.0997385；pass=True。
- source_Q_to_output_Q: 0.999999，BASE 0.999726；pass=True。
- source_PSD_to_output_PSD: 0.401017，BASE 0.418992；pass=True。
- 实际未量化source-bin3 pure-had N=213，谷比例 0.262911，BASE 0.056338，差 0.206573；pass=False。
- 实际未量化source-bin4 pure-had N=181，谷比例 0.187845，BASE 0.00552486，差 0.18232；pass=False。
- 失败时间边界：['second_center_150']。
- 同处理REAL失败门视图：[]。

原2000 SOURCE父ID、组成、顺序和权重保留，417支持外行也走同一新流；只有条件索引夹界，没有波形/电荷裁剪或BASE回退。实际输入source-Q与历史ADC参考source-Q分列。pure-had谷比例是冻结源分组代理，不是粒子误分类率。源置换敏感性不证明物理逐事件映射正确。功能分量含末次float32cast且保留交叉项，不能称独立物理电子噪声。

## 执行、失败与交付

| 入口 | 实际worker PID | native exit | 秒 | RSS峰值/字节 |
|---|---:|---:|---:|---:|
| check_models.py | 49200 | 0 | 370.530977 | 340205568 |
| generate_all.py | 21408 | 0 | 6927.145749 | 596807680 |
| evaluate_saved.py | 58492 | 0 | 75.707823 | 1126510592 |
| replay_all.py | 39264 | 0 | 5664.799126 | 390782976 |
| summarize_and_plot.py | 61804 | 0 | 10.979147 | 355270656 |
| render_train.py | 47312 | 0 | 2.662717 | 110391296 |
| render_boundary_layout.py | 64628 | 0 | 1.872391 | 83906560 |

本机科学原生作业累计 13053.697928 秒；新训练0/新校准0。R48训练各50000更新仅作历史输入，不再次计入本轮耗时。
科学/墙钟/deadline/阶段体积配额为null，本机安全RSS4GiB、一次一个本机科学作业和CPU4保持；CPU没有GPU准入门。每作业按历史峰值和系统余量检查可用内存，RSS缺测原因不明确时标unknown。
远端旧尝试已完成2107.3375221949973科学秒；其原重放worker218727最后确认下界120.15029214299284秒，10:22:45 UTC后传输中断、终态unknown。用户明确授权维护期间换本机；停止重连且未取消或重复提交该远端作业，不宣称已确认全局只有一个物理作业。本地重新生成和评价的费用与历史分开。
准备环境探测因psutil缺失和编码名utf8-sig出现两次控制失败；执行期间一次小JSON读取命令因PowerShell管道语法失败，另一次读取尚未出现的管线进度文件返回native1。五份控制事件均保留：另一次文案补丁因上下文匹配失败而零写入，随后正确应用，以既有Windows ctypes监督、正确编码和读取语法处理，没有科学推理、训练或拟合。若存在科学原生失败，其日志和原件列在NATIVE_CHAIN及RESOURCE_RECEIPT。
本次DEV已在历史多轮暴露，只属于同采集开发比较，不称新盲测或独立物理校准。完整私有数组、预测、身份、资源/native回执在本目录；发布版METRICS_COMPACT只保留聚合。历史封存阶段保持不变；R49旧远端尝试仍未完成，当前本机阶段在终端回执后封存。Git/D由根代理完成，正常复制不宣称迁移一致性验证。
早期SOURCE_BINDINGS与LOCAL_SOURCE_FREEZE保留准备时指纹。之后三份文件有工程或报告修订：evaluate_saved.py修正纯日志标签并保存前像；render_figures.py移除重复TRAIN绘制调用；make_report.py补充本机来源、指标解释及配对区间说明。它们不改变科学公式、参数、索引或已执行结果。最终文件字节以FINAL_MANIFEST为准，不声称这三项仍等于早期指纹。
边界图初版的图例遮住标题，原PNG和脚本保留在display_initial；独立render_boundary_layout.py仅重绘这一张已存统计图，实际PID/RSS守护和费用见BOUNDARY_LAYOUT_CORRECTION_001与原生回执，不重复生成或拟合。修复后的最终PNG已再次实际打开。

图件见results/figures，全部最终图实际打开检查的逐图SHA见FIGURE_QA.json。FINAL_MANIFEST显式排除两个自引用终端文件；最终native退出由实际工具回执绑定。

