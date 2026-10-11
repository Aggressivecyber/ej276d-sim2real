# R69 当前完整分布的抽块反传小试

这是一项从原第500步共同起点出发的固定 TRAIN 诊断。先测梯度近似误差，再按预登记条件决定是否比较响应质量。原 A–F 和完整模型目标仍未完成。

梯度准入通过：True。固定一次模型、噪声及方向，先完整反传104个原块，再由各块平方范数独立复算抽4块估计器的条件方差。相对完整 D 梯度平方范数为 0.0248038386，相对 D+C 为 4.52780738e-07，预设上限均为1。模型和 Adam 原状态未变；该步没有更新、clip或Adam step。

无偏性只针对 clip 前、给定当前端点和方向的程序梯度。没有证明 clip/Adam、后续训练轨迹或响应质量无偏。候选每步仍使用全部6593 SOURCE、21047 REAL计算当前五项分布目标，只对4/104原块做参数反传；原四项 C 每步照常计算。

**结论：联合局部收益门未通过，关闭固定K4候选；不恢复R65、不改k或随机种子重试。**

K4完成100步，用时 152.056 秒；原微批完成 1502 步，用时 152.059 秒，实际差值 0.002814 秒。两臂同模型、Adam、原随机流与固定学习率；根复核确认抽块序列和实际父反传曝光吻合。

| 共同指标 | 起点 | K4 | 原微批 |
| --- | ---: | ---: | ---: |
| 九项共同损失均值 | 36.2918952 | 4.96160224 | 4.69452707 |
| 总体 PSD W1 | 0.0105397117 | 0.0106130583 | 0.0128170729 |

K4相对微批的损失降低 -5.689%；相对起点降低 86.329%。共同8探针赢 0/8；主门通过 False，完整102项无实质退化门通过 False。失败 58/102，全部保留。

固定模型噪声/方向配对的K4减微批均值差描述性95%区间为 [0.203774532, 0.34279157]；不是独立训练或独立采集置信区间。

| 未通过项 | 起点 | K4 | 原微批 |
| --- | ---: | ---: | ---: |
| pieces / temporal_joint | 0.15849799 | 0.13170258 | 0.10881588 |
| pieces / source_order | 2.2436324 | 1.8951071 | 1.5776884 |
| pieces / REAL_identity | 0.021059072 | 0.014925431 | 0.0041490282 |
| nonnegative / PSD_W1_region_-1 | 0.10177857 | 0.12025037 | 0.14221707 |
| nonnegative / baseline_RMS_W1_region_-1 | 6.6195671e-05 | 5.6329024e-05 | 3.3182906e-05 |
| nonnegative / baseline_RMS_W1_region_0 | 2.0455819e-05 | 2.2333236e-05 | 1.712424e-05 |
| nonnegative / PSD_W1_region_1 | 0.017721137 | 0.011163475 | 0.0090220091 |
| nonnegative / baseline_RMS_W1_region_1 | 2.0098183e-05 | 2.1804219e-05 | 1.5676475e-05 |
| nonnegative / PSD_W1_region_2 | 0.033335391 | 0.041373894 | 0.037153396 |
| nonnegative / baseline_RMS_W1_region_2 | 2.5667984e-05 | 2.7180147e-05 | 2.4589886e-05 |
| nonnegative / PSD_W1_region_3 | 0.022307549 | 0.031272706 | 0.029063661 |
| nonnegative / baseline_RMS_W1_region_3 | 2.5176432e-05 | 2.83661e-05 | 2.5170445e-05 |
| nonnegative / PSD_W1_region_4 | 0.018160714 | 0.011195131 | 0.0077896228 |
| nonnegative / baseline_RMS_W1_region_4 | 2.1932431e-05 | 2.7460749e-05 | 1.996408e-05 |
| nonnegative / PSD_W1_region_5 | 0.014465235 | 0.014337267 | 0.0060995392 |
| nonnegative / baseline_RMS_W1_region_5 | 2.9239058e-05 | 3.5849955e-05 | 3.778657e-05 |
| nonnegative / PSD_W1_region_6 | 0.011760048 | 0.010429257 | 0.0053369298 |
| nonnegative / baseline_ACF1_W1_region_8 | 0.017686774 | 0.030279658 | 0.017530696 |
| correlations / overall | 0.17380738 | 0.21609692 | 0.25610119 |
| REAL_pieces / full_window | 0.049484368 | 0.045100252 | 0.02475379 |
| REAL_pieces / coordinate_RMS | 0.063379987 | 0.053468689 | 0.027220775 |
| REAL_pieces / PSD | 1.6408703 | 0.98857256 | 0.33316588 |
| REAL_pieces / noise | 0.3892984 | 0.35909877 | 0.13298275 |
| REAL_pieces / temporal_joint | 0.15191951 | 0.12418567 | 0.046483585 |
| REAL_nonnegative / PSD_W1_overall | 0.017041776 | 0.010581388 | 0.0038519973 |
| REAL_nonnegative / baseline_RMS_W1_overall | 3.9434764e-05 | 3.5017129e-05 | 1.5661983e-05 |
| REAL_nonnegative / baseline_ACF1_W1_overall | 0.062892666 | 0.059926239 | 0.018399169 |
| REAL_nonnegative / PSD_W1_region_-1 | 0.05609253 | 0.044441124 | 0.028112788 |
| REAL_nonnegative / baseline_RMS_W1_region_-1 | 2.0932118e-05 | 2.0102075e-05 | 1.3836734e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_-1 | 0.047484156 | 0.042493041 | 0.0056077661 |
| REAL_nonnegative / PSD_W1_region_0 | 0.0233803 | 0.014585429 | 0.003603091 |
| REAL_nonnegative / baseline_RMS_W1_region_0 | 3.8389082e-05 | 3.3397708e-05 | 1.4327371e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_0 | 0.060265717 | 0.05783106 | 0.013328808 |
| REAL_nonnegative / PSD_W1_region_1 | 0.015454089 | 0.0086267771 | 0.0013526636 |
| REAL_nonnegative / baseline_RMS_W1_region_1 | 3.9110923e-05 | 3.309205e-05 | 1.420245e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_1 | 0.060591927 | 0.058139435 | 0.013583454 |
| REAL_nonnegative / PSD_W1_region_2 | 0.017927094 | 0.012627805 | 0.0060907584 |
| REAL_nonnegative / baseline_RMS_W1_region_2 | 4.2198744e-05 | 3.8786558e-05 | 2.3452027e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_2 | 0.066558856 | 0.064111048 | 0.020977916 |
| REAL_nonnegative / PSD_W1_region_3 | 0.017076117 | 0.0099436341 | 0.005769101 |
| REAL_nonnegative / baseline_RMS_W1_region_3 | 3.9042474e-05 | 3.4348345e-05 | 1.7263774e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_3 | 0.063761259 | 0.061319599 | 0.019374278 |
| REAL_nonnegative / PSD_W1_region_4 | 0.014084759 | 0.0071719371 | 0.0016271153 |
| REAL_nonnegative / baseline_RMS_W1_region_4 | 3.8693765e-05 | 3.3491397e-05 | 1.0871407e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_4 | 0.062300617 | 0.05930655 | 0.016894635 |
| REAL_nonnegative / PSD_W1_region_5 | 0.012156295 | 0.0064558185 | 0.00090141236 |
| REAL_nonnegative / baseline_RMS_W1_region_5 | 4.041275e-05 | 3.5679406e-05 | 1.3551895e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_5 | 0.064731218 | 0.061880632 | 0.019807222 |
| REAL_nonnegative / PSD_W1_region_6 | 0.010350763 | 0.0049190388 | 0.00033510061 |
| REAL_nonnegative / baseline_RMS_W1_region_6 | 4.036005e-05 | 3.6161382e-05 | 1.6444063e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_6 | 0.063945372 | 0.061214537 | 0.022496463 |
| REAL_nonnegative / PSD_W1_region_7 | 0.011192111 | 0.0062637419 | 0.0016004842 |
| REAL_nonnegative / baseline_RMS_W1_region_7 | 4.1501188e-05 | 3.7663365e-05 | 1.5711483e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_7 | 0.063893364 | 0.059667344 | 0.019690559 |
| REAL_nonnegative / PSD_W1_region_8 | 0.013311089 | 0.010745386 | 0.0016837709 |
| REAL_nonnegative / baseline_RMS_W1_region_8 | 4.9391841e-05 | 4.4995087e-05 | 1.6972831e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_8 | 0.07028876 | 0.066447457 | 0.030539657 |
| REAL_nonnegative / parentwise_normalized_identity_MSE | 0.021716635 | 0.015806435 | 0.00437097 |

| 全部原区域 PSD W1 | 起点 | K4 | 原微批 |
| --- | ---: | ---: | ---: |
| overall | 0.010539712 | 0.010613058 | 0.012817073 |
| region_0 | 0.0226635 | 0.007490287 | 0.011235895 |
| region_1 | 0.017721137 | 0.011163475 | 0.0090220091 |
| region_2 | 0.033335391 | 0.041373894 | 0.037153396 |
| region_3 | 0.022307549 | 0.031272706 | 0.029063661 |
| region_4 | 0.018160714 | 0.011195131 | 0.0077896228 |
| region_5 | 0.014465235 | 0.014337267 | 0.0060995392 |
| region_6 | 0.011760048 | 0.010429257 | 0.0053369298 |
| region_7 | 0.13006858 | 0.13324675 | 0.13013196 |
| region_-1 | 0.10177857 | 0.12025037 | 0.14221707 |
| region_8 | 0.058175241 | 0.059455634 | 0.062906019 |

实际原生计算 348.048 秒，登记上限600秒；worker已退出，GPU计算进程为空。CPU亲和性0–11，观测worker峰值RSS 3.126 GiB，进程GPU峰值 4.223 GiB。

该结果只适用于原第500步共同起点、固定LR和这一次短程试验。起点由旧全量D训练产生；8组探针是固定模型噪声重复。所有原父、区域和支持外事件保持；未访问DEV/EVAL/CAL，未运行新HGB、峰位/峰宽拟合或D/E任务评价。不能据此宣称原A–F通过或不可行。

复核入口：work/R69_SAMPLED_FULL_GRADIENT_PILOT_001/RESULT.json；reviews/R69_SAMPLED_FULL_GRADIENT_PILOT_001/ROOT_RESULT_REVIEW.json、COLLECTION_RECEIPT.json。协议和冻结源代码同目录保留；完整复现需原R65依赖与第500步原检查点。

后续决策依据已封存的联合证据，遵守当前候选停止边界。总目标保持active，原验收、源人口和源电荷不变。
