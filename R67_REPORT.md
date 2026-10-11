# R67 同起点、等成本小试

这是固定 TRAIN 人口、共同第500步状态和固定学习率下的短程诊断；原 A–F 与 R65 的50,000步终点均未完成。

**结论：本次未建立全量路线的联合等成本收益，不恢复昂贵的 R65 长训练。**

两臂从同一第500步模型、Adam 与全部随机状态开始。全量完成20步，训练 98.520 秒；小批完成 979 步，训练 98.615 秒。时间差 0.095422 秒，匹配门通过。固定学习率 0.000199999999819。

| 共同指标 | 起点 | 全量 | 小批 |
| --- | ---: | ---: | ---: |
| 九项共同损失均值 | 40.220085 | 5.9281736 | 4.5463601 |
| 总体 PSD W1 | 0.010475867 | 0.00706299 | 0.0065725843 |
| 源 PSD 秩相关 | 0.17467053 | 0.19050557 | 0.23467534 |

全量相对小批的共同损失改善为 -30.394%（负值代表更差）；相对起点为 85.261%。全量在8组中赢 0 组。主门通过：False；无实质退化门通过：False。

共有 63/102 项退化门未通过，全部保存在 ROOT_RESULT_REVIEW.json；不删除失败区域。

| 未通过的退化门 | 起点 | 全量 | 小批 |
| --- | ---: | ---: | ---: |
| pieces / noise | 0.16960984 | 0.14475523 | 0.12885801 |
| pieces / temporal_joint | 0.15495993 | 0.15747813 | 0.091224456 |
| pieces / source_order | 2.0996914 | 1.9786055 | 1.7073965 |
| pieces / REAL_identity | 0.022610119 | 0.015455718 | 0.0050965255 |
| nonnegative / baseline_RMS_W1_overall | 2.3122816e-05 | 1.7907469e-05 | 1.509471e-05 |
| nonnegative / baseline_RMS_W1_region_-1 | 6.6084678e-05 | 6.2167916e-05 | 4.2267983e-05 |
| nonnegative / baseline_RMS_W1_region_0 | 2.0146172e-05 | 2.5841155e-05 | 1.9335817e-05 |
| nonnegative / baseline_ACF1_W1_region_0 | 0.02829247 | 0.030069974 | 0.033715748 |
| nonnegative / baseline_RMS_W1_region_1 | 2.0401089e-05 | 2.5814174e-05 | 1.4156412e-05 |
| nonnegative / baseline_RMS_W1_region_2 | 2.5881284e-05 | 3.2638979e-05 | 2.2443479e-05 |
| nonnegative / baseline_ACF1_W1_region_2 | 0.022514527 | 0.023853962 | 0.025971875 |
| nonnegative / PSD_W1_region_3 | 0.022182355 | 0.024327009 | 0.028309225 |
| nonnegative / baseline_RMS_W1_region_3 | 2.4379279e-05 | 3.1622464e-05 | 2.2065866e-05 |
| nonnegative / PSD_W1_region_4 | 0.017537652 | 0.013536015 | 0.0079789894 |
| nonnegative / baseline_RMS_W1_region_4 | 2.2856557e-05 | 2.8747584e-05 | 1.7159464e-05 |
| nonnegative / PSD_W1_region_5 | 0.014339318 | 0.011566165 | 0.0093877178 |
| nonnegative / baseline_RMS_W1_region_5 | 3.0352638e-05 | 3.8996984e-05 | 2.7808101e-05 |
| nonnegative / baseline_RMS_W1_region_6 | 3.0114767e-05 | 3.7264726e-05 | 5.1581407e-05 |
| nonnegative / baseline_RMS_W1_region_8 | 1.7323646e-05 | 1.4777787e-05 | 1.261655e-05 |
| nonnegative / baseline_ACF1_W1_region_8 | 0.018133657 | 0.02310502 | 0.033102741 |
| nonnegative / projection_fraction_overall | 0.12450705 | 0.11825042 | 0.11100789 |
| nonnegative / projection_fraction_support_out | 0.28904049 | 0.27451585 | 0.25770246 |
| correlations / overall | 0.17467053 | 0.19050557 | 0.23467534 |
| correlations / region_7 | 0.56449468 | 0.54334852 | 0.58803291 |
| valleys / Q3_h_one | 0.033713693 | 0.032417012 | 0 |
| REAL_pieces / full_window | 0.049703481 | 0.045740945 | 0.027929499 |
| REAL_pieces / coordinate_RMS | 0.06336255 | 0.055145364 | 0.030500599 |
| REAL_pieces / PSD | 1.6412252 | 1.1791189 | 0.31936736 |
| REAL_pieces / noise | 0.38914077 | 0.37897251 | 0.1861752 |
| REAL_pieces / temporal_joint | 0.13411868 | 0.12531904 | 0.055719562 |
| REAL_nonnegative / PSD_W1_overall | 0.017041883 | 0.01229699 | 0.0035616063 |
| REAL_nonnegative / baseline_RMS_W1_overall | 3.9386294e-05 | 3.722873e-05 | 2.1652462e-05 |
| REAL_nonnegative / baseline_ACF1_W1_overall | 0.062912957 | 0.062857549 | 0.026144808 |
| REAL_nonnegative / PSD_W1_region_-1 | 0.056016542 | 0.041903466 | 0.026606323 |
| REAL_nonnegative / baseline_ACF1_W1_region_-1 | 0.047836789 | 0.047611271 | 0.0099377854 |
| REAL_nonnegative / PSD_W1_region_0 | 0.02341071 | 0.016236053 | 0.0035112222 |
| REAL_nonnegative / baseline_RMS_W1_region_0 | 3.8459292e-05 | 3.5626613e-05 | 2.0857427e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_0 | 0.060386567 | 0.061945247 | 0.022471637 |
| REAL_nonnegative / PSD_W1_region_1 | 0.015460611 | 0.010715352 | 0.0029278391 |
| REAL_nonnegative / baseline_RMS_W1_region_1 | 3.9211961e-05 | 3.6272832e-05 | 1.9793303e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_1 | 0.060663476 | 0.061921537 | 0.021862044 |
| REAL_nonnegative / PSD_W1_region_2 | 0.017934795 | 0.013491978 | 0.0050393692 |
| REAL_nonnegative / baseline_RMS_W1_region_2 | 4.1991999e-05 | 4.004282e-05 | 3.0267353e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_2 | 0.066400315 | 0.067715949 | 0.03120848 |
| REAL_nonnegative / PSD_W1_region_3 | 0.017091592 | 0.011930006 | 0.0052595603 |
| REAL_nonnegative / baseline_RMS_W1_region_3 | 3.9095883e-05 | 3.6826545e-05 | 2.3414868e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_3 | 0.063782538 | 0.063258836 | 0.026987865 |
| REAL_nonnegative / PSD_W1_region_4 | 0.014074204 | 0.010029376 | 0.001565054 |
| REAL_nonnegative / baseline_RMS_W1_region_4 | 3.8672643e-05 | 3.6325388e-05 | 1.6837118e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_4 | 0.062101636 | 0.061886686 | 0.024438726 |
| REAL_nonnegative / PSD_W1_region_5 | 0.01215589 | 0.0090146944 | 0.00076361067 |
| REAL_nonnegative / baseline_RMS_W1_region_5 | 4.0351136e-05 | 3.8284931e-05 | 1.9144731e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_5 | 0.064852997 | 0.064221802 | 0.027957534 |
| REAL_nonnegative / PSD_W1_region_6 | 0.010350552 | 0.0075802482 | 0.00071353032 |
| REAL_nonnegative / baseline_RMS_W1_region_6 | 4.0182784e-05 | 3.8391586e-05 | 2.1221455e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_6 | 0.06370891 | 0.062918172 | 0.029203273 |
| REAL_nonnegative / PSD_W1_region_7 | 0.011183963 | 0.0077912839 | 0.0020457348 |
| REAL_nonnegative / baseline_RMS_W1_region_7 | 4.1446189e-05 | 3.9436492e-05 | 2.2132626e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_7 | 0.063965252 | 0.06205224 | 0.026856815 |
| REAL_nonnegative / PSD_W1_region_8 | 0.013318677 | 0.0096620741 | 0.0026189056 |
| REAL_nonnegative / baseline_RMS_W1_region_8 | 4.9177737e-05 | 4.833527e-05 | 2.4211476e-05 |
| REAL_nonnegative / baseline_ACF1_W1_region_8 | 0.070952135 | 0.070491817 | 0.035886264 |
| REAL_nonnegative / parentwise_normalized_identity_MSE | 0.021707427 | 0.015000782 | 0.0043053206 |

| 所有原区域 PSD W1 | 起点 | 全量 | 小批 |
| --- | ---: | ---: | ---: |
| PSD_W1_overall | 0.010475867 | 0.00706299 | 0.0065725843 |
| PSD_W1_region_0 | 0.021597135 | 0.0074902618 | 0.012637653 |
| PSD_W1_region_1 | 0.017436414 | 0.010559613 | 0.013168661 |
| PSD_W1_region_2 | 0.033796878 | 0.034604558 | 0.037312699 |
| PSD_W1_region_3 | 0.022182355 | 0.024327009 | 0.028309225 |
| PSD_W1_region_4 | 0.017537652 | 0.013536015 | 0.0079789894 |
| PSD_W1_region_5 | 0.014339318 | 0.011566165 | 0.0093877178 |
| PSD_W1_region_6 | 0.011501467 | 0.0059103065 | 0.0080235215 |
| PSD_W1_region_7 | 0.12956755 | 0.12901037 | 0.13623516 |
| PSD_W1_region_-1 | 0.10181512 | 0.1005194 | 0.10343208 |
| PSD_W1_region_8 | 0.058168105 | 0.059209403 | 0.066640068 |

本次原生运行 235.939 秒；前次启动失败消耗 201.553 秒；累计 437.493 秒，未超过原600秒总上限。worker 与 supervisor 已由原生回执和进程消失核验结束。GPU 计算进程列表为空：True。

R66 初次试验因单核亲和性启动问题无效，已单独封存。本次启动前强制验证实际 CPU0–11，未在运行中改资源、方法或终点。把预定全量步数改为20，是在没有任何训练终点结果时按剩余预算登记的新试验；原 R66 的30步终点从未改写。

这些数据只描述共同第500步起点、固定学习率和本次短程训练。八组是固定模型和 TRAIN 父人口上的噪声/方向重复，不是独立训练或独立采集重复。共同起点此前由 full-D 训练产生，也不能支持从原始初始化开始的优劣结论。原 A–F、峰位/峰宽、五视图判别、数值重放和任务效用没有在本次完成或重开。

冻结方案：tasks/R67_EQUAL_COST_REPAIR_001.md。原始结果：work/R67_EQUAL_COST_REPAIR_001/RESULT.json。全部退化门、数值核验、成本与原生回执：reviews/R67_EQUAL_COST_REPAIR_001/ROOT_RESULT_REVIEW.json 与 COLLECTION_RECEIPT.json。

后续仅整理此次已有证据并准备可检验的机制解释，不自动恢复 R65，不继续追加本次比较。总体模型目标仍未完成。
