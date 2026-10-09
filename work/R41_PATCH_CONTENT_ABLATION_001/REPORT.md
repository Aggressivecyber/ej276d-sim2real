# R41：移除 PatchNCE 的固定消融

六项共同门槛为 **失败／失败／通过／失败／通过／通过**，结论为 `close_this_fixed_no_PatchNCE_recipe`。shape493 BA=0.986，减BASE配对差+0.008，95%CI[-0.00594,+0.02264]；纯u480 BA=0.982，减BASE配对差+0.004，95%CI[-0.01040,+0.02004]。完整响应点值门槛仍为shape≤.934/u≤.928且两项配对95%上界<0；数值可复现不能替代科学通过。本结论只关闭或保留这一预定系数0配方，不外推所有内容约束或无配对转换的可行性。

完整480点G4派生源波形直接进入原R32 CNN，保持源父和沉积组成。唯一训练改动是删除既有PatchNCE专用source/output编码、投影和交叉熵，不进入G目标；它没有被计算，`NCE_evaluated=false`、损失值null、目标系数0、专用前向0。原三层位置列表8850份仍消费保存为未使用的审计流，不能称作执行了NCE。没有R40时间层、新参数、MMD、额外均值项、后置校准或替代内容约束。

原网络没有BN/dropout。投影头3680参数留在state及optimizer中，2950次反传全部grad=None，六个weight/bias张量初末逐值相同。全部G参数原requires_grad保持；总G参数221874，实际参与218194，末次其他G参数均有非None梯度且所有G/reader值有限，未冻结encoder。省去5900次专用encode调用；运算量、全局clip与Adam梯度路径随消融改变，FLOP未实测，不声称等算力或唯一优化机制。

第一更新前严格核对原G/reader/D的全部state键和值，并实际validate(0)立即断言J20差≤1e-6：0.6166301966，差0，D/G/reader更新计数均0。原94400行、七项实际源/REAL/bin/ordinal/NCE审计/随机流和8850 fresh方向全部匹配冻结对照。载入6593/21047，实际唯一参与3752源/18739实测；这些数量不等于独立采集。新Adam原lr/.5,.999/clip5及D1不变。G/D/reader均2950。

内部所选step1500、J20=0.532822430；实际2950末点J20=0.576148748。仅固定末点用于DEV，不按DEV改选内部checkpoint。记录训练85.227584s，原同服务器BASE单臂记录114.033771s；删除NCE减少专用前向，但耗时不是硬件独立的因果量。所有训练/验证轨迹及实际目标分项保留。

| 指标 | 冻结BASE | 固定2950 | 相对变化 |
| --- | ---: | ---: | ---: |
| 总体PSD W1 | 0.0230654614 | 0.023581522 | +2.24% |
| 共同Q区PSD W1 | 0.0347657952 | 0.0376238657 | +8.22% |
| 低侧峰位MAE | 0.00585717151 | 0.0153508469 | +162.09% |
| 高侧峰位MAE | 0.00265866336 | 0.0059826563 | +125.02% |
| 低侧raw-width MAE | 0.0103927218 | 0.0149300316 | +43.66% |
| 高侧raw-width MAE | 0.0130511744 | 0.0133019699 | +1.92% |
| 基线RMS W1（V） | 1.93583857e-05 | 2.21838895e-05 | +14.60% |
| 基线ACF1 W1 | 0.0642349714 | 0.032329459 | -49.67% |

峰位/raw-width只用共同合格区并同时保留全部支持缺口；BASE所需bins2–7新增失败为[]。两高斯成分仅为PSD密度描述，不能称独立粒子真值。未通过区不删掉，最高区缺口见完整门槛文件。完整条件均值/中位、PSD/Q、噪声、曲率与DEV相关矩阵均报告，不能用某个均值曲线或单项局部收益作整体通过。

2000原源父身份、组成、顺序和权重保留；417个旧TRAIN支持外也用新G，无BASE回退，无新增无效或ADC裁剪。actual-source bin3 N213：5.6338%→2.3474%（-3.286 pp）；actual-source bin4 N181：0.5525%→0.5525%（+0.000 pp）。这些actual未量化NN输入分区N213/181与历史postADC参考N207/182单列，不能混称。source_Q_to_output_Q 0.9997263→0.9994488；source_PSD_to_output_PSD 0.4189917→0.4431621。源相关与跨域谷区是代理，不证明逐事件物理类别对应。纯EE/had/mixed及入射PDG证据边界沿用原来源，不随机重标源类。

9个固定边界失败位置为['second_center_150']。前128支持内完整SOURCE输入150:400零积分/RMS1mV扰动使用原随机量，非相同比例1.000；preADC全480输出RMS中位1.369790mV。源敏感性不是正确物理响应；prompt100:150包括峰与早期下降，不能把整段曲率称纯上升沿。

旧同处理BASE_REAL完全复用封存R39的2000波形、10评价器/密度和已执行独立重放，零重生成/拟合。新G处理相同TRAIN_REFERENCE与z/white/style；不赋予源组成或粒子标签。三门如下，V480等其它视图仅描述，对RAW绝对变化完整报告，不预设随机G(REAL)恒等。

| 同处理REAL视图 | BASE BA | 新BA | 配对差与95%CI | 门槛 |
| --- | ---: | ---: | --- | --- |
| shape493 | 0.970 | 0.962 | -0.008 [-0.02691,+0.01312] | 通过 |
| 纯u480 | 0.964 | 0.950 | -0.014 [-0.03862,+0.01271] | 通过 |
| baseline | 0.658 | 0.578 | -0.080 [-0.12391,-0.03281] | 通过 |

主SOURCE2000+新REAL2000、joint1024、原probe832和SOURCE扰动128全部精确保存。独立本机CPU完整链4000逐值通过，原chunk128和运算次序，输入为原源/REAL与随机量；保存输出仅用于最终比较。signal/electronic共享输出归一化和交叉项，功能分量不是独立物理噪声。没有模块置零第二候选。

服务器实际worker83027、监督83026，native0；原生训练87.942935s，RSS峰2649534464B，GPU allocated峰245770752B。查询gap1次，具体时间保留且原因unknown。其余5本机科学或纯绘图作业native0；累计178.259609/1800s，20新HGB、16密度描述槽/15实际EM，参考新增拟合0。12张最终图已实际打开；source rank旧TIME标签与训练图注重叠已仅按保存结果修复，原图与执行脚本保留，没有重复拟合或生成。实际资源和原生链见对应JSON。

本轮为历史暴露同采集TRAIN/DEV开发比较，人工50记录组不是独立批次；500区组bootstrap条件于固定分类器，不包括响应重训或独立采集不确定性。无新REAL/EVAL/CAL/G4/安装/代理/Git操作或迁移内容测试。若门槛失败，不用新的NCE系数或其它损失救本轮；下一问题须由根依据完整证据登记。

[METRICS_COMPACT.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R41_PATCH_CONTENT_ABLATION_001/METRICS_COMPACT.json)、[GATE_METRICS.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R41_PATCH_CONTENT_ABLATION_001/GATE_METRICS.json)、[TRAINING_COMPACT.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R41_PATCH_CONTENT_ABLATION_001/TRAINING_COMPACT.json)、[REPLAY_RECEIPT.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R41_PATCH_CONTENT_ABLATION_001/REPLAY_RECEIPT.json)、[FIGURE_QA.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R41_PATCH_CONTENT_ABLATION_001/FIGURE_QA.json)、[RESOURCE_RECEIPT.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R41_PATCH_CONTENT_ABLATION_001/RESOURCE_RECEIPT.json)。完整逐事件载荷、模型/随机流/预测留E/D，聚合发布文件不含身份和预测。
