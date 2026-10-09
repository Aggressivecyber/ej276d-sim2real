# R25：损失诊断支持检验条件批次，未定位唯一失败原因

固定REAL TRAIN投影与三个保存模型的诊断已完成，没有新响应拟合或参数更新。原加权分布组合未显示明确的轻度收缩偏好，实际加权梯度也没有全面抵消至近零。下一轮检验训练批次的电荷条件结构，保留现有网络、损失及权重。

在预定c=[.5,.75,1,1.25]中，MIXED批次的physical/pooled-shape两项在c=.75时较小，但三组fine项抵消这一优势；组合相对c=1的差为-0.000005812，固定TRAIN经验池的MC 95%区间[-0.0059466,0.0059350]。HOMOGENEOUS组合的相同差为+0.0290695，[0.0217459,0.0363932]。这支持检查混合训练与条件验证之间的差别，尚不证明新批次训练会改善响应。

五组energy U参照在两种抽样中均惩罚所试收缩与扩张。它排除三个对角线并保留负样本估计；MIXED c=1组合均值-0.00259178。energy U与分位数W1的目标和样本基准不同，不能比较绝对值大小来选系数。投影空间收缩不是有效波形生成器，区间不覆盖新采集、训练或多重比较。

INITIAL/D1末点/D5末点的总G梯度范数中位为19.8979/30.8595/101.5954，梯度和范数与分项范数和之比中位为.6009/.5268/.7353。各32批都会触发原clip5，假定缩放中位.2513/.1618/.0492；本轮实际没有裁剪或optimizer步骤。NCE-domain平均余弦为-.0744/.0471/.1777，不能称两末点稳定冲突。NCE的encoder范数份额60.49%/53.25%/68.59%，domain_SW的electronic份额79.65%/71.16%/44.52%；这些是梯度范数比，不是Adam更新或因果归属。

PSD-rank仅7/7/10批有非零梯度，其中4/4/7批与总梯度负点积，分母不是32。positive-charge梯度全部为0，余弦/负对齐率保留undefined。三次直接总梯度与分项加和核对通过，relative L2误差1.28e-6/7.12e-7/3.81e-7，模型参数前后不变。起点D没有学习，两个末点的G和D都已变化；局部原始梯度不能直接解释有限步Adam行为。

实际完成512对批样本和96批梯度；source、identity、NCE源编码、NCE输出编码、critic、reader各3072行。全部输入来自既有TRAIN与模型，没有读取新DEV/EVAL/CAL波形，也没有主响应生成、HGB/EM、新G4、SSH或安装。一次重复保存键工程失败发生在模型前向之前，原件、native1及6.852151秒保留；修复只改输入保存字段名。

包含失败、汇总绘图和两处显示语义修复，科学累计74.373780秒；峰真实RSS1,718,902,784字节，GPU allocated792,177,664字节，所有实际worker导入前均绑定RSS守护，查询失败0。八张最终图已实际查看，零值显示下限与n/a明确标注，旧图另名保留。最终89文件/8,320,993字节，12:39:13.631542 UTC停止阶段写入。

R26已登记：将R23原94400行及逐行随机变量尽量重组为同Q区批次，完整保留余数；原D1训练配方与J20选择不变。只训练一个新臂，预指定新selected/final及旧D1 final的完整响应评价。批次变化影响多项损失和训练顺序，不能称只隔离fine条件化；末点DEV结果不会事后改选模型。

[完整报告](work/R25_LOSS_GEOMETRY_DIAGNOSTIC_001/REPORT.md) · [收缩配对结果](work/R25_LOSS_GEOMETRY_DIAGNOSTIC_001/figures/contraction_paired_W1.png) · [梯度相互作用](work/R25_LOSS_GEOMETRY_DIAGNOSTIC_001/figures/G_gradient_cosines.png) · [关键复核](reviews/ROOT_R25_KEY_RESULT_REVIEW_001.json) · [复现入口](work/R25_LOSS_GEOMETRY_DIAGNOSTIC_001/REPRODUCTION.md) · [R26任务](tasks/R26_CONDITIONAL_BATCH_TRANSLATION_001.md)。

项目保持active，完整响应验收仍未通过。私有Git与D盘普通复制状态读取R25_PUBLICATION_RECEIPT.json、R25_BACKUP_RECEIPT.json。
