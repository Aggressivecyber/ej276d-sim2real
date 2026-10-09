# R12：固定主要分布与随机输入，隔离剩余坐标协方差收缩

本任务属于用户已激活的整个项目持续目标。R11全维逆变换精度良好，但完整shape BA约0.946–0.948，基线RMS均值0.940/1.013mV而REAL约0.646mV，峰后波动的电荷依赖不足。下一步先隔离协方差估计对这些残差的贡献。R11两臂约0.0885的LedoitWolf收缩可能抬高低方差方向；这是假设，不能预写成已证明原因。这里tail指PCA剩余坐标，影响整段480点。

## 范围与预算

- 新阶段work/R12_PAIRED_RESIDUAL_COVARIANCE_001，新独立服务器目录及绝对UTC见reviews/R12_REGISTRATION_001/RECEIPT.json。阶段3600秒，执行器在整体截止前300秒封存；累计数值900秒，CPU4、RSS4GiB、输出512MiB、单作业。服务器沿用R11 Python；不安装环境。
- 只读复用R11参数、实际随机输入、输出及评价源，R10 inputs中的旧TRAIN/val/DEV。TRAIN-fit2592、内部val599、DEV2000及CAL边界角色不变。EVAL、新G4、CAL波形解码、PCA/head GMM/ridge重拟合均0。完整QC人口及unknown保留。迁移正常复制、正常加载，不做源/目的内容一致性测试或来源再审计。
- 同一执行器连续实现、运行、数学自检、绘图、短报告和一份最终清单，无中途许可。复用现有评价/绘图函数，保留6–8张关键图即可；时钟到线时直接交付已完成部分并记录缺项，不继续整理到超时。根代理负责后续Git和D盘复制。

## 唯一模型改变

保留R11 RAW K4和FACTOR K8各自完整表示、尺度、GMM、tail条件均值、固定数值对角底epsilon=1e-12 V²、ADC以及不再CFD裁窗的输出定义。新增RAW_EMPIRICAL_COV、FACTOR_EMPIRICAL_COV两臂；各只改变tail协方差。旧两臂输出直接复用，不重训练或重抽。

设R11保存的最终tail协方差C_old=(1-alpha)S+alpha*mu*I+epsilon*I，维数d=448或446，alpha取R11实际值。由保存协方差或其Cholesky得到C_old，C0=C_old-epsilon*I，mu=trace(C0)/d，S=(C0-alpha*mu*I)/(1-alpha)，C_new=S+epsilon*I。这是原TRAIN经验协方差的解析恢复，不需新的统计拟合。数值上仅允许对称化(S+S.T)/2。报告alpha、mu、特征值范围、条件数、恢复后收缩回C0的误差；默认相对Frobenius误差阈值1e-10、绝对max误差阈值1e-12 V²。C_new不正定或绑定公式不符即保留该臂失败，不扫描新epsilon或偷偷改模型。只有真实保存格式无法恢复S时，允许从原2592 TRAIN-fit和固定R11变换计算同一中心化S=E.T@E/n，并明确为何需要；仍无参数搜索、DEV估计或head重拟合。

从R11实际component/head_white/tail_white重建相同head及标准Gaussian创新，用新Cholesky映射相同tail_white。每新臂2000条，共4000；保存原draw位置及pre/post ADC。明确科学配对：head、条件均值、latent输入一致，协方差变化是唯一受控项；不能将这些配对检查称迁移测试。不拒绝重抽，不替换无效事件，不从REAL波形或残差模板重采样。

全部599内部val重新计算旧/新完整波形连续NLL及其head、tail、Jacobian分量；不按DEV换K、rank或正则系数。两种固定协方差都报告，本轮不靠单一NLL或单个BA选择最终项目模型。原R11数学自检直接引用，只对新增协方差恢复和密度需要的运算做必要检查。

## 检验改变解释了多少差距

1. 计算固定2000个head条件均值下、pre-ADC基线样本方差的期望分解。对基线选择矩阵和中心化H=I-11.T/100，期望为[mu_b.T@H@mu_b+trace(H@C_b)]/99，C_b为U_tail C_tail U_tail.T的前100点子块。分别报告固定均值项、原经验协方差项、收缩引入项和数值底；旧/新差异与实际pre-ADC基线均方统计并列，不增加Monte Carlo输出。可同法报告既有prompt/postpeak/tail区域二阶差分均方，使用原曲率定义对应的线性算子和正确分母，不把窗口曲率当纯上升沿物理。
2. 四臂完整QC统计、电荷–PSD总体及原4电荷分区、基线RMS/ACF/谱、前沿和峰后曲率/波形继续使用R11定义。共坐标全范围图保留离群；主体放大图统一PSD[0,.6]并注明图外数量。
3. 四臂加REAL–REAL共5组、各6视图，最多30个固定HGB。使用原R7训练600/留出250位置池，取四臂共同有效draw位置，再与REAL和TRAIN-reference确定相同N；不得桶外补样。每域训练<150或留出<75时unsupported。全部分类器用同一冻结配置/种子，新对照需要的共同支持评价属于本轮比较，不重做R11其他实验。
4. 保存留出概率，给每种表示新减旧BA/AUC的配对500次源draw/REAL人工组bootstrap；同一个draw和REAL组在两臂共同重抽。区间条件于已拟合分类器，不能说包含训练/独立采集不确定性。不要把不同支持数量的旧R11 BA直接相减当配对效果。
5. 新两臂8个固定窄电荷区间共16描述槽；实际EM次数另记。旧REAL、R11及R3B表直接复用，只有共同合格区间比较峰位/原宽/去趋势宽。保留所有不支持原因，不能让边际PSD改善代替两峰证据。最近邻新两臂各全2000query对2592 TRAIN，两种视图；旧R11及REAL结果复用，固定例图可沿用draw位置。无精确重复不证明无记忆。

## 复现与方向判断

保存两新协方差/Cholesky及实际随机输入后，用新独立进程重放全部4000条，不拟合、不重新生成随机数或重做全评价。pre-ADC沿用R11明确容差，post-ADC要求array_equal，任何差异原样保留。旧模型/输出不覆盖。全部计入900秒数值预算。

交付config/PROTOCOL、原生主运行/重放/资源/数学检查回执、聚合METRICS_COMPACT、短REPORT和REPRODUCTION、关键图及实际图检、一份最终版本清单。逐事件选择/预测/邻居、模型和噪声留E/D；执行器不操作Git。首个科学PID/UTC、关键异常和封存结果回传root即可，不建立逐文件台账。

若仅噪声幅度修复而整体/条件峰宽仍偏，保留该部分成功与残差，下一步再考虑条件尺度或更灵活的联合密度。若效果小或验证NLL恶化，也保留，不追加同轮正则扫参。项目目标继续，由root据这轮证据登记下一轮；R12不是最终G4映射，也不是项目结束。

公式依据为当前服务器版本的[LedoitWolf官方定义](https://scikit-learn.org/1.8/modules/generated/sklearn.covariance.LedoitWolf.html)及[经验协方差定义](https://scikit-learn.org/1.8/modules/generated/sklearn.covariance.EmpiricalCovariance.html)。本轮检验该已知估计规则在此数据中的影响，不声称提出新协方差方法。
