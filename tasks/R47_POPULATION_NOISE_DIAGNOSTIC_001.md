# R47：电荷人口与初始噪声的固定诊断

本协议在reviews/R47_REGISTRATION_001/RECEIPT.json生成后生效。依据用户“构建计划推进”，由根登记、既有执行器连续实施，无新增代理或逐步骤审批。旧R46及其方法家族继续保持关闭。本阶段不是新响应候选，不改变A–F，也不占第3个方法名额。

## 问题与范围

只回答两件事：原SOURCE/REAL电荷人口本身在固定诊断下有多大差异；R46的初始投影Gaussian与已学习增量分别伴随何种基线失真。使用全部已有2000 SOURCE、2000 REAL DEV及2000 TRAIN_REFERENCE，固定短/长生成结果和其完整功能分量。所有父、顺序、权重、支持外、QC失败保留。

无新响应训练、优化器、ODE、密度EM、归一化常数、ADC参数或随机生成流拟合；不读取新采集、CAL波形、EVAL或G4。不能删除一项已保存增量后宣称实现了新的噪声替换模型，不能把功能分量命名成已识别的物理信号/电子噪声。

## 冻结输入

- R46 ACTUAL_GENERATION_INPUTS.npz、DEV_SOURCE_METADATA.npz、UNIT_CHARGE_CONSTANTS及四个SOURCE/REAL SHORT/LONG generated.npz；同一R46远端已封存副本可直接引用，不重新运行模型。
- REAL_DEV_AND_REFERENCE_PRIVATE.npz源于R43，沿用原ID/group/QC；旧R46远端有相同引用，路径见R46 CPU_PAYLOAD_PACKAGE_RECEIPT.json records。
- R26 results/C2ST_PRIVATE.npz原phase0/phase1三域选择；每域训练600、留出250，不重分区。原R7/R27 RAW REAL–REAL baseline预测及原R39 BASE_REAL结果/预测只复用，不重拟合。
- 原R3B DIAGNOSTIC_DEFINITION.json的8个Q区与CAL_REFERENCE_METADATA只复用保存的定义，不解码CAL波形。

准备时绑定实际路径、关键文件/原清单版本和数组键。科学输入检查只核对本阶段必要的行身份/分区/分量定义，不做全量来源审计或迁移内容一致性测试。已有文件路径变化可正常修正并记录，不重复提交科学作业。

## 1. 电荷人口描述与三个固定诊断分类器

对原SOURCE preADC、同一SOURCE一次ADC、LONG输出postADC、REAL TRAIN_REFERENCE、REAL DEV逐域报告Q分布、KS/W1、原8区及上下支持外数量。用原未ADC SOURCE的Q区固定源组成描述，同时另列各域自身观察Q区，禁止换一种分区掩盖迁移。沿用PSD=1-Qfast/Qtotal，不据DEV拟合电荷校准、权重或标签。

新HGB仅三个charge-only视图：SOURCE_INPUT对REAL、LONG_SOURCE对REAL、RAW TRAIN_REFERENCE对REAL。输入只有Qtotal_mVns一列，使用原C2ST训练集合拟合的缺失值/标准化及同一HGB参数：max_iter100、learning_rate0.06、max_leaf_nodes15、min_samples_leaf20、l2=1、early_stopping=false、random_state20261008。分组留出、bootstrap500次、seed2026100871均沿用R46；REAL/REAL的真实group统一R命名空间，SOURCE使用独立源命名空间。保留训练/留出点值、95%区间、实际组数、缺失/QC数量，禁止换种子重拟合。

若charge-only区间下界>0.60，称为当前固定诊断检出的强人口差异；若全区间位于[0.40,0.60]，仅称此一维诊断未检出超限差异；其他情形称不确定。它不是原A中Q–PSD二维分类器的代替或性能下界。新RAW charge-only结果不能补填历史缺失的RAW二维参照。

## 2. 初始噪声与学习增量分开描述

针对SOURCE和REAL控制，按实际保存的S原输入、G projected_initial_Gaussian_f64_V、F learned_flow_increment_f64_V、R final_cast_rounding_f64_V读取；原式为preADC_f32=S+G+F+R。保存数值分量和检查误差，检查是本诊断的科学定义验证，不是迁移验证。SHORT/LONG的S/G若应相同，只说明其实际定义及差异，不通过新推理重建。

并列RAW输入、S+G初态、SHORT末点、LONG末点，以及可用的原BASE/BASE_REAL保存输出。初态同时保留preADC结果和按原公式一次ADC后的结果；这只做确定性特征诊断，不称新响应生成。报告baseline100的RMS/ACF1分布、对REAL DEV的W1、原固定Q区的结果、零方差/非有限/ADC裁剪数量及配对变化。

在各域全部行的中心化baseline100上计算S/G/F/R各自方差，以及所有六个2Cov交叉项，检查这些项之和与实际preADC基线方差一致。报告均值及完整分布描述，保留负交叉项；不可直接按方差百分比声明某项独立物理噪声。预/后ADC差分开，信号窗和已有噪声的身份不由统计分解推断。

再拟合且仅拟合两个baseline_local诊断HGB：SOURCE初态postADC对REAL、REAL初态postADC对REAL，使用与上一节完全相同的固定选择/分类器/500次组bootstrap。baseline_local严格复用原diagnostic_core.matrix定义。REAL初态和RAW REAL–REAL的配对比较复用已保存预测与原真实group对应；若参照缺失或身份不匹配，该配对项记unknown，不新增第6个HGB。与旧BASE_REAL及R46长/短已保存baseline结果并列，但不把不同处理对象的差异冒称响应训练不确定性。

本阶段总计新HGB最多5，EM0、响应拟合0、ODE0；保存预测以便核查，但不重复拟合已有模型。初始化与末点的差别是路径位置描述，不是把某输入机制关掉重新执行的因果贡献估计。

## 图件与结论

至少4张清晰图：电荷CDF与原8区人口；三个charge-only BA/区间；SOURCE与REAL的RAW/initial/SHORT/LONG baseline RMS和ACF1；基线方差功能分解及全部交叉项。图使用同一坐标和定义，原失败/缺失保留。输出一个紧凑诊断结论：人口差异在此评价器下是否明确，初始化与学习增量分别在哪里发生噪声变化，以及这支持/不支持下一机制的哪一部分。没有模型候选通过/完成的判定。

## 预算与正常交付

科学900秒，含必要统计、5个HGB/组bootstrap、作图及所有失败尝试；执行器墙钟3600秒，根交付再留1800秒。该900秒计入原90000秒总预算和3600秒诊断额度；R44已用28.913469秒。服务器CPU12、RSS48GiB上限，无GPU科学作业，一次一个科学进程，启动前查询资源和本项目作业，保留8GiB空闲。不动共享环境/他人任务。

本机辅助CPU4/RSS4GiB，所有文件E/D。新增最终阶段文件≤64MiB，包含私有诊断预测、图、代码、回执；引用旧输入和数组，不复制或再打包整个R46。传输使用新的小包/定点文件，不保留重复大归档。本机D备份现有空间有限，此限制不授权删除历史数据或丢弃失败结果；若意外超限，应保留E上已有产物并报告。

既有执行器连续完成实现、必要修复、顺序计算、实际查看全部最终图、报告、资源/native/failure回执和最终清单后停止阶段写入。源代码、紧凑指标及图可发布私有Git，逐事件预测/身份留E/D。根检查关键证据并按明确路径推送，D采用依赖既有R44/R45/R46的正常增量复制与完整恢复说明，不做迁移内容测试。此阶段结束后由根依证据登记新方法；本协议不自动授权未定义的训练或任何A–F口径修改。
