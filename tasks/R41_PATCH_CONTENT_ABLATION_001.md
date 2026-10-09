# R41：原完整源生成器的局部内容约束消融

本协议在R40实际封存、根完成关键判断并写入`reviews/R41_REGISTRATION_001/RECEIPT.json`后生效。登记前只是草案，不能启动科学计算。根负责方向、登记、关键判断和私有Git/D；既有执行器连续实现、运行、自检、看图及交付，无新代理或子步骤审批。

## 一个问题与已有证据

R40完整shape493/纯u480 BA为.984/.982，对BASE的.978/.978没有改善；总体PSD W1增加20.37%、baseline RMS W1增加52.69%，三个二阶边界及同处理REAL的baseline点差也失败。其新增时间层确实更新和影响输出，源组/秩代理通过。关闭这一固定time-MLP配方，不能用改层数、宽度、位置、损失或训练终点救该扩展。

下一问题改为：原生成器中的局部PatchNCE内容约束，在保留电荷/PSD秩、identity、随机风格和分布约束时，是否妨碍联合响应匹配？R25在保存起点与两个末点的梯度诊断中，NCE的encoder范数份额约60.49%/53.25%/68.59%，但没有证明它是唯一原因或与分布项稳定冲突。R40的负结果也不能证明架构普遍无效。因此本轮用一个直接消融检验该假设，不再重复局部梯度归因或扫描NCE权重。

局部输入/输出内容对应的依据是Park等[CUT原论文，ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/792_ECCV_2020_paper.php)及[作者实现](https://github.com/taesungp/contrastive-unpaired-translation)。本项目的一维三层12位置NCE是既有适配，不是完整CUT复现。此次只检验删除已有约束的效果，不提出新算法，也不预先声称内容约束有害。

## 唯一改动：移除训练PatchNCE项

恢复R32 `SERVER_FRESH_CONTROL` 的原完整卷积G、StyleReader和STANDARDIZED D；不保留R40时间MLP，不使用R39 MMD、不使用额外均值项、不加后置校准。原`model.py`/`legacy_model.py`保持R32结构和推理公式。G/reader从原R21 step2000加载，D从原R18初始加载，全部原state键逐名一致，没有新增参数。

唯一训练改动是`content_NCE`项不进入G目标，执行系数明确为0，并且不执行NCE专用的source/output编码、投影或交叉熵。其余adversarial、identity、Q_relative、Q_rank、PSD_rank、domain_SW、positive_charge及fresh-fine、style reconstruction的公式、系数和运算顺序不变。日志不能把“没有计算NCE”写成“NCE实测损失为0”；保存`NCE_evaluated=false`、`NCE_forward_calls=0`和本轮实际目标的分项定义。

原三组`project`投影头仍保留在state中以保持原架构/检查点兼容，但不被任何实际损失使用；其梯度为None、初末权重逐值不变须明确核对。其余G参数及reader/D按原方式共同更新。不得增加替代内容损失、冻结encoder、改随机风格维度/恢复权重、减少identity、重标定其他项或做梯度平衡。

保留原NCE位置绑定作未使用的审计流：每步按原三层480/240/120长度、原80:420相对范围及原12位置，消费/记录原已保存位置列表，零新增随机抽样。明确这些位置本轮没有进入损失，不能声称仍执行了NCE。同一位置hash只是未扰动原流的证据。

## 同起点、同行流和固定末点

以完整R32配置为来源；seed20261009230、Adam lr.0002/betas(.5,.999)、clip5、G:D:reader=1:1:1，均为新Adam。2950×32=94400实际更新行，原R26重组2946同Q批和4余数混合批，source/target/bin/original ordinal/z-white/style及上述保留NCE位置流逐序一致；实际唯一source/REAL为3752/18739，6593/21047仅是载入池。

原fresh-fine三组`u480`、`diff479`、`baseline100_V`及TRAIN常数、每组128方向、独立PCG64(20261009281)和三组顺序全部保留。2950调用/8850组实际float32方向与保存R32 CONTROL匹配，验证不得消费该方向流。

原512/599内部J20、固定验证投影、step0/每250步/末步验证保持不变。第一更新前对实际加载G/reader/D逐名核对，并立即检验初始J20与同服务器保存BASE初始值之差<=1e-6；失败先定位实现，不放宽容差。无新模块、无需额外旧新模型256行等效前向。保存内部所选同步权重仅作记录，唯一DEV候选是固定第2950步`NO_PATCH_NCE_FINAL`；预算不足不改选前缀。

旧R32 CONTROL_FINAL为唯一主要训练对照，原数据、初始化、更新数、随机流与其匹配，零重训；R39/R40只作已失败方法背景。删除NCE减少专用前向和计算量，也改变全局clip及Adam梯度序列；这些是该消融的实际影响。报告耗时、有效参与参数与前向差异，不声称等FLOP或单独隔离所有优化机制。只试系数0这一个预定消融，不试中间权重或新的种子/学习率/步数。

## 完整生成、参照与回放

原2000源父全部经过新完整G，包括417个TRAIN支持外源；无BASE回退、不删除、不按组成改权重。相同ID、组成、顺序、权重及主z/white/style，单次14bit ADC。保存完整pre/postADC、signal/electronic和sum-rounding，分量只作网络功能描述。

原重复随机性128×8=1024条和832源探针照原口径生成。沿用前128个原支持内SOURCE的150:400段均值0、RMS1mV固定正弦扰动，同原随机量；它作用于完整源输入，报告float32 Qfast/Qtotal/PSD实际差、全480 preADC输出RMS和非相同比例。原SOURCE_CLEAN postADC与actual source preADC分区仍分开，purehad主分区N213/181不换用历史N207/182。

新G对原2000 TRAIN_REFERENCE加相同主z/white/style生成`NO_PATCH_NCE_REAL_CONTROL`，不赋予SOURCE组成或推定PDG。旧BASE_REAL_CONTROL直接复用封存R39的相同波形、随机绑定、10评价器/预测、密度及6000回放中的既有2000条证据；零重新生成/拟合。RAW REAL–REAL九视图继续复用，二维配对预测unknown不补拟合。这些控制仍是同一历史采集、部分训练暴露样本，不是独立新采集。

仅主源和新REAL两个新端点，各10视图，共20 HGB；16个密度描述槽，最多16 EM。固定每域600训练/250留出、行组/分类器和500次区组bootstrap；原端点不重拟合。完整/局部波形、电压、Q–PSD、峰位/raw及detrended宽度、baseline RMS/ACF、曲率、投影、均值/SD、源组及9边界全部保留。不得只挑通过的区或视图。

独立本机CPU进程从完整新G、精确源/REAL输入和z-white-style重放主2000+新REAL2000，共4000，原chunk128和运算次序；全部身份/随机输入/功能分量及pre/postADC逐值相同。推理不读取已保存输出作为输入，不读取TRAIN/REAL事件库、reader或评价拟合。本轮没有模块关闭第二端点，也没有额外候选。

## 六项共同验收与退出

1. 主源shape493/u480减BASE的配对BA差95%上界均<0，且点值分别<=.934/.928。阈值延续R39/R40，不能因消融失败更改。
2. 对BASE总体/共同观察Q1–7 PSD W1、低高raw-width MAE、低高峰位MAE、baseline RMS/ACF1 W1均不恶化>10%；BASE有效密度bins2–7继续有效，不缩小交集掩盖失败。
3. 2000源父的身份/组成/顺序/权重保留，无新增无效/非有限/ADC裁剪；source-Q/PSD全人口Spearman下降各<=.05；actual-source purehad bins3/4谷比例增加各<=.02；417支持外单列、原源探针确有作用。删除NCE不意味着放弃源保真门槛。
4. 9个边界W1各<=max(BASE,原INDEPENDENT_CFM对应值)+1/16383V；128完整SOURCE扰动非相同比例>=.95且全480 preADC RMS中位数>1/16383V。无分段保护、平滑或后处理修复。
5. 新REAL减封存BASE_REAL在shape493/u480/baseline三个视图的配对ΔBA95%上界各<=.10、点差各<=.01。RAW差距全部报告，不宣称REAL恒等、物理不变或独立采集等效。
6. 完成2950步与初始J20核对，原实际行/保留位置/随机流及8850 fresh方向匹配；NCE专用前向和实际损失参与为0，三投影头初末逐值不变，实际参与参数有限；4000独立回放通过，图/资源/失败记录完整。undefined不算通过。

六门全过才保留为下一开发候选，仍需随后冻结方法和任务效用/适用范围验证才能结束项目。任一失败就关闭这个固定“无PatchNCE”配方，不扫描非零系数、改其他损失或回到R40层救援。不把一次消融的成败外推为所有内容约束或无配对转换的结论。区间条件于当前固定分类器和同采集人工组，不包括响应重训或独立采集不确定性。

## 预算与交付

阶段5400秒、执行器4800秒封存、根600秒Git/D；科学累计1800秒、唯一训练含全部J20/初始化检查<=900秒。一次一个科学作业，CPU4、真实worker RSS4GiB、GPU allocated4GiB、输出512MiB。先查现有服务器资源和本项目作业，在新独立目录运行；未知状态不重提交、不改共享环境/他人作业。

本机生成/评价/重放CPU4，全部数据、源码、cache/temp仅E/D。无新REAL、CAL/EVAL波形、G4、安装或迁移内容一致性测试。交付权重、完整推理、原更新/随机绑定及未用NCE位置、精确输出、评价器和聚合指标、直观波形/PSD/宽度/噪声/REAL控制图、训练损失参与说明、命令/原生退出/资源/失败、重放、报告和复现入口；每张最终图实际打开。根关键审核后以明确路径提交私有Git、正常复制D。
