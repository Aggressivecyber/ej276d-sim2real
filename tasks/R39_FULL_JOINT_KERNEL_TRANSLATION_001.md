# R39：完整源波形神经转换的联合核分布训练

本协议在R38实际封存、根完成关键图判断并写入`reviews/R39_REGISTRATION_001/RECEIPT.json`后生效。根负责问题、登记、关键结果与Git/D；既有执行器连续实现、运行、自检、看图、封存，不新增代理或逐步骤批准。

## 唯一问题与关闭边界

R38完整shape493/纯u480 BA均.926，总体PSD W1下降44.25%；但baseline RMS W1增加583.06%，高侧峰位MAE增加51.28%，原有效密度bin2失败，actual-source pure-had bins3/4谷比例增加10.33/17.13个百分点，9个边界中5个失败。门槛为通过/失败/失败/失败/通过，固定切片配方关闭。R33–R35冻结G加MLP-CFM、R34分段保护、R36扩散、R37协方差及R38切片的已关闭变体全部保持关闭。

本轮回到完整源驱动的端到端神经G训练，只检验一个新的分布目标：用完整时间特征的联合Gaussian-kernel MMD，替换R28/R32 CONTROL的三组逐投影分位数训练损失。保持原卷积G、STANDARDIZED critic、reader、源内容/NCE、电荷与PSD秩、identity、低维domain_SW和其权重。没有冻结G后置校准层、不复用R33–38变换、不做保护/平滑或目标事件检索。固定同一初始化和2950更新末点；R32同服务器CONTROL_FINAL作为相同数据流/更新预算的保存对照，不重训旧臂。

方法参考[Gretton等，A Kernel Two-Sample Test，2012](https://jmlr.org/papers/v13/gretton12a.html)及[Li等，Generative Moment Matching Networks，2015](https://proceedings.mlr.press/v37/li15.html)。本项目将核分布损失用于已有完整源神经转换，不采用论文的目标随机自生成结构，不称算法创新或完整复现。总体特征分布的理论可辨识性不能保证有限批次优化、物理对应或独立采集泛化；本轮高斯核/带宽组/初始梯度尺度配方失败即关闭，不扫描。

## 固定数据、初始化与神经训练

复用R32 CONTROL的R21所选step2000 G与reader、R18初始D、新Adam、seed20261009230、lr.0002/betas(.5,.999)、clip5、G:D:reader更新1:1:1。原R26重组的94400行（2950×32，2946同Q区批及4混合余数批）、source/target/bin/original_row_ordinal/z_white/style/NCE位置全部逐序一致；实际唯一源/REAL行3752/18739，载入池6593/21047不得冒充实际暴露。源与REAL无逐事件配对，不重采样或按PSD排序。

使用原`FINE_CONSTANTS.npz`的REAL TRAIN-fit均值/尺度和原critic/physical常数，不重估，不读DEV/CAL/EVAL拟合。原source_val/REAL_val及J20每250步和0/末步检查维持原公式、原固定投影、权重与随机输入；旧J20可能偏好旧目标，所选step仅训练记录/权重保存，不作为新DEV端点。唯一主端点为第2950步`JOINT_KERNEL_FINAL`；预算不足则关闭该未完成配方，不能另挑已完成前缀。

保留全部原基础损失与style重建，mean额外系数为0；仅把fine训练项替换为下述`c * MMD_U`。不改变NCE、输入/输出归一化、噪声或源波形结构，不额外训练新reader/分类器。初始、最终G/reader/D及优化器、实际流绑定、完整trace保存；部署只需要G与原source/z/white/style，不读REAL库或reader。

## 唯一联合核与TRAIN初始尺度

对电压波形y，先按原代码求`s=y-mean(y[0:100])`、`u=s/max(max(s),.0025)`。三个原fine组为`u480`、`diff(u)479`、`s[0:100]V`；各组仍用原对应TRAIN mean/scale标准化。先按原float32特征运算，再将标准化特征升float64计算核与标量损失，梯度回传原G。

两条标准化特征a,b的距离固定为`D(a,b)=(mean((a_shape-b_shape)^2)+mean((a_diff-b_diff)^2)+mean((a_baseline-b_baseline)^2))/3`。它等于三个组分别除sqrt(3*d)后拼接向量的平方欧氏距离。核固定为`k(a,b)=mean_{sigma in [.25,.5,1,2,4]} exp(-D(a,b)/(2*sigma^2))`；五个带宽共同构成一个核，不比较/选择单独带宽，无随机投影、带宽搜索或可学习critic特征进入该核。采用显式逐对差平方计算距离，不以裁掉负MMD修正数值。

每批n=m=32，`MMD_U = sum_{i!=j} k(G_i,G_j)/(n*(n-1)) + sum_{i!=j} k(R_i,R_j)/(m*(m-1)) - 2*sum_{i,j} k(G_i,R_j)/(n*m)`。只去掉两域各自同样本对角；交叉项保留全部n*m，包括相同行号，行号不赋予事件配对。两个生成参数位置均保留梯度；有限负MMD原样进入损失，不取sqrt、不截零。当前固定重用行流及4个异质余数批不称理想iid无偏抽样。

为避免仅由新旧单位相差导致弱监督，在任何参数更新前，固定使用原更新流的前64批、同初始G和原z/white/style，做一次TRAIN尺度确定。G在no_grad下前向后将y作为独立requires_grad张量；不更新任何网络/optimizer。旧fine参考用独立复制的原PCG64(20261009281)方向流第1–64次调用，三组128方向及原分位数L1完全相同；此复制不推进训练噪声/style/NCE或验证流。分别求旧fine和新联合MMD对物理y全部32×480元素的梯度，设`c=sqrt(sum_64 ||grad_y old_fine||_F^2 / sum_64 ||grad_y MMD_U||_F^2)`。分子/分母float64累积；任何非有限、任一和<=1e-24或c<=0则保留失败并停止，不更换批、加epsilon、截断c或试第二系数。

只固定这一个全局正系数c，后续2950步不更新。记录64个原批及各Q区覆盖、两损失、梯度范数、192个参考方向hash及c；这仅匹配初始TRAIN输出梯度的合计尺度，不宣称参数梯度、后续路径或算力完全相同。64×32=2048次额外源前向/REAL重复使用与梯度计算必须计成本；原94400更新流不变，不称总前向次数相同。独立toy float64直接循环与向量式值/梯度、交换两域/打乱行的对称不变性、对角计数及允许负估计检查一次；不训练玩具响应模型。

实际训练记录每步三个Gram摘要、raw/scaled MMD、有限负批数、基础损失和G/D裁剪比例；每250步汇总。原fine投影只出现在上述64批尺度确定及原J20验证，MMD更新本身不再产生fresh方向。报告实际2950步、8850域内/交叉Gram组与五核项计数，不能沿用旧fresh8850投影计数。

## 生成、实测控制与独立重放

候选固定末点生成原主2000源父、128×8=1024原重复噪声、832原源探针。全部父ID、组成、顺序、权重和原随机输入保留；G原样作用于全部源，包括原417个TRAIN支持外父，不改为后置映射的BASE回退，支持内外单列。主源actual preADC与历史SOURCE_CLEAN ADC口径不混用，纯had主组N213/181。保存精确signal/electronic/sum-rounding、preADC与单次14bit ADC，不额外裁波、投影或平滑。

增加一个完整输入作用探针：前128个原支持内源，原source/z/white/style，在源波形150:400加该段零均值RMS1mV固定正弦；原source bin标签固定作描述，记录实际float32 Qfast/Qtotal/PSD差。候选G直接读扰动后完整源；比较原主父preADC，非相同率>=.95且全480点RMS中位数>1/16383V才通过。它是完整source扰动，不得称R36–38的post-G guide扰动；源敏感性不证明唯一物理映射。

新增两个同处理REAL控制端点：既有2000条`TRAIN_REFERENCE`按原行序分别通过冻结R32 CONTROL_FINAL G与新G，使用原主2000同序z/white/style，单次ADC。不能给REAL清零随机量而给source加噪，不能删支持外REAL。保存BASE_REAL_CONTROL与JOINT_KERNEL_REAL_CONTROL精确输入/随机绑定/分量/输出。它们是已有TRAIN参照上的处理控制，部分记录曾参与训练，非新实测采集或盲泛化。原RAW REAL–REAL九视图参照原样复用，Q/PSD二维配对预测仍unknown，不补旧HGB。

独立本机CPU进程只凭权重与输入从头重放候选主2000及上述两个REAL控制各2000，合计6000。原chunk128与推理运算顺序，所有父/行ID、实际随机流、signal/electronic/sum-rounding和pre/postADC逐值一致；推理不读原保存输出作计算输入，不读TRAIN/REAL库拟合、reader或评价器。分支是功能分解，非已识别的物理signal/noise真值。

## 固定评价和门槛

只新增三个端点各10视图HGB，共30次（每域600训练/250留出，原组和500次bootstrap），24个密度描述槽/最多24次EM；固定全量REAL DEV与旧BASE/R33/R35–38/RAW参照不重拟合。候选源与BASE进行全部同口径比较，两个REAL控制相对RAW及彼此比较。新控制只用于判断不回流训练；禁止据DEV选择c、核、权重、步数或端点。

主源完整/局部波形、纯u480/电压、Q–PSD全人口/共同观察Q区、峰位/raw与detrended宽度、baseline RMS/ACF/曲率、原/开发投影、条件均值/SD、source关联/源组谷、9边界、重复随机性和完整源扰动全部报告。相同REAL控制报告全部主要视图和Q/PSD/噪声描述；未定义项保留缺口。以下共同通过才保留为下一个开发对象，仍不代表项目完整科学验收：

1. 候选源full493/纯u480对BASE配对BA差95%上界均<0，且点值分别<=R37 .924/.918+.01。旧R37/R38仅比较参照，不复用其转换。
2. 对BASE总体/共同Q区PSD W1、低高raw-width MAE、低高峰位MAE、baseline RMS/ACF1 W1均不恶化>10%；BASE原有效密度bins2–7均保持有效，不缩交集掩盖失败。
3. 全部源ID/组成/顺序/权重不变，无新增无效/非有限/ADC裁剪，原全人口source-Q/PSD Spearman下降各<=.05，actual-source purehad bins3/4谷增加各<=.02。支持外417行全部保留并描述，源探针确有作用。
4. 9固定边界W1各<=max(BASE,原Independent对应值)+1/16383V，完整source 128扰动满足前述非相同率和RMS门槛，无后处理补救。
5. 新REAL控制相对RAW REAL–REAL在full493/纯u480/baseline三个视图配对ΔBA95%上界各<=.10，且三点值各<=同处理BASE_REAL_CONTROL+.01。完整电压和其他视图即使不是此门槛也须报告，不以raw对照代替同处理对照。原2D配对unknown不补拟合。
6. 唯一训练2950完成、初始尺度/核公式检查、主及REAL共6000完整链回放、图/资源/失败/计数完整。undefined不算通过，500次区间条件于固定分类器，不包含响应重训或独立采集不确定性。

任一失败即关闭这一初始梯度定标的固定联合Gaussian-MMD神经训练配方；不换bandwidth/c/种子/训练步数/初始化/endpoint，也不再加后置映射、保护或新增损失救该轮。不把失败外推为所有核方法不可行；根依据实际结果决定后续单一问题，项目保持active。

## 预算与交付

阶段墙钟5400秒、执行器4800秒封存、根600秒Git/D。科学累计1800秒，其中唯一神经训练含64批初始尺度确定和全部原J20验证<=900秒；toy自检、生成、评价、重放、绘图/修复及失败均计总科学成本。旧对照不重训；同数据池/更新流与预算上限，额外初始尺度前向和实际FLOP不同另报。

沿用原服务器环境，先只读资源/本项目PID/argv/cwd，新独立目录唯一启动；未知状态不重提交。一次一个科学作业、CPU4、真实worker RSS<=4GiB、GPU allocated<=4GiB。本机CPU4后续生成/评价/回放。全部数据/源码/cache/temp留E/D，输出512MiB（依赖/传输包分列），无新REAL/CAL/EVAL/G4/安装、共享环境或他人作业改动，无迁移内容一致性测试。

交付可独立部署G、训练与确定性输入/随机流、精确波形与功能分量、30评价器及全部聚合指标、易读直观图（含源/候选/实测、控制、训练尺度与轨迹）、实际命令/退出/资源/失败、6000重放、报告/复现说明和一次最终清单。所有最终新图实际打开；根关键复核，私有Git保存代码/聚合证据，完整模型与事件数据留E并正常备份D。
