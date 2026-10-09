# R37：冻结源神经转换器的完整联合协方差校准

本协议在R36实际封存、根完成关键图表判断并写入`reviews/R37_REGISTRATION_001/RECEIPT.json`后生效。既有执行器连续实现、计算、自检、看图和封存，根负责问题、预算及Git/D；不新增代理、安装或逐步骤批准。

## 唯一问题与适用范围

R36固定扩散修正在局部波形、总体PSD及部分峰宽上有收益，但完整shape493未证明优于BASE，噪声RMS距离恶化59.42%，actual-source pure-had bins3/4谷比例增加约8.92/9.39个百分点，9个边界中second_center_150超限，且REAL guide控制也可辨。完整4000条独立重放通过不能代替科学验收；该固定扩散配方关闭。旧MLP-CFM耦合及分段保护分支保持关闭。

本轮只问：以完整480点源波形驱动的冻结G为起点，仅用原TRAIN流估计条件均值及完整480×480时序协方差，作一次最小位移的高斯仿射校准，能否同时改善完整响应并保住源信息？R30的域身份诊断及逐点方差处理没有检验这条可部署的完整协方差映射；本轮不使用DEV拟合校准常数、不在推理查询REAL库，不生成与源无关的目标样本。

方法采用[Computational Optimal Transport，Remark 2.31的高斯仿射映射](https://arxiv.org/html/1803.00567v4)，可核对[POT原生文档公式](https://pythonot.github.io/gen_modules/ot.gaussian.html#ot.gaussian.bures_wasserstein_mapping)。这是既有方法的项目实验，无新颖性主张；最优性只针对所定义标准化坐标下的正定高斯近似，不表示非高斯混合波形的真实最优映射或物理真值。实现用既有NumPy/SciPy，不安装POT。

候选是“冻结神经转换器G + TRAIN拟合的线性校准层”的混合链；新增神经更新为0，不能写成新训练了一套神经网络。完整source、原z/white/style经过G，完整guide再经过条件矩阵；非源条件随机自生成不是本轮候选。

## 固定拟合数据与一次统计估计

复用R33 `reconstruct_and_reorder`、原TRAIN_PRIVATE和原噪声/style、冻结R32 CONTROL_FINAL G、原8个source-Q支持区、`AFFINE_CONSTANTS.npz`。使用R33原source/target/bin/original_row_ordinal的94400条实际行序，按原2950×32顺序，每条source只前向G一次；target沿用原始顺序，无PSD重排、临时配对监督或重采样。来源、权重和分区不变，禁止REAL扩容及DEV/EVAL拟合。复用原18895条支持内REAL TRAIN估出的逐点mu/scale，不重新估计仿射常数。

源guide g由原G产生preADC float32后转float64；REAL原TRAIN波形也转float64。在每条原bin b下定义`ys=(g-mu_b)/scale_b`和`yt=(REAL-mu_b)/scale_b`，这一步运算为float64。各bin分别积累两域均值ms/mt与无偏sample covariance Cs/Ct（ddof=1），使用float64批次合并的稳定中心二阶矩算法。各域共94400行，重复行按原流自然计数，不另去重或改权重；报告每bin行数与实际不同source/target行数。独立两域的边缘矩估计不赋予行号配对任何物理意义。

保存实际source/target/bin/ordinal/z_white/style流hash，与R33原记录逐项核对；同服务器环境且G批32不变时，base_preADC流hash也应一致。这里只绑定实验输入和实际G前向，不作迁移文件全量一致性测试。若运行平台导致G输出不同，先记录具体差异与环境并停止本次拟合，不重提作业或暗改参考。

一次拟合包含16份均值/协方差与8个映射。每个原Cs/Ct先对称化，特征值分解；唯一预设正定化为各自`tau=1e-6*trace(C)/480`，将其特征值下限置tau得到Cs_reg/Ct_reg。需要N>480、trace>0、所有值有限；原最小特征值不得低于`-1e-8*max_eigenvalue`。失败则保留并结束本配方，不删区、不增数据、不调floor。保存原/正定化特征谱、floor坐标数及增加trace比例，不把重复行数称有效独立样本量。

每bin唯一矩阵为

`A = Cs_reg^(-1/2) @ (Cs_reg^(1/2) @ Ct_reg @ Cs_reg^(1/2))^(1/2) @ Cs_reg^(-1/2)`，

数值中间矩阵均先对称化；中间矩阵只有机器误差范围的负特征值可置0，若小于上述相对-1e-8限则停止。最后A对称化，不截A的特征值、不做低秩化、任意旋转、分段拼接或结果后校正。保存float64 A、ms、mt、原和正定化Cs/Ct、谱及实际版本。

在人工小矩阵执行一次同分布恒等、对角缩放及协方差等式检查；实际8区报告A对称误差、最小特征值、条件数以及`||A Cs_reg A.T-Ct_reg||F/||Ct_reg||F`，后者须<=1e-6且A正定。再报告真实未正定化训练矩的`A Cs A.T`与Ct差异；不把正定化矩阵的代数闭合作为真实训练或DEV分布已匹配。无第二次真实样本拟合、无floor/分区/矩阵方向扫描。

## 唯一部署、探针与回放

支持内逐行`out64 = mu_b + scale_b*(mt_b + A_b @ ((g-mu_b)/scale_b-ms_b))`，再转float32，最后原14bit ADC一次。以原完整source的Q决定支持与bin，不改成输出Q，不对输出积分、PSD、基线均值或幅度作再次保护/重标/裁剪/平滑。无扩散及新随机源；随机性完全继承G的原z/white/style。

支持外仍原G回退，保留全部源行。保存精确base signal/electronic/sum-rounding、calibration delta和pre/postADC，分量只是加法账本，最终物理signal/electronic拆分unknown。推理入口仅加载冻结G、固定8矩阵/常数及输入，不加载REAL库、伪标签或读出器。

唯一主生成为原DEV源2000（417 pureEE/1518 pure-had/65 mixed），原父128×8 joint1024、原源探针832、以及R36同定义的额外128完整guide扰动：前128支持内主父，保留主source bin与原随机输入，仅对guide的150:400施加去均值且该段RMS1mV的固定正弦，其余0。报告实际float32 perturbation的Qtotal/Qfast/PSD误差和输出preADC差；无需条件c，不能因扰动重分bin。原832探针各配对仍复用相同G随机性。

独立本机CPU重新加载部署入口，从完整source与原z/white/style实际重放主2000；要求IDs/顺序/支持、base与calibration分量、pre/postADC逐值一致。生成与回放都固定G原CPU chunk128和矩阵float64运算，不为凑bitwise而加载已保存输出当答案。无额外2000 REAL映射生成：本轮是Cs→Ct的单向校准，对真实输入直接套源矩阵并不定义恒等对照；原RAW REAL TRAIN-reference对REAL DEV的同评价处理参照继续报告，不能把它称校准后恒等或独立采集验证。

## 完整评价与预定继续规则

使用原10视图、600训练/250留出、500次固定分类器区组bootstrap；只拟合新候选10个HGB、8个密度描述槽/最多8次EM。BASE、R33 Independent、R35、R36及RAW REAL–REAL全复用已有结果，不重拟合旧端点。RAW REAL–REAL旧Q/PSD二维预测不可得，延续unknown，不补拟合或补造配对区间；其余9视图沿用已保存匹配参考。

全部人口与支持外回退均报告：完整/局部波形，原电压与峰归一化、平均/逐点SD，Q–PSD总体及8观察Q区CDF，低高峰位/raw与detrended峰宽，噪声RMS/ACF及时间曲率，原/开发投影，source-Q/PSD关联和actual-source pure-had bins3/4谷比例，9个固定边界。原source组213/181与历史ADC组207/182分别标注，密度失败保持缺口。新增描述为8区训练原/正定化矩的闭合和保存DEV输出的完整相关矩阵与非对角协方差差；只描述，不拟合DEV矩阵用于变换。所有新图实际打开。

只在以下全部通过时保留候选作下一开发对象，仍不代表完整项目验收：

1. full493及纯u480对BASE的配对BA差95%区间上界均<0，且两个BA点值相对R35均不恶化>.01；完整电压及绝对REAL–REAL差仍完整呈现。
2. 对BASE，总体/共同观察Q区PSD W1、低高两成分raw-width MAE、低高峰位MAE、baseline RMS/ACF1 W1均不恶化>10%；原为0不能增加。峰位门槛本轮预先加入，避免再用峰宽收益掩盖中心漂移。BASE原有效密度区2–7全部保持有效，不缩交集掩盖失败。
3. 全部源ID、组成、权重和顺序不变，无新增无效/非有限/ADC裁剪；actual-source pure-had bins3/4谷比例各增幅<=2个百分点，全体source-Q与PSD Spearman各下降<=.05，原源探针有响应。
4. 固定9边界W1均<=max(BASE,原Independent相应W1)+1/16383V；额外128 guide探针>=95% preADC输出非逐值相同且配对RMS中位数>1/16383V。敏感性只证明完整输入被使用，不是物理真值。
5. 实际8矩阵数值检查与主2000 CPU重放通过，所有计数/失败/图和资源完整。

undefined不通过。任一失败关闭这条固定条件高斯协方差校准配方，不扫描floor、秩、bin、方向、目标重配或再叠保护；分别判断协方差闭合、DEV外推、源结构与高阶分布。不能以协方差矩阵匹配或单项PSD收益宣布成功，也不能将一个固定失败写成所有矩方法不可能。项目保持active。

## 资源与交付

墙钟5400秒，执行器4800秒封存，根600秒交付；累计科学1800秒，其中一次94400行G前向与统计拟合<=900秒。到截止时保存实际状态；不以部分数据凑足8矩阵并称完整拟合。服务器沿用R33相同环境、GPU及原chunk32来生成TRAIN guide，矩计算CPU4；后续本机CPU4生成/评价/回放。先只读确认资源及本人项目argv+cwd，在新独立远端目录唯一提交，无未知作业才能启动。

真实worker RSS<=4GiB、GPU allocated<=4GiB、一次一个科学作业；保留nonce/PID/实际监控缺测。新增神经拟合/更新0，唯一统计拟合流，source与REAL各94400训练行；主2000/joint1024/probe832/额外guide128，主独立回放2000；HGB<=10/EM<=8。独有输出<=512MiB，依赖副本及正常打包分列。无新REAL/CAL/EVAL/G4、无新安装、无共享环境或他人作业修改，迁移只正常传输和实际加载，不另加内容一致性测试。本机数据、代码、日志、缓存与临时文件均E/D。

交付冻结G入口、8个float64映射及训练统计、原输入与RNG关联、完整评价/源保真/协方差图、实际命令/计数/资源/失败、独立重放、简洁报告与复现说明及一次最终清单。根关键结果复核后私有Git提交聚合证据，模型与事件数据留E并正常备份D，不重跑已完成科学。
