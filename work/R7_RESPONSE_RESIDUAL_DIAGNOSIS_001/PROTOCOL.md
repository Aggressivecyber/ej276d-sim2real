R7 固定开发诊断；不是新盲测或响应优化。所有定义在本次新数值之前冻结。

输入：固定R3B模型/主DEV2000/主latent白噪声，P1K REAL DEV2000和TRAIN3200（QC fit2592/val599），既有CAL边界/失败参考和R3B固定窄logQ区间。只读取允许的本地文件。P1K字节文件可能含CAL6800行，浮点视图限定前5200；EVAL文件和预测均不访问。SIM元数据只读取real_event_ids/sim_event_ids。所有同父变体复用R1父/人工组划分及固定SHA选序。

52个固定HGB矩阵见config：12原始视图×三比较36；四共同操作×历史shape×三比较12；两干预×两完整视图4。无超参数搜索/响应优化。HGB参数为历史100迭代、lr.06、15叶、最小叶20、L2=1、不early-stop、seed20261008。分类训练侧中位填补/标准化，holdout无估计。目标每域600分类训练、250组/父隔离留出，若某固定池不足则所有核心比较共用可行上界并报告。500次簇bootstrap条件于固定分类器，不是训练不确定性。

视图：full_voltage原始480点电压；full_historical_shape复用全局peak归一化+PSD/rise/tail/RMS比/ACF的旧完整视图。区域0:100、100:150、150:220、220:400，以及去prompt的0:100+150:480，均直接来自原始V。voltage视图保持本段V，不先用区域外baseline/peak归一化；local视图仅减本段均值/除本段总体RMS。baseline另含本段ACF1..8与非零归一化功率谱，电压视图含本段log样本RMS，local不含绝对RMS。去prompt不含PSD/rise/全局peak/跨prompt特征。数值前做区域外改值不变性断言。完整视图特意允许全局信息，不能混为局部证据。

阴性：从允许REAL TRAIN fit2592按固定身份SHA取2000，对照REAL DEV2000；角色人工组不交叠，同一次采集。同N不增加独立采集数，也不是不保证chance的物理等效界。阳性：原REAL DEV加入原先规定2%自身peak exp(-(t-120)/80) t>=120尾和1.5%peak独立Gaussian噪声(seed202610081)；同父/组留出，簇bootstrap将两个视图视为同组。阴性及阳性在每个核心视图/共同操作各有独立拟合；干预的同定义完整视图控制直接引用native控制，不重复拟合。

共同操作：50%最后主峰上升支线性交点移到120，y_new(t)=interp(t+cross50-120,y)，边缘值填充且统计支持变化；ADC4=-.5+4*Delta*round((V+.5)/(4*Delta))，clip[-.5,.5]；Gaussian sigma1ns/truncate4/nearest；先fraction align再同Gaussian。两域和两控制完全同操作。以原始QC/固定身份保持比较对象，重新报告QC/电荷/PSD/rise/峰宽变化。信息丢失引起分类下降不是修复或物理原因。宽度只用原8窄logQ定义/单初始化500更新固定EM，失败undefined，不重拟合CAL参考。

固定模型干预：ADC_off调用保存model.forward(quantize=False)；signal_block_mean20仅将解析信号核/RBF按t=k+j/20,j=0..19计算并块均值到512，在原512噪声、ADC及CFD之前归一化；保存网络权重/latent/white不变，只有派生time buffer更长。它不是完整真实20GS/s采集链，未模拟0.05ns电子噪声/带宽，不能据此唯一归因。两干预共4000×480个输出，不重复已完成主回放；同主权重/latent/white，全部保留。无新采样种子/响应权重更新。

处理链：REAL raw10000→500均值→float64 ADC14bit→float32基线前100及全峰→主峰相连整数CFD→480窗；GEN连续解析信号1ns点采样+512噪声→Torch f32 ADC→前102基线/全峰→同主峰相连整数CFD→480窗。报告点采样/块平均、前100/102、float32量化、完整500/512支持的差异，代码不等于原因。检查phase、peak index、ADC lattice residual、padding/clipping；用固定人工脉冲比较NumPy/Torch整数分支以及边界，人工工程样本不用于物理结论。

条件图及同N距离按各自native CAL Qt桶，包括unknown，不假造粒子真值/q4参考。baseline RMS/ACF/非DC谱、phase、rise、局部二阶差分/相关给出TRAIN fit/val/DEV、GEN及每区同N REAL TRAIN/DEV参照。条件分组使用完整电荷只用于图/摘要，不带入区域分类器。次轮只建议一个优先修正配方和一个机制消融，本轮不执行。

资源：一次一个科学子进程/CPU4，累计数值1800s/90min阶段/RSS4GiB/产物512MiB。所有缓存E/D；失败费用保留，常规工程修复不改科学定义。旧阶段、根控制/Git不可写。
