# R36：完整源波形引导的条件扩散修正

本轮协议在R35实际封存、根完成关键图表判断并写入`reviews/R36_REGISTRATION_001/RECEIPT.json`后生效。仍由既有执行器连续实现、运行、自检、看图和封存，根负责方向与Git/D；不新增代理、安装或逐步骤许可。

## 问题与分支界限

R33独立/OT耦合、R34分段后保护及R35 PSD同序耦合均未使完整响应、源结构和噪声共同通过。R35的full493/u480 BA为.944/.952，总体PSD W1降到约.01531、全体source-PSD秩关联约.420，但actual-source pure-had bins3/4谷比例升到约11.74%/10.50%，噪声退化且原有效bin7密度未收敛。关闭当前“冻结BASE+原480维MLP-CFM”的耦合/代价/种子/网络/步数小变体分支。

本轮检验另一种学习问题：仅从原REAL TRAIN学习在不同噪声级下恢复完整波形的条件模型，推理从完整源驱动BASE波形的有限加噪状态开始，电荷和PSD作为明确的条件。它学习实测域内的去噪，而不是拟合SIM–REAL临时配对间的线性速度；推理必须实际使用完整源波形。不会从纯随机噪声自生成端点替代源转换，也不把条件标量当作完整源输入的替代物。

方法参考[SDEdit的有限加噪后修正思想](https://arxiv.org/html/2108.01073v2)、[DDPM的噪声预测训练和离散反向链](https://arxiv.org/html/2006.11239v2)、[cosine噪声日程](https://arxiv.org/html/2102.09672v1)。这些原工作不能证明本项目波形或源物理保真；本轮是固定的一维条件DDPM修正配方，不声称新颖性、连续SDE的精确解或现成图像方法已经适用于EJ-276D。

## 固定训练数据、模型和预算

新建一个`CONDITIONAL_DIFFUSION_EDIT`模型，从零训练一次。保持与R33/R35相同的原REAL TRAIN 94400条实际target行序、原Q-bin和2950×32更新上限、900秒训练上限；复用R33每区同一480维TRAIN仿射常数与8区边界。只使用原target行，不重新抽取REAL、扩充训练池、改变权重或调用DEV选模。源G/reader/判别器/既有流均零更新；既有源G只用于生成完整推理guide。本次目标、架构及每步计算量变化，报告实际参数量/更新/样本数/时间，不声称FLOP相等或只改变一个系数。

在原18895条共同Q支持REAL TRAIN上，计算`c=(log(Qtotal), PSD)`两坐标的固定均值与sample SD；使用原波形升float64、扣前100点均值、Qtotal=sum100:400×1000、PSD=1-sum100:150/sum100:400。条件仅做上述TRAIN标准化，不裁剪PSD或条件值，不引入粒子伪标签。所有训练条件必须有限且Qtotal>0、SD>0，否则记录并停止，不删行。输入`y0=(REAL-mu_b)/scale_b`。

固定100步离散日程：`f(t)=cos(((t/100+.008)/1.008)*pi/2)^2`，先算`abar_target(t)=f(t)/f(0)`，再`beta_t=min(1-abar_target(t)/abar_target(t-1),.999)`，`alpha_t=1-beta_t`，实际`abar_t=prod(alpha_1:alpha_t)`，`abar_0=1`。系数在float64计算保存，实际网络与状态float32。不能把clipped beta后的abar仍写成未clip公式。

每训练行独立均匀取整数`t∈1..100`、标准正态`epsilon[480]`，构造`y_t=sqrt(abar_t)*y0+sqrt(1-abar_t)*epsilon`，仅最小化完整480点`MSE(epsilon_hat(y_t,t,b,c),epsilon)`。原target/bin同序hash需与R33记录匹配；训练t/noise使用预定PCG64种子20261010361、逐批记录状态/实际流hash，不需要保存全部噪声矩阵。模型初始化20261010360。Adam lr2e-4、betas(.9,.999)、eps1e-8、wd0、clip5，固定实际末点，不使用EMA、数据增强、自条件、辅助损失、classifier-free guidance或checkpoint选取。

固定1D U-Net：输入为`y_t`与固定绝对位置`linspace(-1,1,480)`两个通道，先kernel3卷积2→32；位置通道用来保留逐坐标仿射后不能由局部幅值可靠推知的采样位置。长度480→240→120→60，通道32/64/128/128；每尺度2个残差块，kernel3/padding1、GroupNorm8、SiLU，无dropout；下采样stride2卷积、上采样nearest×2后kernel3卷积，连接对应encoder skip。残差块为GN→SiLU→conv3，再由128维条件嵌入对第二个GN结果作scale/shift，再SiLU→conv3，形状变化使用1×1残差支路。最低分辨率两个残差块之间有一个4-head自注意力块（LayerNorm、MultiheadAttention、残差；无额外FFN），以覆盖完整时序。时间嵌入为`t/100`及2*pi*(t/100)*[1,2,4,8,16,32,64,128]的sin/cos，连接原8维bin one-hot与标准化c，共27维，MLP27→128→128 SiLU；各残差块linear128→2C。最后GN8→SiLU→conv3输出1通道epsilon，末层weight/bias全0。禁止因结果不好更改深度、通道、注意力或日程；参数量以实际源码为准并在训练前保存。

## 唯一源修正链与实测控制

唯一预指定修正强度为`t_start=50`，不扫描强弱或选择较好实例。源候选仍从完整480点source及原z/white/style运行冻结R32 CONTROL_FINAL G得preADC guide `g`，条件c由实际g计算；source-Q原支持标志及bin沿用R33，支持外原G回退，全部行保留。支持内条件无效则停止并报告，禁止增加隐式回退。原G输出不是经过R35的波形，不串联已关闭流。

令`guide=(g-mu_b)/scale_b`，采样`y50=sqrt(abar50)*guide+sqrt(1-abar50)*z50`，依次50→1完整执行离散DDPM反向步骤：

`mean_t=(y_t-beta_t/sqrt(1-abar_t)*epsilon_hat(y_t,t,b,c))/sqrt(alpha_t)`；
`v_t=beta_t*(1-abar_(t-1))/(1-abar_t)`；
`y_(t-1)=mean_t+sqrt(v_t)*z_t`（t=1无随机项）。

所有步保留固定c和b，最后反仿射，再原float32 14bit ADC一次。不裁预测x0，不去均值、按段修正、积分重标、平滑、投影或改ADC。这是定义好的50步离散随机模型，不附加未登记的步数扫描或ODE收敛比较；前向加噪、已知epsilon反解、两种posterior mean公式相等、v1=0等在小人工数组上作一次必要公式检查。

新生成全部2000原DEV源父（417 pureEE/1518 pure-had/65 mixed），保留IDs/顺序/权重和每个支持外回退。原128父×8 joint=1024、原源探针832继续完整报告；保存精确preADC、base原分量和edit增量，最终物理signal/electronic拆分unknown。禁止推理读取REAL训练库、近邻检索或读出器。

另外用同一模型/强度/日程处理既有固定2000行`TRAIN_REFERENCE`完整实测guide，构成`REAL_GUIDE_EDIT`控制。沿用该参考的实际行序/组和原REAL–REAL分类选择；条件取各自原guide，Q在原8区外的实测guide原样保留。该控制包含已训练域样本，明确标记，不能称未见采集。支持内最终使用相同ADC，支持外原实测ADC记录逐值回退；这一口径全部披露。控制用来检查方法对已实测输入造成的分布变形，不能当主仿真转换结果。

随机性用单独PCG64种子20261010362派生每记录、每重复的uint64种子；每记录按固定row-major `(initial+49 reverse,480)`生成float32标准正态，t=1不消耗额外noise。保存种子、映射、NumPy版本与实际noise流hash，种子不随chunk或域内排序变化；源与REAL控制采用不同record namespace。允许按需重建精确噪声，避免保存数百MB重复矩阵；主独立重放必须实际证明重建一致。

额外128行“完整guide作用”探针：按原source-Q支持选择前128个可修正父，复用其主输出完整随机流/固定c/b，只给preADC guide的150:400加入固定sin(2*pi*j/249), j=0..249，经float64去该段均值后缩放为该段RMS1mV、其余点0的扰动。记录数值Qtotal/fast/PSD误差，固定c禁止重新估计，比较主输出和扰动输出的preADC配对差。它只检验在相同标量条件/随机性下完整guide还能影响结果，不是物理事件真值。原832源输入探针仍检验完整source→G→编辑链。

## 完整评价、控制和退出

源候选使用原10视图（7历史+3个480视图）、600训练/250留出和500固定分类器区组bootstrap；与保存BASE以及R35固定末点比较，旧Independent/OT仅描述，不重拟合任何旧端点或RAW REAL–REAL。新REAL_GUIDE_EDIT也按同10视图与原REAL DEV比较，最多再10 HGB；保留完整493、纯u480和baseline的相对原RAW REAL–REAL配对BA差。

源候选密度最多8描述槽/8 EM，REAL控制也最多8槽/8 EM，不补拟合；旧参考和失败原样复用。完整及条件平均/SD、原电压/峰归一化视图、Q–PSD全人口/8观察Q区CDF、低高峰位与raw/detrended宽度、噪声RMS/ACF和时间结构、原/开发投影、源Q/PSD关联与actual-source pure-had bins3/4谷比例都报告。源主口径213/181与历史ADC参考207/182分开。R34固定9个边界差继续检查，不增窗口。

只有以下全部通过才保留固定端点作下一开发对象：

- 源候选full493/纯u480相对BASE配对BA差95%区间上界均<0；相对保存R35两个BA点值均不恶化>.01。它仍是相对开发门槛，完整绝对可分性和REAL–REAL差距必须明确。
- 源候选相对BASE的总体/共同Q区PSD W1、两成分raw-width MAE、baseline RMS/ACF1 W1均不恶化>10%；零基线不增加。BASE原有效密度区2–7不新增支持/收敛失败，不缩小交集掩盖失败。
- 全部父/组成/权重/顺序不变，无新增无效、非有限或ADC裁剪；两个actual-source pure-had区谷比例各增幅<=2个百分点，全体source-Q和PSD Spearman各下降<=.05；原源探针有响应。
- 固定9边界W1不超过max(BASE,原Independent对应距离)+1/16383V。额外128 guide探针至少95% preADC输出不逐值相同，配对preADC RMS中位数>1/16383V；完整guide作用与原输入Q/PSD数值守恒分开描述。
- REAL_GUIDE_EDIT的full493/纯u480/baseline相对原RAW REAL–REAL配对BA差，三个95%区间上界都<=.10。Q/PSD总体W1、baseline RMS/ACF1 W1及条件峰位/峰宽继续完整描述，同时给出RAW REAL–REAL参照；不把其较小抽样距离的固定百分比当作有根据的等效阈值。上述BA限是本轮预定方法变形容差，不是物理等效或新采集证明；undefined不通过。
- 独立本机CPU完整重放源主2000和REAL控制2000，共4000，pre/postADC、实际noise流、父ID、condition/support和base/edit分量逐值通过；可部署入口不加载REAL训练库。全部新图实际查看。

任一失败则关闭此固定条件扩散修正配方，不扫描t_start、beta日程、反向方差、网络、种子或步数。先区分REAL控制本身失败、源内容失败和分布差距，不能只称“diffusion不行”。不宣称所有扩散方法不可能；项目本身继续active。无关的后处理P和旧CFM配对分支不重开。

## 执行与交付

- 新服务器独立目录，原Python环境；启动前只读确认资源与本人项目argv+cwd，无未知作业才提交。既有实际PID/nonce/RSS守护，所有缺测时间保留，未知状态不重复提交。无新安装或共享环境修改。
- 墙钟7200秒，执行器6600秒封存，根600秒交付。累计科学3000秒，唯一训练<=900秒/2950更新/94400 REAL行。一次一个科学作业，CPU4、真实RSS/GPU allocated各4GiB。
- 本机CPU4生成/评价/回放：主源2000、REAL控制2000、joint1024、原探针832、额外guide探针128；独立主回放4000。HGB<=20、密度槽/EM<=16。小人工公式检查无真实新样本；无第二响应拟合。输出独有<=512MiB，依赖副本/正常打包分列。
- 无新REAL/CAL/EVAL/G4或长采集；TRAIN只训练、DEV只评价，所有实际生成全部保存包括失败。原始阶段只读，本机所有数据/日志/源码/缓存/临时文件均E/D。
- 交付权重、固定日程/条件常数、完整源推理入口、实际输入/RNG关联、全部计数/失败/资源、公式与独立重放、完整评价及直观源/BASE/编辑/同Q实测图、REAL控制图、复现和最终清单。根只做关键结果判断、紧凑记录与私有Git/D交付，不重跑整轮科学。
