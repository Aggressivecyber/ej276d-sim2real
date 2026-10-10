# R48 只读设计咨询：已有噪声补全 + 完整 guide 条件流 + REAL 身份轨迹

本文件是一个可登记的固定方案建议，尚非科学执行协议。实际只读开始：2026-10-10T07:13:09.2261682Z，PowerShell PID23300，工具 cf5a87/native0；咨询截止07:28:09.2261682Z。本轮没有读取科学数组、计算统计量、前向、训练、生成或SSH，也未修改R47或旧阶段。所有参数和文件量是建议或代数估算，不能记成已运行验证。

**建议：可以形成一个有明确可检验假设的新框架，但不能宣称已识别物理噪声或已经解决目标兼容性。两臂都加入固定完整guide和同REAL身份轨迹；唯一臂间差别是是否根据输入基线估计已有噪声，减少需要补入的创新量。** 不把这项比较写成相对R46仅改一个系数的因果实验；与R46相比，框架、训练约束和初始化噪声都改变。

## 已核对的旧实现与实质区别

| 项目证据 | 实际已有机制 | 本次所需区别 |
|---|---|---|
| R43 source_flow_core.py:9–20、network.py:27–43、train_server.py:68–73 | 初态统一加1mV白噪声并作全段电荷切向投影；场只接当前状态y、t和Q-bin；唯一速度MSE | 没有输入噪声估计、固定完整guide通道或同REAL身份轨迹 |
| R45 train_extended.py:129–139及PROTOCOL | 保留同一initial/field/MSE，延续Adam与t流；新增white，只扩大到50000 | 延长曝光没有改变上述结构；并非本框架的等预算对照 |
| R46 source_flow_core.py及check_models.py:27–44 | 同一初态与场；共同64步候选、128步仅检查 | 只改变已登记积分预算，无新增身份训练或噪声识别 |
| R35协议15–28行 | 冻结G后完整480维MLP-CFM，Q内按preADC PSD同序耦合 | 它解决耦合，不估计输入已有噪声，不提供REAL自身固定guide身份轨迹 |
| R36协议15–41行 | REAL目标噪声预测、有限加噪guide后DDPM编辑；标量c固定条件 | 没有本方案的恒定完整原输入通道或联合SOURCE-CFM/REAL身份目标；不能以本方案重开t_start扫描 |

以上是所读项目版本的差异判断，不是文献新颖性声明。R35/R36/R42–46固定配方及其负结果保持关闭。本次不是调小1mV、改变solver或重新删除保存的S/G/F/R分量。

完整源码路径：

- E:/GAN/research/sim2real_two_papers_20261008_0305/work/R43_SCALAR_TIME_NEURAL_FLOW_001/{network.py,source_flow_core.py,train_server.py,PROTOCOL.md}
- E:/GAN/research/sim2real_two_papers_20261008_0305/work/R45_EXTENDED_BUDGET_NEURAL_FLOW_001/{train_extended.py,PROTOCOL.md}
- E:/GAN/research/sim2real_two_papers_20261008_0305/work/R46_FIXED_INTEGRATION_BUDGET_001/{source_flow_core.py,check_models.py,PROTOCOL.md}
- E:/GAN/research/sim2real_two_papers_20261008_0305/tasks/R35_PSD_ORDERED_FLOW_001.md
- E:/GAN/research/sim2real_two_papers_20261008_0305/tasks/R36_CONDITIONAL_DIFFUSION_EDIT_001.md
- E:/GAN/research/sim2real_two_papers_20261008_0305/reviews/R47_ROUTE_DECISION_001.md

## 固定数学算子建议

### 1. 不变的电荷坐标与统一调用

对任一完整480点输入x（SOURCE或REAL都完全同一函数），沿用：

q(x)=sum(x[100:400])−3 sum(x[0:100])，T_b(x)=(x/q(x)−mu_b)/s_b。

q、PSD、原8区边界、mu_b/s_b及逐点SD floor继承已有TRAIN常数；不重定义积分窗、不量化guide、不改变真实source-Q与历史postADC参考的命名。支持外只使用原最近边区conditioning，全部行经过新流；无REAL特判、源/目标域flag或BASE回退。

P_b(v)=v−mu_b q(v)，其中q(mu_b)=1。P仍是原全段切向投影，用在创新和速度场上，不是R34的硬分段后处理。推理q非有限或非正时保留失败，不裁值、不最终重归一化。因P保留q，人口Q不会由本机制调整；这不是charge-only uncertain证明A–F不可能，也不是修改源权重或总体标准的许可。

### 2. 唯一全局 colored 核与固定5%标量补全

校准只使用冻结原训练行流实际 unique matched_target 的REAL事件，每个唯一事件计一次；不另开全TRAIN池、不新增REAL暴露，不读DEV/CAL/EVAL。实际唯一数在未来注册后读取确定，当前未知。原训练行的重复曝光和样本权重不改。

令B选择前100点，H为100点居中矩阵。每条REAL的e_i=H B r_i，v_i=||e_i||²/99（V²）。提前固定唯一 tau²=np.quantile(v_i,.05,method='linear')，不扫分位。

从同一批e_i计算100点酉rFFT系数平均功率 p_k=mean_i |rfft(e_i,norm='ortho')_k|²，频率k/100、k=0..50；不乘单边谱的2，DC置0。将p按数字频率线性插值至j/480、j=0..240，得到非负lambda_j，DC再次置0。完整实对称核定义：

K0 epsilon = irfft(sqrt(lambda) * rfft(epsilon,norm='ortho'), n=480,norm='ortho')。

epsilon为480维独立标准正态；正频率共轭镜像、Nyquist一次。因此K0为实对称循环滤波，创新协方差是K0 K0^T。这里使用的是酉频率系数功率作核特征值，不能混用V²/Hz单边谱单位。这是100点谱向480点平稳形状的固定估计，不证明目标完整窗协方差或物理噪声模型。

每原Q条件保留P_b=I−mu_b a^T，q(v)=a^T v、q(mu_b)=1。唯一归一化：

kappa_b²=trace(H B P_b K0 K0^T P_b^T B^T H)/99=||H B P_b K0||_F²/99。

控制：n_U(x,epsilon)=sqrt(tau²) P_b K0 epsilon/kappa_b。

主候选：n_C(x,epsilon)=sqrt(max(tau²−v(x),0)) P_b K0 epsilon/kappa_b，v(x)=||H B x||²/99。

给定固定x，独立epsilon的期望为0且q(n)=0；条件期望样本方差E[v(n)|x]=创新幅度平方，交叉项均值为0。因此主候选 E[v(x+n)|x]=v(x)+max(tau²−v(x),0)=max(v(x),tau²)。这是期望下限，不是逐条实际方差硬设定；不能将投影后的频带功率或协方差差写成精确目标。P会改变频谱并引入相关性，以上kappa已计入该影响。

tau²、lambda和kappa需有限且tau²/kappa严格正；失败停止固定配方，不添加谱floor、抖动或换分位。SOURCE若基线0，两臂都只补至REAL低5%水平；剩余噪声与信号相关涨落由学习路径承担，不保证完整随机响应。已有噪声较高的REAL不补，低噪声尾仍可能被抬升或改变分布，身份轨迹是否纠正须实测。没有域标签、REAL特判或BASE回退。

初态I=x+n，保q。float64创新/坐标/切向算术、float32网络、preADC最终float32及原ADC一次；不对x降噪、全窗RMS缩放、拼接基线或重新归一化。两臂同K0、epsilon、guide、网络、损失和目标，仅标量幅度不同。

此前四频带/多个阈值草稿不采用：它同时增加频带门控、谱形与幅值变化，也容易把投影后泄漏混成精确方差补偿。本方案只检验保守标量已有噪声补全，不能冒称物理去噪。周期FFT、基线代表整窗及平稳假设保留为限制。

## 网络与训练损失（两臂完全相同）

保留R43的32/64/128/128、GN/FiLM、60点4-head attention、scalar-t+zero16+bin8、零final的U-Net。唯一公共结构变化：输入卷积从2通道变成3通道，接 `[当前状态y, 固定g=T_b(x), 绝对position]`。完整原x贯穿每个t，不能只用4个条件或用已加噪状态代替固定guide。多96个卷积参数，代数参数数约1,371,937，实际实现后才计数。

速度仍是 fbar_theta=P_b(s_b*f_theta(y,t,b,g))/s_b。两臂从相同全新初始化和新Adam开始，不能加载R43/R45末点后称仅新方法初始化；建议固定seed20261010480。没有domain embedding、true粒子标签、targetID或邻近REAL检索。推理只加载权重、全局谱、tau²与8个kappa常数、旧TRAIN坐标常数和原输入/创新随机量。

每步保持原Q内来源/REAL边缘、原耦合周期的32个SOURCE和32个REAL行，不改变源采样权重或类别占比。主SOURCE任务沿用原未加噪SOURCE PSD与原REAL PSD的Q内稳定同序统计耦合（并非真实跨域配对）。两臂同source/REAL/bin/ordinal/t/white序列；z/style/NCE保持原unused审计地位，不突然加成损失。

SOURCE路径：a=T_b(I_b(s,xi_s))，c=T_b(r_permuted)，y_t=(1−t)a+t c，v=c−a，g=T_b(s)。

REAL身份路径：a_id=T_b(I_b(r,xi_id))，c_id=T_b(r)，y_id=(1−t_id)a_id+t_id c_id，v_id=c_id−a_id，g_id=T_b(r)。

L_SOURCE=mean480,rows ||fbar(y_t,t,b,g)−v||²。

L_IDENTITY=mean480,rows ||fbar(y_id,t_id,b,g_id)−v_id||²。

固定L=L_SOURCE+L_IDENTITY（两项系数均1，不扫描）；REAL身份只来自相同r事件。它学习恢复自造创新前的r，不宣称去除r原来的探测器噪声或跨域真配对。identity随机流用独立固定PCG64(20261010483)，t_id独立PCG64(20261010484)，两臂实际值严格相同；SOURCE t/white采用注册前固定流，序列不能因创新开关改变。两任务可合成64行同次前向再分项均值，等更新不等于与旧32行训练等FLOP。

该目标对COMP低创新REAL可接近零速度身份轨迹，对UNCONDITIONAL则需要学回新增噪声；这正是输入噪声干预的整体效果。它不是“去噪+多一种loss”相对旧R46的单因素实验；只有新两臂内部的噪声估计开关是受控差别。

建议每臂固定总50000更新、lr2e−4/Adam(.9,.999)/eps1e−8/wd0/clip5，每臂最多5400秒，总训练最多10800秒；全阶段科学14400秒、墙钟21600秒，剩余3600科学秒用于检查/生成/评价/回放，固定末点，0 DEV选点和0其它候选。保留原组隔离内部val，均只监测原SOURCE速度与REAL身份两项、创新触发和谱；不用其选择分位数、网络或checkpoint。实际训练曝光为Monte Carlo行，不新增唯一父人口。

## 稳定性与可识别性：不能省略的漏洞

1. **不是旧无guide确定性双射。** 旧场不接固定guide，同一可逆映射严格保持目标分布又精确把不同源分布推到目标会有冲突；这里是给定不同guide的随机核，不能套用同一个无条件双射论证。反过来，这也不保证A–F或物理正确性。
2. **guide可以成为域捷径。** 当前SOURCE baseline近零、REAL已有噪声，网络可据此学习两类行为；不是显式域标签，但不证明对未知噪声水平/另一采集泛化。原source交换、同父重复、零积分SOURCE扰动都要检验，非零敏感性不是正确物理关联。
3. **100点噪声估计不可辨别全部噪声。** 脉冲前漂移、低频结构、ADC与非平稳/信号相关涨落可能不由单一基线谱描述；补全不能消除已有过量噪声。触发很少仍不保证低噪声REAL尾保持。
4. **身份轨迹不是任意路径的稳定性定理。** 学习有限线性轨迹可能留下off-path误差；只测训练过REAL reference也可能偏乐观。至少按原既有内部val身份事件作TRAIN检查，并完整报告原REAL同处理control；若现行A–F指定未见组，必须使用原指定组，不偷换到TRAIN或先挑低噪声r。
5. **SOURCE固定guide可能容许配位记忆/忽略创新。** 同一个source周期与同序统计目标不是物理配对；保留完整源父、组成、源PSD/Q秩和同父随机变化，还要看输出最近邻/模板趋势（若新协议已预算，不自行增加评估器）。不能从整体相似推出逐事件真值。
6. **q守恒不等于PSD守恒。** P保证电荷切向，但创新与学场仍可改变Qfast、PSD中心宽度和纯组谷代理；原门完整保留，不能以创新方差校准或identity loss低代替源保真通过。

## 两臂与判定

| 固定项 | UNCONDITIONAL控制 | COMPLEMENT主候选 |
|---|---|---|
| 原数据/顺序/群/权重/耦合/t/white；guide网络；身份轨迹；优化器/50000末点；目标TRAIN全局谱和唯一5%分位 | 完全相同 | 完全相同 |
| 创新幅度平方 | tau² | max(tau²−v(x),0) |
| 域标签或REAL回退 | 无 | 无 |

“OFF”若直接把全部创新设为0，会回答有没有随机初态而非有没有已有噪声处理；因此不建议另开零噪声第三臂。两臂都使用相同有色谱，这样臂间差别不混入white-vs-colored变化。

训练前锁定创新校准与数值检查，候选64步Heun、128步只作固定TRAIN参考，CPU12/chunk64；保留median<=.25 LSB、p95<=1 LSB、所有preADC q相对误差<=1e−5。不通过停止DEV，不加256/改solver/最终归一化。两臂数据/更新完整程度如实报告，不能因某臂慢或失败后补训择优。

主评价按原A–F，原源2000父/组成/权重/顺序全部保留，417支持外仍运行新流；原600/250/组/500bootstrap，原Q1–7与密度2–7固定，缺失/不收敛不删区。完整电压/形状、总体和条件Q/PSD中心宽度、baseline RMS/ACF、固定9边界、同处理REAL和完整重放全部报告。噪声主图增加raw→innovation initial→final三点，但不增加新有利窗口或新阈值；REAL触发率和谱只作机制描述。

预定两臂配对比较完整shape、PSD宽度/中心、baseline RMS/ACF和REAL同处理BA/CI，不把COMPLEMENT较好自动记成联合成功。只有原A–F通过的固定主候选才能进入后续原冻结效用/FULL-GAP；若联合失败，关闭本最后方法配方，不调分位数/谱核/身份权重。

## <=160 MiB 紧凑产物建议（包含本地必要回传）

不复制TRAIN/SOURCE/REAL原数组、原流历史、整旧stage，不建立重复全部大数组的tar。普通小文件传输，不做额外迁移一致性测试。固定权重/模型一次存储；重放逐chunk比较既有pre/postADC，保存摘要和失败前缀，不重复保存全体replay输出。

| 必需新内容，两臂合计 | 保守预算 |
|---|---:|
| 两份float32末权重；一份共同float32初态 | 16 MiB |
| 两份末Adam状态（可恢复训练证据，计入同上限） | 24 MiB |
| 两臂SOURCE2000+REAL2000主preADC/postADC各float32[480]，合8000行 | 30 MiB |
| joint1024+probe832+SOURCE扰动128 / 臂，pre/postADC，合3968行 | 15 MiB |
| 创新种子/原输入引用/支持/条件/定义/校准小常数和固定索引 | 3 MiB |
| 分类器预测、密度/距离、曲线/资源/失败、必要训练随机状态 | 10 MiB |
| 可视化、源码、图/报告QA、一个最终清单 | 12 MiB |
| 小包/意外失败前缀/真实序列和末位差记录预留 | 30 MiB |
| 合计建议上界 | 140 MiB |

该估算不假定压缩收益。原完整输入引用已绑定父ID/位置，实际创新由种子+固定全局谱/tau²/kappa常数重建；不保存4份S/G/F/R大矩阵。需要展示的代数分量可按固定小父集保存，完整重放时逐块重建并比较，但不得借精度缺口近似原x1。每臂50k更新日志保存分块聚合与真正失败记录，不保存两份1.6M×480训练white。共有必要随机生成算法/状态必须能重建，两臂所消费流的实际hash仅科学同序证据，不做迁移测试。

上述140MiB是新阶段独有输出及必要回传总预算建议，不是已生成文件量。服务器和本地临时归档若形成重复副本也要计入后续<=160MiB，因此优先逐个紧凑文件取回，避免回传大重复归档。完整两臂主replay约8000行的计算预算需根单独注册，存储不要求再保存8000份数组。

## 注册前需要固定的剩余内容

仅需根将本单一算子/架构/身份loss、随机流、既有训练输入路径与原A–F原文绑定为科学协议，明确两次训练和后续控制/评价/重放计数及存储上限。当前剩余方法1、训练5不意味着必须全部花完；本方案使用两次训练，不追加结构消融或噪声公式扫参。咨询可据此收束；在根完整登记前我保持0科学执行。

本次控制读取证据：cf5a87/PID23300、172692、629eed、7d3658、26f345，均native0；协议/源码静态核对不等于公式数值测试、统计校准或GPU执行。
