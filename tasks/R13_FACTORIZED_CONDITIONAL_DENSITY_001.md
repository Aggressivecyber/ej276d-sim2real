# R13：用四组对照分开检验主要分布与条件残差

整个项目目标保持active。本轮由R11/R12证据直接推进：表示全维可逆；去除各向同性协方差收缩可修复大部分基线幅度，但完整shape BA仍0.924–0.948，峰后曲率随电荷的关系仍过平，FACTOR的两峰描述无合格区间。电荷–PSD图中的条件密度结构与共享残差都需要检验，不能继续把单个噪声幅度当作全部问题。

## 固定部分、数据和预算

- 本轮只使用FACTOR完整480维表示：标准化head34、物理单位tail446、已保存的basis/mean/尺度/ridge与反变换。采用FACTOR是因为它显式分开正电荷、基线和shape，方便解释条件响应；不是宣称它优于RAW。PCA、GMM、ridge及旧统计不再拟合。
- TRAIN-fit2592、内部val599、DEV2000及CAL边界沿用R11/R12；新网络只用TRAIN-fit，checkpoint只按内部val。EVAL读取、新G4、CAL波形解码均0。unknown、QC失败和不支持区间保留；不使用真实波形/残差模板重采样。
- 新独立服务器目录与绝对UTC见reviews/R13_REGISTRATION_001/RECEIPT.json。阶段5400秒，执行器在整体截止前600秒停止写入；累计数值1800秒，其中训练合计最多1200秒，单个网络最多600秒。最多CPU4、进程RSS4GiB、本进程GPU allocated4GiB、输出512MiB、一次一个科学作业。复用既有Torch环境和可用GPU；不安装、不改共享环境、不干扰他人。CPU可用时直接按同一预算运行，不另开作业。
- 旧R10/R11/R12只读，迁移正常传输和真实加载，不新增内容一致性测试。执行器连续实现、必要数学检查、训练、评价、绘图和短报告；无子代理、无Git、无中途许可。普通实现错误在原预算内修复并保留失败；科学不达标不追加参数搜索。

## 两个可独立组合的新模块

**Head FLOW。** 在冻结的34维标准化head上拟合一个RealNVP仿射耦合流，标准正态base，8层，17/17交替mask和固定保存的坐标排列。各层scale/shift使用2个64宽SiLU隐藏层，输出层零初始化；log-scale=3*tanh(raw/3)，shift不截断。保存准确forward/inverse、排列和logdet。没有BatchNorm/dropout/额外PCA，不根据DEV改层数或换模型。仿射耦合与精确密度是已有方法，不声称方法本身创新，依据[Real NVP原论文](https://arxiv.org/abs/1605.08803)。

**Tail CONDITIONAL。** 保留R11固定线性条件均值m0(h)，用R12 FACTOR_EMPIRICAL_COV保存的L作为固定可逆基底。对TRAIN实际tail r计算e=L^(-1)(r-m0(h))。学习m(h)、ell(h)，生成

`r = m0(h) + L @ (m(h) + exp(ell(h)) * epsilon)`。

m和raw-ell由一个head34→64→64→892的SiLU MLP同时输出，末层零初始化，ell=3*tanh(raw_ell/3)。初始化正好对应旧共享Gaussian；m无额外截断。条件协方差为L diag(exp(2ell)) L.T，正定且无需新增协方差扫描。此模型允许条件均值和不同创新方向的尺度变化，但不声称能表达任意条件相关矩阵。报告这些限制。

两个模块各只有一个训练seed（head20261009131、tail20261009132），各最多2000更新，batch128，Adam lr3e-4、weight_decay1e-5、梯度范数上限5，无scheduler或重启。训练损失为相应条件/边际负对数密度除以维数；报告的NLL使用完整nat/event。每100步及step0在全部599内部val上计算对应NLL，保留最小者，完全相同则较早step；tail step0参与选择。各网络可独立选型，因为观察到的head及冻结反变换使两个似然项可分。head和tail顺序运行，实际更新/中断计数如实记录。

允许float32训练；选型使用一致的明确算术，最终完整连续密度和固定参数生成用CPU float64的保存权重副本，以便与旧模型的数值定义一致。既有Torch/cuDNN/TF32与determinism设定记录，不更改系统级设置。所有599条val均报告完整NLL及head/tail/Jacobian分量，仍包含R11已经验证的表示Jacobian，不能只报head分数。

## 2×2固定比较与随机输入

| 名称 | head密度 | tail条件密度 |
|---|---|---|
| GMM_GLOBAL | R11固定K8 | R12经验协方差与固定m0，既存输出复用 |
| FLOW_GLOBAL | 新FLOW | 同一R12共享Gaussian |
| GMM_CONDITIONAL | R11固定K8 | 新条件m/scale网络 |
| FLOW_CONDITIONAL | 新FLOW | 同一个条件m/scale网络 |

R11 FACTOR_PCA_GMM的旧LW版作为额外固定参照；不重新拟合它。三个新臂各生成2000条，共6000条。沿用R11 FACTOR实际head_white/tail_white及draw位置：FLOW把同一head_white当标准正态base而不使用GMM component；两个条件残差对照必须各自共享完全相同的head和tail_white。不得拒绝重抽、换无效事件或夹断生成的q/PSD来改善指标。保持原14bit ADC、480点输出、无追加CFD裁窗。生成非有限、异常q、削顶、QC失败均保留计数；严重数值失败停止受影响臂而不是隐藏。

保存两个网络、固定L及表示引用、实际输入、各新臂pre/post ADC。旧模型与输出不可覆盖。head改变的比较是两种联合密度的比较，不能说共享了相同物理head；tail改变的比较才具有固定head的严格配对。

## 必要数值检查与完整评价

- 对4个固定val位置检查新head forward/inverse与logdet抵消（float64误差≤1e-8）；在1个固定位置用完整34×34自动微分Jacobian核对解析logdet（绝对误差≤1e-6）。检查条件tail的Gaussian公式及初始化identity，不再重做R11整套表示审计。失败先修实现，计入本轮预算。
- 四组加旧LW参照与REAL–REAL，共最多36个固定6视图HGB。使用原R7训练600/留出250位置池，按全部候选共同QC取同N，不桶外补样；每域训练不足150或留出不足75则unsupported。完整人口及QC统计另报。
- 保存留出概率，500次同draw/REAL人工组bootstrap。报告GMM条件−GMM共享、FLOW条件−FLOW共享，以及两种tail下FLOW−GMM四个预定BA/AUC差；区间条件于固定分类器，不含网络重训/独立采集不确定性，不做事后挑视图。
- 共用既有总体电荷–PSD、4电荷分区、基线RMS/ACF/谱、prompt/postpeak曲率与条件波形、8窄电荷区间峰位/原宽/去趋势宽定义。新3臂最多24个描述槽/24次EM；REAL/R11/R12旧表复用。逐臂与REAL列出实际共同合格bin，再列四组共同支持；不同支持的平均值不能直接比较，交集为空明确undefined，失败槽保留。不由旧失败臂把新臂逐箱证据隐藏掉。
- 三个新臂各全2000query对完整2592 TRAIN作V和own-shape最近邻，旧REAL/R12结果复用。无精确拷贝不证明无记忆，较小距离也可能是收缩。补充记录已存在的cross50/peak位置统计；本轮不加新的对齐干预。
- 在独立进程从保存权重和实际输入重放全部新6000条；不重新fit、随机重抽或重复全评价。pre-ADC最大绝对差≤1e-10V，post-ADC要求array_equal；任何差异如实交付。主评价、数学检查、训练和重放都计入数值预算。

## 交付和后续

只需config/PROTOCOL、运行/资源/数学检查与重放回执、紧凑指标、短REPORT/REPRODUCTION、6–8幅实际目检的关键图和一份最终版本清单。优先复用R12绘图和统计函数，完整图外点及失败支持都保留；源、逐事件载荷和模型只E/D，Git由root处理。首个科学PID/UTC、重大异常、关键结果和完成封存回传root即可。

本轮回答主要密度灵活性与条件残差分别解释多少剩余差距。所有四组都报告；最低验证NLL只是选型证据之一，完整分布仍需共同评价，不能将接近0.5的单个BA当成等效。如果显著改善，root再冻结候选并推进G4输入条件与任务效用；若仍失败，按明确残差选下一条有界路线。R13完成不关闭整个项目目标。
