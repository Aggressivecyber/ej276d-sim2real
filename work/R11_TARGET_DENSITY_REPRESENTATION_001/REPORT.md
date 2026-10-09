# R11：可逆目标密度模型仍未达到完整响应分布相似

本轮连续执行已完成，两个固定模型、6 个单次 head GMM、18 个 HGB 评价和独立保存后重放全部正常退出。可逆表示没有损失完整波形信息；独立生成的电荷/PSD及局部前沿明显比旧目标只固定源模型更合理，但完整波形仍高度可分，不能作为项目完成或 G4→真实响应已解决的证据。

## 固定数据与表示

只使用同一次真实采集：REAL TRAIN 的 3200 条中 3191 条 QC 有效，固定人工组划分 TRAIN-fit2592 / 内部 val599。PCA、尺度、ridge、LedoitWolf 和 GMM 都只在 fit 数据估计，K 只按完整599条 val 连续 NLL 选择。DEV2000只报告。CAL 波形解码0，EVAL读取0；已冻结 CAL 电荷边界和高置信参考、8个描述区间原样继承。人工50条组和新随机draw不是新的独立采集。

RAW 使用完整480维原V PCA，head32 / tail448。FACTOR 使用 baseline b、positive signed charge q 和478维受约束 shape坐标；head34=[logq,b,shape32]，tail446 为余下shape系数乘q、保留V单位。完整反变换 V=b·1+q(ms+Uhead·z)+Utail·r。rank32重建只作投影诊断，真正生成包含完整 Gaussian tail。

每臂用一个共享的条件线性 Gaussian tail、固定ridge1e-3、LedoitWolf加1e-12V²；head GMM仅K=1/4/8、一次初始化、最多300迭代。FACTOR完整连续log-density含head尺度和33logq+log|det(1,ms,Uhead,Utail)|，RAW正交变换绝对行列式1。两者均为连续密度代理，不能直接当ADC格点概率质量。负NLL来自V坐标的连续密度单位，不等于错误或等效证明。

新生成每臂2000条独立密度draw，不携带G4父事件，不使用真实事件模板/拒绝重抽。pre-ADC float64，旧14bit[-.5,.5]V round/clamp后保存float32。没有额外CFD裁窗，因为输入/表示已在对齐后的480点上建模；这与旧R8C的完整512点再裁窗处理有区别。

## 重建与连续密度选型

| 表示 | 所选K | 完整val NLL nat/event | rank32 val RMS mV | rank32 PSD重建MAE | 全维val最大误差V |
|---|---:|---:|---:|---:|---:|
|RAW_PCA_GMM|4|-2379.94|2.20458|0.00984244|4.996e-16|
|FACTOR_PCA_GMM|8|-2373.85|2.23626|0.00145259|4.44089e-16|

6个候选均收敛；所有599验证行都得到有限完整密度。RAW K4比K8仅好约0.012nat/event，严格按已冻规则保留K4，没有因为DEV改变选择。FACTOR K8最优，但同V坐标完整NLL仍弱于RAW；FACTOR的PSD重建/生成边际更好不改变该事实。

完整逆变换、Gaussian tail公式、head mixture公式及FACTOR Jacobian自检均通过。固定4个val位置只用于数学自检；完整val重建误差已单独统计。FACTOR Jacobian logdet公式差0，tail密度与独立SciPy Gaussian公式差≤1.37e-12nat。有限差分检查也保存，最大导数差约9.4e-9，不作为新增选型。

## 独立生成：整体仍失败

| 指标 | RAW | FACTOR | REAL–REAL采样参照 |
|---|---:|---:|---:|
| C2ST full_historical_shape BA |0.947791|0.945783|0.512048|
| C2ST full_voltage BA |0.96988|0.985944|0.546185|
| C2ST prompt_local BA |0.722892|0.664659|0.48996|
| C2ST postpeak_local BA |0.7751|0.740964|0.46988|
| C2ST tail_local BA |0.925703|0.935743|0.48996|
| C2ST baseline_local BA |0.588353|0.684739|0.52008|
| PSD W1 |0.0196218|0.00926155|本轮未新计算该标量参照|
| Qtotal_mVns W1 |195.402|113.47|本轮未新计算该标量参照|
| rise_10_90_ns W1 |0.119121|0.177126|本轮未新计算该标量参照|
| baseline_RMS_V W1 |0.000298005|0.0003665|本轮未新计算该标量参照|

C2ST严格使用原R7位置，先取两新臂有效draw位置交集，再与REAL/REAL-TRAIN-reference共同定量截取：600训练、249留出每域。RAW有1条原留出位置无效，因此未用桶外替换事件；实际全人口RAW1994/2000有效、FACTOR2000/2000，原人口统计均保留。500次区组bootstrap仅反映固定分类器条件下的抽样波动。REAL–REAL是同N的TRAIN-reference与DEV比较，不能当物理等效界限。

FACTOR的PSD W1约0.00926、Q W1约113.47mVns，边际覆盖比旧R10固定源明显合理；RAW约0.01962/195.40。完整shape BA仍约0.946，完整V约0.986（RAW约0.948/0.970）。仅匹配PSD、均值曲线、rise时间边际无法证明完整响应相似。旧R3B/R10是历史设计/环境参照，不是本轮严格配对训练对照。

剩余差异有清楚位置：baseline RMS均值REAL0.646mV、RAW0.940mV、FACTOR1.013mV；RMS KS约0.940/0.946。baseline_local视图按本区尺度归一化，本来就不包含该绝对RMS幅度，所以其较低BA不否定这个差异。tail局部BA约0.926/0.936。峰后曲率随电荷的中位数变化在实测约2.5→6.2mV，新模型约4.6→4.9mV，固定Gaussian残差没有恢复实测的条件涨落跨度。prompt100:150包括峰/早期下降，不能把该窗口曲率差全部称纯上升沿问题。

## 条件峰宽与未知项

8区新描述槽共16个，实际EM也16次。RAW仅bin7、8满足冻结的双成分资格；FACTOR0/8满足。未收敛、分离不足、有效成分数不足均保留，不能将不合格参数作为两峰真值。两新臂与REAL共同合格区间为空，因此共同峰位/原宽/去趋势宽度MAE全部undefined；没有拿不同支持区间的平均值作优劣判定。历史REAL/R3B/R10表原样复用、未重新拟合。

这意味着本轮没有复现实测两峰随能区收窄的完整合格证据；不合格描述拟合本身也不能证明物理两类不存在。两成分与高置信伪标签都不是独立粒子真值。旧CAL q4参考失败仍全unknown/FoM undefined；q1高置信neutron-like仍0，没有补参考或fallback。

## 最近邻与复现

每种新生成与REAL DEV各2000条query，完整2592条TRAIN-fit作参考，在V和own-shape两个视图均2000可定义。无重抽/缩减；无完全相同最近邻记录。V最近邻RMS均值REAL4.100mV、RAW4.396mV、FACTOR4.389mV；shape均值约0.02643/0.03108/0.02963。少量固定hash例图可看出类似形状，但不是配对真值；无精确复制不能证明无记忆。逐query距离/邻居索引、选择和实际随机输入留本地。

新进程从保存参数及component/head_white/tail_white重放全部4000条，无重新fit、无完整评价重跑。pre-ADC与主保存bitwise相同、最大差0，post-ADC array_equal通过；共1920000个点。该证据是科学模型重放，不是迁移测试。用户要求的普通传输执行正常exit0，没有源/目的内容一致性检测。

## 预算与可复现交付

服务器主worker134962 native0、16.430760s；新进程replay135146 native0、0.803527s。合计17.234288s/1800s，fit阶段3.928613s/1200s，观测RSS峰1051115520B/4GiB，4CPU线程。无工程失败/训练重试；Matplotlib Axes3D导入警告保留，所有图为2D并已实际目检。无EVAL、新G4、响应模型训练、安装和Git操作。

实际Python3.12.3/NumPy1.26.4/sklearn1.8.0/SciPy1.11.4。各模型representation.npz、所选GMM参数、GENERATION_INPUTS及pre/post生成均保存。metrics.json完整本地；METRICS_COMPACT仅聚合、没有逐事件身份/预测/邻居/噪声载荷。REPRODUCTION.md说明原路径和新目录恢复入口。READINESS.json绑定实际执行源码，后续控制报告代码仅进入最终清单。

## 下一轮方向建议

全维逆变换误差仅浮点级，现有证据没有指向丢弃完整坐标；rank32单独投影仍有约2.2mV误差，但本模型已用完整tail，不能把所有失败简单归为维数不足。目标分布在该坐标系下可以表达，却没有被这套GMM+共享线性Gaussian充分学到。优先检验tail创新的条件尺度/时窗结构：当前全波形物理单位的共享LedoitWolf收缩可能向基线分配信号残差方差，而同一协方差不能表达q/shape相关的峰后涨落范围。这是基于结果的待检验假设，不是因果结论；仅缩小baseline噪声也未必修复tail联合分布。下一轮应在TRAIN/val定义结构化条件残差密度，并保留本轮两个固定参考，完整评价而不最大化FoM。只有目标密度充分接近后再讨论如何绑定G4条件；本轮不是G4输入映射且不能据此评价G4物理正确性。

API依据：[PCA官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)、[GaussianMixture官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)、[LedoitWolf官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html)。网页版本可能不同，实际执行环境和保存源码为准。

补充图 charge_psd_main_body.png 在全部五域使用共同PSD[0,0.6]和共同电荷范围，原全范围图保留。图外点REAL/R3B/R10均0、RAW41（低于0的30、高于0.6的11）、FACTOR8（全部低于0）。该图只使用已保存特征重绘，没有重新计算指标或拟合。服务器绘图PID137126，native0，自计0.960703s；按整个传输/命令3.112239s保守记账，累计数值/绘图预算约20.346526s。
