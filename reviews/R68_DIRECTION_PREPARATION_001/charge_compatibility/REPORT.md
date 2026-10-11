# R68 静态审核：原电荷约束与人口兼容性

结论：**当前有效 A–F 同时要求源 Q/PSD 秩保真，以及各事件 preADC 电荷对原输入的最大相对误差 ≤1e-5。不能只保留秩后更换目标电荷或分母。** 这属于继承的数值/方法不变量，并非已证明的逐事件物理对应关系；postADC 不要求电荷逐位等于 SOURCE。

## 原条款及实现证据

以下路径相对项目根 E:/GAN/research/sim2real_two_papers_20261008_0305；行号为本次只读内容的物理行。

| 证据 | 精确定义与含义 |
|---|---|
| PROJECT_ACCEPTANCE.md:15,19–24,36；PROJECT_ACCEPTANCE.json:36,76,100–113 | 同一冻结方法须满足 A–F，C 明确继承全部原六门、preADC 电荷 1e-5；原约束没有被“只保留秩”替代。 |
| tasks/R43_SCALAR_TIME_NEURAL_FLOW_001.md:23,31,42–46 | q=sum(x[100:400])−3sum(x[0:100])；切空间保电荷；数值电荷检查及全部主输出门仍强制。第3门另外规定源 Q/PSD Spearman 相对 BASE 下降 ≤0.05。 |
| work/R43_SCALAR_TIME_NEURAL_FLOW_001/source_flow_core.py:5–14,39–51,59 | 原输入先按 float32 存储，再以 float64 求 q；e64=abs(q(out64)−q_input)/q_input，e32=abs(q(out32转f64)−q_input)/q_input。分母无下限替换；正电荷不合格不会删行。 |
| work/R43_SCALAR_TIME_NEURAL_FLOW_001/check_train_solver.py:12–13；work/R49_FROZEN_HEUN128_RESPONSE_001/check_models.py:19–30 | N/2N 的逐行电压 RMS 差与上述原输入电荷残差是两项检查。电荷门不是 abs(q_N−q_2N)/q_2N。R49 聚合两解的 f64/f32 原输入残差。 |
| tasks/R49_FROZEN_HEUN128_RESPONSE_001.md:17,23–29；work/R49_FROZEN_HEUN128_RESPONSE_001/source_flow_core.py:40,47–55 | R49 的目标仍为 bins_for 从自身输入求得的 q，SOURCE 和同处理 REAL 各保留自己的原输入 q。128/256不改变目标电荷。 |
| work/R43_SCALAR_TIME_NEURAL_FLOW_001/summarize_and_plot.py:103–110,123–131；work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/summarize_and_plot.py:28–35,89–98 | gate6 还要求全部 SOURCE/同处理 REAL 主输出 f32 preADC 残差 ≤1e-5；postADC 误差只描述，明确 postADC_not_claimed_exact。 |
| work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/source_flow_core.py:13–30,64–66,73–85；numerical_check.py:24–39 | R63 的 Box/Charge 投影目标仍为原 q_input；加范围投影没有授权另定电荷。数值门仍相同。 |
| work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/evaluation.py:83–87 | 评价电荷是减去前100点基线后的100:400积分，再乘1000，等价于同一 q 的 mV ns 尺度；不能换成另一积分窗或目标。 |

PROJECT_ACCEPTANCE 表格将“源信息保真”写成秩/组成等条件，没有独立声称精确电荷是普遍物理定律。但 C 继承的六门包含上述可执行电荷检查，因而**当前验收操作上不能放弃原输入电荷不变量**。单调 Q 分位数映射在数学上可以保留秩，却会破坏原输入电荷门；不是本次已授权的补救。也不能把新电荷写回 source_q_Vns 再报原门通过。

## 哪些兼容性结论成立

- R45_GOAL_COMPATIBILITY_NOTE_001.md:5–9：封存2000 SOURCE/REAL 的电荷经验 KS=0.2685、W1=1842.558452 mV ns。若严格逐事件保电荷及权重，则该**固定经验** Q 边缘不变，联合 Q–PSD 不可能严格相等。实际 1e-5 与 ADC 舍入需分开处理，不能声称 postADC 严格不变。
- 同文:9：在同一完整样本上事后选双方向最佳 Q 阈值，描述性最大 BA=(1+KS)/2=0.63425。它不是原固定 HGB 的留出 BA/分组区间，不能证明 HGB 的区间必超0.60；有限经验人口也不直接等于未知采集总体。
- 同文:11；PROJECT_ACCEPTANCE.md:26：共同确定性可逆映射在严格等分布下有反推逻辑；当前噪声、ADC、条件化和有限视图未被证明符合该前提。近似固定评价器门不等于任意判别器不可分。
- R58_REPORT.md:3,9–11,26,42,49,51–73：6593源等权/21047 REAL 保留的有限联合 PSD 族有数值可行解；没有通过全窗、密度峰宽、噪声或 A。2.9210% 不可实现质量仅针对该保存联合律、已观测 REAL 基线范围及 TRAIN 人口；自由 PSD 电荷硬界未排除源。它不能证明每个原源、所有联合律或原 DEV/完整模型均不可行。
- R64_REPORT.md:71–73：B_V=0.146816248 是该浮点端点与固定参照/标准化下的名义电荷均值下界。V-only 比值0.243179632与四视图几何比值0.100123179分母不同；均非因果归属比例，更不把 W1/坐标损失转换成固定 HGB BA95CI 的下界。R64 的 B128 梯度阴性也限于其固定参照与32宏组设计。
- R63_REPORT.md:3,9–15,36–43,52–53,119,129：同一冻结候选源秩与数值门通过，A/B/C仍失败；证明该候选未达标，不证明所有满足原 q 约束的方法失败。

不得改原2000父ID/组成/顺序/权重、删417支持外、换电荷区或隐瞒总体失败（PROJECT_ACCEPTANCE.md:21,32；R45说明:13）。现有报告足以确认电荷人口是残差来源之一；重算同一 KS、重复 R58 LP、重复 R64 下界或再扩大该梯度批量，均不会产生 A–F 普遍不可行证明。

## 唯一建议的低成本 TRAIN 诊断（仅设计，未登记/运行）

做一次**原电荷容差的可达分布包络**，不加载模型。输入只用已保存合法 TRAIN 的6593原 SOURCE q标量、21047 canonical REAL q标量、父ID/原十区及原权重；不解码波形、读DEV/EVAL/CAL、推理、训练、拟合HGB或LP。缺少准确标量则报告输入缺项，不重定义 q。

令 ε=1e-5；逐父 preADC 可达电荷区间为 L_i=q_i(1−ε), U_i=q_i(1+ε)。共同 ADC 分支另外扩大经原量化实现验证的绝对误差界 Δ_ADC；理想无裁剪均匀量化的电荷界为300/16383（q原单位），实际 float32 运算及末次cast的保守界另列，不能把理想界当已验证实现界；无法证明时该分支记 unknown。保留原分母和 q目标，包络不是放宽验收。

一次排序/扫描计算 F_low(t)=sum w_i 1[U_i≤t]、F_high(t)=sum w_i 1[L_i≤t]，以及固定 REAL F_R(t)。报告
kappa=sup_t max(F_low−F_R, F_R−F_high, 0)，以及 integral distance(F_R,[F_low,F_high]) dt 的 W1下界。总体与原十区全部保留；不用输出区重选或改变原人口质量。

该结果区分“允许的数值漂移/量化仍不能消除的电荷经验差异”与尚可达部分。即便 kappa>0.20，也只保证完整固定经验样本存在 Q阈值描述性 BA>0.60，**仍不是固定训练 HGB、留出样本或其95%区间的失败证明**。kappa≤0.20也不是 A/B/C可行证书。仅一次报告，无阈值/窗口/分母扫描。

R65退休与成本hold维持；根提供的 R67 full20/mini979 等成本 score 5.928/4.546 尚不支持 fullD收益，本审核没有读取或重算该阶段。本诊断需由根另行决定登记执行；本次只交付静态审核。完整 A–F 仍未完成，局部阴性不记总体阻塞。

## 对根新增 TRAIN 标量证书草案的静态意见

根拟从 R64 ENDPOINT_GEOMETRY_PRIVATE.npz 仅读取 SOURCE_original_q / REAL_canonical_q，以6593/21047全人口等权，构造闭区间 q_i±1e-5*q_i±600/16383±1e-7；本审核没有打开该数组，字段语义由根已核的 endpoint_geometry.py:375–382提供。

边界可采用，但证书类须包含**量化输入f32 preADC有限、逐父原q门有效、ADC没有饱和裁剪、使用原R63 canonical链**。仅有保电荷而允许clamp时，量化偏差不受该界。600为q线性系数L1；600/16383是ADC后保守误差，不是允许preADC改变电荷的新容差。另列1e-7的浮点求和/存档q算术余量。

原R63 source_flow_core.py:41–47给出可静态证明的余量：设M=16383、p为范围内f32量化输入，首次缩放误差≤M*2^-25+2^-11<0.001个code；round后电压误差≤0.501/M。legacy码电压的f32除法/减法表示误差≤2^-25+2^-26 V，再编码距首次整数code小于0.001，故canonical恢复该code；最后f64舍入余量仍远小于剩余0.499/M。于是每坐标 abs(canonical−p)<1/M，600/M的q界成立。此论证绑定e32与原实现，不能仅用“两次各半LSB”口头相加或忽略裁剪。

闭区间使用 L(t)=#upper≤t/N、U(t)=#lower≤t/N 正确；REAL也须用≤的右连续CDF。全部唯一精确端点及REAL值覆盖分段常数的sup，显式初始/终态；相同值跳跃必须同时结算。若另报t−，所有计数统一为<，不混用端点策略。

直接整数交叉相乘：D=max(cUpper*N_R−cReal*N_S, cReal*N_S−cLower*N_R, 0)，LB=D/(N_S*N_R)；判定LB>0.20用5D>N_S*N_R。q_i取原f64值的精确Fraction；600/16383用精确有理数；epsilon宜绑定Fraction.from_float(1e-5)，或证明十进制常数与旧二进制常数微差被单列余量向外包住。支持值、单位与源q目标不得改变。

该证书只在上述全TRAIN逐父约束类内说明不可消除的有限经验q差异；无bootstrap或未知总体含义，不证明原2000 DEV固定HGB及A–F不可行。本次不注册、不执行该证书。
