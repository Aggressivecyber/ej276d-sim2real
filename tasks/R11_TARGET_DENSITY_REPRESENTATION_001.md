# R11：非对抗目标分布与可逆表示基线

本轮来自用户已设置的整个项目持续目标，根代理已选定方向。既有执行器连续实现、调试、计算、绘图和封存，无中间许可。上一轮配平后shape BA仍0.9938，固定源随机模型分布收缩；本轮检验一种明确不同的非对抗概率模型，同时把表示重建误差和分布学习误差分开报告。不是再次修改GAN系数，也不是最终G4响应映射。

## 范围与预算

- 本地新目录work/R11_TARGET_DENSITY_REPRESENTATION_001；远端本项目根下新R11目录，禁止覆盖R10。可只读复用远端R10的inputs和已保存源码/结果，按需正常复制；不重做来源审计、迁移哈希或安装。
- 从R11_REGISTRATION_001/RECEIPT.json的UTC起90分钟，最后3分钟留给根代理；累计数值1800秒，拟合软预算1200秒，失败/必要科学自检/重放均计。CPU4、RSS4GiB、阶段512MiB、单作业。这是NumPy/sklearn概率模型，可用服务器CPU；不为使用GPU改算法。
- 只用已有REAL TRAIN-fit2592、内部val599、DEV2000及同样本量TRAIN参考、既有固定CAL定义/8个电荷区间、R7评价分组和R3B/R10历史参照。EVAL、新G4、旧阶段写入均0。未知/重叠全保留。
- 两种预定表示，每种3个GMM组件数，共最多6次GMM fit；每次n_init=1、固定seed、max_iter=300、tol=1e-3、full covariance、标准化空间reg_covar=1e-4。不追加组件/种子/正则扫描。非收敛结果保留，但不参加本表示选型；全部失败则该表示undefined。

## 两种表示与精确逆变换

全部均值、PCA、尺度、回归与协方差只在TRAIN-fit拟合。采用float64，PCA/SVD使用确定性的full分解；固定生成主维数k=32。诊断8/16/32/64/128及全维的TRAIN-fit/val重建误差，只用同一全基展开，不据DEV改k。

### RAW_PCA_GMM

对480点原物理电压直接拟合全PCA。前32个系数为head，其余448为tail。head按TRAIN均值/标准差标准化（纯数值scale下限1e-12，记录受影响项），tail保留物理电压单位。全维逆变换保留全部480点，不把低秩无噪声重建充当完整输出。

### FACTOR_PCA_GMM

采用原共同积分：b=mean(V[0:100])，q=sum(V[100:400]-b)>0（V·ns），s=(V-b)/q。合法TRAIN/val样本须满足q>0；如有不满足，保留计数，不私自换角色或定义。

定义固定约束a为前100点平均、w为100:400求和减300a，因此a(s)=0,w(s)=1。取s0在100:400等于1/300、其余0，取Z为a与w共同零空间的确定性正交基（480×478）。在(s-s0)Z上做TRAIN-fit全PCA，得到480×478的U、均值m_s。这样数值逆变换保留a=0,w=1，不靠生成后强行配回电荷。

head=[log(q), b, normalized-shape前32个PCA系数]，共34维，按TRAIN均值/标准差标准化。tail为其余446个shape系数乘以本事件q，保留物理电压单位。逆变换：V=b·1+q(m_s+U_head z_head)+U_tail r_tail。

这是可逆的数据坐标，不是物理模型。全维TRAIN-val重建应只存在浮点误差；保存最大误差及基线/积分约束残差，工程错误可在预算内修复，不增加模型搜索。低秩重建用于定位被删信息，不作为随机生成成功。

## 密度、选择与采样

每种表示分别在标准化head拟合组件数[1,4,8]的GMM。固定基础seed2026100911，组件/表示seed由预先明确的确定公式派生，不选种子。

每种表示共用一个tail条件模型：r_tail = intercept + head_std·B + epsilon。TRAIN上带截距的ridge最小二乘，系数惩罚1e-3、截距不惩罚；epsilon用LedoitWolf收缩协方差，再加固定1e-12 V²对角数值底。保存回归、收缩系数和Cholesky等模型参数。tail不按组件重拟合，不抽取训练波形或训练残差模板。

每个GMM记录全部599内部val记录的连续完整波形近似log-density：head score_samples减head标准化log-Jacobian，加tail条件Gaussian log-density。RAW正交逆变换绝对行列式1。FACTOR须再减[(k+1)log(q)+log|det(1,m_s,U_head,U_tail)|]；后一个常数可用固定480×480矩阵slogdet求得。解释推导并做一次必要的逆变换/密度数值自检，不将不同维度的裸head NLL直接比较。该分数是量化样本的连续密度代理，不是精确ADC格概率。

每种表示按TRAIN-val平均完整NLL选择唯一收敛模型，严格较小优先；完全相等取较小组件数。两种已选模型都保留，DEV不得选择组件、k、协方差或尺度，也不因输出不好删臂。全维NLL和重建诊断均报告，不能用较低NLL自动认定统计等效。

每模型生成2000个新随机draw，采样GMM head及独立Gaussian tail创新，经保存逆变换得到pre-ADC480点，再采用旧WholeResponse的[-0.5,0.5]、14bit round/clamp生成评价波形。保存pre/post、裁剪/失效计数和实际采样输入；不重新做CFD裁窗，因为建模对象已是保存的对齐480点。这个输出阶段差异要声明，不能把与R8C的差异只归因于损失。

随机draw不是G4父事件，无原事件配对、无物理转换器称谓。不得拒绝采样后按结果重抽补足，不从REAL波形或最近邻模板生成。

## DEV评价与记忆边界

- 两臂各6个既有视图HGB，另6个未加权REAL TRAIN-reference–DEV对照，总上限18；参数沿用原定义。共同使用原600/250位置桶，在两生成臂合法位置交集内筛选，再确定性截到各域相同N，不从桶外补样。每域训练<150或留出<75时记unsupported。
- 500次源draw/REAL人工组bootstrap，条件于已拟合模型与评价器，非训练/独立采集不确定性；完整原始人口QC/失效/保留率单列，不把筛后指标冒充全人口。
- 全体QC有效电荷–PSD/波形/前沿/局部曲率/基线RMS、ACF及频谱沿用现有定义。两臂各8个固定电荷区密度描述槽位，最多16槽；实际EM次数与槽数分开。复用原REAL、R3B、R10的已有描述，不重拟合旧模型。仅共同支持区比较峰位/原宽度/去趋势宽度，并显示未支持区域。
- 对完整2592 TRAIN参考做两臂各2000query的V和own-shape最近邻，与同N REAL DEV参考比较；记录精确复制/极近邻及距离分布。小距离可能是收缩，不能证明无记忆。固定哈希选择少数例图，完整逐事件索引只留本地。
- 固定一组主要图：TRAIN-val重建误差随秩、6次拟合及NLL、共同坐标电荷–PSD、C2ST含REAL–REAL、条件峰宽和支持、波形/前沿/噪声、最近邻。轴限可比、失败点不隐藏。

## 必要重现与交付

新独立进程从两个已存模型参数和实际随机采样输入重放全部4000输出，不重训、不重新抽样、不重新拟合指标。比较pre-ADC使用明确浮点容差、post-ADC要求数组一致；任何差异保留实际规模和原因，不通过改变已保存主结果“修成功”。这属于科学推理重现，迁移不做一致性测试。

保存简短config/PROTOCOL、真实计数与资源/失败/重放回执、模型、实际输入、完整私有指标、仅聚合METRICS_COMPACT、REPORT/REPRODUCTION和实际目检图；一个最终版本清单足够。必要源/输入版本直接引用旧绑定，不另建多层读取台账。远端启动及时报实际PID/目录/时间，未知远端状态不重提。根代理负责Git和D盘交付，执行器不操作Git。

本轮完成后根代理按下列证据决定下一轮：重建限制明显则先改表示；重建好而概率模型分布差则定位具体相关/非Gaussian/条件噪声结构；目标分布明显改善则再设计G4条件接入。任一分支均需本轮证据，不自动认定PCA/GMM成功，也不把本轮封存当整个项目完成。

方法/API参考为官方文档：[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)、[GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)、[LedoitWolf](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html)。网页当前版本与服务器1.8.0可不同，实际调用以现有环境记录为准，不升级环境。
