# R37：二阶结构与完整分类改善，峰形和源组未保住

唯一TRAIN完整协方差校准使完整shape493/纯u480 BA从.978/.978降至.924/.918，并改善部分条件时间相关；低侧PSD峰位/宽度、基线RMS和纯had源组谷占比仍失败。按预定规则关闭固定高斯协方差配方，不调floor、秩、分区、方向或再叠保护。项目继续，完整响应尚未验收。

|同2000源父、同评价口径|BASE|R37|
|---|---:|---:|
|完整shape493 BA|.978|.924|
|纯u480 BA|.978|.918|
|完整电压480 BA|.982|.954|
|总体PSD W1|.02306546|.01550574|
|共同观察Q区1–7等权PSD W1|.03476580|.03039434|
|基线RMS W1，mV|.01935839|.03111536|
|基线ACF1 W1|.06423497|.03860939|
|actual-source purehad bin3谷比例，N213|5.6338%|14.0845%|
|actual-source purehad bin4谷比例，N181|.5525%|12.1547%|

对BASE的full493/纯u480配对ΔBA为−.054/−.060，95%区间[−.079205,−.029132]/[−.084645,−.034543]；对R35点值也改善。区间来自500次固定分类器区组重抽，不覆盖响应训练种子及独立采集不确定性。完整BA仍高；baseline局部.564、prompt .704等改善不能代替完整响应等效。

总体/共同Q区PSD距离下降32.78%/12.57%。但原有效密度区2–7内，低/高峰位MAE由.005857/.002659增至.035334/.008622，分别恶化503.25%/224.29%；低raw-width MAE .010393→.020122（+93.62%），高侧.013051→.007277（−44.24%）。raw-width为样本SD描述，非FWHM。原有效区全部仍有效，bin1未分离和bin8 N36支持不足保留。密度成分不是粒子真值。

purehad两区谷占比增加8.45/11.60个百分点，超过2个百分点限制。全体source-Q/PSD Spearman .999046/.408773通过相对降幅门槛，不能掩盖源组内变化。所有父ID、组成、顺序与权重保留，1583支持内校准、417支持外原G回退，无新增非有限、无效或ADC裁剪。实际source由已有EE/had总沉积派生；微观沉积时序未恢复，推定入射PDG不等于逐事件已知真值。

严格复用R33原94400条source/REAL/bin/ordinal/z-white/style与G preADC，七项流hash全部匹配；实际不同source/REAL TRAIN行3752/18739，94400不是独立事件数。新增神经权重更新0，唯一拟合16份float64协方差及8个正定高斯映射。16份原协方差均未触及固定特征值floor，正则/原矩最大相对闭合残差约1.903e−13/1.910e−13。

TRAIN数值闭合、DEV区内改善与DEV全人口结果须分开：观察Q区1–7的协方差差均改善；全人口协方差相对Fro误差.457723→.474279略差，而非对角相关RMS差.181458→.107835改善。该结果不证明高阶分布、事件映射或全人口协方差匹配，DEV矩阵仅用于描述。

9个固定边界全过。128个guide探针的150:400段仅加入零均值RMS1mV正弦，固定原bin和G随机流；全部输出改变，全480点preADC差RMS中位11.190mV，显示扰动被放大，不能解释成已验证的物理增益。最终signal/electronic分解unknown。独立CPU从完整源和原随机量重放主2000，所有保存中间量与pre/postADC逐值通过；推理不读取REAL库。RAW REAL–REAL旧参照继续报告，2D预测缺失保持unknown；单向源校准不声称对REAL恒等。

五门为通过/失败/失败/通过/通过。7个科学进程全native0，累计88.169105/1800秒，唯一统计拟合14.347015秒；真实观测峰RSS 1,709,957,120字节、GPU allocated 57,150,976字节、CPU4。两次RSS查询缺测原因unknown。新10HGB、8密度槽/7实际EM，旧端点零重复拟合，3次控制路径查找错误如实保留。11张最终图实际查看，2版布局旧图保留，根实际查看8张关键图。实际封存native0，274文件/521,325,612字节。

R38已登记一个不同问题：从原冻结G开始，用TRAIN逐层学习完整投影的单调分布变换，检验非高斯结构。固定8区、每区32层、每层16正交方向及257结点，一次拟合900秒/累计科学1800秒；无DEV选择、无参数扫描、无后保护。门槛继续同时约束完整波形、峰位/峰宽、噪声和源保真。

[完整报告](work/R37_FULL_COVARIANCE_CALIBRATION_001/REPORT.md) · [相关矩阵](work/R37_FULL_COVARIANCE_CALIBRATION_001/results/figures/DEV_full_correlation.png) · [峰位与宽度](work/R37_FULL_COVARIANCE_CALIBRATION_001/results/figures/energy_peak_width.png) · [源组保持](work/R37_FULL_COVARIANCE_CALIBRATION_001/results/figures/source_preservation.png) · [复现入口](work/R37_FULL_COVARIANCE_CALIBRATION_001/REPRODUCTION.md) · [R38协议](tasks/R38_SLICED_MONOTONE_TRANSPORT_001.md)。

实际私有Git与D副本状态见R37_PUBLICATION_RECEIPT.json、R37_BACKUP_RECEIPT.json。同次历史采集与反复DEV开发不能替代独立采集或完整项目验收。
