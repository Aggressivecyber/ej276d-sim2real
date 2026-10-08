R8固定配对方案：JOINT、JOINT_SIGNAL；共同R3B初始化（bb45da6e...）、原全部损失、三区未平滑局部SW额外系数1；过程开关是两臂唯一科学配方差别。TRAIN fit2592/val599和G4缓存原internal_training划分，DEV不选点，EVAL/SSH/新G4为0。

局部向量[logQ_mVns,PSD,logpeak_V,loglocalRMS_V,z,dz,d2z]，本区均值/RMS，RMS下限1e-6V；TRAIN fit总体均值和type7 IQR一次固定；4标量IQR下限.1/.05/.1/.1，其余.05。各块除sqrt维度，各区64 Gaussian L2单位投影(seed2026100882)，排序batch平均绝对差，三区等权。此训练联合向量有全局条件标量，不能称R7严格局部分类输入。

保留原8维边际/.35平滑SW/.08协方差/4时间点边际、.2弱组成、.002时间系数平滑/.0001原mean/random正则、4条件CDF/1占比；旧全局投影按R3B原seed2026100836独立构造并保存。内部验证沿用2026100821的原父/真实384选序、z/white；signal white以独立2026100886固定。条件CDF fit2592及完整val599目标分别保存，局部尺度仅fit。验证值不包含原composition/regular/temporal罚项，与旧选择口径一致。

所有模型构造前Torch/NumPy2026100881。每臂训练Generator重置同2026100881，并按ii,jj,z,baseline white,signal white相同顺序每步抽取，即使过程关闭亦抽signal white。初始可学习参数与全部buffer逐位核对；JOINT新增未用参数没有梯度且保持初值。Adam lr.0005/batch256/2500上限，每100验证incl0；每臂900秒硬timer，接近墙钟留5秒用于末次验证/保存；每个有效checkpoint/失败全部保留，不DEV补训。

eta=.02sigmoid(b+w*c)*sqrt(max(clean,0)/.1V)*[1-exp(-max(t-first_argmax(clean),0)/10ns)]*xi；峰位整数detached。xi为3抽头数学卷积，核[.5tanh(a0),1,.5tanh(a1)]按L2单位化，零边缘；a/w零初值，b=logit(.1)。sqrt在clean=0处采用精确零且零梯度分支以避免0*infinity，正值公式不变。此为统计假设，不能称光子数/硬件已标定机制；原平稳噪声/ADC/CFD完全保留。

主2000使用R3B实际保存main z/white（从其绑定RNG恢复），signal独立2026100883；两额外视图2026100884/85顺序抽z/white/signal且共用父及两臂，保存全部实际输入。附加随机视图不是独立真实事件。原始R3B已保存三视图直接复用，不重复旧推理。

评价共同signed charge/PSD和冻结CAL后验0.99；q4参考失败无fallback，不筛unknown。8窄区TRAIN冻结，旧REAL/R3B描述EM结果复用、新臂各固定8区单初始化；未收敛/floor/支持不足均undefined。宽度误差必须用REAL/R3B/两臂共同合格bin集合并列缺口。源名义Qsource*1000四区的全部父与输出自身四区双报告、迁移和条件关联保留，不把输出分区当固定父条件。

12新HGB（每新臂6固定视图），与R7同600/250父/组索引和分类尺度规则。R7旧GEN、阴性TRAIN–DEV及阳性控制复用；若新臂在固定选择上无效不悄悄换样本，报告支持失败。500簇bootstrap、两新臂共测试父/REAL组配对差异；条件于固定评估器，不含响应训练不确定性。局部分类本区demean/RMS，不夹入全局峰/PSD/rise；源总体QC条件另披露。

一个本地科学作业依次训练/生成/评价/绘图；独立进程仅保存权重+全部主输入重放两臂各2000x480，逐位及max差/父顺序，不重训不重评。4CPU/4GiB/512MiB/累计2400秒，阶段从START_RECEIPT到120min。系统依赖装载不作原数据字节账本；日志、实际进程/退出/RSS/输出大小明示。
