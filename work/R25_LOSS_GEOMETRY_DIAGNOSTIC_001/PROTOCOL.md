# R25 固定诊断

本轮不拟合响应、不选择模型、不更新G/D/reader。只读R23已缓存TRAIN与实际流，及R21起点、R23/R24同步末点；所有旧阶段不可变。DEV/EVAL/CAL波形访问为0。

A复用原Torch float32投影定义、中心/尺度和投影列。每Q区投影均值仅作受控特征诊断，以float64积累后转回float32。四个c共享实际X/Y抽样；c=1直接使用原投影行，避免一次算术roundtrip。原128分位数L1距离按原float32实现；energy U用float64积累绝对差，三个对角线均剔除。统计组合为2×(physical+pooled_shape)+三fine平均。MIXED每行独立均匀Q区，HOMOGENEOUS每区32重复，共各256批对；完整整数PCG64种子20261009251/252。95%区间使用1.96×MC标准误；HOMOGENEOUS方差为各区均值方差之和/64。配对差先逐批相减，再按相应方差计算。区间仅条件于固定TRAIN经验池，不涵盖新采集、训练或多重比较。

B严格复用R23实际消耗的前1024source/REAL/bin索引和96行NCE位置。z/white/style由R23保存的准确PCG64初态按原逐批调用顺序重建，实际数组保存一次给三个状态共同使用，不重新抽样。三状态均STANDARDIZED critic，原key.detach、identity全零随机量、同Q的PSD排序、原物理/fine定义完整保持。10项系数按task：1/1/2/.5/2/5/2/1/1/1。D参数冻结而D输入梯度保留；G/reader原train模式，无BN/Dropout状态修改。没有optimizer构造/step。每状态32批，逐项autograd.grad保留计算图，各项梯度加和；第一批另求总损失直接梯度一次，记录float32误差与atol1e-5/rtol1e-4核对。零范数余弦为undefined/null。

G分encoder、NCE_project、decoder_mean、style_FiLM、electronic；reader独立。范数/点积/余弦均针对原加权梯度；clip5只报告假定缩放，不实际裁剪。负点积对应负总梯度方向的局部一阶增项，不等于Adam有限步更新或唯一失败机制。共同起点D未学习，不能代表末点D。保存参数在诊断前后逐值核对没有变化。

一项CPU作业做A，再一项CUDA作业做B，最后仅汇总和绘图；全部计入600秒。沿用R23实际science PID导入前nonce握手/RSS监督，不重做探针。4线程，真实RSS/GPU allocated各4GiB，累计输出128MiB；13:02:24.088820UTC前封存。没有SSH、安装、新G4、长波形解码、新HGB/EM和额外响应生成。结果不触发本轮新配方。
