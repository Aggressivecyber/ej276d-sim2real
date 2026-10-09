# R11：表示可逆，生成分布仍失配；项目继续

两种目标密度模型已在服务器完成6个预定GMM拟合、18个判别评价、16个条件密度描述及4000条独立生成。完整坐标能在浮点精度内还原真实波形，但生成分布仍不满足完整响应要求。详见[阶段报告](work/R11_TARGET_DENSITY_REPRESENTATION_001/REPORT.md)和[关键结果判断](reviews/ROOT_R11_KEY_RESULT_REVIEW_001.json)。

| 指标 | RAW | FACTOR | 实测对实测 |
|---|---:|---:|---:|
| 完整shape判别BA |0.94779|0.94578|0.51205|
| 完整V判别BA |0.96988|0.98594|0.54618|
| prompt局部BA |0.72289|0.66466|0.48996|
| 电荷W1，mV·ns |195.40|113.47|本轮未新增该标量参照|
| PSD W1 |0.01962|0.00926|本轮未新增该标量参照|
| 基线RMS KS |0.940|0.946|本轮未新增该标量参照|

每域判别使用600条训练、249条留出，共同有效位置一致；RAW原始2000条中6条QC无效，FACTOR全部有效。内部599条验证按完整连续NLL选择RAW K4、FACTOR K8。rank32投影仍有约2.2mV误差，真正生成已包含其余全部坐标，因此不能将剩余失败简单归为只保存32维。

基线RMS均值约为实测0.646mV、RAW0.940mV、FACTOR1.013mV。峰后波动随电荷变化的幅度也未恢复。RAW仅第7/8格的两成分描述合格，FACTOR没有合格区间；两新模型共同峰宽误差未定义，不能由较小的总体PSD距离宣称峰宽成功。参见[噪声图](work/R11_TARGET_DENSITY_REPRESENTATION_001/results/figures/baseline_noise.png)、[条件波动图](work/R11_TARGET_DENSITY_REPRESENTATION_001/results/figures/conditional_curvature.png)和[完整判别图](work/R11_TARGET_DENSITY_REPRESENTATION_001/results/figures/c2st.png)。

下一轮固定两种表示、已选主要分布模型和实际随机输入，只改变剩余坐标的协方差估计，检验收缩规则对基线及完整波形差距的影响。当前协方差收缩系数约0.0885；它向整体平均方差收缩，可能抬高低方差基线方向，这是待检验假设。若此项改善后仍存在电荷相关波动与峰宽缺口，再据证据建立条件残差模型，不重新打开EVAL调参。

所有4000条输出在新进程中从保存参数及实际随机输入重放，pre/post ADC均逐位一致。主计算与重放合计17.234秒，另有只读保存特征的补充绘图，最终总费用见[资源回执](work/R11_TARGET_DENSITY_REPRESENTATION_001/RESOURCE_RECEIPT.json)。没有新G4、EVAL读取或数据迁移一致性测试。

科学计算在数值预算内完成，但执行器末次控制组装在03:37:58 UTC，超过03:36:15停写线约103.7秒；根代理随后完成文件封存时也已过本轮03:39:15墙钟截止。本轮墙钟约束未通过，不能把阶段记录写成全部按时完成。原始超时和根代理仅收尾、无新增数值的事实见[超时封存记录](work/R11_TARGET_DENSITY_REPRESENTATION_001/ROOT_LATE_CLOSURE_RECEIPT.json)。

本轮输出是实测目标分布样本，尚未绑定G4事件。较旧方案的局部改善不构成严格配对因果证据，实测对实测也不是等效阈值。项目目标保持active，R11完成不等于项目完成。代码/聚合图表进入私有Git，模型及逐事件载荷只保存在E/D；最终发布和正常复制状态读取R11_PUBLICATION_RECEIPT.json、R11_BACKUP_RECEIPT.json。[复现说明](work/R11_TARGET_DENSITY_REPRESENTATION_001/REPRODUCTION.md) · [交付索引](R11_DELIVERY_INDEX.json)。
