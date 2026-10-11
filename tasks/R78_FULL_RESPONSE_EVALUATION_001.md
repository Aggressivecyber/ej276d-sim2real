# R78：唯一 R77 完整响应的冻结评价

本任务只评价已完成 R77 唯一25000 shape更新与冻结R71唯一50000 Q-head。全部6593 SOURCE和21047 REAL的TRAIN方法、九项与原始父权重已固定；不得按本次评价选择检查点、重训、修改随机流或选择对照替代。原生R77退出0、原定TRAIN独立NumPy64+64参考与TRAIN4000序列化重放已完成，直接引用原生回执。用户已明确要求进行评价并取消额外重复复算；不再执行根侧重复TRAIN数值计算，也不将Git/备份设成评价启动前置。当前登记不构成A–F成功。

生成使用完全相同的公共SOURCE/REAL算子：完整480输入→独立预测Q*→电荷lifting与有色噪声→482→128→128→480残差→原Box约束→原float32 ADC及严格共同canonical码表示。无域标记、REAL身份旁路或BASE回退。raw明确float64、原保存white明确float32，chunk256含末批、straight_through=False；保留原z/style但其不进入算子。绑定唯一权重、统计、四份原R77运行模块及全部输入哈希。

按原顺序一次生成SOURCE2000、同处理TRAIN_REFERENCE REAL2000、8噪声重复1024、源扰动832和guide128，共5984。REAL比较目标仍为原real_* DEV2000，不与输入reference_*对调；原SOURCE组成1518 pure_had、417 pure_EE、65 mixed，原支持外417和原谷母群213/181完整保留。历史ADC谷母群207/182仅另报。原父身份、组成、权重、原始条件/随机流、独立Q*、preADC和canonical Q、全部条件迁移与支持外质量保存。

生成后新进程重新载入同一权重与原输入/随机流，先完整重算两域4000，再读取主保存输出比较全部payload字段、preADC、canonical电压和整数码。此独立held-out4000与TRAIN序列化检查分开计数。任何数值或进程失败保留原始结果并停止本执行，不能删除行后通过。

评价只读取保存响应，不进行神经前向/更新。固定原R26三个角色选择槽：SOURCE0、REAL DEV target1、REAL TRAIN_REFERENCE control2，每域600分类训练+250分类留出，不重分。原五视图、HGB超参数、500次分组bootstrap与种子2026100871保持；新增HGB正好10次，密度描述槽正好16（两域各8），EM尝试至多16，失败/空槽完整保留。BASE、BASE_REAL、RAW_RR的原R51分类预测与原R39参考宽度复用，没有重拟合历史对照。11份历史参考首次读取前逐项验证绑定。最终EVAL与CAL波形解码均0；CAL既有固定描述和已暴露DEV的使用不代表新独立采集证据。

所有现行v5 A/B数字及原六项门保留：两域各五视图BA95%CI全部包含于[0.40,0.60]；总体及固定共同Q1–7 PSD W1≤0.01；必需Q2–7两峰位MAE≤0.01、原REAL raw SD平均相对差≤10%，支持/拟合失败不删除；原噪声、源谷/秩、九边界、guide与同处理REAL门保持。第6门使用事前R77非迭代独立NumPy参考（median≤0.25LSB、p95≤1LSB）与独立预测Q*的float32 preADC积分相对差≤1e-5及新4000重放。不能换成原SOURCE Q硬守恒或虚构迭代N/2N结果。

Q描述保留原2000均匀质量，正式评价一次计算输入Q0、预测Q*、preADC和共同canonical四轴CDF/KS/W1/计数/权重/支持外，以及三个相邻11×11迁移表（含低/高支持外与非法格）。聚合入口直接引用正式评价导出的数值证据与原生回执，不重复读取原波形或重算Q/ADC/身份/CDF/迁移，不将引用称为独立复算；不另加结果驱动电荷阈值。所有真实非法值作为失败保留；不能归零、删除或重新归一化冒充通过。

执行顺序固定为generate_full_response.py、replay_full_response.py、evaluate_saved.py、summarize_and_plot.py。每个入口由原受控Linux监督器核验真实PID、原生退出、RSS及适用GPU峰值，顺序单科学作业；GPU入口仅前两个。CPU0..11/RSS48GiB/GPU10GiB，无科学墙钟/阶段体积预算；本机辅助共享CPU4/RSS4GiB，每个辅助1CPU/1GiB。新增本地文件仅E/D。bash协调器不是科学回执，真实科学终态以四份native worker回执和进程消失为准。

主A–C任一失败则封存这个固定R77方案的负结果、实际图像、成本、原生记录与交付；不能降低门槛或把工程/局部通过说成模型完成。A–C全部通过才另行绑定D任务效用与E的REAL-only FULL/GAP设计；本任务不执行D/E、不验证真实无观测事件率或物理能量标定，它们仍unknown。当前应用长期目标读取为paused，本轮直接用户指令授权本次评价，不据此篡改应用目标状态或自动启动下一模型。完成后根实际打开8类科学图复核，私有Git仅保存源码/配置/紧凑指标/图，私有逐事件数组和权重保留E及D备份。


## 实际完成补记

正式生成、独立重放和评价exit0；A/B/C失败。历史REAL密度表缺少绘图字段造成原summary exit1，原记录保留。只用已保存结果修复出图最终exit0，8图完成目检，没有额外科学复算。详见R78_REPORT.md、R78_RESULT_SUMMARY.json及LOCAL_COMPOSITE_COMPLETION.json。目标paused，D/E与后续训练未执行。实际Git/D交付由R78_DELIVERY_RECEIPT.json记录。
