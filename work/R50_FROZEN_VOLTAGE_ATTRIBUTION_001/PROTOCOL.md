# R50：冻结输出的电压表示与基线归因

本协议在任何 R50 数值评价之前登记，由主助手独立实施。使用已封存 R49 本机两臂 REAL 对照输出、R42 保存的 REAL DEV/REFERENCE 和 R26 固定分组索引；全部父 ID、顺序、分组和权重保持。服务器仅运行本阶段独立目录，既有环境、CPU12、作业树 RSS48GiB，单科学作业；没有计算时长、方法数或文件体积预算。EVAL 不打开，CAL 不解码，响应训练、推理生成和密度拟合均为0。

科学问题：R49 的 REAL 原始电压可分性，是 ADC 浮点表示产生的，还是模型对幅度、基线或噪声的实际改变？本研究是诊断，不产生合格响应模型，不替换原 A–F。

## 固定比较

六个人口为 RAW_REFERENCE、RAW_LEGACY_ADC、FIXED_PRE、FIXED_POST、COMPLETION_PRE、COMPLETION_POST。前者为保存原始 REAL 参考；第二者仅应用 R49 原 float32 ADC；其余为两臂保存的量化前/后波形。每个人口均相对同一 RAW_DEV。

1. 六个人口各在 native_voltage 和 canonical_ADC_voltage 两种表示下评价，共12个 HGB。canonical 操作在两侧共同施加：先以 float64 计算 clip(round((V+0.5)*16383),0,16383) 整数码，再按同一 float64 除法重建电压。报告码变化，不能把所有微小改变事先视为纯表示。
2. RAW_REFERENCE、FIXED_POST、COMPLETION_POST 各在 baseline_subtracted_voltage、baseline_mean、baseline_RMS、pulse_Q_PSD_peak 四种诊断视图下评价，共12个 HGB。基线使用前100点，RMS使用去均值样本标准差，Qtotal/fast 使用100:400和100:150，PSD=1-Qfast/Qtotal；三列为Qtotal、PSD、去基线峰值。

合计24个固定 HGB，600/250各域训练/留出行、原分组索引、原100次迭代参数、原500次分组bootstrap和种子保持。每个变换也作用于参照域，不改 QC 选行或重新划分。变换间及相对同视图 RAW 的 BA 差使用相同留出组配对bootstrap。报告多比较的探索性质，不声称跨评价器或跨采集置信。

逐事件描述在全2000参考行上报告：ADC码是否改变、最大/均方表示变化、量化前后及RAW ADC前后的baseline mean/RMS、Q、PSD、峰值变化。保存完整24行指标、配对区间和紧凑图。现有R49原始验收保持失败。

## 判定与结束条件

- 若 RAW 仅经过旧ADC便出现可分性，RAW码未变化，共同canonical重建后 RAW_ADC 与 RAW 的特征/预测一致，而处理输出的差异也明显减弱，则支持 ADC 表示是可分性来源之一；不由此宣布全部物理响应合格。
- 若共同canonical后处理输出仍超原[0.4,0.6]区间，或量化前差异与幅度/基线/噪声诊断一致，则记录剩余模型失配，进入针对该机制的新协议；不得以去基线视图取代原电压验收。
- 若结果混合或不明确，报告未归因部分，以预先列出的比较结束本轮，不追加阈值扫描或事后挑变换。SOURCE形状/PSD和Q3失败仍是独立待解问题。
- 24次评价、固定描述、图形复核、原生回执和结果/后续方向写齐后，本诊断阶段结束。任何固定方案失败只关闭方案，长期模型目标仍未完成。

输入来源：work/R49_LOCAL_HEUN128_RECOVERY_001；原始ADC实现：legacy_model.py；固定索引：work/R26_CONDITIONAL_BATCH_TRANSLATION_001/results/C2ST_PRIVATE.npz。部署只做正常传输与实际加载，不新增迁移内容一致性测试。
