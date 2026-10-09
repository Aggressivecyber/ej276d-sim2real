# 仿真输入与同父重建输出

主图为 `simulation_to_reconstruction.png`，SVG同名。三行分别为电子沉积、强子沉积和混合沉积；左列是实际进入完整波形网络的G4派生清洁仿真输入，右列是R28 FRESH_FINAL第2950步已保存的post-ADC输出。没有重新推理或训练，没有读取实测波形。

选择规则：每种源组成中，选择源侧冻结Qtotal窗口电荷最接近该组成中位数的事件，并列取原序号较小者。输出完全不参与选择。三个CSV保留完整480点，绘图只把V换为mV，不平滑、也不分别归一化；每行左右时间和电压轴相同。索引120为对齐参考0，网格间隔1ns，并非绝对采集时间。尾段图为辅助，不能替代完整主图。

按已完成DEV完整shape493 C2ST BA较低选择R28末点作开发展示（0.970仍明显可区分）。旧J20所选2250步模型保持不变；这不是最终验收通过，也不是按图挑好看的事件。来源组成桶不自动等同独立粒子真值，图形接近不能证明逐事件物理对应。完整输入/输出/权重身份与选中事件、PSD/Q口径见SOURCE_OUTPUT_REFERENCE.json。

复现：在新的E/D目录复制本预览脚本、config、supervise和owned_worker，将config source_stage指向只读R28存档。使用既有CPU Python，4线程，所有缓存写新目录；本次真实PID/退出/RSS见make_preview.py.NATIVE_RECEIPT.json。不要覆盖旧阶段或删除本次.used重提。
