# R34 复现

在新目录恢复，禁止覆盖本轮或R33。唯一投影在projection_core.project/protect：x1和x0分别转f64后相减，0:100零、100:150与150:400各减均值、400:480原样。保存f64 preADC、f32 ADC输入、最终postADC、原/保留/移除增量。没有目标域数据输入该部署变换。

本机解释器 C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe，实际Torch/NumPy/sklearn版本在metrics.versions；CPU4。supervise.py设置E阶段cache下TEMP/TMP、Torch/Matplotlib/joblib缓存，owned_worker导入前报告PID/nonce，确认真实RSS后执行。

实际入口：python -B supervise.py evaluate_protected.py <该CPU解释器>。worker42452/super56960/native0；主保存输出来自R33两末点的generated.npz，不重新主推理。辅助原preADC未另存，精确恢复补充见tasks/R34_AUXILIARY_PRECISION_RECOVERY_ADDENDUM_001.md；从R33实际joint/probe输入复跑固定权重/CPUchunk128/32步链3712，核对全部原分量和原ADC后才P。恢复输入/模型路径与重要旧字节清单均由DEPENDENCY_INDEX/config和R33 immutable FINAL_MANIFEST绑定；无需重新做迁移内容测试。

部署独立回放入口：python -B supervise.py replay_protected.py <解释器>。worker51120/super25704/native0，4000完整实际source→G→固定flow→P→1ADC逐值一致；该入口只读R33 ACTUAL_GENERATION_INPUTS、冻结R32 G、R33两流权重和AFFINE_CONSTANTS，不读取REAL/TRAIN样本、reader、内部val或DEV统计，无拟合更新。REPLAY_RECEIPT包含所有字段证据。

固定评价复用R26 600/250源与REAL索引/组以及R32/R33旧预测；新候选仅20HGB。宽度仅16描述槽/14实际EM，旧REAL/BASE/两原流描述直接复用；所有新fit/slot立即落盘。不得通过重跑初始化填补bin1/8缺口。保存完整metrics/预测/参数可重建所有聚合结论；发布聚合METRICS_COMPACT不含身份载荷。

汇总绘图入口：python -B supervise.py summarize_and_plot.py <解释器>。worker10324/super26120/native0，13图、0新HGB/EM/响应更新/神经生成。图例明确旧流与保护流；固定边界例父0/500/1000/1999，同Q REAL仅未配对参照。源主条件采用实际未量化source-Q213/181，历史量化参考207/182另列。

所有保护候选联合门槛失败，复现是开发负结果证据，不是自动授权再次科学工作。旧base S/E与flow增量只有代数意义，最终物理分解unknown。完整数据和随机输入保留E/D，未上传逐事件科学载荷。
