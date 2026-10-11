# R75 条件特征 VJP 候选封存

本机纯人工 CPU 检查通过，native exit 0；当前没有发现阻断集成的实现问题。仅改变 R74 `iter_gradient_groups` 首组 full_window + coordinate_RMS 的一阶计算方法，未改变九项定义、区域、人口、权重、尺度、分母、稳定排序/tie 规则或随机数。此报告不证明 CUDA 加速、实际人口数值等价、显存足够或科学质量改善。

## 固定接口与数学

入口为 `conditional_vjp_loss.FullWaveformLoss`，构造参数和状态继承 R74 `combined_loss.FullWaveformLoss`。依赖原模块名 `combined_loss`、`full_waveform_loss` 和 `exact_uniform_w1`；不得将候选本身重命名为 `combined_loss.py`，否则会形成循环导入。`iter_gradient_groups` 仍返回 `(names, differentiable_total, detached_reports, diagnostics)`，共八组九项；其余七项实现原样继承。普通 `iter_terms/__call__` 保持旧实现，不走本优化。首组仅支持一阶微分；两个报告值明确不可单独 backward。

设域 d 的父人口为 N_d，视图 F_v、当前输出区域 B_db 的人数 n_db、冻结尺度 s_vb，局部标准化特征 A_dbv=(F_v[B_db]-center_vb)/s_vb。设原区域 full 与 coordinate 项为 f_dbv、c_dbv。原首组为

`L = (1/2) sum_d (1/4) sum_v sum_b (n_db/N_d) (f_dbv+c_dbv)`。

候选逐区计算 `g_A = ((n_db/N_d)/4)*0.5 * grad_A(f_dbv+c_dbv)`，唯一写入 `G_v[B_db] = g_A/s_vb`，最后对四个完整视图执行一次 `sum_v J(F_v)^T G_v` 的域内 VJP。当前区域是原 `_context` 的 detached bins；其十个互斥区域覆盖全部父行，每行每视图只有一次写入。没有把未选父从目标删除、重复索引累加或改变原区域重分配语义。每区精确 W1、投影方向与参考排序沿用原实现，源与同处理 REAL 共用同一参考投影缓存。值的逐区、逐视图、逐域标量表达式及求和顺序保持原式；局部图在写入后释放。

外层 custom Function 仅保存两域完整 canonical 梯度，backward 乘任意上游 scalar；该乘法位置可造成末位舍入或正负零差异，不能宣称普遍逐位等价。

## 实际人工证据

`ATTEMPT_002_ARTIFICIAL_CHECK.json` 为完成记录；`ATTEMPT_002_NATIVE_RESOURCE_RECEIPT.json` 记录 mask 2、一个逻辑核、Torch/interop 各一线程、1 GiB 原生硬限、峰值 RSS 442,691,584 bytes、运行 19.0655 秒。所有缓存位于本候选目录的 E 盘 `runtime_cache`，Python `-B`。`ATTEMPT_002_NATIVE_EXIT_CODE.txt` 实际为 0。

- 20 SOURCE + 30 REAL 的人工 480 点输入，沿原 13 行 NN 分块，包含尾块；完整九项值、六叶梯度、首组梯度、其余七组梯度、诊断、完整 NN 参数 VJP、clip5 后一次 fresh Adam 更新全部 byte 相同，最大差 0。
- 逐区重复值/特征 tie、REAL 精确零 marginal/coordinate 范数、两域各一父当前输出区域迁移：九值及六叶梯度均最大差 0、byte 相同。输入父信息保留。
- 上游 -3.25、0、0.125、1000 均通过紧数值核对。最大绝对差 5.8208e-11（上游 1000），最大相对 L2 2.2305e-16；零上游数值全为零，但 signed-zero 字节可不同。SOURCE 单侧、REAL 单侧及两侧均不 requires_grad 的接口通过。
- 原 required 当前区域为空、端点 nonfinite、占用区域无参考三种拒绝的异常类型与文字保持相同；不裁行、不回退。调用者 CPU RNG 已恢复。

首轮 `ATTEMPT_001` native exit 1，原因是人工比较器把零维 scalar 直接 `.view(uint8)`，与损失实现无关。失败 JSON、资源回执、exit 和当时比较器源码快照均保留；仅将比较器改为展平后 byte view，再运行 002。候选损失源码在两轮之间未改。

以上是人工固定小样本的一阶工程证据，fresh Adam 检查不等于实际 checkpoint 后的多步训练证明。未读取科学数组/权重，未启动 GPU、训练、拟合、probe、SSH、网络或 Git；没有写候选目录之外的文件。

## 性能与集成边界

本实现取消每区将局部 feature 梯度散射回完整矩阵的反向图，改为逐行唯一写入，再合并四视图 VJP。但内层 autograd 工作移入外层 forward，所以必须评估完整更新耗时，不能仅比较表面 backward 时间。

固定实际人口 6593+21047 时，两域 canonical 梯度保存需要 `27640*480*8 = 106,137,600 bytes`；每域还暂存四视图梯度、视图图和当前局部图。域顺序计算且局部图及时释放，仍不能静态证明峰值 GPU 低于旧版或满足 10 GiB。需由 root 按原绑定人口、原损失、原随机流及原同起点第 1/25/50 步门验证九值、完整参数梯度、Adam 和资源，再作 50,000 更新成本准入。root 的 retained neural graph 是另一个由其负责验证的集成变化，本人工检查只审核条件 VJP 模块。

`SOURCE_MANIFEST.json` 绑定实际输入源码、最终三份候选源码及失败/通过回执；封存后不得静默改源码。
