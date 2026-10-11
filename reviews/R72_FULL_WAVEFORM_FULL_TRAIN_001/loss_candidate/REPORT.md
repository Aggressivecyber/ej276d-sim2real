CPU 人工实现检查已通过；当前 trainer 接口静态无阻断。仅实现/人工/静态证据，不构成真实训练收益或 v5 A–F 通过。此目录源码现已锁定，不再修改。

API：FullWaveformLoss(real_reference_canonical, edges_logQ, source_psd, source_interval_low, source_interval_high, source_bin_original, source_h, valley_bounds, config)。call 按 SOURCE/REAL 顺序输入 canonical、preADC、raw 六端点，再输入四组64方向及31维方向，返回 (total, terms, diagnostics)；iter_terms 每次 yield (name, scalar, diagnostics)，每项创建新图；stats_serializable() 提供 JSON 常量回执。canonical/raw 为 float64，preADC 可为 float32，所有常量为同设备 Tensor。检查入口保持 run_checks(require_cuda=True, require_numba=True)，一次覆盖 CPU/CUDA，未添加或重跑包装。

九项权重均1。五个分布项分别比较完整 SOURCE、同处理 REAL 与原未处理 canonical REAL，然后两域平均。全窗/coordinate/temporal 用当前 canonicalQ raw 区 -1..8 的 count/N 加权；有质量的支持外独立纳入。PSD 使用整体与0..6等权条件项，必需空区失败；占用区无合法原 REAL 同区参照失败，非必需零质量区保持0质量。REAL identity 保留原行配对和原 REAL bin 尺度。谷仍按原 SOURCE bin2/3、原 h==1 全母数，原 fraction+0.02。box 使用 preADC/raw 两域均方修正。

不均人口 W1 对共同均匀 CDF 的有理断点精确积分；仅将最终区间质量转 float64。量化划分 LRU 最多32项/262144 cells，固定参照特征可缓存。source order 的分母 D=N²-sum(original PSD tie_count²)，保留全部原 PSD 不同 pair 的原1/N²权重；只在精度区间严格分离的 pair 上计逆序。两次整数 Fenwick 扫线给出值和完整梯度，无 NxN 图；Numba cache=False、fastmath=False，pure/compiled 与独立原矩阵人工值/梯度一致。

原生 CPU suite exit0，5类检查实际 PASS：不均N/重复值 W1 值与梯度及 SciPy，原矩阵 order/精度 mask/接触与重复值，置换/动态 bins/支持外全质量/empty fail/配对 identity，九项 direct 与 iter 六 leaf 梯度，全部人工父事件 VJP 和人工6593 order。W1 dense 最大值/梯度差4.44e-16/1.39e-16，SciPy值差0；order旧矩阵最大值/梯度差5.33e-15/3.55e-15。检查2.9267秒含首次 JIT2.3062秒。人工6593 scalar order compiled1.100ms、pure39.714ms只描述该人工整数/标量模块，不能预测实际完整步时。CUDA本机无设备，状态unknown；注册作业仍须一次实际双设备 suite，compiled 不可用明确失败。

资源：affinity mask4 /1 CPU，RSS硬上限1073741824 bytes，CPU suite峰值462725120 bytes；全部新增文件/cache在本目录E盘。读取真实科学数组0，实际科学模型拟合/推理0，SSH/Git0。

最终静态 trainer 对照 train_full.py SHA928e5786218d871e29d7965fc8ee21c8ec46c07159f2888220de73c92eeed249：53/60–63行已将常量转同设备float64 Tensor；114–124行六 leaf 顺序和三元 iter_terms、f64累加正确；129–138行 ADC/pre/raw 梯度与所有原父回放对应，配置chunk27640；146–161行工程测时后恢复初始权重、空Adam及随机流，成本只决定完整终点是否启动，未用新科学质量评价择模型。root报告其独立完整NN CPU人工检查通过；本审核未重复执行它。GPU实际资源/全部步时仍由注册监督与实测判定。

锁定 SHA：
full_waveform_loss.py aa9c713fc40ed1a9c63c5d5a3bd75d094855e1480967ea0d596389e8041ce59d
check_full_waveform_loss.py 4d7416cde2c2a5c660c0e0f405ed5d19df770b3e9a960e034b13e546f31eac86
实际 CPU 回执 ARTIFICIAL_CHECK_RECEIPT.json；输入 FINAL_INPUT_SHA256.json；原生/资源 NATIVE_RESOURCE_RECEIPT.json；全输出锁定 FINAL_OUTPUT_SHA256.json。
