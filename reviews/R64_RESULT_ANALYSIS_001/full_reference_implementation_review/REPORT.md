完整有限 TRAIN D + 一个原 C 微批：静态实现审查

结论：现有源码支持一个简单实现基线：**原五项 D 联合调用一次得到总端点梯度，随后对所有 SOURCE 按原 64 行顺序做完整参数 VJP；C 仍调用原 DistributionLoss，只取其四项 C。** 本审查没有编写训练实现、运行网络/损失/VJP、读取科学数组或权重，也没有注册新实验。根决策 [R64_ROOT_DIRECTION_001.json](E:/GAN/research/sim2real_two_papers_20261008_0305/reviews/R64_ROOT_DIRECTION_001.json) 已选择无额外 REAL 特征缓存、无 C 平均、无采样 VJP，并保留原初始化、Adam、学习率和 50,000 次更新唯一末点；此处仅审查该选择的实现。

**最小 D 接口和每步路径**

建议薄封装 `full_D_value_and_endpoint_gradient(Y, raw_REAL, stats, cfg, directions, temporal_directions, regions, full_layout)`。内部沿用 FrozenDistributionObjective：每步传入当步新方向，用一个 `Y.detach().clone().requires_grad_(True)` 叶子调用现有 `__call__(terms=D_NAMES)`，对其原样返回的 `result.total` 求一次 `autograd.grad`，返回五项值、总值与单个 `GY[N_SOURCE,480]`。

[现有联合 evaluator](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001/distribution_adapter/distribution_gradient.py:252) 已在每个区域/特征族复用 marginal W1 来计算 full_window 和 coordinate_RMS。现有 `endpoint_gradients()` 在第 327 行仍逐项调用五次；新薄封装不需要沿用这个五次路径。返回 total 的现有求和顺序为 full_window、coordinate_RMS、PSD、noise、temporal_joint，应记录并保留；不同浮点求和分组不承诺逐位相同。

每个更新先为全部 6,593 SOURCE 父事件各生成一份新噪声，原档案顺序、固定原区域标签与 frozen statistics 不变。全部 21,047 REAL 仅通过原 canonical_REAL_torch 作参考，无 REAL 全人口网络前向或参考噪声。原方向流每步各抽一次，五项 D 使用同一组当步方向。

第一遍在 no-grad 下按 103 个完整 64 行块加末尾 1 行缓存 SOURCE 的 projected Y、canonical codes、strict Box free mask，以及本步精确 innovation。第二遍参数保持不变、噪声不重抽、模式和数值开关不变，按同样边界重放；Y 逐位、codes 和 free mask 必须一致。每块将总 GY 注入原 endpoint 返回的 `out` 一次，获得全部参数梯度并按原参数/块顺序作 float64 累加。[原 R64 实现](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001/diagnose_train.py:363) 已提供这条完整重放路径。

GY 已含原 ADC STE、质量、区域权重和全部损失归约；不得再除以父样本数、块数或五项数，不得再 canonicalize 梯度或将其注入 raw。Box backward 在原网络 endpoint 重放中进入一次。每个更新都必须重算 SOURCE endpoint/GY；跨更新复用旧结果不属于该基线。

原初始化证据为 seed 20261010590、state SHA `4a51097e257d4b35976452982aa9a84f48e321d425e0ba3859e8afe8a7bd239b`、1,369,889 参数、零输出头；未来实现应重新构造并核对该 SHA，不加载 R63 训练末态或 R64 冻结权重。[原训练初始化及 Adam](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/train_final.py:103)

**一个原 C 微批保持实际计算路径**

每步仍抽原每区 32 行 SOURCE，共 320 行，以及 64 行 REAL identity；一次原 384 行网络前向，同时保留 out 和 raw。继续调用原九项 `DistributionLoss`，再直接相加 `source_order, source_valley, REAL_identity, box_correction` 四项的原权重。五项微批 D 仍实际计算，只从优化目标中丢弃；不采用“九项总损失减 D 总损失”的抵消写法，也不把微批 D 误写成完整 D 的训练日志。

[原 C 源码](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/R61_distribution_loss.py:89) 必须保留以下细节：

- source_order 分母是原 SOURCE PSD 不等的全部有序对质量总和，区间可分辨条件只作用于分子；不能把分母改成已分辨对质量。[precision_order.py](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/precision_order.py:25)
- source_valley 保留 Q3/Q4 各 32 槽的 mean、原 h=1 mask 和 frozen population h=1 fraction；不能改为采样 h 数作分母。
- REAL_identity 使用原 REAL raw bin +1 取 V480 scale，保留 64×480 mean；网络 clipped bin 不能替代 raw bin。
- box_correction 保留 `0.5*(SOURCE 加权 mean480((out-raw)^2) + identity mean((out-raw)^2))/0.5^2`。必须保留 out 与 raw 的共同网络图；只计算 dC/dout 会漏掉 box 的直接 raw 路径。

原 C SOURCE、REAL-reference、identity 索引流及 SOURCE/identity occurrence-noise 流保持。重复 SOURCE 槽仍各自有独立噪声，不能从完整 D 的每父样本噪声表中 gather 来代替。完整 D 使用另行声明的新噪声流。保留原 reference 抽样以供这次完整原损失调用；原 full-window/temporal 方向每步抽一次并可同时传给完整 D 和原微批调用，不额外推进方向流。

将完整 D 参数梯度与这一次 C 梯度相加后，只执行一次原 gradient clip=5 和一次原 Adam.step。保留原 LR 日程、betas=(0.9,0.999)、eps=1e-8、weight_decay=0。C 系数仍为 1，不平均四个 C 微批。新 D/C 使用不同 endpoint realizations，不能声称联合随机耦合与 R63 完全相同。

**只按源码计数，不估算耗时**

| 每个更新的网络操作 | 行数 | 调用次数 |
|---|---:|---:|
| 完整 SOURCE no-grad 前向 | 6,593 | 104 |
| 完整 SOURCE VJP 重放前向 | 6,593 | 104 |
| 一个原 C 微批前向 | 384 | 1 |
| 合计前向 | 13,570 | 209 |
| 完整 D 参数 VJP | 全部 SOURCE | 104 |
| 原 C 反传 | 一次原微批 | 1 |

联合 D 的一个端点反传不计作网络 VJP。完整 D 共 22,690 个一维 W1 law：共享坐标 19,170、full-window slices 2,560、PSD 8、noise 2、temporal 950。当前 feature_chunk=64 的 uniform fastpath 共 389 个坐标块、778 次 torch.sort API 调用；每个 sort 仍覆盖所属分布的全部行，没有沿父样本截断。

作为 C 的原九项损失调用仍计算并丢弃 41,863 个一维微批 D law，包含 RMS 再算坐标 W1，以及原 PSD 先计算全部 10 区再取 Q1–Q7；它们通过 36 次值排序和 3 次 CDF 边界排序完成。因此不能把默认 C 路径记为“免 D 开销”。law 数和 API 调用数不同，且各 law 行数不同，不能据此作时间比例预测。C 的原 source_order 矩阵为 320²=102,400 个有序槽对。

50,000 步若完整执行，对应 678,500,000 前向行、10,450,000 次前向调用、5,200,000 次 D 参数 VJP 和 50,000 次 C 反传。这是循环计数，不是耗时、吞吐或峰值资源测量。

**实现前仍需的人工证明**

已有 R64 CPU/CUDA 的 69 项分布适配器检查，以及非零人工头的真实架构 replay 检查，分别支持现有 W1 规则和 64+1 重放；它们没有证明这个新增“联合五 D + 原四 C + 一次 optimizer”整体。当前审查没有运行新测试。

| 必需证明 | 需要保留的判定 |
|---|---|
| 联合五 D | 一次 joint call 的五项值、total GY 对照原五次 helper 与适用的精确 CDF oracle；涵盖等 32、区域不等人口、重复/相同值、abs(0)、零 RMS、质量差异和原 float32 cast 位置。 |
| 完整参数 VJP | 原架构人工非零头、同 64+1 调度，对照 joint 直接图及五项 VJP 之和；所有参数参与，Y/codes/free mask 精确重放，CPU/CUDA 各自核对。只用零头初始化不足以证明内部梯度。 |
| 原 C 取项 | 对照直接原 DistributionLoss 的四项值；独立 out/raw 叶子的两组偏导和耦合 endpoint 的参数梯度都要核对，覆盖 Box 激活、非零 identity、PSD 对与 valley 分母边界。 |
| 合并与一次更新 | 人工完整 D+C 直接图对照分开累加的参数梯度、合并 norm、一次 clipping 后梯度及一次 Adam 状态/参数更新；无隐藏除法或重复系数。 |
| 初始化与随机流 | 原 init SHA、原 C draws 与方向 draw 次数保持；完整 D 新噪声跨步改变、步内重放固定；参数更新后不复用旧 endpoint/GY。 |
| 资源与数值容差 | 在执行前冻结原风格的 value/gradient 容差；人工检查联合图容量，保留失败。feature_chunk=64 只分块排序，autograd 会保留所有相关块，不能据此声称峰值只占一个块。 |

“精确”指对当步有限完整父样本及固定噪声/方向 realization 的全部参数一阶 selected algorithmic derivative，没有采样 VJP。它不表示对所有噪声/方向积分、硬 ADC 的唯一经典导数，或不同浮点累加分组逐位相等。

**可选优化均不进入基线**

REAL canonical/标准化 marginal/PSD/noise、固定 target sort、按区域计数的 CDF 计划理论上可缓存；fresh-direction sliced projection 及排序每步仍须重算。任何缓存都需证明值和 selected derivative 不变，SOURCE endpoint/特征不能跨更新缓存。

若以后实现 feature streaming，coordinate_RMS 必须使用完整坐标 W1 向量的 L2 norm；对每块分别求 norm 再平均会改变目标。若以后抽取 C 来跳过旧 D，必须先通过上述四 C 值、独立 raw/out 偏导和参数梯度等价证明。更大网络块也需单独 replay 证明。当前无额外缓存、C 抽取、feature streaming 或 chunk 改动。

本目录仅交付静态报告、结构化 REVIEW.json 与绑定回执，原封存文件及根元数据未修改。后续实现必须进入根代理另行指定的新准备目录，不能追加到本 R64 目录。当前审查进程单 CPU（affinity mask=1）、1 GiB RSS 监控、native 0；绑定阶段峰值采样 18,481,152 字节。所有新文件位于 E 盘。

