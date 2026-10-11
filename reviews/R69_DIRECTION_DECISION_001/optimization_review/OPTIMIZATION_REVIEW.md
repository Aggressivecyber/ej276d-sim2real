# R69 K4 静态审阅

结论：K4 有可核查的计算机制，适合先做协议规定的单次、不更新模型的完整梯度准入；目前没有其质量收益证据。未发现公式或现有网络结构上的阻断。只有准入通过且预算预留通过，才支持这一个候选的封顶等成本 pilot。R65 保持退休，原 50,000 步端点与 A–F 仍未完成。

本审阅仅读取代码、封存报告及标量 JSON；没有打开 NPZ/PT/CSV、运行网络或新增科学指标。审阅的协议为 `tasks/R69_SAMPLED_FULL_GRADIENT_PILOT_001.md`，最终复核共 45 行，SHA256 `55007dfe51fea5e59c002462d53b23a867762f12ac21edeff3584d7566429a06`。第 27 行包含前瞻性双范数门，新增第 33 行明确两臂完整步超 200 秒均拒绝；其余输入 SHA 见同目录清单。

## 证据与候选理由

R67 的固定起点、固定 LR、等时间端点中，full 的九项均值 5.928 高于 mini 的 4.546，8/8 探针更差。R68 分解指出 full 的 D5 略小，但 C4 较大，噪声和 temporal D 本身也较差；扣除整个 valley 项之后仍 8/8 更差，配对噪声描述区间仍在零以上。这支持停止原昂贵路线，不能把 48.95 倍 C 更新机会或任何单一梯度估计器认定为差异原因。依据：`R68_REPORT.md`、`work/R67_EQUAL_COST_REPAIR_001/RESULT.json` 及两臂 `TRAINING_RECEIPT.json`。

固定 1 full + 49 C-only 会改变 C 的更新权重和 Adam 轨迹，49 仅来自先前机会比，没有直接梯度冲突/收益证据，且会减少每单位时间完整 D 的机会。本次拒绝这一提案的理由是证据不足；旧 R65 的 D:C=1:1 是其自身绑定，并非所有未来独立登记算法不可改权重的用户限制。

旧 R65 吞吐优化只验证了参数打包，不曾实测排除更大 VJP 块或 GPU 累加；原 64 行、CPU float64 顺序限制是旧方案的逐位冻结条件。现有网络没有跨事件状态，数学上可以改变分块，但没有该变更的整步耗时/显存证据，本轮不再提出第二个候选。依据：`reviews/R65_THROUGHPUT_OPTIMIZATION_001/agent_candidate/REVIEW.md:7-16`、`work/R65_FULL_EMPIRICAL_D_RESPONSE_001/implementation_upgrade/CUDA_EQUIVALENCE.json:80-89`。

## 数学及实现边界

固定当前 theta、全部 SOURCE 噪声、方向和完整联合 D 的端点导数 GY。原块 b 的参数梯度为 G_b = J_b^T GY_b，G_D = sum_b G_b。M=104，独立均匀无放回抽 k=4 块，包含最后单父块，令

    Ghat_D = (M/k) * sum_{b in S} G_b,
    E[Ghat_D | theta, noise, directions, GY] = G_D,
    E[||Ghat_D-G_D||^2 | fixed draw]
      = M*(M-k)/(k*(M-1)) * (sum_b ||G_b||^2 - ||G_D||^2/M).

等概率的单位是原块。最后 1 行块的包含概率仍为 4/104，不能再乘块长权重、选中父平均或区域归一化。上述无偏性针对原 autograd/ADC STE 所定义的 clip 前程序梯度及理想算术；有限精度、clip5、Adam 和不同轨迹均不具有同一保证。

该方案与 R64 的小人口分布拟合不同：先保留全部 6593 SOURCE 端点，针对完整 21047 REAL 一次计算原五项 D/GY，然后只抽参数反传。每父的 forward/分布目标曝光保留；实际每步 backward 曝光只有 4 块，不能称为每步全父反向曝光。现有 GY 已包含区域质量、科学权重、归约与 ADC STE，必须一次注入 projected out，不应注入 raw 或再次除人数。依据：`work/R65_FULL_EMPIRICAL_D_RESPONSE_001/objective_adapter/objective_adapter.py:16-37`；`work/R67_EQUAL_COST_REPAIR_001/full_step.py:31-75`。

网络的 GroupNorm 沿每个事件内部归一化，attention 输入是 N×L×C、dropout=0，position 是固定 buffer；投影的电荷和自由掩码也是逐行操作。原 64 行 replay 因而可以独立求各块 VJP。仍须保持 no_grad/replay 后端设置、当前端点/ADC/free-mask 一致，诊断与更新中不得插入参数或 buffer 改动。依据：`work/R65_FULL_EMPIRICAL_D_RESPONSE_001/network.py:7-25,35-47`；`source_flow_core.py:13-37`；`full_step.py:65-79`。最终 C 原样一次、D+C 一次 cast/clip/Adam：`full_step.py:83-113`。

## 对协议的判断与停止规则

第 27 行的 V <= ||G_D||² 且 V <= ||G_D+G_C||² 比仅用联合范数更稳妥，避免大 C 梯度掩盖 D 相对误差。计算使用同一参数坐标下 CPU float64 的平方范数，不除维数；所有量有限，零范数只允许 V=0。只允许舍入尺度内的微小负 centered sum 置零，较大负值必须报数值失败。该门是前瞻性工程成本准入，不是 A–F，也不是一般 SGD 无效证明。

剩余实质风险是：单个 theta/噪声/方向/C 批的 V 不能约束后续 100 步的方差；排序、Box 活跃集、clip 和 Adam 都可能放大轨迹差异。K4 节省 100 次 replay/VJP/打包，全部 104 次 first-pass、完整分布排序/GY 和原 C 仍需执行，不能据局部 VJP 或旧微基准保证 100 步能在 200 秒内完成。现有数据未给出块梯度同质性，也未给出 K4 的质量或实际整步加速。

协议第 21/25/27 行的执行顺序正确：人工不等长块证明先于科学读取；单次完整 D+C 诊断不做 optimizer.step；失败则此固定 K4 关闭、零响应更新、无 k/seed 扫描。诊断虽可推进 RNG，两个训练臂必须分别重载原 update500 的模型、Adam、八条原流和 torch RNG；额外 PCG64 块流必须独立并在实现中显式持有，不能从既有噪声流抽索引。

第 31/33/35/37/41 行允许的比较只能是共享 fullD 500 起点、固定 update501 LR 条件下的等时间路线差异。candidate 必须完整 100 更新并完成同步计时，mini 在第一个完整步达到同一 T，耗时差超过 max(0.5s,1%T) 则比较无效；任一臂完整步耗时超 200 秒同样拒绝，容差不豁免此门。不能把部分 candidate 步数另选为终点。九项均值至少改善 10% 且优于 baseline、至少 6/8 胜出及全部伤害门共同通过，才是局部阳性；8 组噪声 bootstrap 只描述固定模型/固定人口的 MC 变化。

600 秒足够性尚无实测证据。第 41 行先用实际 baseline 耗时，扣除已花掉的加载/人工/诊断成本，预留两臂各 200 秒、剩余完整评价和 20 秒落盘；预算不足必须 INCONCLUSIVE。不得删探针、缩人口、超 cap 追加或依据中途结果改 k。一次准入失败不启动响应探针；一次阳性也只允许另行登记的封顶确认，不恢复长期训练。

R68 的固定 TRAIN 电荷包络不随本优化消失；它界定该人口与精确联合 Q/PSD 匹配的限制，不能外推为原固定 HGB/A–F 一般不可能。原源 Q、事件人口/权重和所有 A–F 推理/验收阈值保持。依据：`reviews/R68_DIRECTION_PREPARATION_001/charge_compatibility/REPORT.md`、`PROJECT_ACCEPTANCE.md:15-21,32-36`。

最终可用结论：先允许根实现和审阅 K4 的单点完整梯度准入；没有证据直接批准响应训练、质量提升或最终候选。静态数学无阻断；独立流、最后单父块、同一 GY、投影 replay、单次合并更新、完整预算/状态恢复必须由根的实际实现核验。辅助模块与 Fraction 枚举只是人工 CPU 证据，不属于 CUDA 或科学通过。

人工 native 交付：`sampled_blocks.py` 的三个纯 API 及 `check_sampled_blocks.py:run_checks()` 已执行，35 项通过、进程 exit0。5 个不等长块长度 2/3/4/5/1、k=2 全部 10 个子集的 Fraction 均值精确等于完整梯度，误差方差精确等于公式 `5083466702147/389577988800`；104 块末块 1 行、k=M 零方差及独立 PCG64 行为通过。原 guard 核验 affinity2/1CPU、hard RSS1GiB，实际峰值 RSS35,299,328 字节，检查耗时 0.598 秒。实际退出、资源与 SHA 分别保存于 `NATIVE_EXIT_RECEIPT.json`、`RESOURCE_RECEIPT.json`、`SHA256_MANIFEST.json`；没有科学数组读取、模型/GPU运行或训练更新。
