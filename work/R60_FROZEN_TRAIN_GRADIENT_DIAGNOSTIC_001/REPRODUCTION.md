# R60 复现与范围

本轮是冻结 R59 第 50000 步权重的 TRAIN 诊断，不训练新模型，不构造教师，不重新打开 DEV/EVAL/CAL，也不更改原 A–F。实际完成状态以 `diagnose_train.py.NATIVE_RECEIPT.json`、`RESULT.json`、`FIGURE_QA.json` 和项目根交付回执为准。

`config.json` 给出服务器环境、八个冻结输入引用、九个 PCG64 种子和固定 32 批设计。`INPUT_BINDING.json` 绑定输入及原验收；`ROOT_IMPLEMENTATION_REVIEW.json` 绑定本轮执行源码。`network.py`、`source_flow_core.py`、`noise_core.py`、`distribution_loss.py` 是 R59 原字节副本。`diagnostic_math.py` 只增加诊断分解，不改原函数。

完整科学复现需要既有 Linux/CUDA 环境（Python 3.12.3、NumPy 1.26.4、SciPy 1.11.4、PyTorch 2.12.0+cu130）和配置中列出的原 TRAIN、R59 权重/标准化/噪声常数、训练配置/末点回执/逐更新损失。它们作为冻结历史依赖引用；R60不重复备份相同权重和大型输入。R59权重和必要推理文件已在其D盘交付中。

在新的独立复现目录保留本目录的执行源码、配置、协议、输入绑定及实现检查记录，确认引用仍指向原输入并满足 CPU12/RSS48GiB/GPU10GiB、单科学作业要求后，用原环境执行：

```sh
python -B linux_supervise.py diagnose_train.py
```

原执行目录含 `.used` 标记，会拒绝重复提交；不能删除标记后重复覆盖。运行时保存真实PID握手和原生退出。全部新增本机文件须在E/D盘，Windows本机的人工检查为CPU4；本轮没有执行完整Windows模型推理，也不声称跨平台逐位一致。

固定科学量：全部6593 SOURCE和21047 REAL各一个固定噪声；32批中每批320 SOURCE和64 REAL保持，共12288额外神经前向行。合计39928行，0优化器步。全TRAIN使用原Heun128，梯度使用原训练解析端点。17个参数梯度共享每批一次前向，544次求导；四类分解须还原原损失及梯度。

全TRAIN输出只在内存完成描述，`FULL_TRAIN_DIAGNOSTICS_PRIVATE.npz` 保留逐父PSD/Q/噪声/投影与精度区间；`GRADIENT_DIAGNOSTICS_PRIVATE.npz` 保留抽样索引、17项损失、参数Gram矩阵和全480点梯度摘要。固定输入、权重和随机流可复现全窗输出；没有为存储节省而删除人口或计算区间。原始逐参数梯度不长期保存，其几何量可由同一固定批重建。

`RANDOM_STREAM_RECEIPT.json` 给出实际消费流哈希和类型；初/末 RNG 状态完整保存。全TRAIN父顺序不变，批索引为uint16，白噪声/方向为float32。没有声称已从初始状态独立重建全部流。

预定标记只反映冻结点处的欧氏梯度几何；不是精确Adam方向或因果证明。零范数的余弦保存为null/NaN而非伪造0。储存舍入区间不代表物理测量不确定性，原源PSD关联验收不变。若需新训练，必须据本轮证据另行预登记。

全部图来自实际保存的TRAIN描述、梯度记录或原R59损失轨迹；保存最终图后由主助手实际打开复核。私有Git与D盘交付以项目根的 `R60_DELIVERY_COMPLETION_RECEIPT.json` 为准。诊断完成不代表长期模型目标完成。
