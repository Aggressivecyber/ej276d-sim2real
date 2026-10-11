# R63 复现与可移动推理

训练与全部预登记评价、独立重放已完成；具体结果以原生退出、`TRAINING_FROZEN_AUDIT.json`、`ACCEPTANCE_A_C.json`、`REPLAY_RECEIPT.json` 和实际 Git/D 盘交付回执为准。原 A–F 是完整模型目标；运行推理或独立重放成功不代表完整响应或任务效用通过。

## 冻结方法与保存项

`config.json` 和 `PROTOCOL.md` 是预登记源。原 R61 网络、全部6593 SOURCE/21047 REAL TRAIN父事件、噪声、区域质量、原七项损失和精确电荷/范围/ADC构造不变；新增全坐标RMS经验W1和31维基线时序联合分布两项，系数各1。末点只能是50000更新。原六条随机流的实际消费及曝光必须与R61一致；独立第七条方向流单独保存和重构。

- `COVERAGE_RESPONSE/FINAL_FLOW.pt`：唯一第 50000 步末点；单一冻结末点的神经权重。
- `network.py`、`source_flow_core.py`、`noise_core.py`：原网络、固定Heun128及显式范围/电荷投影、输入噪声构造。
- `UNIT_CHARGE_CONSTANTS.npz`、`NOISE_CALIBRATION.npz`：推理所需冻结常数。
- `R61_distribution_loss.py`、`distribution_loss.py`、`precision_order.py`：原七项损失及新增两项的可检查实现。
- `TRAIN_DISTRIBUTION_CONSTANTS.npz`：原常数逐字节复用；`TRAIN_TEMPORAL_CONSTANTS.npz/.json`：仅用REAL TRAIN、首次更新前拟合的新时序标准化。
- `COVERAGE_RESPONSE/LOSS_BY_UPDATE_PRIVATE.npz`、`COVERAGE_BY_UPDATE_PRIVATE.npz`、`PRECISION_ORDER_BY_UPDATE_PRIVATE.npz`：全部50000步的总损失/组成和保持约束分解。
- `INITIAL_RANDOM_STATES.json`、`FINAL_RANDOM_STATES.json`、`PARENT_EXPOSURES_PRIVATE.npz`：均位于权重子目录，记录随机流与原父事件实际曝光。

原始TRAIN与历史依赖路径/大小/SHA见`INPUT_BINDING.json`。私有D盘备份采用原R62/R61及更早依赖链的增量叠加；历史文件不删除，未把完整原TRAIN重复放进每个阶段。源码/图示/元数据进入私有Git，数组、权重和日志放在E/D私有备份及原服务器。

## 推理入口

将整个本目录与权重/常数一起保存在E或D盘，或服务器自有项目目录。输入NPZ包含`source_waveforms_V`和`white`，两者都为N×480；白噪声必须显式提供，使结果可重放。可附带`source_ids`、`reference_ids`、`reference_groups`，入口原样保留。SOURCE与REAL调用同一算子，不输入域标志或粒子真值。

```text
python -B infer_saved.py --input /path/to/input.npz --output /path/to/new_output.npz --device cuda:0
```

Windows输出路径须位于E/D，缓存也固定在本目录。入口拒绝覆盖已有输出及回执。`--device cpu`是实现提供的接口；跨平台逐位相同必须由对应实际执行证明，不能由加载权重或CUDA重放推断。

输出包含`pre_ADC_V`、`waveforms_V`、输入身份以及逐事件电荷/支持范围/约束投影诊断，另写`.receipt.json`。内部范围与电荷联合投影是方法的一部分；ADC零裁剪不代表内部无修正。初始化噪声和神经位移共覆盖完整480点，最终使用原共同整数ADC表示进行评价。

## 训练及固定评价顺序

精确训练复现使用`config.json`中固定服务器环境和依赖字节，并在新的登记目录中执行，避免覆盖本冻结阶段或原始数据。原目录的单次启动标记用于阻止重复提交。`train_final.py`先做人工CPU/CUDA检查、输入与代码SHA核验，再只解码合法TRAIN字段；不加载旧训练权重。

原生工作进程由`linux_supervise.py`与`linux_owned_worker.py`进行PID握手并约束CPU12、RSS48GiB、GPU10GiB与单科学作业。末点核验要求原六条流、原父事件曝光及旧常数与R61相同，独立重构新方向流。训练/评价依次为：

1. `train_final.py`：唯一50000更新及完整记录。
2. `numerical_check.py`：原128条TRAIN SOURCE/REAL的Heun128/256检查，原0.25/1 LSB与1e−5电荷限值。
3. `generate_all.py`：原固定人口及全部探针，共5984网络输出；数值门不通过时不会打开DEV。
4. `evaluate_saved.py`：原五视图10个HGB、16个密度槽、完整源/PSD/宽度/噪声/边界评价及冻结R61配对比较。
5. `replay_all.py`：独立进程通过可移动推理入口重放4000主输出。
6. `summarize_and_plot.py`：汇总原A–C和原六门并输出全部图示。

本机串行控制器为`reviews/watch_R63_pipeline_root.py`。观察超时只重读同一进程，不自动重启科学作业。滚动恢复缓存仅供同一运行故障恢复，不能当作另一个候选或按评价选末点。原A–C失败则关闭该固定方案；通过后还必须按单独冻结协议完成D任务效用与E FULL/GAP。

## 证据边界

训练损失是优化代理；原验收仍对SOURCE和同处理REAL全部五视图、PSD、峰宽、源关联、噪声、边界及数值/重放共同判断。历史评价来自已暴露的同一次采集，原父事件重复曝光不是新独立样本。保存的配对区间条件于固定分类器与一个神经训练种子，不能解释成跨采集或重复训练的置信保证。
