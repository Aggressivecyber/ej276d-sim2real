# R53 冻结推理与复现

本目录对应唯一的 50,000 步 `JOINT_PULSE_RESPONSE`。是否达到目标须读取 `ACCEPTANCE_A_C.json` 和最终报告；权重存在或程序能运行不能代替联合验收。

推理所需文件为 `infer_saved.py`、`flow_runtime.py`、`source_flow_core.py`、`network.py`、`noise_core.py`、`model.py`、`legacy_model.py`、`UNIT_CHARGE_CONSTANTS.npz`、`NOISE_CALIBRATION.npz` 和 `JOINT_PULSE_RESPONSE/FINAL_FLOW.pt`。它们共同构成可移动的推理目录，不需要 TRAIN、CAL 或评价集。Python 3.12.3、NumPy 1.26.4、PyTorch 2.12.0+cu130 是本轮服务器环境。

输入 NPZ 必须含 `source_waveforms_V` 和 `white`，均为 N×480。`white` 明确保存各样本的白噪声输入以便复现；`source_ids`、`reference_ids`、`reference_groups` 如存在会原样传递。SOURCE 与 REAL 使用同一算子，不按数据类别选择不同网络。

```text
python -B infer_saved.py --input /data/input.npz --output /data/new_output.npz --device cuda:0
```

Windows 的输入、输出和整个推理目录应置于 E/D。输出示例为 `E:/GAN/inference/new_output.npz`。脚本拒绝覆盖已有输出；缓存和临时文件写入包内 `cache/`。固定 Heun 128 步、分块 64，保存预 ADC、一次 ADC 后的波形、电荷误差及条件元数据。`cpu` 是可用代码路径，本轮独立位级回放使用的是相同服务器 GPU 环境，不能据此宣称跨平台位级一致。

完整训练复现需要另行恢复 `INPUT_BINDING.json` 所列的原 TRAIN、R43 暴露流、R52 完整与末轮耦合、R48 初始化/训练对照和噪声常量。需在新的隔离目录修正配置中的路径，保留所有种子、50000 步终点、损失权重与训练流。原工作目录的 `.used` 文件防止重复提交；不要删除它们来覆盖本轮记录。

固定执行顺序：`linux_supervise.py train_final.py`；成功完成且教师检查通过后执行 `linux_supervise.py numerical_check.py`；数值门通过后依次执行 `generate_all.py`、`evaluate_saved.py`、`replay_all.py`，均由同一监督器启动；最后由 `linux_supervise_reporting.py summarize_and_plot.py` 汇总既有输出。数值门失败须在读取 DEV 前关闭该固定方案，不自动增加积分步数。

本轮完整回放在独立进程通过 `infer_saved.build_pipeline` 加载包内权重/常量，再重算 2000 SOURCE 和 2000 同处理 REAL；具体结果由 `REPLAY_RECEIPT.json` 证明。原 5984 条主输出及探针、10 次固定 HGB、16 个密度分区均保留。已量化数据统一重建为 float64 ADC 电压，原始 SOURCE 输入的电荷/PSD 分箱保持未量化定义。失败分区、历史对照、所有原始权重与事件组成均不删除。

服务器使用 CPU12、作业树 RSS48GiB、GPU10GiB，单科学作业；科学计算次数和墙钟时间未设置配额。训练、数值检查、生成、评估、回放的原生进程回执与资源记录分别保存。数据、权重、优化器及私有数组随 E/D 交付，Git 只保存代码、配置、汇总指标、图和回执。
