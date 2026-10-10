# R54 推理与复现

本目录对应唯一 50,000 步 `CONTINUOUS_Q_RESPONSE`。当前执行状态以原生作业回执为准；模型是否达标须读取 `ACCEPTANCE_A_C.json` 及最终报告。权重存在、代码可运行或重放通过均不能代替原 A–F。

可移动推理包包含 `infer_saved.py`、`flow_runtime.py`、`source_flow_core.py`、`network.py`、`noise_core.py`、`model.py`、`legacy_model.py`、`UNIT_CHARGE_CONSTANTS.npz`、`NOISE_CALIBRATION.npz` 和 `CONTINUOUS_Q_RESPONSE/FINAL_FLOW.pt`。推理不需要 TRAIN/CAL/DEV。服务器环境沿用 Python3.12.3、NumPy1.26.4、PyTorch2.12.0+cu130。

输入 NPZ 含 `source_waveforms_V` 与 `white`，均为 N×480。white 是显式给定的白噪声，实现可重复生成；可选身份数组原样传递。SOURCE 和同处理 REAL 调用同一算子。连续条件由原始电荷与包内冻结边界计算，不选择域，不查询实测模板。

```text
python -B infer_saved.py --input /data/input.npz --output /data/new_output.npz --device cuda:0
```

Windows 输入、输出、工作目录与缓存须在 E/D。脚本拒绝覆盖既有输出；固定 Heun128、chunk64，输出 preADC、一次 ADC 后波形、连续 logQ 条件与守恒/噪声元数据。独立重放按同一 GPU 环境经 `infer_saved.build_pipeline` 加载包内权重/常量，重算 2000 SOURCE + 2000 REAL 后再读取历史输出作位级比较。只有 `REPLAY_RECEIPT.json` 的实际 4000 条结果可证明该项；CPU 路径可用不代表跨平台位级一致。

训练复现需要 `INPUT_BINDING.json` 和 `config.json` 中的原 TRAIN、R43 暴露流和坐标常量、R52 完整/末周期匹配、R48 初始化及噪声、R53 流一致性回执。复制到新的隔离目录并改成本机实际路径，保留公式、参数、随机种子和唯一 50000 更新终点。不得删除原 `.used` 标记重跑并覆盖既有记录。

固定顺序为监督器下 `train_final.py` → `numerical_check.py`；只有原 128/256 数值门通过才运行 `generate_all.py`、`evaluate_saved.py`、`replay_all.py`；最后以报告监督器执行 `summarize_and_plot.py`。任何数值失败都关闭固定配方，不自动增加积分步数。训练参数初值与 R48 相同、原始与教师匹配流与 R53 相同的断言均在训练原生进程内执行。

新生成 5984 条、新 HGB10 次、密度槽16个；全部失败区间与原始 SOURCE 人口保留。所有已量化域共同重建 ADC 电压，原始 SOURCE 分箱保持未量化输入。R53 和 R49 的固定对照不重拟合，R54 对 R53 的五视图差异用同一留出身份/组做配对 bootstrap。

服务器 CPU12/RSS48GiB/GPU10GiB、单科学作业；本机辅助 CPU4/RSS4GiB。无计算配额，50000 是固定科学终点。数据、权重、优化器、私有预测与日志只进 E/D 及已授权服务器；Git 只保存代码、协议、汇总指标、图与回执。
