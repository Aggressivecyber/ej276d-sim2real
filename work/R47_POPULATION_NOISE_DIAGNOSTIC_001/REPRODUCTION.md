# R47 复现入口

实际科学入口为 `numeric_worker.py`，由 `launch_server.py` 通过已保存的 `linux_supervise.py` / `linux_owned_worker.py` 导入前 PID/nonce/RSS 握手启动。实际 worker159116、supervisor159111；原生终态见 `numeric_worker.py.NATIVE_RECEIPT.json` 和 `numeric_worker.py.LOCAL_NATIVE_RECEIPT.json`。

服务器目录：`/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R47_REMOTE_20261010_053712_UTC`；解释器：`/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`。本轮实际环境 Python3.12.3 / NumPy1.26.4 / sklearn1.8.0 / SciPy1.11.4；CPU12、GPU0。

输入均只读引用 `config.json.inputs` 指定的 R46 服务器原件：实际 SOURCE/IDs、DEV_SOURCE_METADATA、REAL_DEV_AND_REFERENCE、原 C2ST phase0/phase1 选择、冻结 Q-bin 常数、SHORT/LONG 的保存输出及 S/G/F/R 分量、原 RAW RR 预测与分组。`SOURCE_REUSE_BINDINGS.json` 记录历史版本；正常上传/下载 native0，不宣称迁移位级验证。

在另一个新工作目录复制本轮必要脚本和 config，修改该新目录 remote_root，保留原 R46 输入路径，只在新目录部署并运行 `launch_server.py numeric_worker.py`。不能在本封存目录重跑或覆盖结果。该入口仅五个预定 HGB 与保存数组描述，不运行任何响应网络/ODE。

完整 `metrics.json` 和 `results/*PRIVATE.npz` 包含身份、索引与固定预测；发布汇总为 `METRICS_COMPACT.json`。`METRICS_COMPACT_EXECUTED.json` 保留原字段，元数据差异见 `METADATA_CORRECTION_RECEIPT.json`。

原图 `render_figures.py` 已执行；其一次性元数据澄清不可重复。`repair_display.py` 只从保存汇总修两幅显示，原图在 `results/figures/display_initial`，原脚本为 `render_figures_display_initial.py`。本机绘图解释器 `C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`，CPU4，缓存在本阶段 E 盘 cache。

原 RAW RR 分类器和 R46 全部旧结果直接复用。SOURCE 与 REAL 本来不是事件物理配对，人工组/区组区间也不能换成独立采集不确定性。本轮没有可保存/重放的新响应权重或候选。
