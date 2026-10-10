# R43 复现入口与范围

训练：`train_server.py`，独立服务器目录 `/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R43_REMOTE_20261009_233131_UTC`，Python `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`；Linux3.12.3/NumPy1.26.4/Torch2.12.0+cu130、RTX5070。原R42常数与saved耦合只读复用。实际初态、optimizer、2950末点、CFM配位/源目标/t流及随机状态在 `SCALAR_TIME_FLOW_FINAL`。seed20261010420新建模型，原R42 INITIAL仅比较，不作为加载初始化。

本机CPU入口：`supervise.py`→`owned_worker.py`导入前nonce/actualPID/RSS守护；Python `C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`，3.13.3/Torch2.10+cpu/NumPy2.2.6、4线程。实际顺序 `check_train_solver.py`→`generate_all.py`→首失败`evaluate_saved.py`→固定分区路径修复后的`evaluate_recovery_001.py`→`replay_all.py`→`summarize_and_plot.py`。原评价入口/配置/日志保留engineering_failure_001，恢复只补既有R26路径与独立失败文件名，不改科学定义。Pipeline为 `network.py/source_flow_core.py/flow_runtime.py`，权重 `SCALAR_TIME_FLOW_FINAL/FINAL_FLOW.pt` 和 `UNIT_CHARGE_CONSTANTS.npz`。读config中已绑定输入，无原始长波形/新G4。

独立CPU重放已实际执行4000条，原输入/white→新流32Heun→一次ADC及所有保存分量逐值通过；另一个进程从新初始化训练未重复执行，不能宣称CPU/CUDA训练逐位一致。初态比较是科学对照证据，正常scp/copy不做迁移一致性测试。复现应在新E/D目录，保留封存stage和 `.used` 原件；本轮结果失败不能用64步/额外编码或改门槛救援。完整私人数组/行流/预测/模型保留E/D，发布用METRICS_COMPACT。
