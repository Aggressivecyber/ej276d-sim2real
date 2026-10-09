本轮服务器目录：/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R12_REMOTE_20261009_034933，解释器 /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python；实际入口 numeric_worker.py，由 remote_supervise.py 启动并随后执行 replay.py。CPU4，现有环境，不安装或升级。

只读依赖为兄弟 R11_REMOTE_20261009_020915 下 RAW_PCA_GMM/FACTOR_PCA_GMM 的 representation、冻结K4/K8、GENERATION_INPUTS、generated、metrics/results，及 R10_REMOTE_20261009_012338/inputs。精确路径映射在 config.json。新stage保存了两套恢复的representation、GMM和实际创新输入；这些科学文件可在新目录用 replay.py 重放，无需新抽样/拟合或调用旧R11入口。正式再次执行需作为后续任务登记，避免覆盖本轮一次性token与旧结果。

已实际执行的重放PID153727/exit0：4000条/1920000点，pre-ADC与post-ADC均bitwise，通过回执 REPLAY_RECEIPT.json。对照头、条件均值、创新保持的配对检查在 metrics.json。这里的位级检查是模型科学重放，不是迁移完整性测试。正常文件迁移只保留SCP/SSH退出与实际加载；未声称远端/本地逐文件内容相等。

完整 metrics.json、NPZ、NN索引、分类预测及验证逐值保留E/D；METRICS_COMPACT.json仅聚合结果用于版本管理。所有模型均同一次采集TRAIN/DEV诊断，不称外部盲测或G4条件物理映射。
