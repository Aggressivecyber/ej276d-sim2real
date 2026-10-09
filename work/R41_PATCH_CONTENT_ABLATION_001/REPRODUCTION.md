# R41 执行与复现

实际服务器入口 `/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R41_REMOTE_20261009_224636_UTC/linux_supervise.py train_server.py`，Python `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`，Python3.12.3/NumPy1.26.4/torch2.12.0+cu130/scipy1.11.4/cuda:0。先资源/本人argv+cwd预检，再正常源码scp；训练只读复用原R32/R33的缓存，不做迁移一致性测试。

准备 `prepare_stage.py`、`prepare_summary.py`，资源 `resource_check.py`，`deploy_training.py` 冻结源码与配置；`launch_remote.py train_server.py` 唯一启动，`retrieve.py`正常取回。执行源与版本在PRETRAIN_FREEZE/EVALUATION_SOURCE_FREEZE；原model/legacy_model保持R32，`online_training.py`只取消实际NCE与目标并加入审计。cfg中少量remote历史字段来自完整R32配置，以execution/remote_root/实际native为准；evaluation结束日志的R40字样为模板标签，实际域键NO_PATCH_NCE_FINAL/REAL_CONTROL明确。

本机CPU入口 `supervise.py generate_all.py <CPU_python>`、`evaluate_saved.py`、`replay_all.py`、`summarize_and_plot.py`，CPU Python `C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`；全部TEMP/TMP/cache在本轮E目录，-B/PYTHONDONTWRITEBYTECODE/UTF8，4线程。不得运行旧R40的prepare/summary来覆盖本轮文件。

固定模型NO_PATCH_NCE_FINAL/FINAL_TRANSLATOR.pt（2950）；TRANSLATOR.pt为内部step1500记录，不是本轮DEV终点。原2000完整输入/ID/z/white/style在ACTUAL_GENERATION_INPUTS，REAL与reference在REAL_DEV_AND_REFERENCE_PRIVATE，joint/probe及CAL_REFERENCE_METADATA仅沿用既有定义。新权重、optimizer/reader/D、未用NCE位置流、投影头初末、实际随机流均保存。BASE_REAL直接引用R39，零再次推理/拟合。

独立4000主完整链已经执行native0、逐值pass，证据REPLAY_RECEIPT和对应NATIVE_RECEIPT，未读保存输出作输入。不同CPU/CUDA库训练不保证跨平台位等，重放只比较本轮同CPU生成。未来恢复在新E/D目录重新登记时间/路径，保存科学config/实际输入/权重/随机量不变；不得删除.used/SSH标记重复提交已封存作业。本文件不触发新的科学计算。
