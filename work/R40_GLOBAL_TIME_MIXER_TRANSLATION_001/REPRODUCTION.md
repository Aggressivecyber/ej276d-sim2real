# R40 执行与复现

实际训练在 `rylan@100.82.231.102` 的 `/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R40_REMOTE_20261009_221922_UTC`，解释器 `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`；Python3.12.3 / NumPy1.26.4 / torch2.12.0+cu130 / scipy1.11.4 / cuda:0（RTX5070）。本机生成、20HGB/15EM、独立4000重放和绘图使用 `C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`，torch2.10.0+cpu / NumPy2.2.6；两端不是逐位训练平台对照，CPU重放只与本轮CPU保存输出逐值比较。

实际输入与版本在 `SOURCE_BINDINGS.json`、`PRETRAIN_FREEZE.json`、`EVALUATION_SOURCE_FREEZE.json`。本轮 `ACTUAL_GENERATION_INPUTS.npz` 保留原2000完整source480、IDs、z/white/style；`REAL_DEV_AND_REFERENCE_PRIVATE.npz` 保留既有DEV与2000TRAIN_REFERENCE；`JOINT8_INPUTS_PRIVATE.npz`、`PROBE_INPUTS_CORRECTED.npz` 保留预定随机/源控制。服务器训练只读复用原R32/R33的训练载荷，原94400行和七流/8850方向的核对记录在 `MATCHED_TRAINING_FLOW_RECEIPT.json`。没有迁移逐文件相等测试；传输以正常scp/ssh退出与实际加载为据。

唯一科学模型为 `GLOBAL_TIME_MIXER_FINAL/FINAL_TRANSLATOR.pt`（2950）。`TRANSLATOR.pt` 是内部1250所选记录，不能替代本轮唯一DEV终点。`INITIAL_FULL_TRANSLATOR.pt`、`INITIAL_TIME_MIXER.pt`、reader/D初态与最终状态、optimizer和每步随机/梯度轨迹都保留。源结果在 `GLOBAL_TIME_MIXER_FINAL/generated.npz`，REAL处理控制在 `GLOBAL_TIME_MIXER_REAL_CONTROL/generated.npz`；旧BASE_REAL由R39封存目录引用，不能再次生成来替换历史对照。

本轮实际顺序：`prepare_stage.py` → `finish_prep.py` → 资源预检/`deploy_training.py` → `launch_remote.py train_server.py`（调用Linux `linux_supervise.py train_server.py`）→ `retrieve.py` → `prepare_evaluation_r40.py` → 本机监督 `generate_all.py`、`evaluate_saved.py`、`replay_all.py` → `prepare_summary.py`、监督 `summarize_and_plot.py`。历史模板 `prepare_evaluation.py` 明确禁用，不能用来覆盖本轮入口。

本机实际监督命令形式（cwd为本轮E目录，所有TEMP/TMP/缓存在项目E盘）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:TEMP='E:/GAN/research/sim2real_two_papers_20261008_0305/.runtime-cache/tmp'
$env:TMP=$env:TEMP
& 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe' -B ./supervise.py replay_all.py 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe'
```

该主重放已经独立执行：真实PID16360/监督40420、native0、7.0646635s；4000行全部preADC/postADC、signal/electronic/sum-rounding、父/参考ID与支持分支逐值通过。推理没有把保存输出或reader/评价器作为计算输入；保存输出仅作为最终比较对象。旧BASE_REAL独立回放直接引用R39证据。本轮再现事实见 `REPLAY_RECEIPT.json` 与完整原生日志。

未来恢复须使用新E/D目录及新资源/时间登记，保持科学config/源/权重/实际输入/随机流固定，只调整执行路径和时限。当前目录已封存，`.used`/SSH一次性标记与旧期限不得删掉后重复提交。训练从头再现仍受平台/库版本影响，不保证跨CPU/CUDA位等；本轮独立推理4000位等是已执行的有限证据。无需为此文档再启动计算。
