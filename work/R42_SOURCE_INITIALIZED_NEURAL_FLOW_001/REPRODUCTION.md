# R42 复现与停止位置

完整执行模型为 `network.py` + `source_flow_core.py`，唯一权重 `SOURCE_NEURAL_FLOW_FINAL/FINAL_FLOW.pt` 和固定 `UNIT_CHARGE_CONSTANTS.npz`。训练入口 `train_server.py`，服务器 `/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R42_REMOTE_20261009_231107_UTC`，Python `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`，实际NumPy1.26.4/Torch2.12.0+cu130/CUDA RTX5070；初态、optimizer、actual CFM行/耦合/t与未使用审计流均保存。输入从config中已绑定R23 TRAIN池/R26重组及R33原t流只读复用，没有重跑G4或获取新实测。

本机检查用 `C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`，Python3.13.3/Torch2.10+cpu/NumPy2.2.6，CPU4。入口 `check_train_solver.py` 由 `supervise.py`→`owned_worker.py` 导入前nonce/PID/RSS监督。固定chunk64对TRAIN128行做32/64检查，保存精确输入与解；这是数值参考检查，不是迁移测试。复现须在新E/D目录恢复，保留旧阶段和原 `.used` 文件，不能覆写本stage。

两环境不同，不宣称CPU/CUDA训练逐位一致。已保存末点CPU TRAIN检查正常执行且数值失败；未执行DEV生成、20HGB/16槽或4000主链回放，因此不声称推理主输出/训练初始化全链已通过独立复现。预备 `generate_all.py/evaluate_saved.py/replay_all.py/summarize_and_plot.py` 本轮未执行，必须尊重 `STOP_CANDIDATE.json`；不能将它们当作获准绕过检查的复现实验。

必要图入口 `plot_train_failure.py` 只读取本轮已保存TRAIN trace、128两解和固定坐标，不再求解、拟合或生成。正常迁移/取回pack和scp均native0，未做迁移内容一致性测试；原输入版本绑定保留。所有本机缓存/TEMP/MPL路径由监督器限定E盘。根负责Git及D盘交付。
