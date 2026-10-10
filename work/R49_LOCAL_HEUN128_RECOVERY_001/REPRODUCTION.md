# R49 本机恢复复现

本次两臂128/256数值检查均通过；8000条独立主输出重放通过。指定主候选A/B/C为False/False/False，项目联合目标尚未完成。

本轮复用R48两个50000更新末点，新增训练/校准为0。实际本机数值、生成和独立重放见TRAIN_SOLVER_CHECK、GENERATION_RECEIPT、REPLAY_RECEIPT与原生链；科学通过情况见GATE_METRICS及ACCEPTANCE_A_C。远端旧尝试的重放终态仍unknown，未恢复或冒充为本地结果。

实际本机根：E:\GAN\research\sim2real_two_papers_20261008_0305\work\R49_LOCAL_HEUN128_RECOVERY_001
Python：C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe

在新的独立E/D目录恢复本轮源码及config，把既有R48权重/核、R43坐标常数、R46检查包和冻结评价输入以及旧reference目录映射到实际只读位置。所需逐项路径在config_cpu.json和LOCAL_INPUT_INVENTORY.json；历史stage/模型不重复复制，原白噪声/ID/组成必须完整保留。Windows Python3.13/Torch2.10 CPU/NumPy2.2与原服务器版本不同，不宣称跨平台逐位一致。

先用python -B supervise_local.py check_models.py；只有两臂原128条TRAIN均通过才能启动python -B run_local_pipeline.py，依次运行generate_all.py、evaluate_saved.py、replay_all.py、summarize_and_plot.py、render_train.py。每入口实际PID/nonce在科学导入前与监督器握手。已有.used/active标记须先辨认真实终态，不在未知状态下重提；在新目录复现不改封存目录。

候选Heun128，参考Heun256只用于TRAIN；CPU4、chunk64、同一本机确定性后端。独立重放对8000主行pre/postADC及身份/辅助字段逐值比较。参数零更新；没有加载Adam或重新生成训练随机流。样本/40分类器/32密度槽是固定设计，失败必需区不可删除。

早期SOURCE_BINDINGS与LOCAL_SOURCE_FREEZE为准备时指纹。evaluate_saved.py纯日志标签修正、render_figures.py移除重复TRAIN绘制、make_report.py本机来源和指标解释补充属于已记录的工程/报告修订；科学定义及已执行结果不变，最终字节以FINAL_MANIFEST为准。

时间/体积预算字段保持null；机器4GiB安全内存、一次一个本机科学作业和真实磁盘容量依然检查，CPU不受GPU忙的准入条件限制。源码、日志、缓存、临时与结果仅E/D。科学replay是实际算法检查，不是迁移一致性测试。报告、图件实际QA和最终清单必须在最终封存前完成；封存后停止写入。

本次边界图还执行了python -B supervise_boundary_layout.py render_boundary_layout.py作单张布局修复；原图/脚本在display_initial，实际费用进入RUN_HISTORY。它只读已保存九个统计量，不重新生成、拟合或评价。
