# R33 复现入口

这是固定配方负结果的复现资料，不默认授权再次训练；复现应放新目录，保持本阶段只读。

实际训练位置：rylan@100.82.231.102，
`/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R33_REMOTE_20261009_175239_UTC`。
既有解释器 `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`，CUDA:0、CPU4。
入口 `linux_supervise.py train_server.py <解释器>` 经 `launch_remote.py` 发起，实际worker31584/super31583/native0；实际环境在remote_receipts/SERVER_ENVIRONMENT.json。

正常上传/返回由deploy.py、retrieve.py完成。现有输入为R23 TRAIN_PRIVATE、R26完整同区94400行及原R23 z/white/style流、R32 CONTROL_FINAL权重；精确路径配置在remote_config/config及PRETRAIN_FREEZE。
仅复用了已有TRAIN/DEV，未取EVAL/CAL波形或长采集。部署使用正常传输退出和加载状态，没有迁移对等哈希测试。

保存的INITIAL_FLOW和实际t/置换/匹配目标、训练全stream、初末RNG均保留；初态完全相同、2950实际末点、只速度MSE。
表示/仿射常数在AFFINE_CONSTANTS与AFFINE_METADATA，源Q的8区定义不改。匹配按每批原Q子组，不跨Q；NCE旧位置只记录而不用于本轮CFM。

本机实际解释器 `C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`，Torch2.10.0+cpu/NumPy2.2.6。
所有TEMP/TMP/TORCH_HOME/CUDA_CACHE_PATH/MPLCONFIGDIR/JOBLIB临时路径由supervise放到本阶段cache，线程4，导入前PID握手/RSS守护。

本轮实际成功评价入口：
`python -B supervise.py evaluate_endpoints_attempt002.py <本机解释器>`。
attempt001因缺少random_diagnostics在导入时退出1，保留原件，attempt002仅补必要旧辅助源码，不变数学。
实际worker33800/super35696/native0，20HGB、14EM、16描述槽；每fit/slot逐项持久化。

保存权重独立推理入口：
`python -B supervise.py replay_endpoints.py <本机解释器>`。
实际worker21608/super43484/native0。读取ACTUAL_GENERATION_INPUTS、冻结BASE权重、两份FINAL_FLOW、AFFINE_CONSTANTS；不读取REAL/TRAIN/内部验证，不拟合或更新。
完整源→G→固定32步Heun→一次14bit ADC的4000行全部逐位通过，结果REPLAY_RECEIPT。
32vs64只预定前32父/臂检查，未据结果选步数；不以64替代发布32。

纯保存结果汇总/绘图入口 `summarize_and_plot.py`；实际worker61876/super44116/native0，11图，0fit/0生成/0新随机投影。完整数值metrics留本地；METRICS_COMPACT是无身份/预测聚合版。

文件传输或新平台训练不承诺逐位一致；本轮bitwise只指已执行的本机保存权重完整链回放。
最终代数分量base_signal/base_electronic/flow_increment可重构总输出，不能当作新的物理信号/独立电子噪声分解。
