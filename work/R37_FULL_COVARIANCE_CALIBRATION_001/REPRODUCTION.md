# R37 复现入口

此固定科学配方已关闭，旧stage不可覆盖、不可删一次性标记后重跑。需要复现时在新的E/D目录复制必要代码/配置并建立新运行记录；本文件不授权新实验或新拟合。

实际统计入口为服务器 `/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R37_REMOTE_20261009_204109_UTC/train_server.py`，由`linux_supervise.py`和`linux_owned_worker.py`以真实PID/nonce监督执行。Python是`/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`；R33原输入只读，现场Python3.12.3/NumPy1.26.4/Torch2.12.0+cu130/SciPy1.11.4、RTX5070，四CPU线程。固定R32 G以原GPUchunk32处理2950批，生成/REAL统计流与R33七项hash匹配，随后8个float64高斯map。无新神经训练。正常SSH/scp全部成功；没有为迁移新做对等内容测试。

本机CPU入口使用`C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`（Python3.13.3/NumPy2.2.6/Torch2.10.0+cpu/sklearn1.7.2）；单作业顺序为`supervise.py generate_all.py <CPU_python>`、`supervise.py evaluate_saved.py <CPU_python>`、`supervise.py replay_all.py <CPU_python>`。原本机监督还需要E项目`.runtime-cache/tmp`的TEMP/TMP、-B、PYTHONUTF8、PYTHONDONTWRITEBYTECODE和线程4。重放实际worker62156、supervisor60340，tool7b00fc/native0，2000条完整链各字段逐位一致；这属于科学模型复现，不是迁移测试。

最小推理输入为完整480点`ACTUAL_GENERATION_INPUTS.npz`（实际source/ID与z-white/style）、原冻结`E:\GAN\research\sim2real_two_papers_20261008_0305\work\R32_CONDITIONAL_MEAN_TRANSLATION_001\SERVER_FRESH_CONTROL\FINAL_TRANSLATOR.pt`、本轮`FULL_COVARIANCE_CALIBRATION/CALIBRATION_MAPS.npz`中的8个A/ms/mt及原TRAIN mu/scale/edges、本轮model/legacy_model/flow_runtime/covariance_core实现。实际source-Q选bin，支持外保留G；支持内原TRAIN坐标中的mean+线性map，转float32后仅一次原14bit[-.5,.5]V ADC。推理不读取REAL库、不重拟合矩阵、不修改父事件。`flow_*`名字为兼容模板字段，实际不是ODE流。

独立回放加载保存输入与map，完整重新计算G和map；不把保存的输出当答案输入。`REPLAY_RECEIPT.json`保存所有14字段核对和pre/post数组SHA，科学输入/执行源冻结分别见`PRETRAIN_FREEZE.json`、`LOCAL_PREGEN_FREEZE.json`、`STATISTICS_INPUT_BINDING.json`，最终文件版本见`FINAL_MANIFEST.json`。Linux统计与本机CPU是不同执行环境，不能称全平台浮点逐位相同；本机生成和本机独立重放逐位相同已执行。

评价复用原600/250分组与旧RAW RR/BASE/R33/R35/R36结果；没有新旧REAL重抽或旧模型重新拟合。完整metrics/逐事件预测/身份/私有矩阵留本地，METRICS_COMPACT只含聚合量。11张最终图QA记录及初图/第一次短标签布局原件均保留。报告、资源与最终manifest由纯控制脚本汇总，不再解码科学NPZ或运行模型。
