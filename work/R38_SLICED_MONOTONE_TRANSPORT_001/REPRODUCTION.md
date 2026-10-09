# R38 复现

旧stage已封存且固定科学配方关闭。需要复现时在新的E/D目录建立新记录，不覆盖原stage、不删除一次性标记；本文不额外授权重拟合。

实际唯一统计入口：/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R38_REMOTE_20261009_211706_UTC/train_server.py，经linux_supervise.py/linux_owned_worker.py真实PID/nonce守护，Python /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python，Python3.12.3/NumPy1.26.4/Torch2.12.0+cu130/SciPy1.11.4、RTX5070。原G chunk32及R33原七条实际流只读，G前后相同；随后CPU4 float64方向学习/分位map。正常SSH/scp/native完成，没有迁移内容测试。

本机实际Python：C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe，Python3.13.3/NumPy2.2.6/Torch2.10.0+cpu/sklearn1.7.2。顺序入口：supervise.py generate_all.py <CPU_python>；evaluate_saved.py；replay_all.py；summarize_and_plot.py。-B/PYTHONUTF8/PYTHONDONTWRITEBYTECODE及E/D TEMP/TMP、cache，4线程与4GiB真实workerRSS守护。图显示修复入口repair_display.py不重拟合或生成。

完整推理需要ACTUAL_GENERATION_INPUTS.npz的完整source480/父ID/z/white/style、冻结G E:\GAN\research\sim2real_two_papers_20261008_0305\work\R32_CONDITIONAL_MEAN_TRANSLATION_001\SERVER_FRESH_CONTROL\FINAL_TRANSLATOR.pt、SLICED_TRANSPORT_PARAMS.npz中A[8,32,480,16]/xk/yk[8,32,257,16]/TRAIN mu/scale/8bin edges、model/legacy_model/flow_runtime/sliced_core。按未量化实际源Q选bin，支持外G回退；同full-bin变换顺序，每层完整矩乘与标量插值，转f32仅一次ADC。推理不查REAL库、不独立目标采样、不重拟合。

已执行独立2000条重放：actual3840/super12588，tool6df45b/native0，6.855693s，所有字段bitwise。记录含完整源、support/bin、base分量、out64、cast增量/舍入、pre/post、ID；保存输出不作为推理输入。Linux拟合与Windows生成环境不同，不称跨平台逐位相同；本机生成与独立本机重放一致已执行。256组训练轨迹、实际初始化hash/seed/版本与严格结点/逆检查分别在DIRECTION_INITIALIZATION.json、LAYER_*、STATISTICS_FIT_RECEIPT.json和SLICED_FORMULA_CHECK.json。

模板flow/covariance/calibration字段只是增量与二阶描述兼容名，实际不是CFM/ODE或R37高斯map。最终物理分支分解unknown。SOURCE_BINDINGS/FINAL_MANIFEST绑定本轮源码与产物，FIGURE_QA标最终两图修复版本，原图在layout_initial。报告/封存阶段只读JSON和纯字节hash，无数组解码/模型运行。
