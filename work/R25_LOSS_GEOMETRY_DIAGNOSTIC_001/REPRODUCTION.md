本轮是冻结损失诊断，重现不训练或选择响应模型。实际CPU解释器为C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe，B用E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe（Torch2.12.0+cu130，已有RTX4060）。所有TEMP/TMP/Torch/CUDA/Matplotlib/Joblib缓存由supervise.py设置在本轮E盘cache；4线程，实际science PID导入前握手，launcher PID单列。

在新E/D目录复制本轮必要源码/config/PROTOCOL，保留科学定义及INPUT_VERSIONS.json中旧R21/R18/R23/R24只读输入路径。按新的有界执行安排更新仅deadline/runtime控制，不能在已封存本轮目录重跑或覆盖.used标记。

CPU运行supervise.py contraction_cpu.py <CPU_python>，再CPU监督supervise.py gradient_gpu_repair001.py <GPU_python>，最后supervise.py summarize_and_plot.py <CPU_python>及repair_display_semantics.py <CPU_python>。原gradient_gpu.py是失败历史，不作为重现入口。顺序只有一个科学作业；无optimizer构造/step。

A实际512对抽样及PCG64初末态已保存。B前32批source/REAL/bin/NCE索引直接读取R23实际消耗流；z/white/style按其保存初态和逐批z→white调用重建，B_ACTUAL_INPUTS_PRIVATE.npz保存同一实际输入供三个状态。原权重路径见config/INPUT_VERSIONS，模型及其梯度没有写回旧目录。B有每状态一次直接总梯度核对及全参数未更新证据，没有新增2000 DEV主波形或科学重放。

原生作业：A actual61776/native0；B失败actual24636/native1（duplicate serialized source key，模型前向0）；B修复actual57620/native0；汇总绘图actual59272/native0；显示修复actual60552/native0。RUN_HISTORY及各entry.NATIVE_RECEIPT、SCIENCE_PID、WORKER_RECEIPT保留真实链。输入迁移/内容一致性测试0，科学输出诊断核对不代表跨Python/Torch设备逐位一致。
