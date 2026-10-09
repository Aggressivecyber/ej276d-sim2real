# R32复现入口

实际训练位置为 `/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R32_REMOTE_20261009_170201_UTC`，Python `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`（Python3.12.3，NumPy1.26.4，Torch2.12.0+cu130，RTX5070）。先在全新同任务目录准备依赖，再以 `linux_supervise.py train_server.py` 执行。已有目录的一次性token已消费，不在原目录重复启动。DEPLOY/TRANSFER/RETURN回执记录正常传输退出，未做迁移一致性测试；不安装共享环境。

本机CPU入口 `C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`，`supervise.py` 使用真实PID/nonce导入前握手及RSS守护。环境变量TEMP/TMP指E项目缓存，-B/UTF8，4线程。原 `evaluate_endpoints.py` 已消费且在最终汇总断言处失败；已生成/拟合的数值不重做。`assemble_saved_results.py`仅从保存输出/逐fit/slot组装，`source_semantics_and_gate.py`仅做授权保存数组ADC语义核对及门槛汇总，`render_figures.py`读取保存图数据。

独立主推理 `replay_endpoints.py` 实际已执行（worker37712，native0），8000条pre/postADC/signal/electronic逐值及源ID次序一致，证据在REPLAY_RECEIPT。可在全新恢复目录重放，模型输入为两个臂所选/末点TRANSLATOR.pt、实际ACTUAL_GENERATION_INPUTS的source/z/white/style、原model/legacy_model；推理不需要TRAIN/REAL或reader。不能把新推理写进已封存目录。

PRETRAIN_FREEZE、remote_receipts保存服务器实际执行配置/源码。config的R28_root是在首CPU评价前补的本地历史结果路径，未修改已执行远端配方；原remote_config及freeze保留。继承的旧numeric_wall_limit描述不控制本轮2400科学预算，实际执行字段/RUN_HISTORY/监督器和task为准。源码/配置/权重及科学输入在FINAL_MANIFEST引用。所需复用依赖来自R23 TRAIN/VALIDATION/FINE/critic scaler、R26行流、R21初G/reader、R18初D；11依赖正常上传，原目录只读。CPU Torch2.10.0+cpu/NumPy2.2.6与服务器环境不同，因此历史R28不作新损失的同平台因果对照。

科学公式、角色和固定停止规则见本轮PROTOCOL与任务。不是新盲测，不恢复被输入遗漏的微观沉积时间；不根据DEV重选端点。固定lambda=1门槛失败已关闭，复现不意味着授权扩训练。
