# R39复现与部署

本轮训练实际路径是服务器R39_REMOTE_20261009_214432_UTC下的train_recover.py；第一次train_server.py失败原件保留，不把它写为成功入口。执行器原环境Python3.12.3/NumPy1.26.4/torch2.12.0+cu130，GPU RTX5070；本机后续Python3.13.3/NumPy2.2.6/torch2.10.0+cpu，CPU4。

训练采用原R32 inputs只读（路径见config.json）：TRAIN_PRIVATE.npz、VALIDATION_INPUTS.npz、FINE_CONSTANTS.npz、CRITIC_SCALER.npz、R26_TRAINING_STREAMS_PRIVATE.npz、原R23随机与R26回执、INITIAL_TRANSLATOR/STYLE_READER/CRITIC.pt。恢复入口只复用已保存64批c，原失败与恢复源码/配置均保存。要重训须新独立目录并获得对应新任务；本封存目录不可直接重运行或解除.used。

本轮实际命令链：launch_remote.py train_server.py（失败）→launch_remote.py train_recover.py（成功）；科学前由linux_supervise.py和本地SSH回执绑定实际PID。正常返回tar保留，无迁移一致性测试。初态/流/参数由PRETRAIN_FREEZE、ENGINEERING_RECOVERY_RECEIPT、MATCHED_TRAINING_FLOW_RECEIPT和最终清单区分绑定，不把修复后配置冒充首份原字节。

本机后续通过supervise.py <entry> <Python绝对路径>启动；TEMP/TMP精确E项目/.runtime-cache/tmp，PYTHONUTF8=1/PYTHONDONTWRITEBYTECODE=1与-B，CPU4。实际顺序generate_all.py → evaluate_saved.py → replay_all.py，各自nonce/PID/RSS导入前握手。评价按原固定600/250与同人工组，不能据DEV重选2750或调核。

部署仅用model.py/legacy_model.py、config.json记录的源定义与JOINT_KERNEL_FINAL/FINAL_TRANSLATOR.pt，再提供实际完整source480 V、z/white/style。G不读取REAL库、reader或评价器。输出只经过既定一次14bit ADC；全部历史支持外source仍用新G。单父源关联不意味着无配对学习识别出唯一真实事件映射。

ACTUAL_GENERATION_INPUTS.npz保存主2000实际source/ID/z/white/style；REAL_DEV_AND_REFERENCE_PRIVATE.npz中reference_V/reference_ids/reference_groups保存同处理REAL输入。两REAL控制复用完全相同主随机量，不得给REAL改为零随机量。各端点generated.npz保存preADC/postADC、功能分量、身份，SOURCE_PROBES_PRIVATE与JOINT8_PRIVATE保存辅助。

独立完整CPU replay_all.py已在worker53388/super52640实际原生0执行：6000条，所有身份/功能分量/pre/postADC逐值通过，见REPLAY_RECEIPT与NATIVE_CHAIN。从新目录恢复时先复制必要执行源/权重/实际输入/比较载荷及配置路径，保持chunk128和运算顺序；只做推理，不重新拟合。旧目录中的.used专门防重复启动。

本轮绘图实际由summarize_and_plot_recover.py成功落8图后在trace初始行失败，finish_plots.py完成余4图，fix_final_figures.py仅修三图显示。失败源/回执/原图保留；最终SHA以FIGURE_QA及FINAL_MANIFEST为准，首份PLOT_SUMMARY的三旧SHA已被修图回执取代。所有图来自保存的结果，无重新评价。

完整逐事件位置、随机流、预测、权重/科学数组仅E/D；Git发布只用聚合compact与代码图文。INITIAL_GRADIENT_SCALE_COMPACT不含source_rows/target_rows/ordinal数组，原全文本地保留。旧R11–R17目标密度自生成、R33–38后处理路线均不是本轮部署链。
