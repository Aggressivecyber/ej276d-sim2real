本轮在本机E盘执行，无SSH。训练Python：E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe；公式检查也使用该既有Torch。CPU评价/重放/绘图Python：C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe，只读解释器，数据/临时/缓存写stage/cache。线程4，supervise.py/owned_worker.py在科学导入前绑定真实PID/nonce并确认RSS守护，launcher单独记录。

顺序是supervise.py check_energy_formula.py GPU_PYTHON → train_gpu.py GPU_PYTHON → evaluate_endpoints.py CPU_PYTHON → replay_endpoints.py CPU_PYTHON → render_figures.py CPU_PYTHON；上述所有入口均受同supervise守护，具体native链在NATIVE_CHAIN/RUN_HISTORY。原目录.used禁止重复提交；复现应在新E/D目录复制必要文件并调整路径，独立注册预算/期限，不能删除原.used或覆盖旧阶段。

config/PROTOCOL/PRETRAIN_FREEZE保存首次训练前的方法和执行源SHA；EVALUATION_SOURCE_FREEZE固定评价与绘图源。R23 TRAIN_PRIVATE/VALIDATION_INPUTS/FINE_CONSTANTS/CRITIC_SCALER只读复用；reorder_core用原R23流和R26实际重排行/状态重建全部94400行，R21 step2000 G/reader与R18 INITIAL_CRITIC共同起点。SLICED_ENERGY/TRANSLATOR.pt、STYLE_READER.pt、CRITIC.pt、OPTIMIZER_SELECTED.pt为2750同步所选；FINAL_*和OPTIMIZER_FINAL.pt为2950末点。

energy_core.py只覆盖训练距离，validation继承原W1。ENERGY_TERMS_PRIVATE.json保存每次调用三个交叉/自项、U/负方向比例/混合bin上下文；FRESH_PROJECTION_STREAM.json保存逐ordinal与R28匹配，FIRST/LAST_PRIVATE和TRAINING_RANDOM_STATES存原随机状态。config记录独立PCG64种子，验证不消费它；训练项自然尺度未重权重。

ACTUAL_GENERATION_INPUTS/JOINT8_INPUTS_PRIVATE/PROBE_INPUTS_CORRECTED是保存的实际推理输入。ENERGY_SELECTED/generated.npz与ENERGY_FINAL/generated.npz各2000 pre/post ADC、signal/electronic、源ID；model.py纯推理重放用同步保存权重。REPLAY_RECEIPT及真实native回执证明4000条独立重放已执行且逐值一致；不再重做完整评价。

旧R28控制输出/预测/密度和R27固定线性空间/四投影块按config只读复用，固定600/250行及组保持，不重fit旧结果。这些DEV方向已暴露，不称新外部盲测。完整metrics/predictions/源身份/actual stream/模型留E/D，METRICS_COMPACT为聚合发布版。最终清单只封存本stage字节SHA，不重审3GB历史源或运行迁移一致性测试。
