# R35复现入口与固定依赖

已执行唯一PSD_ORDERED_CFM，2950末点权重`PSD_ORDERED_CFM/FINAL_FLOW.pt`；初态来自R33/INITIAL_FLOW.pt，冻结G来自R32/SERVER_FRESH_CONTROL/FINAL_TRANSLATOR.pt。完整source480波形、父ID及实际z/white/style见ACTUAL_GENERATION_INPUTS.npz；joint8与832探针原输入独立保存。支持/仿射常数AFFINE_CONSTANTS.npz直接复用R33，solver32与唯一14bitADC未改。

远端根：`/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R35_REMOTE_20261009_190721_UTC`。既有解释器：`/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`，Python3.12.3/NumPy1.26.4/Torch2.12.0+cu130/SciPy1.11.4/RTX5070，与R33环境匹配；读取R33远端inputs而非重新上传TRAIN。Linux真正科学PID43603受导入前nonce/RSS守护；launch_remote.py调用linux_supervise.py train_server.py。deploy_training.py、launch_remote.py、retrieve.py及TRANSFER/RETURN回执给出正常传输退出码，未声称位级迁移验证。

训练按原R26逐行源/REAL/bin/ordinal和原z-white/style/t读取，源preADC来自固定G，目标仅区内稳定PSD排列。原始流和实际目标排列分别为PSD_ORDERED_CFM/ACTUAL_CFM_STREAM_PRIVATE.npz、COUPLING_BY_UPDATE_PRIVATE.json，配位摘要COUPLING_SUMMARY.json。训练耗时9.241214s，INITIAL_FLOW科学初态摘要、八组实际流摘要及每批BASE比较见PAIRED_FLOW_RECEIPT.json；它们是科学配对核验，不是迁移测试。没有DEV checkpoint选择。

CPU评价/重放：`C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`，Torch2.10.0+cpu/NumPy2.2.6；主与独立重放同本机CPU环境。已执行入口：

```powershell
# 在新E/D目录恢复，不能向已封存目录写入，也不能删除原used标记重提。
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP='E:/GAN/research/sim2real_two_papers_20261008_0305/.runtime-cache/tmp'
$env:TMP=$env:TEMP
# 新目录config路径指向固定原依赖，deadline按新登记设置，保留实际科学定义。
& 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe' -B -u supervise.py replay_endpoints.py 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe'
```

原执行replay PID19328/native0、7.064447s已核对2000/2000 pre/postADC、source IDs、base各分量、精确f64增量和支持标记逐值一致，证据REPLAY_RECEIPT.json与replay_endpoints.py.NATIVE_RECEIPT.json。重放无需REAL/TRAIN搜索，不拟合。正常恢复需原R32/R33/R23/R26/R27固定依赖；未随阶段包重复迁移大库。evaluation/data_support及 config 路径给出旧REAL DEV/TRAIN参照、600/250固定行/组、旧预测及原密度控制位置。

首次evaluate_endpoints.py保存所有10fit/8slot及生成后仅聚合字段错误，保留native1原件；resume_evaluation.py直接读取保存C_*/WIDTH_*/generated/joint/probes完成缺失摘要，0重复HGB/EM/推理，不能重新运行原失败入口。summarize_and_plot.py与fix_CDF_layout.py均只读保存结果；CDF原图保留。完整metrics/权重/逐行身份私有，Git只应收录聚合、源码、报告和关键图。

本次用途是历史暴露同采集DEV开发诊断。postADC SOURCE_CLEAN参考source-Q与未量化实际NN source-Q分列；统计PSD成分/谷不是独立粒子真值。后流代数拆分可复现，物理signal/electronic归因unknown。config继承旧字段未追溯改写，实际运行以本轮protocol/native/execution/flow参数为准。
