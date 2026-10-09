# R10 实际入口与恢复说明

本轮已完成；以下记录用于恢复结果和后续明确登记的复现，不在封存目录再次训练。

- 本地阶段：`E:/GAN/research/sim2real_two_papers_20261008_0305/work/R10_MATCHED_COMPOSITION_AND_TARGET_ONLY_001`。
- 实际服务器：`rylan@100.82.231.102`，`rylan-GTi15`。
- 本轮新远端根：`/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R10_REMOTE_20261009_012338`。
- 已实际使用解释器：`/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`；Torch2.12.0+cu130、NumPy1.26.4、RTX5070 cuda:0。现有环境，无安装。
- 入口：在远端根运行该解释器的 `-B remote_supervise.py`；它只启动 `run.py`，成功后串行启动 `replay.py`。四个CPU线程，GPU训练/生成，4GiB RSS和allocated上限；`config.json`保存实际定义，`READINESS.json`绑定执行源码。原launch token、预算和绝对截止不支持再次提交。

## 输入恢复

`config.json.input_files`精确列出26项本地已有输入路径、当时版本SHA和大小；远端实际读取路径为本轮根的 `inputs/<key>`。原Windows路径只是来源绑定；远端不从这些Windows路径读取。`UPLOAD.tar`包含这些输入、必要源码、任务和位置修订。它是正常传输包，没有源/目的逐文件迁移一致性验证。`TRANSFER_RECEIPT.json`及`RETURN_TRANSFER_RECEIPT.json`记录正常退出。

需要在一个新的独立恢复目录解包必要源码/配置/26项输入，保留 `inputs/<key>`映射。不要在旧阶段覆盖源码或运行。原输入角色为：G4 TRAIN缓存8192（fit6593/val1599），REAL DEV2000/原TRAIN3200（QC3191，fit2592/val599），R5已存TRAIN生成仅取R3B_V及其父身份，R3B/FULL/OFF既有DEV输出、固定CAL参考和8-bin定义，以及原R8C初态/目标/验证/RNG。没有EVAL，也不调用含EVAL的R5原run。完整重新训练或评价应作为下一次有界任务另记实际环境、绝对时钟与成本，不能通过改旧token悄悄复跑。

## 保存模型的推理复现

将 `model.py`、`observable_source.py`、`cuda_runtime.py`、`replay.py`，以及 `arm/model.pt`、`GENERATION_INPUTS.npz`、`generated.npz`按原相对结构放入新的独立目录。`replay.py`只加载选中step1500的实际权重和保存的2000条实际source/noise输入，在相同新CUDA后端以64行块推理，与保存主波形比较；无随机重抽、拟合或训练。需要预先把临时目录和缓存放在自己的项目目录，保留4线程；设备/库漂移应记录，不能预先保证跨环境位级相同。

这项科学重放已经由监督器执行一次：远端PID125361，native0，2000行/960000值 bitwise_equal=true、maxabs0；数组SHA为 `15e3773068b12bb2d6b6aecd8495992e3d400c172cc07bb5601d87a1064f721d`。证据：`REPLAY_RECEIPT.json`、`RUN_HISTORY.json`、`REMOTE_NATIVE_RECEIPT.json`、`SSH.stdout.log`、`LOCAL_NATIVE_RECEIPT.json`。它验证保存模型和实际生成输入的科学推理复现，不是迁移一致性测试。本次不再次执行。

## 比较限制

原R8C在本机Torch2.10.0+cpu环境完成。R10复用了保存G/D参数、CPU随机状态和实际TRAIN目标/验证输入；固定验证输入摘要一致，但实际训练合并采样摘要0/2000与旧记录匹配，差异原因未定位。因此FULL/OFF是同设计历史参照，不能称严格采样配对或CPU/GPU训练逐位等价；未为定位差异追加训练。A的共同支持未加权/加权/REAL–REAL在本轮同一服务器环境计算；原总体R7明确为历史引用。

完整私有身份、预测、近邻索引、训练每步摘要、模型与数组保留本地/本轮远端目录。发布用 `METRICS_COMPACT.json`、`REPORT.md`和最终 `charge_psd_uniform.png`等关键图，不上传完整私有载荷。
