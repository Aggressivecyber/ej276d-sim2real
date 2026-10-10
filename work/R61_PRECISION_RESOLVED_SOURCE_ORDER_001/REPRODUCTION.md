# R61 推理、单因素核验与复现

本轮相对封存 R59 只改变源 PSD 排序损失的分子：源 float32 PSD 保守区间重叠时不施加顺序罚。分母、源人口、其余六项损失、全部外层系数、网络与训练终点保持。区间是输入存储精度包络，不是粒子标签或物理测量不确定性。

唯一可交付候选是预登记的第 50000 次更新。训练与全部预登记评价、独立重放已完成；具体结果以原生退出、`TRAINING_RECEIPT.json`、`TRAINING_FROZEN_AUDIT.json`、`ACCEPTANCE_A_C.json` 和 `REPLAY_RECEIPT.json` 为准。

## 可移动推理

将下列文件保持相对位置；服务器 TRAIN、DEV 和历史评价目录不参与推理。

- `infer_saved.py`、`flow_runtime.py`、`source_flow_core.py`、`network.py`、`noise_core.py`
- `model.py`、`legacy_model.py`，用于原固定 ADC 操作
- `UNIT_CHARGE_CONSTANTS.npz`、`NOISE_CALIBRATION.npz`
- `PRECISION_RESOLVED_RESPONSE/FINAL_FLOW.pt`，唯一第 50000 步末点

输入 NPZ 提供 `source_waveforms_V` 和 `white`，均为 `N × 480`。波形按原保存窗口对齐，单位为 V；`white` 为显式保存的标准正态数组。可附带 `source_ids`。同处理 REAL 使用完全相同的网络、噪声、积分、约束投影和 ADC，不使用域标志旁路。

```powershell
python -B E:/path/to/R61/infer_saved.py --input E:/path/to/input.npz --output E:/path/to/new-output.npz --device cuda:0
```

入口也接受 `--device cpu`；跨设备位级一致性未经证明。输出必须尚不存在，Windows 下包、缓存和输出只写 E 或 D。预登记重放在独立服务器 CUDA 进程中，通过该可移动加载入口核对原 2000 SOURCE 与 2000 同处理 REAL 的全流程输出、身份和每事件诊断。

电荷 Q 必须有限且为正。`abs(Q)>300 V ns` 时全窗电压箱和电荷条件不相容，入口会报错，不删事件。支持外输入仍经过网络，使用最近分区标准化常数和未经截断的连续 logQ。该处理规则本身不证明支持外准确性。

网络由原全窗、投影有色噪声和电荷条件预测切空间位移。同一位移在 Heun128 节点间缓存，随后执行全窗电压 `[-0.5,0.5] V` 和原电荷的联合欧氏投影，再执行原 ADC。数学场求值与实际神经调用分别计数。内部投影修正单独报告；后续 ADC 零 clip 不代表模型内部没有修改波形。

## 训练条件与单因素证据

`config.json`、`PROTOCOL.md`、`INPUT_BINDING.json` 和 `ROOT_IMPLEMENTATION_REVIEW.json` 绑定注册条件、输入与实际执行源码。R59 的唯一末点是固定历史比较，不用来初始化 R61。

R61 从同一随机初始化重新训练。模型种子 20261010590；初始逐张量 SHA 必须为 `4a51097e257d4b35976452982aa9a84f48e321d425e0ba3859e8afe8a7bd239b`。六条 PCG64 流及调用顺序与 R59 完全相同。训练末点不仅比较种子，还核对实际消费字节的六条链式 SHA256、全部父事件曝光次数和最终 RNG 状态。

全部 6593 SOURCE / 21047 REAL 父事件、十个电荷区域保留。每更新各区域各抽取 32 SOURCE 和 32 REAL 参考，另抽取 64 REAL 保持事件；原区域质量恢复全局等父事件目标。期望权重等父事件不代表有限曝光次数完全均匀。

冻结 R59 的 TRAIN 分布标准化文件直接复用，不重新拟合。网络、单位电荷常数、有色噪声、原 Adam 与余弦学习率、梯度裁剪、50000 更新终点、原 Heun128/参考256 均保持。训练解码范围限于原合法 TRAIN，原数据包内的验证数组不打开。

训练目录保存初始/最终 RNG、初始权重摘要、六条消费哈希、父事件曝光、原 20 列逐更新记录及 `PRECISION_ORDER_BY_UPDATE_PRIVATE.npz` 的四列排序分解。逐步检查原排序罚等于保留罚加省略罚。`cache/RECOVERY_SAME_RUN.pt` 为单个滚动恢复缓存，不能用作另一个科学候选。

`precision_audit.py` 重新读取保存记录，对照冻结 R59。`TRAINING_FROZEN_AUDIT.json` 必须通过后才进入本轮数值前向和 DEV。检查包括常数字节、源身份/原始统计、曝光、六条实际消费哈希、最终 RNG、区间包含性与全部 50000 次排序分解。`POST_AUDIT_ARTIFICIAL_CHECK.json` 的正例和六类独立扰动反例仅用人工数组，不读取科学数据。

## 固定执行顺序与评价

服务器环境沿用 Python 3.12.3、NumPy 1.26.4、SciPy 1.11.4、PyTorch 2.12.0+cu130、RTX 5070。CPU12、作业树 RSS48 GiB、GPU allocated/reserved 各10 GiB，单科学作业；没有计算时间配额。

唯一训练末点通过一致性审计后，按顺序执行 `numerical_check.py`、`generate_all.py`、`evaluate_saved.py`、`replay_all.py`、`summarize_and_plot.py`。串行控制器只在前项原生退出 0 且真实进程不存在后启动下一项；观察超时不重提训练。原生资源和退出回执逐项保存。后处理 31 个文件由 `POSTPROCESS_FILE_BINDING.json` 冻结。

原 128 条 TRAIN SOURCE/REAL 的 Heun128/256 数值门通过后，一次性生成原主输出与全部固定探针，共 5984 条；完成五视图 10 个新 HGB、16 个密度描述槽和 4000 条重放。SOURCE 身份、组成、顺序、等父事件权重及评价划分均沿用原协议。

R61−R59 配对差复用 R59 已冻结预测；RAW REAL–REAL、BASE 和 R49 比较也复用封存结果，不重新拟合历史对照。初始化、随机流及曝光一致支持本固定运行的单因素解释；配对置信区间仍条件于固定分类器，单个训练种子不能估计重复训练或独立采集的不确定性。

原 A–F 不变。训练损失、数值稳定、投影可行或位级重放均不替代完整响应、PSD/峰位峰宽、源关联、噪声、边界与同处理 REAL 的联合要求。A–C 全部通过后才另行登记 D 任务效用与 E FULL/GAP。历史 DEV 已暴露，独立采集泛化仍未验证。

## 实际交付

私有 Git 与 D 盘交付以项目根的 `R61_PUBLICATION_RECEIPT.json`、`R61_BACKUP_RECEIPT.json`、`R61_DELIVERY_COMPLETION_RECEIPT.json` 为准。文件尚不存在时不能宣称交付完成；单轮研究结束也不等于原完整模型目标完成。
