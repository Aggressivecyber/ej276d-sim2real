# R59 推理与复现

本轮使用一个全 480 点神经网络，直接优化无配对源波形和实测波形的分布及事件保持损失。没有固定目标教师波形。唯一可交付模型是预登记的第 50000 次更新；训练与全部预登记评价、独立重放已完成；具体结果以原生退出、`TRAINING_RECEIPT.json`、`ACCEPTANCE_A_C.json` 和 `REPLAY_RECEIPT.json` 为准。

## 可移动推理

将本目录的下列文件保持相对位置。服务器外部 TRAIN、DEV 或历史评价目录不参与推理。

- `infer_saved.py`、`flow_runtime.py`、`source_flow_core.py`、`network.py`、`noise_core.py`
- `model.py`、`legacy_model.py`，用于原固定 ADC 操作
- `UNIT_CHARGE_CONSTANTS.npz`、`NOISE_CALIBRATION.npz`
- `DIRECT_DISTRIBUTION_RESPONSE/FINAL_FLOW.pt`，唯一第 50000 步末点

输入 NPZ 必须提供 `source_waveforms_V` 和 `white`，均为 `N × 480`，按原保存窗口对齐。前者以 V 为单位，后者为显式保存的标准正态随机数组；可附带 `source_ids`。同一个输入和 white 可独立重复运行。所有事件使用相同算子，不依据来源标志对 REAL 旁路。`z` 与 `style` 不参与本轮网络。

```powershell
python -B E:/path/to/R59/infer_saved.py --input E:/path/to/input.npz --output E:/path/to/new-output.npz --device cuda:0
```

入口也接受 `--device cpu`；跨设备位级一致性未经证明。输出路径必须尚不存在，Windows 下包、缓存和输出只能在 E 或 D 盘。实际独立重放使用服务器 CUDA 环境、64 条分块，必须逐位比较 2000 个 SOURCE 与 2000 个同处理 REAL 的全流程结果、身份及每事件诊断。

原始电荷必须有限且为正，`abs(Q)>300 V ns` 的 box-charge 问题无解时会报错，不删事件。支持外事件仍经过网络，使用最近分区的标准化常数和未经截断的连续 logQ 坐标。这个处理规则本身不证明支持外具有实测准确性。

网络从原全窗、投影有色噪声及连续电荷条件预测切空间位移；同一位移在 Heun128 节点间缓存。随后明确执行全窗电压 `[-0.5,0.5] V` 与原电荷的联合欧氏投影，再执行原 ADC。报告区分数学场求值与实际神经调用。内部投影的事件数、修正 RMS/最大值和到达电压上下限的点数必须保留；不能用后续 ADC 的零 clip 数声称内部没有约束修正。

## 完整训练记录

`config.json` 和 `PROTOCOL.md` 固定本轮方法；`INPUT_BINDING.json` 绑定合法 TRAIN、坐标、噪声、数值探针和谷定义。服务器环境为 Python 3.12.3、NumPy 1.26.4、SciPy 1.11.4、PyTorch 2.12.0+cu130、RTX 5070。CPU 12、RSS 48 GiB、GPU 10 GiB，单科学作业。

源和 REAL 原始父事件分别为 6593 / 21047。每步在十个原电荷区域内分别放回抽取 32 条源和 32 条 REAL，整体损失显式使用各自原区域质量；额外抽取 64 条等概率 REAL 计算同处理保持损失。全局等父事件目标是在抽样期望意义上恢复，实际有限抽样次数不声称完全一致。

初始化种子和六条 PCG64 流见配置。训练目录保存初始/最终 RNG 状态、实际消耗字节的六条链式 SHA256、每个父事件的累计出现次数、逐更新全部损失和唯一末点权重。`TRAINING_RECEIPT.json` 给出流消费顺序；重建历史流需要完全相同的 NumPy 版本和调用顺序。`cache/RECOVERY_SAME_RUN.pt` 是运行中覆盖保存的恢复缓存，不是额外候选末点，不进入交付模型选择。

## 评价与证据界限

冻结权重后，先执行原 128 条 TRAIN SOURCE/REAL 的 Heun128/256 数值检查。通过后按原冻结身份、随机数组、源组成和等父事件权重，完成 5984 条主输出及探针、10 个新 HGB、16 个密度描述槽和 4000 条独立重放。RAW REAL–REAL、BASE、R49 和 R54 的比较使用既有冻结结果，不重新拟合对照。相对 R54 的差异涉及新损失、抽样和初始化等多个改变，不能归因于单一因素。

原 A–F 判据不变。训练损失、数值稳定、投影可行和重放一致均不能替代完整响应、PSD/峰宽、源保持、噪声、边界及同处理 REAL 的联合验收。D/E 必须在同一末点通过 A–C 后另行登记；历史 DEV 已反复暴露，独立采集泛化仍未验证。

最终私有 Git 和 D 盘交付以项目根的 `R59_PUBLICATION_RECEIPT.json`、`R59_BACKUP_RECEIPT.json`、`R59_DELIVERY_COMPLETION_RECEIPT.json` 为准；文件尚不存在时不代表完成。
