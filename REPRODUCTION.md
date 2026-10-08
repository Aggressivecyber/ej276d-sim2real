# 本机复现与交付入口

研究目录：`E:/GAN/research/sim2real_two_papers_20261008_0305`。当前研究数据、模型、源码和缓存保存在 E/D；2026-10-08 11:41 UTC 之后的工作全部在本机执行，不需要连接服务器。GitHub 保存源码、定义、报告、汇总指标和图，模型权重及逐事件载荷保留在本地，单独克隆 GitHub 不会获得这些文件。

## 已验证的复现范围

| 对象 | 本地入口 | 已执行证据 |
| --- | --- | --- |
| 最新开发候选 R3B 的固定生成 | `work/R3B_ENERGY_CONDITIONAL_WIDTH_001/replay.py` | 保存权重、输入及原 RNG 在独立进程生成 2000×480 波形；960000 个 float32 值逐位一致，父事件顺序一致。见同目录 `REPLAY_CHECK.json` 与 `RESOURCE_RECEIPT.json`。 |
| R3 的固定生成 | `work/R3_FLEXIBLE_TEMPORAL_RESPONSE_001/replay.py` | 同样的完整 2000×480 主生成逐位重放通过。 |
| R5 的保存估计器预测 | `work/R5_PREFIX_RECONSTRUCTION_UTILITY_001/replay.py` | 44 个估计器对 64 个固定前缀输入的 2816 个预测值逐位一致；没有重新训练。见同目录 `REPLAY_CHECK.json`。 |
| R6 的 FULL/GAP 固定生成 | `work/R6_CHARGE_GAP_GENERALIZATION_001/replay.py` | 两臂各 2000×480、合计 1920000 个值逐位一致，最大绝对差为 0；使用保存权重和已保存的条件、latent、white 输入，没有重新拟合或评价。 |
| R2 原始训练 | `work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/REPORT.md` | 其初始隐藏层构造前未设置主 Torch 种子，也未保存初始 RNG。保存权重的固定输入复载通过，但不能声称从头训练逐位可复现；后续固定模型结果不掩盖此限制。 |

这些结果是已保存实现的重放证据，不证明响应与实测物理等效。R4 的留出 C2ST 仍能清楚区分生成与实测；R5 的效用依赖估计器、样本量和目标。R6 的配对区间排除仍有响应与覆盖失配，完整结果见 [FINAL_REPORT.md](FINAL_REPORT.md)。

## 环境

2026-10-08 本机实际环境为 Python 3.13.3、NumPy 2.2.6、SciPy 1.16.3、PyTorch 2.10.0、scikit-learn 1.7.2、joblib 1.5.2、Matplotlib 3.10.3，Windows 11。当前程序复用已安装的 Python，所有新增研究文件及缓存定向到 E/D。R4 此前的远端提取使用 Python 3.12.3 / NumPy 1.26.4，其已成功取得的 `EVAL_EXTRACT.npz` 现已在 E 盘；本机后续工作不再要求原始服务器在线。

固定权重的位级重放证据仅对应实际记录的环境与计算路径；其他版本、设备或线程设置须重新记录结果，不能沿用旧的通过结论。每个阶段的 `config.json`、原生回执、清单与保存随机状态是参数依据。

## 复现固定 R3B 生成

已封存的研究目录保持只读使用。需要自行复现时，将必要文件复制到新的 E 盘目录，运行副本，使新回执不会覆盖原有证据。以下命令只做固定权重推理，不训练、不运行 C2ST，也不读取 EVAL。

```powershell
$studyRoot = 'E:\GAN\research\sim2real_two_papers_20261008_0305'
$stageRoot = Join-Path $studyRoot 'work\R3B_ENERGY_CONDITIONAL_WIDTH_001'
$replayRoot = Join-Path 'E:\GAN\replay_checks' ('R3B_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
if (Test-Path -LiteralPath $replayRoot) { throw 'Use a new replay directory.' }
New-Item -ItemType Directory -Path $replayRoot | Out-Null
$replayFiles = @('model.py', 'config.json', 'model.pt', 'GENERATION_RANDOM_STATE.pt', 'generation_inputs.npz', 'generated.npz', 'replay.py')
foreach ($replayFile in $replayFiles) {
    Copy-Item -LiteralPath (Join-Path $stageRoot $replayFile) -Destination (Join-Path $replayRoot $replayFile)
}
$cacheRoot = Join-Path $replayRoot 'cache'
New-Item -ItemType Directory -Path $cacheRoot | Out-Null
$env:TEMP = $cacheRoot
$env:TMP = $cacheRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TORCH_HOME = Join-Path $cacheRoot 'torch'
$env:XDG_CACHE_HOME = $cacheRoot
python -B (Join-Path $replayRoot 'replay.py')
```

预期副本中的 `REPLAY_CHECK.json` 报告 `bitwise_equal=true`、`max_absolute_difference_V=0` 和 `parent_ID_order_equal=true`。这里给出复现入口，没有为写说明重复运行已经通过的完整数值检查。

`generated.npz` 的 `waveforms_V` 为电压波形，`sim_event_ids` 为原 G4 父事件身份；`generation_inputs.npz` 保存模型实际使用的 `G4_condition`、`Qsource_Vns` 和相应父事件。主生成使用已保存随机流，不把不同随机实现之间的差异叫作复现失败，也不把随机噪声副本当作新的独立物理事件。

## 复现 R5 固定预测

同样在新的 E 盘副本目录中复制 R5 的 `replay.py`、`task_models.joblib` 和 `REPLAY_INPUTS.npz`，设置上述临时目录后运行 `python -B replay.py`。实际通过记录为 44 个估计器、64 个固定输入、2816 个预测值全部逐位相同。只加载本研究保存且能与清单核对的模型文件。

R5 输入仅包含已对齐保存窗的前 150 点，但保存窗的 CFD 对齐曾使用完整波形。因此这不是端到端因果触发或 FPGA 延迟验证。完整预测及标签在 `predictions.npz`，来源与训练身份在本地 `results/TASK_TRAIN_IDENTITIES.json`，所有比较臂的指标在 `results/metrics.json`。

## 复现 R6 两臂固定生成

在新的 E 盘目录保留以下相对结构，复制共 7 个文件：`model.py`、`replay.py`、`generation_inputs.npz`、`FULL_generated.npz`、`GAP_generated.npz`、`FULL/R3B/model.pt`、`GAP/R3B/model.pt`。源目录为 `work/R6_CHARGE_GAP_GENERALIZATION_001`。对新副本设置与前文相同的 E 盘缓存目录后，执行 `python -B replay.py`。

`generation_inputs.npz` 包含两个臂共同的条件、G4 父身份、latent 与白噪声输入；推理不需要原始实测服务器或训练缓存。预期新 `REPLAY_CHECK.json` 中 `arms.FULL.bitwise` 与 `arms.GAP.bitwise` 均为 true、`maxabs` 均为 0，各比较 960000 个值。该命令只应加载与本研究清单一致的自有检查点。

原封存目录还保留两臂三阶段的初始/最终权重、全局与采样随机状态、训练轨迹及区间排除清单。`INITIAL_PARAMETER_EQUALITY.json` 记录两臂可学习初始参数一致；噪声谱 buffer 各自只由允许的实测拟合记录估计，无须相等。固定生成重放已经实际通过，完整 11200 次训练没有为了文档再跑一次，也不把未执行的跨环境从头训练写成已验证。

## 共同定义与证据边界

评价采用 1 ns 采样、前 100 点基线、fast 100:150、total 100:400 与 signed PSD = 1 − Qfast/Qtotal。电荷是 mV·ns 代理，不是经过独立逐事件校准的入射中子能量。全体预定 QC 有效事件包含 unknown 和重叠事件；未收敛的密度分量或 CAL q4 参考保留未定义。

实测来自同一次采集，人工每 50 条分组用于用途隔离和组抽样，不能当作独立采集批次。当前角色、模型和 R4/R5/R6 任务在首次 EVAL 科学读取前固定，见 `FROZEN_METHOD.json`。此前历史噪声标定和开发暴露另行披露，不能把当前留出叫作整个研究历史都未见过的外部盲测。

各阶段 `FINAL_MANIFEST.json` 记录本地完整产物，部分成员按政策不进入 Git。重新执行训练、评价或修改输入时应另建实验目录并记录新配置、预算、环境及结论；已封存阶段及其原生失败记录保留。
