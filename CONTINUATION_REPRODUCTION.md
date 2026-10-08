# R7–R9 本机复现与交付入口

本文件为已封存R7–R9研究的本机复现说明。R8 与 R8C 的独立进程重放均已完成、原生退出0；每阶段两臂共4000条主生成、1920000个float32值逐位一致。R8结论见 `work/R8_LOCAL_JOINT_FLUCTUATION_001/REPLAY_CHECK.json` 及 `numeric_3/NATIVE_RECEIPT.json`；R8C见对应阶段的 `REPLAY_RECEIPT.json` 与 `numeric_2/NATIVE_RECEIPT.json`。R1–R6 的既有证据见未修改的 [原复现说明](REPRODUCTION.md)。

研究根目录为 `E:/GAN/research/sim2real_two_papers_20261008_0305`。GitHub 只保存源码、定义、配置、报告、汇总和图；模型、随机输入、逐事件数据及日志在 E/D。单独克隆仓库不能执行固定权重重放。冻结目录作为输入使用，新回执写入新目录。

## R8 两臂主生成重放

R8 两臂均使用 2000 个 G4 DEV 父事件，主输出为各 2000×480 的 float32 电压波形。`GENERATION_INPUTS.npz` 保存实际的条件、源电荷、父事件顺序，以及三组显式 latent、baseline white 和 signal white；主重放使用第一组，不从种子重新猜测随机抽取。相同父事件的多视图是随机实现，不是新增独立物理事件。

以下命令只做保存模型的主生成和逐值比较。它不训练、不运行分类器、不读取 EVAL；模型文件应与本研究封存清单一致。脚本本身会把线程数限制为 4。

```powershell
$studyRoot = 'E:\GAN\research\sim2real_two_papers_20261008_0305'
$stageRoot = Join-Path $studyRoot 'work\R8_LOCAL_JOINT_FLUCTUATION_001'
$replayRoot = Join-Path 'E:\GAN\replay_checks' ('R8_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
if (Test-Path -LiteralPath $replayRoot) { throw 'Use a new replay directory.' }
New-Item -ItemType Directory -Path $replayRoot | Out-Null
$replayFiles = @(
    'model.py', 'response_base.py', 'replay.py', 'config.json', 'GENERATION_INPUTS.npz',
    'arms\JOINT\model.pt', 'arms\JOINT\generated.npz',
    'arms\JOINT_SIGNAL\model.pt', 'arms\JOINT_SIGNAL\generated.npz'
)
foreach ($replayFile in $replayFiles) {
    $targetPath = Join-Path $replayRoot $replayFile
    New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $stageRoot $replayFile) -Destination $targetPath
}
$cacheRoot = Join-Path $replayRoot 'cache'
New-Item -ItemType Directory -Path $cacheRoot | Out-Null
$env:TEMP = $cacheRoot
$env:TMP = $cacheRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TORCH_HOME = Join-Path $cacheRoot 'torch'
$env:XDG_CACHE_HOME = $cacheRoot
python -B -X utf8 (Join-Path $replayRoot 'replay.py')
if ($LASTEXITCODE -ne 0) { throw 'Replay failed; retain its failure receipt.' }
```

预期新目录 `REPLAY_CHECK.json` 的 `status` 为 `pass`，两臂各 `values_compared=960000`、`bitwise_equal=true`、`maxabs_V=0`、`same_2000_parent_order=true`，总计 1920000 个值。本轮已由实际独立进程确认上述结果，重放耗时 6.041 秒。不同环境的复现要记录自己的结果，不能继承本机位级通过结论。

## R8C 整段映射的保存输入重放

R8C两臂已由独立进程从保存模型和实际输入重放，REPLAY_RECEIPT.json记载两臂均bitwise_equal=true、parent_order_exact=true、maxabs_V=0；纯推理及比较3.013秒，含进程启动的监督wall为6.448秒，原生退出0。

R8C 输入增加完整 clean_V[2000,480]、z_pulse[16]、z_base[4] 以及两组512点白噪声。脚本读取已保存的 view1 实际数组、模型构造参数和 pulse_enabled 状态，并按固定 generation_chunk 推理；主生成之外的两个视图不在这次主重放中重复计算。

```powershell
$studyRoot = 'E:\GAN\research\sim2real_two_papers_20261008_0305'
$stageRoot = Join-Path $studyRoot 'work\R8C_WHOLE_RESPONSE_MAPPING_001'
$replayRoot = Join-Path 'E:\GAN\replay_checks' ('R8C_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
if (Test-Path -LiteralPath $replayRoot) { throw 'Use a new replay directory.' }
New-Item -ItemType Directory -Path $replayRoot | Out-Null
$replayFiles = @(
    'model.py', 'observable_source.py', 'replay.py', 'config.json', 'GENERATION_INPUTS.npz',
    'arms\FULL_WAVE_STOCHASTIC\model.pt', 'arms\FULL_WAVE_STOCHASTIC\generated.npz',
    'arms\PULSE_STOCHASTIC_OFF\model.pt', 'arms\PULSE_STOCHASTIC_OFF\generated.npz'
)
foreach ($replayFile in $replayFiles) {
    $targetPath = Join-Path $replayRoot $replayFile
    New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $stageRoot $replayFile) -Destination $targetPath
}
$cacheRoot = Join-Path $replayRoot 'cache'
New-Item -ItemType Directory -Path $cacheRoot | Out-Null
$env:TEMP = $cacheRoot
$env:TMP = $cacheRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TORCH_HOME = Join-Path $cacheRoot 'torch'
$env:XDG_CACHE_HOME = $cacheRoot
python -B -X utf8 (Join-Path $replayRoot 'replay.py')
if ($LASTEXITCODE -ne 0) { throw 'Replay failed; retain its failure receipt.' }
```

该脚本的通过条件为新目录 REPLAY_RECEIPT.json 的 status=passed、inference_rows=4000、waveform_values=1920000，两臂 all_arms 下 bitwise_equal=true、parent_order_exact=true、maxabs_V=0。复制命令是复现说明，不代表本轮已额外执行一次复制路径重放。使用前先对照最终冻结清单核对源权重、生成和实际输入哈希。

## 训练、评价和冻结

R8 的训练配置、共享初始参数、TRAIN fit/内部验证目标、检查点轨迹、实际采样摘要和随机状态留在阶段目录。两个臂只由 TRAIN 内部验证选检查点；运行顺序、上限和三组生成输入种子见 [任务](tasks/R8_LOCAL_JOINT_FLUCTUATION_001.md)。固定模型推理重放不等于从头训练已被重复验证，本轮没有为了说明文档再训练一次。

R7 的诊断只比较固定模型与 TRAIN/DEV，并保存固定干预。R9 的共同口径在 R8 之前登记；在用户明确整体数据差距后，候选范围已在R8C计算前通过 [补充协议](tasks/R9_CANDIDATE_SCOPE_ADDENDUM_001.md) 扩展，五个候选模型与生成哈希已统一保存为[续轮冻结文件](CONTINUATION_FROZEN_METHOD.json)，原 `FROZEN_METHOD.json` 保留。旧 EVAL 已经查看，R9 是同次采集上的冻结后回顾性验证，不能称新增盲测。

阶段内工程失败、修复和实际数值时间保留，失败不通过删除目录从预算或证据中消失。新完整训练或评价应使用新实验目录与新记录；源码阅读、清单核对和备份不触发全量重算。

## 环境与数据边界

使用已安装的本机 Python / PyTorch / NumPy。当前同机安装版本的只读记录见 `reviews/CONTINUATION_RUNTIME_METADATA_001.json`；它在 R8C 运行期间查询，不补称历史阶段曾记录全部已加载模块，实际重放结论仍以各阶段原生回执为准。数据、源码、缓存、临时文件都写 E/D；不需要服务器恢复或新安装环境。D盘备份覆盖本地研究目录的科学载荷与Git bundle；实际复制字节与SHA256核对结果由CONTINUATION_BACKUP_RECEIPT.json记录。外部原始数据不会被隐式纳入该项目备份；相关来源和输入绑定仍保留在阶段清单。

## R9验证与回执说明

[R9阶段复现说明](work/R9_FROZEN_RETROSPECTIVE_VALIDATION_001/REPRODUCTION.md)记录原执行入口、固定42评价器、同一2000主REAL与7583补充总体、共享评价事件及500次条件bootstrap。R9没有响应训练或推理；此前的R8/R8C保存权重重放不重复执行。READINESS、INPUTS和FROZEN_BINDING_CHECK保存源/config/输入绑定。

R9原监督回执的EVAL_read=0及响应训练说明是陈旧字段。[独立更正回执](work/R9_FROZEN_RETROSPECTIVE_VALIDATION_001/RECEIPT_FIELD_CLARIFICATION.json)绑定原始SHA，并记录实际7600条读取、7583有效、2000主评价、响应更新0；原进程PID、退出码、时间、内存与源文件不改。

当前代码含原研究路径和输入SHA绑定。R8/R8C上述复制重放说明已做静态依赖核对，实际通过依据是阶段内独立进程回执；没有额外声称该复制路径、移机重训或一键从头评价已执行。重新评价时在新E/D目录保留并显式重定位路径，记录自己的源和输入绑定，不直接在封存目录运行。

## D盘恢复与核对

备份位置由[交付索引](CONTINUATION_DELIVERY_INDEX.json)的backup.directory给出。其project目录为本地研究文件副本，repository.bundle保存版本历史；BACKUP_MANIFEST.json列出每个复制文件的字节数和SHA256。实际状态以CONTINUATION_BACKUP_RECEIPT.json的status及核验字段为准，忽略清单自身与回执自身的循环引用。

恢复时先按备份清单核对SHA256，再复制project到新的E/D目录。可用`git clone <备份目录>/repository.bundle <新的E/D版本目录>`恢复版本历史；该Git副本不含模型和逐事件载荷，仍需同一备份的project内容。不要覆盖现有封存目录。跨目录恢复涉及配置中的绝对输入引用，外部原始数据仍按INPUTS及来源记录定位。

本轮报告入口为[CONTINUATION_REPORT.md](CONTINUATION_REPORT.md)。最终Git分支SHA以本地CONTINUATION_PUBLICATION_RECEIPT.json为准，备份逐文件结果以CONTINUATION_BACKUP_RECEIPT.json为准；这两份执行后回执不通过写入自身SHA制造循环证明。
