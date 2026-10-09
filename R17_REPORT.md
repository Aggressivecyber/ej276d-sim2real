# R17：目标密度诊断封存，恢复无配对仿真波形转换

R17只回答“显式条件PSD损失能否改善实测密度生成”，没有G4波形输入。它不能验证源事件保真，也不能作为已经实现仿真→实测转换的证据。用户明确指出研究核心是让神经网络在无配对两域间学习信号差距；现已停止目标域独立生成支线，下一轮R18直接接收每条480点仿真波形。

本轮固定基分布与残差网络、从相同head开始、各2000更新。加入条件PSD损失后，八个固定电荷区的PSD Wasserstein距离均下降，等权平均由0.01107368降至0.00831564（24.91%）。这支持条件PSD训练项有局部效果，但第2–8区谷区仍过满，最高两区双成分描述未收敛；不能宣布两条响应带已正确恢复。

两臂都用相同PSD参与的内部验证J选点，因此对照并非完全没有PSD选择信息。仅继续NLL的所选点为600步，加入PSD损失的所选点为1200步。训练末点均不优于各自所选点，全部轨迹保留。两臂完整shape判别BA为0.908/0.918，完整电压0.964/0.956；差值区间均含零，完整响应未验证。二维Q/PSD判别器留出BA均0.528，但训练BA约0.741/0.707，不能以近0.5代替实际分布检查。

四千条新输出的独立进程重放在ADC前后逐值一致。主计算、重放及保存结果的CDF标题重绘累计93.238194秒；两臂训练合计71.166753秒，峰RSS约2.622GiB、GPU allocated约92.60MiB。32次HGB、16个宽度拟合，无EVAL、新G4或原始长波形读取。旧重叠标题图保留，新版已实际检查；无重训或额外拟合。

R18的输入是已有G4沉积通过解析快慢响应核派生的完整波形，保留父事件与沉积组成。训练的sim/real样本独立抽取，域匹配学习实测风格，源内容约束保留输入信息；没有逐事件配对监督。PSD允许域间系统偏移，必须检验纯EE、纯had、mixed输入的条件输出及固定噪声下源替换的影响。入射n与pure-had不是同一分组，不把实测密度成分当粒子真值。该映射仍待实际实验验证。

[阶段报告](work/R17_CONDITIONAL_PSD_HEAD_FINETUNE_001/REPORT.md) · [电荷–PSD图](work/R17_CONDITIONAL_PSD_HEAD_FINETUNE_001/results/figures/charge_psd_main_body.png) · [关键结果复核](reviews/ROOT_R17_KEY_RESULT_REVIEW_001.json) · [复现入口](work/R17_CONDITIONAL_PSD_HEAD_FINETUNE_001/REPRODUCTION.md) · [R18任务](tasks/R18_UNPAIRED_WAVEFORM_TRANSLATION_001.md) · [方向纠正](reviews/UNPAIRED_SOURCE_CORRECTION_20261009_001.json)。

项目仍active。版本与D盘副本状态读取R17_PUBLICATION_RECEIPT.json、R17_BACKUP_RECEIPT.json，不用阶段封存代替项目完成。
