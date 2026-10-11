# R69 sampled-full pilot：第二项独立静态实现审核

结论：当前核心实现的三项实质发现已在执行前修正；没有剩余已确认的核心实现漏洞。此结论是源码静态核对，不是科学执行/资源实测/原A–F通过，也不授权启动。根负责复制已完成抽块模块、封存最终绑定与登记。R65原退休及50000终点未完成状态保持。

本轮没有打开NPZ/PT/CSV或权重，没有加载/执行模型、科学函数、训练、推理、SSH或Git。全部新增文件、缓存均在本目录；自身1 CPU、RSS硬上限1 GiB。

路径均相对 E:/GAN/research/sim2real_two_papers_20261008_0305；以下简称 P=work/R69_SAMPLED_FULL_GRADIENT_PILOT_001。

## 已发现并关闭

| 发现 | 原影响 | 当前关闭证据 |
|---|---|---|
| 参考准入改写Adam学习率 | 原reference_gradient改为固定501学习率，而准入前后要求整个Adam摘要相同。原R65更新500处学习率与501不同，导致无step也会执行失败。 | P/reference_gradient.py:23–24现在仅取rate并zero_grad；AST中仅有optimizer.zero_grad，文件终止于105行combined有限性断言。P/pilot.py:320,333–334仍严格检查model/Adam。R65 config.json:60–63、train_final.py:106–110,572–573支持原问题确有影响。 |
| mini缺少200秒臂上限 | 原mini可在首次超过candidate时间后仍处于1%容差内，但超过预算预留的200秒。 | P/pilot.py:264–280现在为两臂设置deadline，每步前phase_check、步后elapsed检查；candidate成功必须恰100步。 |
| mini前向counter_delta为0 | 原mini收据报告384n前向行，但继承的scientific_neural_forward计数没有增加。 | P/pilot.py:273–275每个成功mini步追加384行和1次调用；288–289保存counter_delta与独立joint_microbatch_reverse_calls=n。没有把mini记作全人口D/VJP。 |

这些修复由根完成；本审核没有改训练代码。

## 核心协议逐项核对

- 无更新准入：P/pilot.py:110绑定的是reference_gradient.py，未执行包含Adam step的reference_full_step.py。参考片段先对全部104块求完整D，再求原C并相加；没有参数cast/clip/step调用。P/pilot.py:318–350收集每块norm平方、完整D/C/联合norm、按预登记规则写证书。数学有限性、每父电荷/范围及model/Adam不变均有运行时断言；静态审核没有代替这些原生检查。
- 方差失败边界：P/pilot.py:352–360只有通过后才创建baseline并调用evaluate；不通过直接NEGATIVE_GRADIENT_ADMISSION，明记0更新、0新完整响应probe，不扫描k/seed。该有限准入负结果不能推成原A–F普遍不可行。加载参考统计量或求参考梯度，不被伪称为完整新响应probe。
- 完整原目标：P/sampled_step.py:18–22,35–59对6593 SOURCE全父逐块前向、对21047 REAL完整参考求一次五D与GY。冻结objective_adapter.py:16–39确认GY已含原人口/区域质量、权重、ADC STE；43–70保留原四C、仅将微批五D作为丢弃值。旧R65 config.json:110–119,230–244的九项权重/次序和clip5没有更改。
- 逆概率只一次：新独立PCG64由P/pilot.py:154创建；P/sampled_step.py:27–30抽块，63–83按原次序重放/检查端点、ADC码、Box free mask；84仅一次D_gradient *= ht_scale，111直接加C，114–118仅一次cast/clip/Adam。没有按所抽父数重新归一或增加块长度权重。
- 末1父块保留：optimization_review/sampled_blocks.py:53–57使用range(0,n,chunk)并min保留短末块；60–74只对104块均匀无放回选择4块、排序返回、scale=104/4。因此(6592,6593)与64父块同为4/104入选。方差公式及有限性/小负舍入规则在77–110。check_sampled_blocks.py:18–29任何失败抛错，不能返回伪PASS；97–117覆盖104块、末1父、连续全覆盖、原排序与scale26。本审核只读这些函数，没有执行其人工检查。
- 同起点/随机流：P/pilot.py:129–158分别恢复model、deepcopy Adam、全部原PCG64及torch RNG，并核对摘要；准入的抽样消耗不传给后续两臂。原C仍320 SOURCE+320 REAL-reference+64 REAL-identity；full_SOURCE_epsilon原流每候选步全6593父。block流与原C/noise/directions分离，保存实际块历史、父反向次数与末状态。网络GroupNorm/按事件attention/零dropout见冻结network.py:5–20,34–47。
- 计数保真：P/sampled_step.py:142–168记录实际选中父/4 VJP；全人口forward/distribution计数与抽样backward计数分开。P/pilot.py:290–300不声称每步所有父都反向。mini只报告实际微批前向与联合reverse；R65历史counter保持为恢复起点，以delta报告新工作。中途失败/硬杀不能由这些成功终点收据补造完成次数。
- 成本与完整评价：P/pilot.py:358–363在训练前以实测baseline成本预留两臂200秒、两套完整评价的1.10余量和20秒保存；不够只记INCONCLUSIVE_BUDGET。两臂同固定501学习率、mini首次达到candidate时间，允许差<=max(.5s,1%)；不匹配不解释为科学负结果。评价/门在370–400全部完成后才有局部正/负判断，未截短探测。

## 原102项联合门与资源定义

与R67源码做AST比较，metrics、noise_for、full_forward、common_C、evaluate五个函数完全相同；只解析文本，未导入旧runner。冻结CATEGORIES有10类，SOURCE为9项piece+37项非负距离/投影+11相关+2谷=59；同处理REAL为5项piece+38项非负项=43，共102。P/pilot.py:381–397应用所有对应门，并与主score下降10%及6/8胜出条件取AND；None相关记失败。原charge_range直接保留f64及转f32后相对同一原q的1e-5和范围检查（冻结train_final.py:134–143）。数值目标/分母/人口/权重没有替换。可在最终绑定时加len(harms)==102作防漏列断言，不改变原门。

P/linux_supervise.py:8,25,35–50定义exact CPU0..11、fresh memory/GPU/own-job准入和一次提交；77–97从worker启动前开始计600秒硬watchdog，绑定真实PID/nonce后才允许科学imports；124–149监控实际进程树RSS48GiB和GPU进程10GiB。P/pilot.py:12,62–70,405核对线程/环境及allocated/reserved上限。这些是已读的守护定义，不是已发生的资源回执；本轮没有服务器调用。监测失败、资源/600秒上限或执行异常只能记不确定。

## 封存边界

封存回执时，根已将sampled_blocks.py/check_sampled_blocks.py复制到work并生成FILE_BINDING.json。本审核静态确认两份入口辅助源码与optimization_review中已读版本逐字节SHA一致，记录绑定JSON的文本SHA；没有打开或复算其科学数组/权重绑定对象。最终注册、人工检查原生回执及实际启动仍由根负责，没有用本静态报告代替原生验证或声称整个pilot成功。

根已明确确认完整预声明条件协议：准入失败才零更新/零完整probe退出；准入通过且预算门通过时，允许自动完成baseline、100步candidate、等时mini及原完整评价。当前入口无需另加只准入出口；两臂200秒、总600秒、不扫k/seed、不恢复R65保持。任务协议:3,27,31,39–43保持原A–F及独立登记要求。

当前输入SHA、静态检查回执及本机资源/工具退出记录见同目录JSON。原生执行验证仍由根完成。
