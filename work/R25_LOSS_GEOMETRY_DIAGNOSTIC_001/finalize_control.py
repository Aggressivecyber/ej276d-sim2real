from pathlib import Path
import os,json,datetime,hashlib,ast
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
cfg=read('config.json');a=read('A_METRICS.json');b=read('B_METRICS_COMPACT.json');history=read('RUN_HISTORY.json')
assert len(history)==5 and all(r['handshake_before_imports'] and r['memory_probe_failures']==0 for r in history)
assert [r['actual_worker_native_exit'] for r in history]==[0,1,0,0,0]
assert all(s['parameters_unchanged'] and s['gradient_sum_check_first_batch']['allclose_atol1e-5_rtol1e-4'] for s in b['states'])
def f(x):return f'{x:.8g}'
def ci(cell):return f"{f(cell['mean'])} [{f(cell['CI95'][0])}, {f(cell['CI95'][1])}]"
def delta(reg,g,c=.75,stat='W1_quantile'):return next(r['paired_minus_c1'] for r in a['summary'][reg][g][stat] if r['c']==c)
report=[
'R25没有得到“原组合分布项整体偏好收缩”或“主要梯度全面抵消至近零”的证据。混合Q区下physical和pooled_shape局部偏好c=.75；三fine项抵消了这一优势，原加权组合的配对变化接近零。改为同Q区批次时，各组c=.75均变差。两训练末点的NCE与domain梯度也不是稳定负对齐。下一问题优先是训练与条件验证的批次口径是否一致；本轮没有响应更新或新成功模型。',
'A只使用已缓存REAL TRAIN-fit21047行，其中18895行在冻结8区内。原physical/coarse/fine中心、尺度和投影均未重估；仅按区计算诊断均值。四c=[.5,.75,1,1.25]在同一X/Y对上比较，c=1直接使用原行。MIXED每行独立选Q区，HOMOGENEOUS每区固定32个重复；各256批，每批32。所有向量收缩发生在投影空间，不是合法波形生成或探测响应。',
'c=.75相对c=1的W1配对差如下。区间是固定TRAIN经验池条件下的Monte Carlo估计区间，不覆盖采集、训练或多重比较。HOMOGENEOUS按8区均值等权及分层方差计算。',
'| 投影组 | MIXED ΔW1及95%区间 | HOMOGENEOUS ΔW1及95%区间 |\n|---|---:|---:|'
]
for g in cfg['A']['groups']+['original_distribution_combination']:
    report[-1]+=f'\n| {g} | {ci(delta("MIXED",g))} | {ci(delta("HOMOGENEOUS",g))} |'
report+=['MIXED组合c=.75变化为-0.0000058124，区间跨0，不能称整体支持收缩。c=.5或1.25的组合变化两抽样方式均为正。单区结果也不能自动保证新的训练能改善：它是在REAL自身经验投影上的受控检验，而非源→目标网络训练。',
'energy_U是统计距离参照，不是物理能量。它复用相同冻结投影坐标，排除跨域及两域内部的三个i=j对角线，保留有限样本负值。W1使用分位数L1，U使用成对绝对差；构造、有限样本基准与数值尺度不同，不跨目标比较绝对量大小或以此选择新权重。五组在两种抽样中，c=.5/.75/1.25相对c=1的U变化均为正，配对区间均不含0。MIXED组合c=1的U均值为-0.0025917829，负样本估计没有截零；HOMOGENEOUS为0.0017466528。同一经验分布的c=1小批W1不为0，不能解释成真实域差距。',
'B使用三个同步保存状态：R21所选第2000步G/reader+R18初D、R23/STANDARDIZED末步2950、R24/D5末步2950。共同起点D尚未学习，不能用其弱对抗梯度代表训练后D。每状态复用R23实际前32批1024source/REAL行，z/white/style由旧记录准确PCG64初态和原逐批调用顺序重建，96行NCE位置直接取旧实际流。实际数组保留本地；没有重新抽样后声称配对。',
'STANDARDIZED critic保留固定中心化和尺度。原NCE key.detach、identity零随机量、同Q PSD排序和全部物理/fine定义保持。10项权重为1/1/2/.5/2/5/2/1/1/1。D参数冻结而输入梯度保留，G/reader原train模式无BN/Dropout可变状态；没有构造optimizer、step或裁剪。',
'| 原加权项 | INITIAL G梯度平均L2 | D1末点 | D5末点 |\n|---|---:|---:|---:|'
]
for t in b['loss_names']:
    report[-1]+=f'\n| {t} | '+' | '.join(f(s['terms'][t]['G_weighted_gradient_norm']['mean']) for s in b['states'])+' |'
report+=['NCE的encoder平均范数份额为60.49%、53.25%、68.59%；domain_SW的electronic份额为79.65%、71.16%、44.52%，fine份额仅2.96%、3.81%、5.78%。这些份额是各项平均L2之和归一化，不是实际更新量或因果归属；NCE project仅由NCE影响、reader仅由style项影响，也已分列。Reader总梯度中位为0.1442、1.7530、1.1178，并未主导G+reader总范数。',
'| 状态 | 总G范数中位 | G取消比中位 | clip5假定缩放中位 | 会裁剪批数 |\n|---|---:|---:|---:|---:|'
]
for s in b['states']:
    report[-1]+=f"\n| {s['state']} | {f(s['G_total_norm']['median'])} | {f(s['G_cancellation_ratio_norm_sum_over_sum_norms']['median'])} | {f(s['clip5_reference_scale']['median'])} | {s['clip5_would_apply_batches']}/32 |"
report+=['取消比定义为||各加权G梯度之和||/各项范数之和。三个状态都有部分方向抵消，但没有近零全面抵消；96批按旧clip5规则都会缩放。NCE-domain的平均余弦为-.0744、.0471、.1777，不支持两末点稳定冲突。D1的identity在23/32批与总G梯度负点积，D5则0/32；PSD-rank仅7、7、10批具有非零梯度，其中4、4、7批负点积，不能把这些比例当32批的无条件比例。positive_charge所有批梯度为0，余弦和负对齐比例是undefined/null，图中标n/a；零中位数或分位数在log图统一使用1e-8显示下限，保存值仍为0。',
'负点积表示原始负总梯度方向的一阶局部增项，不是Adam的有限步更新。两个末点G与D都已不同，较强D5对抗梯度不能再细分为仅D更新次数造成。较高NCE范数也不能直接证明NCE阻碍真实响应，更不能据此本轮删项或调权。',
'每状态第一批的逐项加权梯度加和与直接总损失梯度均有限并通过atol1e-5/rtol1e-4核对。最大绝对误差为1.1921e-5、5.4240e-6、5.2452e-6，相对L2误差1.2822e-6、7.1217e-7、3.8126e-7，属于记录的float32算术差，未追求逐位相等重跑。全部G/reader/D参数前后逐值不变。3072source前向、3072identity、3072 NCE源编码+3072输出编码、3072 critic及3072 reader行分别计数；同1024实际批行跨状态复用，不是新增独立事件。',
'建议只提出一个下一问题：保留损失/系数/结构和同一实际训练行多重集，重组同Q区批次以使分布训练口径接近条件验证，再按预指定selected与final分别评价。这个干预也改变批内源排序/NCE相互作用，不能声称只隔离一个W1项。本轮不实施它，也不替换energy、移除NCE或扫描权重。',
'本轮所有统计限于同一次采集的已有TRAIN与历史模型局部位置，无DEV/EVAL/CAL波形、新G4/长波形解码、HGB/EM/新响应拟合。没有新父事件主生成或重放需求。结论不证明物理机制或新采集泛化；旧模型历史DEV曝光边界不变。',
'一次工程失败发生在保存实际输入时波形source与索引source重复关键字，前向/梯度为0。原执行源、native1和6.8521512s保留；修复新入口只改变保存schema，没有更改数学、状态或流。两张图的显示语义另行修复，原图以_display_initial保留；其余6图未改。8张最终图均实际查看。'
]
science=sum(r['elapsed_s'] for r in history);peak=max(r['actual_worker_RSS_peak_bytes'] for r in history)
report.append(f'累计科学（含失败、汇总绘图及显示修复）{science:.7f}s/600s；真实worker峰RSS {peak}B/4GiB，GPU allocated {b["receipt"]["GPU_peak_allocated_bytes"]}B/4GiB。CPU Torch {a["receipt"]["torch"]}/NumPy {a["receipt"]["numpy"]}，GPU Torch {b["receipt"]["torch"]}/NumPy {b["receipt"]["numpy"]}；不作跨平台逐位相同声明。原生回执与计数详见RESOURCE_RECEIPT.json，定义/config和实际输入/逐批诊断保留本地；发布版METRICS_COMPACT仅为聚合。')
report.append('关键文件：A_METRICS.json、B_METRICS_COMPACT.json、METRICS_COMPACT.json；逐批A_VALUES_PRIVATE.npz、A_DRAWS_PRIVATE.npz、B_ACTUAL_INPUTS_PRIVATE.npz、B_GRADIENTS_PRIVATE.json；figures/内8张最终图。输入版本来自既有封存清单，不重复散列源数据。封存清单只散列本轮新文件。')
(ROOT/'REPORT.md').write_text('\n\n'.join(report)+'\n','utf8')
reproduction='''本轮是冻结损失诊断，重现不训练或选择响应模型。实际CPU解释器为C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe，B用E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe（Torch2.12.0+cu130，已有RTX4060）。所有TEMP/TMP/Torch/CUDA/Matplotlib/Joblib缓存由supervise.py设置在本轮E盘cache；4线程，实际science PID导入前握手，launcher PID单列。

在新E/D目录复制本轮必要源码/config/PROTOCOL，保留科学定义及INPUT_VERSIONS.json中旧R21/R18/R23/R24只读输入路径。按新的有界执行安排更新仅deadline/runtime控制，不能在已封存本轮目录重跑或覆盖.used标记。

CPU运行supervise.py contraction_cpu.py <CPU_python>，再CPU监督supervise.py gradient_gpu_repair001.py <GPU_python>，最后supervise.py summarize_and_plot.py <CPU_python>及repair_display_semantics.py <CPU_python>。原gradient_gpu.py是失败历史，不作为重现入口。顺序只有一个科学作业；无optimizer构造/step。

A实际512对抽样及PCG64初末态已保存。B前32批source/REAL/bin/NCE索引直接读取R23实际消耗流；z/white/style按其保存初态和逐批z→white调用重建，B_ACTUAL_INPUTS_PRIVATE.npz保存同一实际输入供三个状态。原权重路径见config/INPUT_VERSIONS，模型及其梯度没有写回旧目录。B有每状态一次直接总梯度核对及全参数未更新证据，没有新增2000 DEV主波形或科学重放。

原生作业：A actual61776/native0；B失败actual24636/native1（duplicate serialized source key，模型前向0）；B修复actual57620/native0；汇总绘图actual59272/native0；显示修复actual60552/native0。RUN_HISTORY及各entry.NATIVE_RECEIPT、SCIENCE_PID、WORKER_RECEIPT保留真实链。输入迁移/内容一致性测试0，科学输出诊断核对不代表跨Python/Torch设备逐位一致。
'''
(ROOT/'REPRODUCTION.md').write_text(reproduction,'utf8')
versions=read('PREEXEC_FROZEN_BINDINGS.json')['inputs']
for key in ['G_initial','reader_initial','D_initial']:
    fpath=Path(cfg[key]);stage=fpath.parent.parent if key!='D_initial' else fpath.parent
    manifest=json.loads((stage/'FINAL_MANIFEST.json').read_text('utf8'))
    entries=manifest.get('members',manifest.get('entries'))
    if not isinstance(entries,list):raise RuntimeError('unknown historical manifest schema')
    rel=fpath.relative_to(stage).as_posix();entry=next(x for x in entries if x['path']==rel)
    versions[str(fpath)]['historical_manifest_binding']=entry
write('INPUT_VERSIONS.json',{'bindings':versions,'source_bodies_rehashed_for_version_audit':False,'reference':'Existing immutable stage manifests; actual runtime loads and save-schema counted separately'})
qa={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actually_viewed':True,'final_figures':read('PLOT_RECEIPT.json')['figures'],'findings':'All eight images viewed; full axes/legends/CI/matrix readable. Corrected zero-norm floor applies to all norms/quantiles and positive-Q undefined rates explicitly n/a. Other six unchanged.','display_repair':read('DISPLAY_REPAIR_RECEIPT.json'),'new_metrics_or_fits':0}
write('VISUAL_QA.json',qa)
tools={'contraction_cpu.py':['5b51dc','39bd3c'],'gradient_gpu.py':['205d16'],'gradient_gpu_repair001.py':['ecde94','1f22b3'],'summarize_and_plot.py':['3223c9'],'repair_display_semantics.py':['439ffa']}
resource={'PID':os.getpid(),'UTC':qa['UTC'],'registered_UTC':cfg['registered_UTC'],'seal_deadline_UTC':cfg['deadline_UTC'],'science_cumulative_s':science,'science_cap_s':600,'jobs':[{**r,'native_tool_chunks':tools[r['entry']]} for r in history],'failure_count':1,'engineering_failure_s':history[1]['elapsed_s'],'response_G_D_reader_updates':0,'optimizer_steps':0,'new_model_fits':0,'HGB':0,'EM':0,'A_batch_pairs':512,'A_coefficients':4,'B':b['receipt'],'actual_worker_RSS_peak_bytes':peak,'GPU_allocated_peak_bytes':b['receipt']['GPU_peak_allocated_bytes'],'actual_PID_handshake_every_job':True,'memory_probe_failures':0,'CPU_threads':4,'new_DEV_main_generation':0,'DEV_EVAL_CAL_waveforms_read':0,'new_G4':0,'long_raw_waveforms':0,'SSH':0,'installation':0,'new_agents':0,'scientific_active':0,'no_scientific_model_replay_needed':'No response weights changed','compact_contains_per_event_indices':False,'output_cap_bytes':cfg['output_limit_bytes'],'output_bytes_before_seal':sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file()),'preparation_native_tools':['3f2783','7ee91d','ae2b7b','9bb68f','0c81d0','dc1c42'],'unknowns':['Adam finite-step causal mechanism not evaluated','New acquisition uncertainty not evaluated','Single controlled feature c does not define a valid waveform response']}
write('RESOURCE_RECEIPT.json',resource)
sources={}
for path in ROOT.glob('*.py'):
    data=path.read_bytes();ast.parse(data.decode('utf8'));sources[path.name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
write('DELIVERY_SOURCE_BINDINGS.json',{'PID':os.getpid(),'UTC':qa['UTC'],'AST_pass':True,'sources':sources,'executed_science_sources_and_original_failure_preserved':True})
assert science<=600 and peak<=cfg['RSS_limit_bytes'] and b['receipt']['GPU_peak_allocated_bytes']<=cfg['GPU_allocated_limit_bytes']
print(json.dumps({'PID':os.getpid(),'UTC':qa['UTC'],'science_s':science,'RSS_peak':peak,'GPU_allocated_peak':b['receipt']['GPU_peak_allocated_bytes'],'report_bytes':(ROOT/'REPORT.md').stat().st_size,'all_figures_viewed':True,'no_science_active':True}))
