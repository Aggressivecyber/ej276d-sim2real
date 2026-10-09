from __future__ import annotations
import pathlib,json,hashlib,datetime,os,time,ast
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic()
UTC=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
DEADLINE=datetime.datetime.fromisoformat('2026-10-09T23:04:32.204052+00:00')
assert datetime.datetime.now(datetime.timezone.utc)<DEADLINE
assert not(ROOT/'FINAL_RECEIPT.json').exists(),'stage already sealed'
def read(n):return json.loads((ROOT/n).read_text('utf-8-sig'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def fmt(v):return 'undefined'if v is None else format(v,'.8g')
gate=read('GATE_METRICS.json');m=read('metrics.json');tc=read('TRAINING_COMPACT.json');cal=read('INITIAL_GRADIENT_SCALE_COMPACT.json')
rr=read('REPLAY_RECEIPT.json');runs=read('RUN_HISTORY.json');tr=read('JOINT_KERNEL_FINAL/TRAINING_RECEIPT.json')
A='JOINT_KERNEL_FINAL';cs=m['C2ST']['results'];pa=m['C2ST']['paired'][A];g2=gate['gate2_PSD_width_centers_noise'];g3=gate['gate3_source_preservation'];g4=gate['gate4_boundary_source'];g5=gate['gate5_same_treated_REAL_control']
assert rr['pass']and rr['rows']==6000
figure_notes={
 'c2st.png':'Full/voltage/local/linear-space views, 500 fixed-classifier group intervals; legend outside data, RAW 2D absent explicitly.',
 'charge_psd.png':'Common Q and PSD axes, whole source and output populations, QC and out-of-view counts visible.',
 'conditional_psd_CDF.png':'All8 predeclared Q bins, sample counts and bin8 low support retained.',
 'energy_peak_width.png':'Two density components, center/raw/detrended widths; failed bins remain gaps, not particle truth.',
 'conditional_waveforms.png':'Complete480 native mV, means/medians and fixed observed-Q counts; tail texture visible.',
 'noise_curvature.png':'Baseline RMS/ACF1 and prompt/postpeak curvature, complete CDFs, units readable.',
 'source_preservation.png':'Actual unADC input bins3/4 N213/181, source ranks, valley fraction not particle error.',
 'boundary_differences.png':'Nine prespecified native first/second-difference positions, no chosen window.',
 'training_joint_kernel.png':'Final2950 recorded J20 point restored; selected2750 record-only, actual Gram summaries and64-batch scale.',
 'DEV_full_correlation.png':'Full480 correlation with common color limits; population description, no DEV fit.',
 'source_and_random_decomposition.png':'FullSOURCE perturbation, correct probe references and functional components/cross terms; no physical noise identification.',
 'same_treated_REAL_controls.png':'Same stochastic reference treatment; gate lines apply only first3 views; V480 descriptive.'}
qa={'UTC':UTC(),'reviewer':'executor actual view_image inspection','method':'Each final PNG opened; three repaired PNGs reopened after repair',
    'all_final_pngs_actual_opened':True,'new_final_figures':len(figure_notes),'old_versions_kept':'results/figures/layout_initial',
    'files':{n:{'bytes':(ROOT/'results/figures'/n).stat().st_size,'sha256':sha(ROOT/'results/figures'/n),
               'actual_opened':True,'labels_axes_layout_pass':True,'observations':note}for n,note in figure_notes.items()},
    'repair_native_tool':'0d7661','repair_actual_worker_PID':27008,'repair_native_exit':0,
    'no_scientific_pass_from_visual_QA':True}
write('FIGURE_QA.json',qa)
tools={
 'train_server.py':{'tool':'142a24','native_exit':1,'local_launch_PID':59924,'SSH_PID':56004},
 'train_recover.py':{'tool':'b9135c','launcher_tool':'ae2dd6','native_exit':0,'local_launch_PID':10740,'SSH_PID':3664},
 'generate_all.py':{'tool':'43e682','launcher_tool':'855b28','native_exit':0},
 'evaluate_saved.py':{'tool':'fb93b7','launcher_tool':'0ef4e8','native_exit':0},
 'replay_all.py':{'tool':'31edb7','native_exit':0},
 'summarize_and_plot.py':{'tool':'288ff0','native_exit':1},
 'summarize_and_plot_recover.py':{'tool':'ca712b','native_exit':1},
 'finish_plots.py':{'tool':'a2b070','native_exit':0},
 'fix_final_figures.py':{'tool':'0d7661','native_exit':0}}
chain=[]
for r in runs:
    assert r['entry']in tools,('unaccounted run',r['entry'])
    chain.append({**r,'native_tool_evidence':tools[r['entry']]})
science=sum(r['elapsed_s']for r in runs);training=sum(r['elapsed_s']for r in runs if r['entry']in ['train_server.py','train_recover.py'])
query_gaps=[{'entry':r['entry'],'event':e,'cause':'unknown'}for r in runs for e in r.get('memory_probe_failure_events',[])]
assert science<=1800 and training<=900
assert all(r['actual_worker_RSS_peak_bytes']<=4*1024**3 for r in runs)
assert all(r['handshake_before_imports']for r in runs)
resource={'UTC':UTC(),'science_elapsed_s_inclusive_failed_checks_and_plots':science,'science_cap_s':1800,
    'training_attempt_native_elapsed_s_inclusive_initial64_scale_J20_and_failed_D':training,'training_cap_s':900,
    'sequential_science_jobs':True,'all_actual_worker_PID_handshakes_before_imports':True,'CPU_threads_cap':4,
    'actual_worker_RSS_peak_bytes':max(r['actual_worker_RSS_peak_bytes']for r in runs),'RSS_cap_bytes':4*1024**3,
    'GPU_peak_allocated_bytes':tc['GPU_peak_allocated_bytes'],'GPU_cap_bytes':4*1024**3,
    'memory_probe_gap_count':len(query_gaps),'memory_probe_gaps':query_gaps,'query_gap_causes':'unknown; temporal proximity to exit is not a demonstrated cause',
    'neural_recipe_count':1,'neural_execution_attempts':2,'unique_original_G_updates':2950,'unique_original_D_updates':2950,'style_reader_updates':2950,
    'actual_D_optimizer_update_compute_calls_including_failed_known_first_D_replay':2951,
    'initial_gradient_calibration_calls':1,'calibration_repeated_calls':0,'calibration_extra_source_G_rows':2048,
    'original_training_row_stream':94400,'actual_unique_source_exposure':3752,'actual_unique_REAL_exposure':18739,
    'loaded_source_REAL_fit_pools':[6593,21047],'J20_step0_compute_calls':2,
    'successful_J20_source_generation_rows':tr['validation_generation_rows'],'extra_failed_attempt_step0_source_generation_rows':512,
    'source_G_forward_rows_for_successful_D_and_G_loss':188800,'extra_failed_attempt_D_source_forward_rows':32,
    'MMD_update_calls':2950,'Gram_groups':8850,'five_bandwidth_group_terms':44250,'negative_U_truncated':False,
    'new_HGB_fits':m['counters']['HGB_attempts'],'density_description_slots':m['counters']['density_description_slots'],
    'actual_EM_calls':m['counters']['density_EM_attempts'],'all_old_reference_fits':0,
    'main_generated_source_rows':2000,'same_treated_REAL_control_rows':4000,'joint8_generated_rows':1024,'source_probe_generated_rows':832,
    'fullSOURCE_extra_perturbation_rows':128,'independent_replayed_rows':6000,
    'responses_or_fits_repeated_for_plot_repair':0,'EVAL_read':0,'CAL_numeric_decode':0,'raw_long_waveform_decode':0,'new_G4':0,'installation':0,
    'execution':'Server CUDA training; local CPU generation/evaluation/replay/plot; no parallel science',
    'all_local_stage_and_temporary_data':'E/D; CPU Python executable on C only',
    'transfers':'normal SSH/SCP exit and actual load; no migration byte equality tests',
    'native_runs':[{k:r[k]for k in ['entry','elapsed_s','actual_worker_PID','supervisor_PID','actual_worker_native_exit','actual_worker_RSS_peak_bytes','start_UTC','stop_UTC']}for r in runs],
    'registered_UTC':'2026-10-09T21:44:32.204052+00:00','executor_seal_deadline_UTC':DEADLINE.isoformat(),
    'output_cap_bytes':512*1024**2,'budget_pass':True,'complete_stage_physical_bytes_final':'FINAL_RECEIPT.json; includes dependency copies, both return packages and failures'}
write('RESOURCE_RECEIPT.json',resource)
write('NATIVE_CHAIN.json',{'UTC':UTC(),'runs':chain,'first_control_tool':'f0da4c','first_control_PID':47824,
    'first_control_UTC':'2026-10-09T21:47:03.0078409+00:00','server_return_tool':'05f8f0','server_return_native_exit':0,
    'actual_native_results_external_to_worker_self_report':True,'final_seal_native_exit':'parent tool observes after receipt is written'})
write('FAILURE_ACCOUNTING.json',{'science_failures':[
    {'entry':'train_server.py','PID':70858,'native_exit':1,'elapsed_s':runs[0]['elapsed_s'],'error':"KeyError('patches_per_layer')",
     'G_updates':0,'reader_updates':0,'D_updates':1,'scale_calibration_completed_once':True,'recovery':'complete original R32 neural fields, reuse saved scale, deterministically replay known firstD from initial state; original firstD optimizer state was not saved'},
    {'entry':'summarize_and_plot.py','PID':54960,'native_exit':1,'elapsed_s':tools and next(r['elapsed_s']for r in runs if r['entry']=='summarize_and_plot.py'),'error':"KeyError('paired_vs_RAW_RR')",'completed_figures':0},
    {'entry':'summarize_and_plot_recover.py','PID':58064,'native_exit':1,'elapsed_s':next(r['elapsed_s']for r in runs if r['entry']=='summarize_and_plot_recover.py'),'error':"KeyError('step') in initial validation-only trace row",'completed_figures':8,'already_saved_figures_not_redrawn_by_finisher':True}],
    'display_repairs':['c2st legend outside data','connect saved final2950 J20','V480 descriptive only; three gates only'],
    'control_read_notes':[{'tool':'94aaf1','missing_path':'FORMULA_CHECK.json','actual_correct_path':'KERNEL_FORMULA_CHECK.json','scientific_work':0},
        {'tool':'f26feb','missing_paths':['assessments.py','JOINT_KERNEL_FINAL/TRAINING_NUMERIC_RECEIPT.json'],'actual_paths':['assessment.py','TRAINING_NUMERIC_RECEIPT.json'],'scientific_work':0}],
    'all_first_sources_logs_receipts_preserved':True,'no_repeated_HGB_EM_or_generation_from_summary_repairs':True})
bits={k:bool(v['pass'])for k,v in gate.items()if k.startswith('gate')}
write('FINAL_RESULT.json',{'UTC':UTC(),'decision':gate['decision'],'scientific_recipe_retained':False,'gates':bits,
    'endpoint':A,'actual_checkpoint_step':2950,'internal_selected2750_not_DEV_evaluated':True,'independent_replay_pass':True,
    'all_final_figures_actual_QA_pass':True,'project_goal_status':'active; next problem belongs to root','further_science_in_this_stage':False})
cov=m['covariance_description'];ranks=g3['whole_population_source_ranks'];gp=m['guide_probe']
report=[]
report.append('# R39 完整源驱动联合核训练：固定配方关闭\n')
report.append('唯一2950步末点没有通过联合响应门槛。完整493列与纯480点归一化波形BA均为0.982；相对冻结BASE的差为+0.004，配对95%区间跨0，未达到相对R37的固定上限0.934/0.928。总体PSD与基线ACF有收益，但峰位、纯源条件结构和同处理REAL对照有退化。所有结果保留，不转而评价内部选中的2750步，不扫描核/系数/种子。\n')
report.append('## 方法、输入与训练\n')
report.append(f'完整G4派生clean480波形直接进入原卷积G，所有源父关联保留。输入描述来自总EE/had沉积的既有解析响应，不能恢复未保存的微观沉积时序。初始化、STANDARDIZED D、style reader、其余原损失和94400行流不变；唯一替换为三块标准化时间特征组成的Gaussian混合核MMD U，带宽固定[.25,.5,1,2,4]。只去GG/RR对角，cross全部保留；负值不截零。64个TRAIN初始批次一次确定c={tc["gradient_coefficient"]:.15g}，匹配物理输出grad-y合计尺度，未匹配后续参数梯度或FLOP。理论和方法依据[MMD原论文](https://jmlr.org/papers/v13/gretton12a.html)及[GMMN原论文](https://proceedings.mlr.press/v37/li15.html)，本轮不称方法创新或论文完整复现。\n')
report.append(f'原流七项全部匹配。载入G4/REAL fit池6593/21047，实际唯一更新暴露3752/18739；这些是本轮更新成本，不是整个历史模型/噪声先验的总实测成本。人工50记录组来自一次真实采集，不是独立批次。2000 DEV构成pure_EE417/pure_had1518/mixed65，全部有实际源父且不重加权；推定PDG22/2112来源边界见DATA_ROLES，不是输出粒子真值。全部417历史TRAIN支持外父仍使用新G，无BASE回退。既有模型和开发路线已历史暴露DEV，本轮仅开发诊断，不是新盲测或独立采集验收。EVAL/CAL新解码、新REAL/长原波形、新G4均0。\n')
report.append(f'完成2950 G/D/reader唯一更新轨迹。内部J20为初始{tc["step0_J20"]:.8f}、记录选点2750 {tc["selected_J20"]:.8f}、唯一末点{tc["final_J20"]:.8f}；旧W1验证未改为MMD，不能把两者当同一个目标。G/reader裁剪比例{tr["global_gradient_norms"]["G_reader_clip_fraction"]:.1%}，D为{tr["global_gradient_norms"]["D_clip_fraction"]:.2%}，仅描述实际优化路径，不据此判定唯一原因。2950次联合U、8850 Gram组、44250组×带宽项；Gram标签明确为withinG/withinREAL/cross，不是三组时间损失。\n')
report.append('## 结果\n\n|共同口径指标|BASE|Joint kernel|变化/判定|\n|---|---:|---:|---|')
for k,row in g2['measures'].items():report.append('|'+k+'|'+fmt(row['BASE'])+'|'+fmt(row['candidate'])+'|'+('undefined'if row['relative_change']is None else f'{100*row["relative_change"]:+.2f}%')+'; '+('通过'if row['pass']else'失败')+'|')
report.append('\n共同PSD人口支持bins1–7，每域N≥40；两成分MAE共同合格bins2–7，各自缺口全部保留。原BASE有效bins2–7均未新增失败；新候选bin8 N36支持不足。raw与去趋势宽度均保存，密度两成分仅统计描述，不是n/g标签。未因宽度小收益隐藏低/高峰位误差+23.01%/+194.50%。\n')
report.append('|完整判别视图|BASE BA|候选BA|候选−BASE 95%CI|\n|---|---:|---:|---|')
for v in ['full_historical_shape','normalized_u480','full_voltage','prompt_local','postpeak_local','tail_local','baseline_local']:
    report.append(f'|{v}|{fmt(cs["BASE"][v]["balanced_accuracy"])}|{fmt(cs[A][v]["balanced_accuracy"])}|{pa[v]["BA_delta_95CI"]}|')
report.append('\n500次区间只条件于固定分类器与同采集人工留出组，不包含响应重训或外部采集不确定性。R37/R38仅历史参照，均未重拟合。整体/条件平均曲线外观接近不代表联合波形分布接近；完整相关矩阵和条件中位尾部仍有固定纹理。\n')
report.append('## 源关联、边界与REAL处理控制\n')
for r in g3['actual_source_pure_had_valleys']:
    report.append(f'actual unADC source {r["actual_source_bin"]}，N{r["N"]}：谷比例{100*r["BASE"]:.4f}%→{100*r["candidate"]:.4f}%，差{100*r["delta"]:+.4f}pp，'+('通过'if r['pass']else'失败')+'。')
for r in ranks:report.append(f'{r["measure"]} Spearman：{fmt(r["BASE"])}→{fmt(r["candidate"])}，下降{fmt(r["BASE_minus_candidate"])}。')
report.append('实际NN source-Q条件主组N213/181；历史postADC SOURCE_CLEAN分组207/182另存，不混称。谷区是冻结实测代理，不等价于粒子误分类。相同父记录及非零源敏感性不证明逐事件物理对应。\n')
report.append(f'128个完整SOURCE扰动preADC输出RMS中位{1000*gp["preADC_output_RMS_V"]["quantiles"][3]:.6f}mV，非同率{gp["output_nonidentical_fraction"]:.1%}；源Qfast严格不变，实际f32 Qtotal/PSD微差完整保存。9边界失败项为{[r["location"]for r in g4["boundaries"]if not r["pass"]]}。功能signal/electronic共享G状态且有交叉协方差，不是已标定物理分解。\n')
report.append('启动前补充已冻结：对已含噪REAL再用非零随机响应，不要求RAW恒等。门5只比较新G与BASE G处理同2000 TRAIN_REFERENCE/同z-white-style。两者相对RAW所有视图照报（RAW Q/PSD配对unknown未补拟合），不能称REAL控制或独立泛化成功。\n\n|同处理REAL视图|BASE G BA|新G BA|差|95%CI|门5|\n|---|---:|---:|---:|---|---|')
for v,r in g5['three_views'].items():report.append(f'|{v}|{fmt(r["BASE_G_REAL_BA"])}|{fmt(r["new_G_REAL_BA"])}|{r["BA_delta_first_minus_second"]:+.6f}|{r["BA_delta_95CI"]}|'+('通过'if r['pass']else'失败')+'|')
report.append('\n纯u480与baseline点差均+.012，超过预定+.01；三项区间上界都≤.10仍不能使门5通过。V480只描述，图上不施加此门。\n')
report.append('## 实现、费用与交付\n')
report.append(f'两个训练原生attempt：70858 exit1 {runs[0]["elapsed_s"]:.6f}s；71473 exit0 {runs[1]["elapsed_s"]:.6f}s。第一入口继承统计配置缺patches_per_layer，完成c和已更新D1后失败；保存旧源码/失败，恢复完整R32字段，复用c，不重新标定。因未保存失败D optimizer状态，从初态重演已知第一D，实际D计算更新2951而唯一轨迹2950；额外32源前向和额外step0 512行验证全部计费。首次失败D状态未作为独立bitwise恢复证据。\n')
report.append(f'生成worker30348、评价26340、独立回放53388均原生0；回放6000行包含新源2000及两REAL控制4000，ID、random与所有分量/pre/postADC逐值通过。30新HGB、24描述槽/{m["counters"]["density_EM_attempts"]}实际EM，旧reference重fit0。两次汇总schema失败3.440621/7.769760s及三个图显示修复费用全部保留，没有重复HGB/EM/生成。\n')
report.append(f'累计科学原生监督墙钟{science:.6f}s（1800s上限），训练attempt合计{training:.6f}s（900s），实际RSS峰{resource["actual_worker_RSS_peak_bytes"]}B，GPU allocated峰{resource["GPU_peak_allocated_bytes"]}B。两服务器attempt各2次RSS查询gap有具体时间戳，原因unknown，不能写0或断言发生在退出后。全部12张最终PNG实际打开；原图保留，图修复未改指标。全部stage物理字节含复制依赖/包/失败由FINAL_RECEIPT登记，整体≤512MiB。\n')
report.append(f'六门结果按登记顺序为{list(bits.values())}；关闭这一固定核+带宽+初始尺度配方，不外推所有核方法不可行。G对source仍有作用且局部PSD/噪声有收益，但不足以完成完整响应任务。后续问题由ROOT另行登记；本阶段不启动新训练。\n')
report.append('报告/聚合指标：METRICS_COMPACT.json、GATE_METRICS.json；完整逐事件/模型留本地。最终权重JOINT_KERNEL_FINAL/FINAL_TRANSLATOR.pt；初始c全文INITIAL_GRADIENT_SCALE.json含逐行位置，仅本地，公开版INITIAL_GRADIENT_SCALE_COMPACT.json。12图位于results/figures；实际命令与进程链NATIVE_CHAIN.json、RUN_HISTORY.json；资源RESOURCE_RECEIPT.json；复现REPRODUCTION.md；最终字节FINAL_MANIFEST.json。\n')
(ROOT/'REPORT.md').write_text('\n'.join(report),'utf8')
repro='''# R39复现与部署

本轮训练实际路径是服务器R39_REMOTE_20261009_214432_UTC下的train_recover.py；第一次train_server.py失败原件保留，不把它写为成功入口。执行器原环境Python3.12.3/NumPy1.26.4/torch2.12.0+cu130，GPU RTX5070；本机后续Python3.13.3/NumPy2.2.6/torch2.10.0+cpu，CPU4。

训练采用原R32 inputs只读（路径见config.json）：TRAIN_PRIVATE.npz、VALIDATION_INPUTS.npz、FINE_CONSTANTS.npz、CRITIC_SCALER.npz、R26_TRAINING_STREAMS_PRIVATE.npz、原R23随机与R26回执、INITIAL_TRANSLATOR/STYLE_READER/CRITIC.pt。恢复入口只复用已保存64批c，原失败与恢复源码/配置均保存。要重训须新独立目录并获得对应新任务；本封存目录不可直接重运行或解除.used。

本轮实际命令链：launch_remote.py train_server.py（失败）→launch_remote.py train_recover.py（成功）；科学前由linux_supervise.py和本地SSH回执绑定实际PID。正常返回tar保留，无迁移一致性测试。初态/流/参数由PRETRAIN_FREEZE、ENGINEERING_RECOVERY_RECEIPT、MATCHED_TRAINING_FLOW_RECEIPT和最终清单区分绑定，不把修复后配置冒充首份原字节。

本机后续通过supervise.py <entry> <Python绝对路径>启动；TEMP/TMP精确E项目/.runtime-cache/tmp，PYTHONUTF8=1/PYTHONDONTWRITEBYTECODE=1与-B，CPU4。实际顺序generate_all.py → evaluate_saved.py → replay_all.py，各自nonce/PID/RSS导入前握手。评价按原固定600/250与同人工组，不能据DEV重选2750或调核。

部署仅用model.py/legacy_model.py、config.json记录的源定义与JOINT_KERNEL_FINAL/FINAL_TRANSLATOR.pt，再提供实际完整source480 V、z/white/style。G不读取REAL库、reader或评价器。输出只经过既定一次14bit ADC；全部历史支持外source仍用新G。单父源关联不意味着无配对学习识别出唯一真实事件映射。

ACTUAL_GENERATION_INPUTS.npz保存主2000实际source/ID/z/white/style；REAL_DEV_AND_REFERENCE_PRIVATE.npz中reference_V/reference_ids/reference_groups保存同处理REAL输入。两REAL控制复用完全相同主随机量，不得给REAL改为零随机量。各端点generated.npz保存preADC/postADC、功能分量、身份，SOURCE_PROBES_PRIVATE与JOINT8_PRIVATE保存辅助。

独立完整CPU replay_all.py已在worker53388/super52640实际原生0执行：6000条，所有身份/功能分量/pre/postADC逐值通过，见REPLAY_RECEIPT与NATIVE_CHAIN。从新目录恢复时先复制必要执行源/权重/实际输入/比较载荷及配置路径，保持chunk128和运算顺序；只做推理，不重新拟合。旧目录中的.used专门防重复启动。

本轮绘图实际由summarize_and_plot_recover.py成功落8图后在trace初始行失败，finish_plots.py完成余4图，fix_final_figures.py仅修三图显示。失败源/回执/原图保留；最终SHA以FIGURE_QA及FINAL_MANIFEST为准，首份PLOT_SUMMARY的三旧SHA已被修图回执取代。所有图来自保存的结果，无重新评价。

完整逐事件位置、随机流、预测、权重/科学数组仅E/D；Git发布只用聚合compact与代码图文。INITIAL_GRADIENT_SCALE_COMPACT不含source_rows/target_rows/ordinal数组，原全文本地保留。旧R11–R17目标密度自生成、R33–38后处理路线均不是本轮部署链。
'''
(ROOT/'REPRODUCTION.md').write_text(repro,'utf8')
prohibited={'source_ids','reference_ids','source_rows','target_rows','original_row_ordinal','probability','label','cluster','predictions','selected_indices'}
issues=[]
def privacy(v,path=''):
    if isinstance(v,dict):
        for k,x in v.items():
            if k in prohibited and isinstance(x,(list,dict)):issues.append(path+'/'+k)
            privacy(x,path+'/'+k)
    elif isinstance(v,list):
        for i,x in enumerate(v):privacy(x,path+'/'+str(i))
privacy(read('METRICS_COMPACT.json'));privacy(cal)
assert not issues,issues
sources={}
for p in sorted(ROOT.glob('*.py')):
    text=p.read_text('utf-8-sig');ast.parse(text);sources[p.name]={'bytes':p.stat().st_size,'sha256':sha(p),'AST':True}
write('DELIVERY_SOURCE_BINDINGS.json',{'UTC':UTC(),'current_sources':sources,
    'first_frozen_train_config_preserved':'failure001/source_snapshot','actual_executed_recovery_sources':['online_recover.py','train_recover.py'],
    'final_plot_producers':['summarize_and_plot_recover.py','finish_plots.py','fix_final_figures.py'],'no_training_source_changed_after_recovery_start':True,
    'final_checkpoint':{'path':'JOINT_KERNEL_FINAL/FINAL_TRANSLATOR.pt','sha256':sha(ROOT/'JOINT_KERNEL_FINAL/FINAL_TRANSLATOR.pt')},
    'task_sha256':'1174a58a7f66a1f8286c5f687b1028f56a1ecf56c91b7aea901da92ea4c4480a',
    'startup_REAL_addendum_sha256':'58ab02ccc12222d89e2bfff97de09cd29f369c0464db6872d7d8d9275f1fe697'})
write('DELIVERY_CHECKS.json',{'UTC':UTC(),'all_new_final_figures_actual_opened':True,'figure_count':12,'privacy_for_compacts':True,
    'sources_AST':True,'all_responses_source_driven':True,'training_actual_unique_trajectory2950':True,
    '6000_independent_replay':True,'30_HGB_24slots_23EM':m['counters']['HGB_attempts']==30 and m['counters']['density_description_slots']==24 and m['counters']['density_EM_attempts']==23,
    'outputs_source_control_reference_unknowns_not_removed':True,'source_and_REAL_roles_separate':True,
    'native_science_failures_preserved':3,'scientific_negative_result_is_completion':True})
# Exactly one manifest: hash bytes only, never decode or fit scientific arrays here.
assert datetime.datetime.now(datetime.timezone.utc)<DEADLINE
excluded=['FINAL_MANIFEST.json','FINAL_RECEIPT.json']
members=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file()and p.relative_to(ROOT).as_posix()not in excluded:
        members.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
total=sum(v['bytes']for v in members)
assert total+1024*1024<512*1024**2
manifest={'UTC':UTC(),'root':str(ROOT),'files':len(members),'member_bytes':total,'members':members,
    'self_reference_exclusions':excluded,'all_byte_hash_no_science_decode':True,'metadata_only_no_refit_or_inference':True}
write('FINAL_MANIFEST.json',manifest)
receipt={'UTC':UTC(),'seal_PID':os.getpid(),'state':'sealed','decision':gate['decision'],'scientific_recipe_retained':False,
    'gates':bits,'native_exit':'parent tool evidence after this process returns; receipt is not a native-exit substitute',
    'manifest_sha256':sha(ROOT/'FINAL_MANIFEST.json'),'manifest_members':len(members),'manifest_member_bytes':total,
    'manifest_exclusions':excluded,'final_outputs_and_all_scientific_payload_bytes_in_manifest':True,
    'science_inclusive_s':science,'training_inclusive_s':training,'6000_bitwise_replay':True,'actual_QA_12final_pngs':True,
    'executor_deadline_UTC':DEADLINE.isoformat(),'no_further_stage_writes':True,'further_science':0,'seal_control_elapsed_s_before_receipt':time.monotonic()-START}
write('FINAL_RECEIPT.json',receipt)
# From here there are no further writes, including no self-hash ledger rewrite.
all_files=[p for p in ROOT.rglob('*')if p.is_file()]
physical=sum(p.stat().st_size for p in all_files)
assert physical<=512*1024**2
last=max(p.stat().st_mtime for p in all_files)
assert last<DEADLINE.timestamp()
print(json.dumps({'seal_PID':os.getpid(),'stop_UTC':UTC(),'actual_last_stage_write_UTC':datetime.datetime.fromtimestamp(last,datetime.timezone.utc).isoformat(),
    'FINAL_MANIFEST_SHA256':sha(ROOT/'FINAL_MANIFEST.json'),'FINAL_RECEIPT_SHA256':sha(ROOT/'FINAL_RECEIPT.json'),
    'manifest_members':len(members),'all_stage_files':len(all_files),'all_stage_bytes':physical,'science_s':science,
    'state':'sealed_no_further_stage_writes','no_new_scientific_work_in_seal':True},ensure_ascii=False),flush=True)
