from pathlib import Path
import datetime,json,hashlib,os
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
m=read('metrics.json');g=read('GATE_METRICS.json');history=read('RUN_HISTORY.json');cfg=read('config.json');tr=read('PSD_ORDERED_CFM/TRAINING_RECEIPT.json');cs=read('COUPLING_SUMMARY.json');replay=read('REPLAY_RECEIPT.json');plot=read('PLOT_SUMMARY_RECEIPT.json');layout=read('CDF_LAYOUT_RECOVERY_RECEIPT.json');transfer=read('RETURN_TRANSFER_RECEIPT.json');prep=read('PREPARATION_RECEIPT.json')
a='PSD_ORDERED_CFM';names=['BASE','INDEPENDENT_CFM','OT_CFM',a]
# No scientific array decoding, fitting, inference or source reads in this entry.
control_failures=[
 {'tool':'5daed8','PID':'unknown','native_exit':1,'elapsed_tool_s':.4800585,'error':'KeyError: ROC_AUC; actual metric key is AUC','scientific_array_decode_fit_inference':0},
 {'tool':'8f8bd1','PID':'unknown','native_exit':1,'elapsed_tool_s':.4301261,'error':'FileNotFoundError: results/metrics.json; actual aggregate is stage-root metrics.json','scientific_array_decode_fit_inference':0},
 {'tool':'428ebc','PID':'unknown','native_exit':1,'elapsed_tool_s':.0093321,'error':'KeyError: INDEPENDENT_CFM in new actual-source association aggregate; immutable R33 control aggregate is reused for its two controls','scientific_array_decode_fit_inference':0},
]
if (ROOT/'ASSEMBLY_FAILURE_001.json').exists():
 saved=read('ASSEMBLY_FAILURE_001.json');control_failures.append({'tool':saved['tool'],'PID':saved['actual_PID_from_saved_os_getpid'],'native_exit':1,'elapsed_tool_s':saved['elapsed_tool_s'],'error':'overbroad compact-key guard rejected original_row_ordinal boolean stream-match flag, not an identity payload','scientific_array_decode_fit_inference':0,'original_partial_files_preserved':saved['original_partial_control_files_retained']})
native_chain=[
 {'entry':'resource_check.py','tool':'049881','PID':57140,'native_exit':0,'kind':'metadata-only SSH preflight'},
 {'entry':'train_server.py','tool_start':'94e35a','tool_terminal':'643e9b','local_launcher_PID':60800,'SSH_PID':62032,'actual_worker_PID':43603,'supervisor_PID':43602,'native_exit':0},
 {'entry':'evaluate_endpoints.py','tool_start':'4944a8','tool_terminal':'cc645f','actual_worker_PID':37968,'supervisor_PID':56476,'native_exit':1},
 {'entry':'resume_evaluation.py','tool_start':'c5bf7d','tool_terminal':'2458e0','actual_worker_PID':61036,'supervisor_PID':56004,'native_exit':0},
 {'entry':'replay_endpoints.py','tool_terminal':'f33108','actual_worker_PID':19328,'supervisor_PID':21916,'native_exit':0},
 {'entry':'summarize_and_plot.py','tool_start':'10a8df','tool_terminal':'1bc5c5','actual_worker_PID':44600,'supervisor_PID':45428,'native_exit':0},
 {'entry':'fix_CDF_layout.py','tool_terminal':'c38733','actual_worker_PID':60552,'supervisor_PID':11508,'native_exit':0},
]
write('NATIVE_EXECUTION_CHAIN.json',{'UTC':now(),'PID':os.getpid(),'native_tools_observed_by_executor':native_chain,'RUN_HISTORY_original_measurements_preserved':True,'native_receipt_not_inferred_from_script_claims':True,'control_read_failures':control_failures,'seal_native_tool_reported_separately_after_return':True})
failed=next(r for r in history if r['entry']=='evaluate_endpoints.py')
write('FAILURES_AND_REPAIRS.json',{'scientific_failure':failed,'original_error':'KeyError INDEPENDENT_CFM in fine_distances after all new fits/generation slots were saved','original_entry_and_logs_retained':True,'resume':'resume_evaluation.py','resume_source_receipt':read('ENGINEERING_RESUME_RECEIPT.json'),'repeat_HGB_EM_generation_solver':0,'metadata_only_failures':control_failures,'layout':'legend moved outside PSD CDF panels; original PNG retained','layout_receipt':layout,'RSS_missing_event_count':1,'RSS_missing_exact_cause':'unknown; timestamp after worker-finally receipt and before supervisor terminal; no rerun'})
misses=[e for r in history for e in r.get('memory_probe_failure_events',[])]
guarded=sum(r['elapsed_s']for r in history);preflight=read('RESOURCE_CHECK_NATIVE.json')['elapsed_s']
gpu=read('TRAINING_NUMERIC_RECEIPT.json')['GPU_peak_allocated_bytes'];peak=max(r['actual_worker_RSS_peak_bytes']for r in history)
dep_names=['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json','VALLEY_DEFINITION.json','AFFINE_CONSTANTS.npz']
dependencies={n:{'bytes':(ROOT/n).stat().st_size,'copied_normal_dependency_not_new_scientific_data':True}for n in dep_names}
packs={n:(ROOT/n).stat().st_size for n in ['UPLOAD.tar','REMOTE_RETURN.tar']}
resource={'UTC':now(),'PID':os.getpid(),'registered_UTC':cfg['registered_UTC'],'first_actual_control':cfg['first_actual_command'],'executor_deadline_UTC':cfg['deadline_UTC'],'root_deadline_UTC':cfg['absolute_deadline_UTC'],'science_guarded_elapsed_s_including_failure_plots':guarded,'science_budget_s':1800,'metadata_only_preflight_elapsed_s':preflight,'conservative_charge_including_preflight_s':guarded+preflight,'control_failure_tool_elapsed_s':sum(x['elapsed_tool_s']for x in control_failures),'training_body_s':tr['training_s'],'training_guarded_elapsed_s':history[0]['elapsed_s'],'training_limit_s':900,'flow_fit_attempts':1,'flow_updates':tr['updates'],'flow_update_limit':2950,'fixed_G_training_forward_rows':tr['actual_rows'],'fixed_G_training_forward_limit':94400,'new_G_D_reader_updates':0,'HGB_attempts':m['counters']['HGB_attempts'],'HGB_limit':10,'density_slots':m['counters']['density_description_slots'],'density_EM_attempts':m['counters']['density_EM_attempts'],'density_EM_limit':8,'main_rows':m['counters']['main_generation_rows'],'joint8_rows':m['counters']['repeat_noise_rows'],'probe_rows':m['counters']['source_probe_rows'],'solver_64step_check_rows':m['solver_check'][a]['rows'],'independent_replay_rows':replay['rows'],'resume_new_HGB_EM_generation':0,'threads':4,'actual_worker_RSS_peak_bytes':peak,'actual_worker_RSS_cap_bytes':4294967296,'GPU_peak_allocated_bytes':gpu,'GPU_allocated_cap_bytes':4294967296,'memory_probe_failure_events':misses,'memory_probe_failure_count':len(misses),'exact_missing_RSS_cause':'unknown; remote event is between worker-finally and supervisor terminal','observed_monitoring_gaps_not_claimed_zero':True,'single_scientific_job_at_a_time':True,'EVAL_new_REAL_CAL_waveform_G4_install':0,'SSH':'own registered training preflight/deploy/one launch/retrieval only; terminal SSH and native exit0','remote_root':cfg['remote_root'],'training_python':cfg['server_python'],'evaluation_python':cfg['CPU_python'],'local_DATA_CACHE_TEMP_only_E_D':True,'dependency_copies':dependencies,'dependency_copy_bytes':sum(x['bytes']for x in dependencies.values()),'packaged_duplicate_bytes':packs,'packaged_duplicate_total_bytes':sum(packs.values()),'local_output_cap_bytes':402653184,'migration_equality_test':False,'local_output_bytes_before_final_control_files':sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file()),'final_output_exact_total_bound_in_final_manifest_and_receipt':True,'run_history':history,'native_chain':native_chain}
assert guarded+preflight<1800 and peak<4294967296 and gpu<4294967296
assert m['counters']['HGB_attempts']==10 and m['counters']['density_EM_attempts']==8 and tr['updates']==2950
write('RESOURCE_ACCOUNTING.json',resource)
figs={}
for n in plot['figures']:
 p=ROOT/'results/figures'/n;figs[n]={'absolute_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'actual_executor_opened_and_visually_inspected':True,'axes_legends_support_gaps_readable':True}
qa={'UTC':now(),'PID':os.getpid(),'final_figure_count':len(figs),'figures':figs,'actual_view_batches':'six then seven original PNGs; final conditional_psd_CDF.png reopened after layout-only correction','CDF_old_layout_preserved':'results/figures/conditional_psd_CDF_layout_initial.png','whole480_raw_mV_no_smoothing':True,'same_raw_Q_PSD_axes':True,'all_density_failure_gaps_preserved':True,'negative_covariance_visible':True,'fixed_parent_examples':'prespecified0/500/1000/1999, unpaired REAL nearestQ reference, no output selection','all_final_figures_have_actual_visual_QA':True}
write('FIGURE_QA.json',qa)
compact=read('METRICS_COMPACT.json');suspect=[]
compact['training_flow']={'fit_attempts':1,'actual_updates':tr['updates'],'fixed_G_forward_training_rows':tr['actual_rows'],'training_s':tr['training_s'],'selection':'actual final only','frozen_G_D_reader_updates':0,'response_updates_counter_scope':'original frozen translator; new CFM flow updates are counted separately here'}
write('METRICS_COMPACT.json',compact)
write('COUNTER_SEMANTICS.json',{'original_metrics_preserved':True,'original_counters_response_updates':m['counters']['response_updates'],'original_counter_scope':'frozen original G/D/reader, not newly fitted CFM flow','new_flow_fit_attempts':1,'new_flow_updates':tr['updates'],'training_body_receipt':'TRAINING_NUMERIC_RECEIPT.json','no_new_scientific_computation':True})
forbidden={'source_ids','real_ids','event_ids','parent_indices','parent_index','REAL_index','original_row_ordinal','predictions','actual_source_index','actual_REAL_index','train_indices','test_indices','holdout_indices','original_target_permutation','initial_t_RNG','final_t_RNG'}
def scan(x,path=''):
 if isinstance(x,dict):
  for k,v in x.items():
   if k in forbidden and not isinstance(v,bool):suspect.append(path+'/'+k)
   scan(v,path+'/'+k)
 elif isinstance(x,list):
  for i,v in enumerate(x):scan(v,path+'/'+str(i))
scan(compact);assert not suspect
write('COMPACT_PRIVACY_RECEIPT.json',{'UTC':now(),'aggregate_file':'METRICS_COMPACT.json','bytes':(ROOT/'METRICS_COMPACT.json').stat().st_size,'sha256':sha(ROOT/'METRICS_COMPACT.json'),'explicit_identity_prediction_RNG_keys_found':suspect,'private_streams_and_predictions_retained_local':True,'no_fit_or_array_decode':True})
br=g['gate4_boundary_solver_replay']['boundaries'];b99=next(x for x in br if x['location']=='first_99_100');p=m['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4'];rows=[]
for n in names:
 r=m['C2ST']['results'][n];rows.append(f"| {n} | {r['full_historical_shape']['balanced_accuracy']:.3f} | {r['normalized_u480']['balanced_accuracy']:.3f} | {m['domain_vs_REAL'][n]['overall']['PSD']['W1']:.6f} |")
report=f'''# R35：Q区内PSD顺序耦合的完整波形转换

固定配方未通过联合继续门槛，关闭当前冻结R32 BASE＋共同仿射＋480维MLP-CFM的耦合/代价小变体分支。新端点保留了相对BASE的完整波形局部收益，也恢复了源PSD秩关联；但噪声、纯had源组谷占比和第7区密度收敛仍失败。项目整体目标继续，有限负结果不证明所有流模型或无配对转换不可行。

## 方法与输入

完整480点G4派生clean输入进入冻结R32 CONTROL_FINAL生成器，再经过一个480维CFM流与一次14bit ADC；不是目标域独立抽样。唯一训练变化是原Q子批内实际BASE preADC PSD与原REAL PSD稳定同序配位，保留两端每行边缘并还原源原顺序。PSD用f32波形升f64、扣前100点均值，1−Qfast(100:150)/Qtotal(100:400)，不裁剪PSD。统计耦合不是物理事件配对，推理并无逐事件单调/守恒保证。

实际完成2950更新、94400行固定G前向，G/D/reader更新0。相同R33环境、初始流、source/原REAL/bin/ordinal/z-white/style/t/base-preADC八组实际流以及2950个BASE逐批摘要全部相符；只有目标排列不同。2957子组含4个混合批和1个singleton，source/target稳定tie数0/649，所有一对一/无跨Q/无逆序/边缘与源顺序检查通过。行加权配位PSD绝对差由{cs['weighted_mean_absolute_PSD_gap_original']:.6f}降到{cs['weighted_mean_absolute_PSD_gap_ordered']:.6f}，这是配位描述，不是验证成功。网络497→512→512→512→480、Adam2e−4(.9,.999)、clip5、原初始化与t流未改；仅发布2950末点，未用DEV选checkpoint。

源TRAIN6593/原内部val1599与REAL TRAIN21047/val599依赖不变。本轮训练复用指定94400行，不把加载库大小当独立采集数。主DEV2000父完整保留：417 pureEE、1518 purehad、65 mixed，原顺序与权重未改；TRAIN PDG显式22/2112，DEV PDG由R2既存CSV仅22/2112与历史source-label推定，不能称本轮逐事件新读PDG。输入只有总EE/had沉积派生解析波形，缺乏微观沉积时序。REAL是同一次采集且已历史暴露的开发参照，人工50记录组不是真实独立run；本轮不读EVAL、CAL波形、新REAL、长波形或新G4。

## 波形收益及PSD退化

| 端点 | shape493 BA | 纯u480 BA | 总体PSD W1 |
|---|---:|---:|---:|
{chr(10).join(rows)}

新臂相对BASE的shape493差−.034，95%区间[−.058929,−.008686]；纯u480差−.026，[−.052292,−.001737]。相对原Independent点差−.004/0，区间均跨0，不能宣称胜过Independent。绝对BA仍.944/.952，完整波形远未匹配。置信区间是固定已拟合分类器的500次留出区组重抽，不涵盖响应训练不确定性；REAL–REAL沿用保存控制，不把其范围当等效阈值。

总体PSD W1降低33.63%，但共同观察Q bins1–7 (各域N≥40) 等权平均W1由.034766升到.037130（+6.80%）。这两个口径并列，不能用总体收益覆盖条件差距。REAL/BASE/新臂密度共同有效bins2–6上，低侧raw-width MAE .011641→.011618（−.20%），高侧.015332→.007952（−48.14%）；两侧中心MAE分别.004416→.022100和.002450→.014941，明显偏移。第7区原BASE有效而新臂500次EM未收敛，预定门槛失败，不能缩小交集或追加拟合消除失败。第8区另有component1支持<20；所有支持/分离/收敛缺口保留在表和图中，密度成分不等于粒子真值。

## 实际源口径、噪声与边界

主源口径为未量化NN输入source-Q，purehad bins3/4固定N213/181。新臂PSD均值.299965/.295390、sample SD .049537/.038893；BASE均值.281402/.280560、SD .026703/.019818。谷占比BASE5.63%/.55%→新11.74%/10.50%，分别增加6.10/9.94个百分点，均超过2个百分点的停止限。相较Independent18.31%/22.10%有恢复，但仍未保住BASE；谷区变化是源条件保真风险，不能仅据此判真实粒子错误。历史postADC SOURCE_CLEAN源Q(N207/182)单列供历史比较，不能称实际NN输入source-Q。

全体source→output Q Spearman .999726→.998535，PSD .418992→.420039；全体秩恢复不保证每组中心/宽度和谷占比保真。原2000行QC全部有效、ADC裁剪0，支持内1583行修正、支持外417行原BASE回退，后者不删。固定源变噪声RMS4.133mV、变源固定噪声14.659mV、相同输入/噪声0，源敏感性成立，但不证明逐事件物理映射正确。joint128×8=1024的重复不增加独立父事件，四个精确代数项含base舍入及协方差交叉项，闭合误差5.9631e−19V²；最终物理signal/electronic拆分unknown。

基线RMS对REAL W1 .019358→.022391mV（+15.67%），ACF1 .064235→.083638（+30.21%），均超10%停止限。主峰/早期下降prompt100:150和峰后/尾部局部分布仍可分，不能把局部曲率仅归因于上升沿。条件中位/均值曲线接近不代表联合完整分布成功。

9个预指定边界差均通过max(BASE,Independent)+1LSB限，未采用P或平滑；99→100 W1为{b99['candidate_W1_V']*1000:.6f}mV，BASE{b99['BASE_W1_V']*1000:.6f}mV。32父的32vs64 Heun最大preADC差1.73524e−5V<1/16383V，发布仍32步。独立CPU进程2000条完整source→固定G→流→一次ADC重放，pre/postADC、ID、支持与全部分量逐值相等。辅助joint/probe本轮从一开始保存精确preADC，未用f32增量近似原端点。

## 结论与执行边界

四门结果为完整响应通过、PSD/宽度/噪声失败、源保真失败、边界/求解/回放通过。关闭这个冻结BASE+MLP-CFM的耦合小变体，不追加PSD权重/cost/种子/网络/步数扫描。证据说明训练统计排序能影响源关联和总体PSD，但尚不足以约束纯源组响应、条件中心与电子涨落；下一问题须由根登记为实质不同的源驱动联合随机响应机制，不能退回无G4条件的目标密度抽样。本报告不启动下一科学工作。

服务器actual worker43603/super43602、native0，训练体9.241214s；CPU首次评价37968 native1在所有10HGB/8密度拟合/主辅助输出保存后因旧域名聚合失败。resume61036 native0仅续接保存结果，0重复拟合/生成；原代码/日志/回执保留。replay19328 native0；绘图44600、CDF布局60552均native0。累计科学{guarded:.9f}s，包括失败、聚合恢复、独立回放及绘图；元数据预检另{preflight:.6f}s，保守合计{guarded+preflight:.9f}s<1800s。峰RSS{peak}B、GPU allocated{gpu}B均<4GiB；远端1次RSS缺测时间在finally回执和进程退出间，确切原因unknown，未写成0。全部6个守护作业顺序单进程运行；新流fit1、2950更新、10HGB、8描述槽/8EM计费完整。

13张最终PNG实际逐张查看，CDF图例仅布局修复，旧图保留。完整metrics、逐行训练配位/实际流、ID、预测与权重保留本地；METRICS_COMPACT只发布聚合。必要入口、依赖、normal copy/pack费用和所有原生链见REPRODUCTION.md、RESOURCE_ACCOUNTING.json、NATIVE_EXECUTION_CHAIN.json，最终字节总量/SHA见FINAL_MANIFEST.json；正常迁移未做对端内容一致性测试。config保留继承的历史字段，R35实际范围以execution、sole_training_intervention、flow_*、first_actual_command对象和本轮native为准，不能据旧first_actual_command_UTC或G_D比例字段宣称R35训练了G/D。
'''
(ROOT/'REPORT.md').write_text(report,'utf8')
repro=f'''# R35复现入口与固定依赖

已执行唯一PSD_ORDERED_CFM，2950末点权重`PSD_ORDERED_CFM/FINAL_FLOW.pt`；初态来自R33/INITIAL_FLOW.pt，冻结G来自R32/SERVER_FRESH_CONTROL/FINAL_TRANSLATOR.pt。完整source480波形、父ID及实际z/white/style见ACTUAL_GENERATION_INPUTS.npz；joint8与832探针原输入独立保存。支持/仿射常数AFFINE_CONSTANTS.npz直接复用R33，solver32与唯一14bitADC未改。

远端根：`{cfg['remote_root']}`。既有解释器：`{cfg['server_python']}`，Python3.12.3/NumPy1.26.4/Torch2.12.0+cu130/SciPy1.11.4/RTX5070，与R33环境匹配；读取R33远端inputs而非重新上传TRAIN。Linux真正科学PID43603受导入前nonce/RSS守护；launch_remote.py调用linux_supervise.py train_server.py。deploy_training.py、launch_remote.py、retrieve.py及TRANSFER/RETURN回执给出正常传输退出码，未声称位级迁移验证。

训练按原R26逐行源/REAL/bin/ordinal和原z-white/style/t读取，源preADC来自固定G，目标仅区内稳定PSD排列。原始流和实际目标排列分别为PSD_ORDERED_CFM/ACTUAL_CFM_STREAM_PRIVATE.npz、COUPLING_BY_UPDATE_PRIVATE.json，配位摘要COUPLING_SUMMARY.json。训练耗时9.241214s，INITIAL_FLOW科学初态摘要、八组实际流摘要及每批BASE比较见PAIRED_FLOW_RECEIPT.json；它们是科学配对核验，不是迁移测试。没有DEV checkpoint选择。

CPU评价/重放：`{cfg['CPU_python']}`，Torch2.10.0+cpu/NumPy2.2.6；主与独立重放同本机CPU环境。已执行入口：

```powershell
# 在新E/D目录恢复，不能向已封存目录写入，也不能删除原used标记重提。
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP='E:/GAN/research/sim2real_two_papers_20261008_0305/.runtime-cache/tmp'
$env:TMP=$env:TEMP
# 新目录config路径指向固定原依赖，deadline按新登记设置，保留实际科学定义。
& '{cfg['CPU_python']}' -B -u supervise.py replay_endpoints.py '{cfg['CPU_python']}'
```

原执行replay PID19328/native0、7.064447s已核对2000/2000 pre/postADC、source IDs、base各分量、精确f64增量和支持标记逐值一致，证据REPLAY_RECEIPT.json与replay_endpoints.py.NATIVE_RECEIPT.json。重放无需REAL/TRAIN搜索，不拟合。正常恢复需原R32/R33/R23/R26/R27固定依赖；未随阶段包重复迁移大库。evaluation/data_support及 config 路径给出旧REAL DEV/TRAIN参照、600/250固定行/组、旧预测及原密度控制位置。

首次evaluate_endpoints.py保存所有10fit/8slot及生成后仅聚合字段错误，保留native1原件；resume_evaluation.py直接读取保存C_*/WIDTH_*/generated/joint/probes完成缺失摘要，0重复HGB/EM/推理，不能重新运行原失败入口。summarize_and_plot.py与fix_CDF_layout.py均只读保存结果；CDF原图保留。完整metrics/权重/逐行身份私有，Git只应收录聚合、源码、报告和关键图。

本次用途是历史暴露同采集DEV开发诊断。postADC SOURCE_CLEAN参考source-Q与未量化实际NN source-Q分列；统计PSD成分/谷不是独立粒子真值。后流代数拆分可复现，物理signal/electronic归因unknown。config继承旧字段未追溯改写，实际运行以本轮protocol/native/execution/flow参数为准。
'''
(ROOT/'REPRODUCTION.md').write_text(repro,'utf8')
required=['REPORT.md','REPRODUCTION.md','GATE_METRICS.json','METRICS_COMPACT.json','RESOURCE_ACCOUNTING.json','NATIVE_EXECUTION_CHAIN.json','FAILURES_AND_REPAIRS.json','FIGURE_QA.json','COMPACT_PRIVACY_RECEIPT.json','REPLAY_RECEIPT.json','PSD_ORDERED_CFM/FINAL_FLOW.pt']
assert all((ROOT/n).is_file()for n in required)
receipt={'PID':os.getpid(),'UTC':now(),'kind':'control-only report assembly','scientific_decode_fit_generation':0,'required_files':required,'figure_count':len(figs),'joint_gate_pass':g['retain'],'decision':g['decision'],'science_guarded_elapsed_s':guarded,'script_native_exit_observed_separately':True,'reports_SHA':{n:sha(ROOT/n)for n in ['REPORT.md','REPRODUCTION.md','RESOURCE_ACCOUNTING.json','FIGURE_QA.json','METRICS_COMPACT.json']}}
write('DELIVERY_ASSEMBLY_RECEIPT.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
