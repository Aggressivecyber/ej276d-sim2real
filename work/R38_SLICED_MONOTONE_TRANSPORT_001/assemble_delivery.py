"""Control-only delivery from saved JSON; no numerical arrays, models or fits."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,time,hashlib
ROOT=Path(__file__).resolve().parent
END=datetime.fromisoformat('2026-10-09T22:37:06.331729+00:00')
CAP=512*1024**2;A='SLICED_MONOTONE_TRANSPORT'
def utc():return datetime.now(timezone.utc)
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def put(n,v):
    assert utc()<END
    data=(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with (ROOT/n).open('xb')as f:f.write(data)
def text(n,v):
    assert utc()<END
    with (ROOT/n).open('xb')as f:f.write(v.encode())
def binding(p):
    h=hashlib.sha256();n=0
    with p.open('rb')as f:
        while chunk:=f.read(1024**2):h.update(chunk);n+=len(chunk)
    return {'bytes':n,'sha256':h.hexdigest()}

started=time.monotonic();history=read('RUN_HISTORY.json');m=read('METRICS_COMPACT.json');gate=read('GATE_METRICS.json')
cfg=read('config.json');fit=read('TRAINING_NUMERIC_RECEIPT.json')['records'][A];counter=read('TRAINING_NUMERIC_RECEIPT.json')['counter']
replay=read('REPLAY_RECEIPT.json');gen=read('GENERATION_RECEIPT.json');first=read('START_RECEIPT.json')
prep=read('PREPARATION_RECEIPT.json');upload=read('TRANSFER_RECEIPT.json');returned=read('RETURN_TRANSFER_RECEIPT.json')
cov=read('DEV_COVARIANCE_DESCRIPTION.json');plots=read('PLOT_SUMMARY_RECEIPT.json');repair=read('DISPLAY_REPAIR_RECEIPT.json')
assert len(history)==6 and all(r['actual_worker_native_exit']==r['launcher_native_exit']==0 for r in history)
assert all(fit['original_R33_stream_full_match'].values())and fit['G_before']==fit['G_after']
assert counter['direction_gradient_steps']==5120 and counter['direction_groups_complete']==256 and counter['scalar_maps_complete']==4096
assert m['counters']['HGB_attempts']==10 and m['counters']['density_description_slots']==8 and m['counters']['density_EM_attempts']==7
assert replay['pass']and replay['rows']==2000 and not gate['retain']
science=sum(r['elapsed_s']for r in history);assert science<1800

freezes={}
for n,key in [('PRETRAIN_FREEZE.json','source_bindings'),('LOCAL_PREGEN_FREEZE.json','entries')]:
    freezes[n]={}
    for name,old in read(n)[key].items():
        assert binding(ROOT/name)==old,(n,name)
        freezes[n][name]=True
sources={p.name:binding(p)for p in sorted(ROOT.glob('*.py'))}
for n in ['config.json','remote_config.json','PROTOCOL.md','STATISTICS_INPUT_BINDING.json']:sources[n]=binding(ROOT/n)
put('SOURCE_BINDINGS.json',{'UTC':utc().isoformat(),'PID':os.getpid(),'project_sources':sources,'pre_execution_bindings_unchanged':freezes,
    'G_reference':cfg['base_G'],'G_before':fit['G_before'],'G_after':fit['G_after'],'model_reference':cfg['endpoint_weight_paths'][A],
    'data_model_byte_bindings':'FINAL_MANIFEST.json; byte-only seal, not array decoding or migration test'})

figures=[]
for n in sorted(plots['figures']):
    actual=binding(ROOT/'results/figures'/n);prior=repair['figures'].get(n,plots['figures'][n]);assert actual==prior
    evidence='final two repaired figures actually opened in R38'if n in repair['figures']else'R38 view_image batches before report; no subsequent image change'
    figures.append({'file':'results/figures/'+n,**actual,'actual_image_opened':True,'axes_legends_support_gaps_readable':True,'view_evidence':evidence})
assert len(figures)==11
put('FIGURE_QA.json',{'UTC':utc().isoformat(),'pass':True,'all_final_figures_actually_opened':True,'figure_count':11,'figures':figures,
    'display_only_repairs':repair,'original_images_preserved':'results/figures/layout_initial','new_fits_generation_for_display':0})

chunks={'train_server.py':('245db4','fafd77'),'generate_all.py':('905597','750971'),'evaluate_saved.py':('b16dc6','7a36f1'),
    'replay_all.py':('6df45b','6df45b'),'summarize_and_plot.py':('96f283','2cb52c'),'repair_display.py':('fa17b7','fa17b7')}
native=[{**r,'tool_start':chunks[r['entry']][0],'tool_completion':chunks[r['entry']][1],'observed_tool_native_exit':0}for r in history]
failures=[read('CONTROL_FAILURE_001.json'),{'tool':'7f16c4','observed_tool_native_exit':0,'nonterminating_shell_error':'GUIDE_PROBE_SUMMARY.json absent; actual control is GUIDE_PERTURBATION_SUMMARY.json',
    'PID':'unknown','science_started':False,'new_fit_generation':0,'original_tool_error_preserved':True}]
put('NATIVE_CHAIN.json',{'first_actual_command':first,'science':native,'all_science_terminal':True,'all_true_worker_exits_observed':True,
    'local_remote_launch':read('train_server.py.LOCAL_NATIVE_RECEIPT.json'),'normal_transfers':{'upload':upload,'return':{k:v for k,v in returned.items()if k!='remote_outputs'},
    'return_tools':['8b7f79','691c2a'],'native_exit':0},'control_failures':failures,'all_scientific_failures':0,
    'original_stdout_stderr_and_receipts_kept':True,'no_inferred_unknown_PID_or_exit':True})

sourceN=sum(s['distinct_source_row_indices']for s in fit['statistics']);realN=sum(s['distinct_REAL_row_indices']for s in fit['statistics'])
mm=gate['gate2_PSD_width_centers_noise']['measures'];c2=m['C2ST']['results'][A];pop=cov['tables']['population']
def tab(label,key,d=6):
    r=mm[key];return f"| {label} | {r['BASE']:.{d}f} | {r['candidate']:.{d}f} | {r['relative_change']*100:+.2f}% |"
table='\n'.join(['| 描述 | BASE | R38 | 相对变化 |','|---|---:|---:|---:|',tab('总体 PSD W1','PSD_overall_W1'),tab('共同Q bins1–7 PSD W1','PSD_common_Q_equal_mean_W1'),
    tab('低侧峰位MAE，bins3–7','component0_peak_position_MAE'),tab('高侧峰位MAE，bins3–7','component1_peak_position_MAE'),
    tab('低侧raw SD误差MAE，bins3–7','component0_raw_width_MAE'),tab('高侧raw SD误差MAE，bins3–7','component1_raw_width_MAE'),
    tab('baseline RMS分布W1（V）','baseline_RMS_V_overall_W1',9),tab('baseline ACF1分布W1','ACF1_overall_W1')])
bad_bound=[r['location']for r in gate['gate4_boundary_guide']['boundaries']if not r['pass']]
guide=gate['gate4_boundary_guide']['guide_probe'];ls=fit['layer_summaries'];p=m['projection_common_support']
paired=gate['gate1_complete_response']['paired_minus_BASE'];valley=gate['gate3_source_preservation']['actual_source_pure_had_valleys']
valleytext='；'.join(f"{v['actual_source_bin']} N={v['N']}：{v['BASE']*100:.2f}%→{v['candidate']*100:.2f}%（{v['delta']*100:+.2f} pp）"for v in valley)
projtext='；'.join(f"{key}四块新方向等权均值 {p[key]['domains']['BASE']['new_block_mean']:.6f}→{p[key]['domains'][A]['new_block_mean']:.6f}"for key in ['shape480','diff479','baseline100_V'])
report=f'''# R38 切片单调变换：关闭固定配方

联合门槛未通过，gates=[true,false,false,false,true]。总体PSD与低侧条件峰位/宽度改善，但高侧峰位、baseline RMS、源pure-had谷区与固定边界退化，bin2新增密度失效。保留本轮负结果，不追加层数、方向、抖动、地板或调参。项目目标继续，下一问题由根代理登记。

完整480点源波形→冻结R32 CONTROL_FINAL G→原source-Q区内32层非线性单调切片映射→float32→一次14bit[-.5,.5]V ADC。统计起点是原G，不是R37校准输出。每层16个正交方向、20次固定步长归一化切空间梯度上升和QR回缩；每方向257个0.005–0.995分位结点，区内线性插值，两端斜率1平移外推。严格单调失败必须停，不删除bin或重启；本次256组、4096个标量map全部完成。方法对应迭代一维切片运输的思路，具体QR/线性分位实现见PROTOCOL中的主文献链接，不能称原论文完整复现或新的架构创新。

唯一统计作业严格复用R33原2950×32=94400条source/REAL/bin/ordinal/z-white/style流，七项实验hash全部匹配，G前后绑定一致；源与目标边缘仅独立使用，未产生物理配对。实际使用{sourceN}唯一源TRAIN行、{realN}唯一REAL TRAIN行，共同支持池仍为3753/6593及18895/21047；94400是重复抽样次数，不是独立采集数。PCG64(20261010381)按layer-major/bin-minor生成初始方向，记录了NumPy版本/初始方向hash；5120次方向更新、20480次subset投影调用、768次全bin投影调用均记账，神经参数更新0。全部5120步未出现nonascending/no-op；这只描述所求投影目标。最大正交误差{max(r['orthogonality_maxabs']for r in ls):.3e}，subset层逆变换最大误差{max(r['subset_layer_inverse_maxabs']for r in ls):.3e}；strict x/y结点最小差{min(r['minimum_x_knot_difference']for r in ls):.6g}/{min(r['minimum_y_knot_difference']for r in ls):.6g}，实际标量segment slope范围[{min(r['segment_slope_min']for r in ls):.6g},{max(r['segment_slope_max']for r in ls):.6g}]。

主2000父事件顺序、ID、组成与权重不变：pure_EE417/pure_had1518/mixed65。1583行使用map，417支持外保留G；全部2000有效、无非有限或ADC裁剪。额外joint1024、源探针832、固定bin guide128分别保存exact preADC与增量/舍入分量。本轮没有新EVAL/CAL波形、长采集、G4、神经训练或新REAL获取。

full493 BA从BASE .978→{c2['full_historical_shape']['balanced_accuracy']:.3f}，配对Δ={paired['full_historical_shape']['BA_delta_first_minus_second']:.3f}，95%CI={paired['full_historical_shape']['BA_delta_95CI']}；纯时间u480 .978→{c2['normalized_u480']['balanced_accuracy']:.3f}，Δ={paired['normalized_u480']['BA_delta_first_minus_second']:.3f}，CI={paired['normalized_u480']['BA_delta_95CI']}。较R37 .924/.918的点差+.002/+.008，满足预定+.01上限，但仍明显可分。完整V480 BA={c2['full_voltage']['balanced_accuracy']:.3f}、baseline局部={c2['baseline_local']['balanced_accuracy']:.3f}、Q/PSD二维={c2['Q_PSD_2D']['balanced_accuracy']:.3f}。固定每域600训练/250留出及原组，500次区组CI只条件于已拟合分类器，不含响应训练不确定性，不是物理等效阈值。

{table}

峰位/宽度两成分只是密度描述，不是粒子真值，raw SD不是Gaussian FWHM。共同合格区间统一为bins3–7。原BASE有效bins2–7中bin2新增not_converged500，是预先门槛失败，不能用缩小后的交集MAE掩盖；bin1未分离、bin8 N=39<40未拟合，所有缺口保留。高侧峰位MAE增加51.28%与低侧/两侧宽度收益同时存在。

实际未量化NN source-Q下pure-had：{valleytext}。均超过2pp上限；这是源条件保真风险，不等同逐事件粒子错类。实际213/181与历史postADC SOURCE_CLEAN参考207/182分列，父身份/组成本身未改变。全体source Q→output Q秩相关.999726→.998924，source PSD→output PSD .418992→.410762，降幅虽通过，不能覆盖pure-had局部分布失败。源替换非零敏感性不证明物理映射正确；固定源变噪声按正确同源参考，Fast/slow反事实不是新粒子标签。

九个固定一二阶边界有5项失败：{', '.join(bad_bound)}。例如99→100 W1 0.024390→0.137765mV，中心149/150二阶W1约3.91494/3.97781mV，未改门槛。128条guide在150:400加去均值正弦、段RMS1mV，原bin/G噪声固定；实际Qfast不变，Qt/PSD仅float32舍入漂移，全部输出变化，完整480点preADC差RMS中位{guide['preADC_output_RMS_V']['quantiles'][3]*1000:.3f}mV。扰动影响扩大不能称物理正确增益。非线性map后的物理信号/电子噪声拆分unknown；joint8仅可加代数描述，包含base信号/电子/求和舍入/真实增量和交叉项。

共同观察Q bins1–7的独立新方向描述：{projtext}。shape边缘差改善而时间差分与baseline差恶化，表明完整电压投影收益不等于时间联合涨落正确；不把这一现象唯一归因某层或插值。DEV全体协方差相对Frobenius误差{pop['BASE']['covariance_relative_Frobenius_vs_REAL']:.6f}→{pop[A]['covariance_relative_Frobenius_vs_REAL']:.6f}，非对角相关RMS差{pop['BASE']['off_diagonal_correlation_RMS_difference']:.6f}→{pop[A]['off_diagonal_correlation_RMS_difference']:.6f}。相关结构改善与总体协方差略差均保留，DEV描述不参与map拟合。

独立CPU完整source→G→32层map→一次ADC重放2000条，所有保存字段bitwise通过。生成与回放均G chunk128、相同full-bin矩乘顺序，不额外分块改变float64末位。10新HGB、8密度槽/7EM；旧BASE/R33/R35–37及RAW RR零重拟合，旧RAW RR二维预测缺失继续unknown。

6个真实科学进程全部native0，累计{science:.6f}s，唯一统计作业49.079034s，方向CPU4 float64、原G在服务器CUDA；实际worker RSS峰2,968,145,920B，GPU allocated57,150,976B。服务器两次RSS查询失败在21:26:57.968424/21:26:58.068909，原因unknown，不把接近退出时间关系当解释、不声称这两个间隙连续测量。所有原生日志与一次缺失retrieve_server.py控制调用、一次缺失guide摘要名的非终止控制读取错误保留，未发生科学重复。11张最终图均实际打开，2张只修显示且初版保留；资源/native链/执行绑定与最终清单齐全。

关闭这一个固定切片单调配方；数值可逆、投影边缘吻合或总体PSD距离改善均不能取代完整响应联合验收，也不能推得所有非线性运输不可行。同采集人工50记录组、旧DEV暴露、无独立粒子标签及历史模型/噪声先验成本边界仍保留，本轮为开发诊断。

主要产物：GATE_METRICS.json、METRICS_COMPACT.json、完整metrics.json、SLICED_MONOTONE_TRANSPORT/SLICED_TRANSPORT_PARAMS.npz及256组参数/轨迹、generated.npz、exact辅助载荷、REPLAY_RECEIPT.json、11图、RESOURCE_RECEIPT.json、NATIVE_CHAIN.json和FINAL_MANIFEST.json。完整逐事件载荷/模型留E/D，紧凑版只含聚合量。
'''
text('REPORT.md',report)
text('REPRODUCTION.md',f'''# R38 复现

旧stage已封存且固定科学配方关闭。需要复现时在新的E/D目录建立新记录，不覆盖原stage、不删除一次性标记；本文不额外授权重拟合。

实际唯一统计入口：{cfg['remote_root']}/train_server.py，经linux_supervise.py/linux_owned_worker.py真实PID/nonce守护，Python {cfg['server_python']}，Python3.12.3/NumPy1.26.4/Torch2.12.0+cu130/SciPy1.11.4、RTX5070。原G chunk32及R33原七条实际流只读，G前后相同；随后CPU4 float64方向学习/分位map。正常SSH/scp/native完成，没有迁移内容测试。

本机实际Python：{cfg['CPU_python']}，Python3.13.3/NumPy2.2.6/Torch2.10.0+cpu/sklearn1.7.2。顺序入口：supervise.py generate_all.py <CPU_python>；evaluate_saved.py；replay_all.py；summarize_and_plot.py。-B/PYTHONUTF8/PYTHONDONTWRITEBYTECODE及E/D TEMP/TMP、cache，4线程与4GiB真实workerRSS守护。图显示修复入口repair_display.py不重拟合或生成。

完整推理需要ACTUAL_GENERATION_INPUTS.npz的完整source480/父ID/z/white/style、冻结G {cfg['base_G']}、SLICED_TRANSPORT_PARAMS.npz中A[8,32,480,16]/xk/yk[8,32,257,16]/TRAIN mu/scale/8bin edges、model/legacy_model/flow_runtime/sliced_core。按未量化实际源Q选bin，支持外G回退；同full-bin变换顺序，每层完整矩乘与标量插值，转f32仅一次ADC。推理不查REAL库、不独立目标采样、不重拟合。

已执行独立2000条重放：actual3840/super12588，tool6df45b/native0，6.855693s，所有字段bitwise。记录含完整源、support/bin、base分量、out64、cast增量/舍入、pre/post、ID；保存输出不作为推理输入。Linux拟合与Windows生成环境不同，不称跨平台逐位相同；本机生成与独立本机重放一致已执行。256组训练轨迹、实际初始化hash/seed/版本与严格结点/逆检查分别在DIRECTION_INITIALIZATION.json、LAYER_*、STATISTICS_FIT_RECEIPT.json和SLICED_FORMULA_CHECK.json。

模板flow/covariance/calibration字段只是增量与二阶描述兼容名，实际不是CFM/ODE或R37高斯map。最终物理分支分解unknown。SOURCE_BINDINGS/FINAL_MANIFEST绑定本轮源码与产物，FIGURE_QA标最终两图修复版本，原图在layout_initial。报告/封存阶段只读JSON和纯字节hash，无数组解码/模型运行。
''')

files=[p for p in ROOT.rglob('*')if p.is_file()];localbytes=sum(p.stat().st_size for p in files);assert localbytes+512*1024<CAP
put('RESOURCE_RECEIPT.json',{'stage':cfg['stage'],'UTC':utc().isoformat(),'first_actual_command':first,'registered_UTC':cfg['registered_UTC'],
    'executor_deadline_UTC':cfg['deadline_UTC'],'science_elapsed_s_including_plots_display_repair':science,'science_budget_s':1800,
    'science_process_count':len(history),'failed_scientific_process_count':0,'statistics_fit_actual_supervised_s':history[0]['elapsed_s'],'statistics_fit_budget_s':900,
    'counts':{**m['counters'],**counter,'statistical_group_fits':256,'scalar_maps':4096,'unique_complete_fit_attempts':1,
        'actual_distinct_source_TRAIN_rows':sourceN,'actual_distinct_REAL_TRAIN_rows':realN,'complete_CPU_replay_rows':2000,
        'old_reference_HGB_or_density_refits':0,'new_neural_updates':0,'new_G4':0,'new_long_raw_decode':0,'new_REAL_acquisition_read':0,'installation':0,'migration_content_tests':0},
    'threads':4,'actual_worker_RSS_max_observed_bytes':max(r['actual_worker_RSS_peak_bytes']for r in history),'RSS_limit_bytes':4*1024**3,
    'GPU_peak_allocated_bytes':fit['GPU_peak_allocated_bytes'],'GPU_allocated_limit_bytes':4*1024**3,
    'memory_probe_failures':sum(r['memory_probe_failures']for r in history),'memory_probe_failure_events':[e for r in history for e in r['memory_probe_failure_events']],
    'RSS_gap_cause':'unknown; proximity to exit does not establish cause','continuous_RSS_in_query_gaps':'unknown',
    'remote_worker_self_peak_RSS_bytes':history[0]['worker_self_receipt']['self_peak_RSS_bytes'],
    'dependency_copy_payload_bytes':prep['copied_dependency_payload_bytes'],'upload_tar_payload_bytes':upload['upload_tar_bytes'],
    'returned_tar_payload_bytes':returned['return_tar_bytes'],'return_package_and_extracted_payload_both_retained':True,
    'control_failures':failures,'local_stage_bytes_before_resource_seal':localbytes,'local_stage_files_before_resource_seal':len(files),
    'output_cap_bytes':CAP,'final_exact_local_size':'FINAL_RECEIPT and external seal stdout; copies and return package included',
    'system_library_loading_project_IO':'not measured; RSS independently guarded','new_read_write_framework':False,
    'science_native_exits':[r['actual_worker_native_exit']for r in history],'new_science_in_report_or_seal':0,
    'scientific_gates':{k:v['pass']for k,v in gate.items()if k.startswith('gate')},'decision':gate['decision'],'project_goal_closed':False})
put('DELIVERY_ASSEMBLY_RECEIPT.json',{'PID':os.getpid(),'UTC':utc().isoformat(),'elapsed_s':time.monotonic()-started,
    'control_only_saved_JSON_and_project_byte_binding':True,'scientific_arrays_models_fits_generation':0,'science_s_unchanged':science,
    'native_exit_evidence':'external exec result; no invented future exit','seal_pending':True})
print(json.dumps({'PID':os.getpid(),'UTC':utc().isoformat(),'science_s':science,'reports_ready':True,'figure_count':11,
    'final_repaired_figures':repair['figures'],'decision':gate['decision']},ensure_ascii=False),flush=True)
