"""Control-only report and terminal seal; no scientific computation."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,math,hashlib
ROOT=Path(__file__).resolve().parent;START=time.monotonic()
DEADLINE=datetime.fromisoformat('2026-10-10T00:06:36.130049+00:00')
def utc():return datetime.now(timezone.utc).isoformat()
def load(name):return json.loads((ROOT/name).read_text('utf8'))
def encode(v):return(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
def put(name,v):
    if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('seal deadline')
    p=ROOT/name;assert not p.exists(),name;p.write_bytes(encode(v))
def text(name,v):
    p=ROOT/name;assert not p.exists(),name;p.write_text(v.strip()+'\n','utf8')
def digest(p):
    h=hashlib.sha256();n=0
    with p.open('rb')as f:
        while b:=f.read(1024**2):n+=len(b);h.update(b)
    return{'bytes':n,'sha256':h.hexdigest()}
def link(n):return f'[{n}]({(ROOT/n).as_posix()})'
def main():
    cfg=load('config.json');h=load('RUN_HISTORY.json');gate=load('GATE_METRICS.json');t=load('TRAINING_COMPACT.json');p=load('PLOT_SUMMARY_RECEIPT.json');qa=load('FIGURE_QA.json')
    ev=load('EVALUATION_NUMERIC_RECEIPT.json');gen=load('GENERATION_NUMERIC_RECEIPT.json');replay=load('REPLAY_RECEIPT.json');tn=load('TRAINING_NUMERIC_RECEIPT.json');tr=load('TRANSFER_RECEIPT.json');ret=load('RETURN_TRANSFER_RECEIPT.json');prep=load('PREPARATION_RECEIPT.json');external=load('EXTERNAL_NATIVE_CHAIN.json')
    assert len(h)==6 and all(x['actual_worker_native_exit']==x['launcher_native_exit']==0 for x in h)
    science=math.fsum(x['elapsed_s']for x in h);assert science<=1800 and h[0]['elapsed_s']<=900
    assert t['updates']==t['D_updates']==t['style_reader_updates']==2950
    assert ev['counters']['HGB_attempts']==20 and ev['counters']['density_description_slots']==16 and ev['counters']['density_EM_attempts']<=16
    assert replay['rows']==4000 and replay['pass']and qa['all_final_figures_opened']and set(qa['figures'])==set(p['figures'])and p['figure_count']==12
    repair=load('FIGURE_REPAIR_RECEIPT.json');final_figures={**p['figures'],**repair['final_figures']}
    for n,q in qa['figures'].items():assert q['sha256']==final_figures[n]['sha256']and q['visual_pass']
    nce=t['NCE_participation'];assert not nce['NCE_evaluated']and nce['NCE_dedicated_forward_calls']==0 and all(nce['project_initial_final_bitwise_equal'].values())
    passes=[gate['gate'+str(i)+'_'+k]['pass']for i,k in enumerate(['complete_response','PSD_width_centers_noise','source_preservation','boundary_source','same_treated_REAL_control','numerical_replay'],1)]
    put('NATIVE_CHAIN.json',{'UTC':utc(),'scientific_processes':h,'external_native_tool_chain':external,'science_total_s':science,'self_receipt_not_external_native_exit':True})
    gaps=[{'entry':x['entry'],'events':x.get('memory_probe_failure_events',[]),'count':x['memory_probe_failures']}for x in h if x['memory_probe_failures']]
    put('FAILURE_ACCOUNTING.json',{'UTC':utc(),'science_native_failures':sum(x['actual_worker_native_exit']!=0 for x in h),'science_cost_s':science,'control_native_failures':external.get('control_native_failures',[]),
        'RSS_query_gaps':gaps,'RSS_gap_causes':'unknown; temporal proximity to exit is not proof of cause','legacy_metadata':{'evaluate_saved_terminal_log_label':'R40 saved evaluation complete inherited template string; R41 actual endpoint keys/config/receipts authoritative'},
        'NCE_loss_value':'not evaluated, not measured zero','negative_scientific_results_preserved':True,'repeat_fits_or_generation':0,
        'display_defects_and_repairs':repair,'original_executed_plot_source_preserved':True})
    g2=gate['gate2_PSD_width_centers_noise'];g3=gate['gate3_source_preservation'];g4=gate['gate4_boundary_source'];g5=gate['gate5_same_treated_REAL_control'];g1=gate['gate1_complete_response']
    labels={'PSD_overall_W1':'总体PSD W1','PSD_common_Q_equal_mean_W1':'共同Q区PSD W1','component0_peak_position_MAE':'低侧峰位MAE','component1_peak_position_MAE':'高侧峰位MAE','component0_raw_width_MAE':'低侧raw-width MAE','component1_raw_width_MAE':'高侧raw-width MAE','baseline_RMS_V_overall_W1':'基线RMS W1（V）','ACF1_overall_W1':'基线ACF1 W1'}
    metric_rows=[]
    for k,label in labels.items():
        x=g2['measures'][k];a=x['candidate'];b=x['BASE'];delta=x['relative_change']
        metric_rows.append(f"| {label} | {b:.9g} | {a:.9g} | {100*delta:+.2f}% |"if a is not None and b is not None and delta is not None else f'| {label} | {b} | {a} | undefined |')
    control_rows=[]
    for k,label in [('full_historical_shape','shape493'),('normalized_u480','纯u480'),('baseline_local','baseline')]:
        x=g5['three_views'][k];ci=x['BA_delta_95CI'];control_rows.append(f"| {label} | {x['BASE_G_REAL_BA']:.3f} | {x['new_G_REAL_BA']:.3f} | {x['BA_delta_first_minus_second']:+.3f} [{ci[0]:+.5f},{ci[1]:+.5f}] | {'通过'if x['pass']else'失败'} |")
    response=[]
    for v,label in [('full_historical_shape','shape493'),('normalized_u480','纯u480')]:
        x=g1['paired_minus_BASE'][v];ci=x['BA_delta_95CI'];response.append(f"{label} BA={g1['candidate_BA'][v]:.3f}，减BASE配对差{x['BA_delta_first_minus_second']:+.3f}，95%CI[{ci[0]:+.5f},{ci[1]:+.5f}]")
    failed_boundaries=[x['location']for x in g4['boundaries']if not x['pass']]
    rms_summary=g4['full_SOURCE_perturbation']['preADC_output_RMS_V'];rms_median=rms_summary['quantiles'][rms_summary['quantile_levels'].index(.5)]
    valleys='；'.join(f"actual-source {x['actual_source_bin']} N{x['N']}：{x['BASE']:.4%}→{x['candidate']:.4%}（{100*x['delta']:+.3f} pp）"for x in g3['actual_source_pure_had_valleys'])
    ranks='；'.join(f"{x['measure']} {x['BASE']:.7f}→{x['candidate']:.7f}"for x in g3['whole_population_source_ranks'])
    training_original=json.loads((ROOT.parent/'R32_CONDITIONAL_MEAN_TRANSLATION_001/SERVER_FRESH_CONTROL/TRAINING_RECEIPT.json').read_text('utf8'))
    report=f'''
# R41：移除 PatchNCE 的固定消融

六项共同门槛为 **{'／'.join('通过'if b else'失败'for b in passes)}**，结论为 `{gate['decision']}`。{'；'.join(response)}。完整响应点值门槛仍为shape≤.934/u≤.928且两项配对95%上界<0；数值可复现不能替代科学通过。本结论只关闭或保留这一预定系数0配方，不外推所有内容约束或无配对转换的可行性。

完整480点G4派生源波形直接进入原R32 CNN，保持源父和沉积组成。唯一训练改动是删除既有PatchNCE专用source/output编码、投影和交叉熵，不进入G目标；它没有被计算，`NCE_evaluated=false`、损失值null、目标系数0、专用前向0。原三层位置列表8850份仍消费保存为未使用的审计流，不能称作执行了NCE。没有R40时间层、新参数、MMD、额外均值项、后置校准或替代内容约束。

原网络没有BN/dropout。投影头3680参数留在state及optimizer中，2950次反传全部grad=None，六个weight/bias张量初末逐值相同。全部G参数原requires_grad保持；总G参数{nce['total_G_parameters']}，实际参与{nce['participating_G_parameters']}，末次其他G参数均有非None梯度且所有G/reader值有限，未冻结encoder。省去5900次专用encode调用；运算量、全局clip与Adam梯度路径随消融改变，FLOP未实测，不声称等算力或唯一优化机制。

第一更新前严格核对原G/reader/D的全部state键和值，并实际validate(0)立即断言J20差≤1e-6：{t['step0_J20']:.10f}，差0，D/G/reader更新计数均0。原94400行、七项实际源/REAL/bin/ordinal/NCE审计/随机流和8850 fresh方向全部匹配冻结对照。载入6593/21047，实际唯一参与3752源/18739实测；这些数量不等于独立采集。新Adam原lr/.5,.999/clip5及D1不变。G/D/reader均2950。

内部所选step{t['selected_step']}、J20={t['selected_J20']:.9f}；实际2950末点J20={t['final_J20']:.9f}。仅固定末点用于DEV，不按DEV改选内部checkpoint。记录训练{t['training_s']:.6f}s，原同服务器BASE单臂记录{training_original['training_s']:.6f}s；删除NCE减少专用前向，但耗时不是硬件独立的因果量。所有训练/验证轨迹及实际目标分项保留。

| 指标 | 冻结BASE | 固定2950 | 相对变化 |
| --- | ---: | ---: | ---: |
{chr(10).join(metric_rows)}

峰位/raw-width只用共同合格区并同时保留全部支持缺口；BASE所需bins2–7新增失败为{g2['new_required_density_failures']}。两高斯成分仅为PSD密度描述，不能称独立粒子真值。未通过区不删掉，最高区缺口见完整门槛文件。完整条件均值/中位、PSD/Q、噪声、曲率与DEV相关矩阵均报告，不能用某个均值曲线或单项局部收益作整体通过。

2000原源父身份、组成、顺序和权重保留；417个旧TRAIN支持外也用新G，无BASE回退，无新增无效或ADC裁剪。{valleys}。这些actual未量化NN输入分区N213/181与历史postADC参考N207/182单列，不能混称。{ranks}。源相关与跨域谷区是代理，不证明逐事件物理类别对应。纯EE/had/mixed及入射PDG证据边界沿用原来源，不随机重标源类。

9个固定边界失败位置为{failed_boundaries}。前128支持内完整SOURCE输入150:400零积分/RMS1mV扰动使用原随机量，非相同比例{g4['full_SOURCE_perturbation']['output_nonidentical_fraction']:.3f}；preADC全480输出RMS中位{1000*rms_median:.6f}mV。源敏感性不是正确物理响应；prompt100:150包括峰与早期下降，不能把整段曲率称纯上升沿。

旧同处理BASE_REAL完全复用封存R39的2000波形、10评价器/密度和已执行独立重放，零重生成/拟合。新G处理相同TRAIN_REFERENCE与z/white/style；不赋予源组成或粒子标签。三门如下，V480等其它视图仅描述，对RAW绝对变化完整报告，不预设随机G(REAL)恒等。

| 同处理REAL视图 | BASE BA | 新BA | 配对差与95%CI | 门槛 |
| --- | ---: | ---: | --- | --- |
{chr(10).join(control_rows)}

主SOURCE2000+新REAL2000、joint1024、原probe832和SOURCE扰动128全部精确保存。独立本机CPU完整链4000逐值通过，原chunk128和运算次序，输入为原源/REAL与随机量；保存输出仅用于最终比较。signal/electronic共享输出归一化和交叉项，功能分量不是独立物理噪声。没有模块置零第二候选。

服务器实际worker{h[0]['actual_worker_PID']}、监督{h[0]['supervisor_PID']}，native0；原生训练{h[0]['elapsed_s']:.6f}s，RSS峰{h[0]['actual_worker_RSS_peak_bytes']}B，GPU allocated峰{tn['GPU_peak_allocated_bytes']}B。查询gap{h[0]['memory_probe_failures']}次，具体时间保留且原因unknown。其余{len(h)-1}本机科学或纯绘图作业native0；累计{science:.6f}/1800s，20新HGB、16密度描述槽/{ev['counters']['density_EM_attempts']}实际EM，参考新增拟合0。12张最终图已实际打开；source rank旧TIME标签与训练图注重叠已仅按保存结果修复，原图与执行脚本保留，没有重复拟合或生成。实际资源和原生链见对应JSON。

本轮为历史暴露同采集TRAIN/DEV开发比较，人工50记录组不是独立批次；500区组bootstrap条件于固定分类器，不包括响应重训或独立采集不确定性。无新REAL/EVAL/CAL/G4/安装/代理/Git操作或迁移内容测试。若门槛失败，不用新的NCE系数或其它损失救本轮；下一问题须由根依据完整证据登记。

{link('METRICS_COMPACT.json')}、{link('GATE_METRICS.json')}、{link('TRAINING_COMPACT.json')}、{link('REPLAY_RECEIPT.json')}、{link('FIGURE_QA.json')}、{link('RESOURCE_RECEIPT.json')}。完整逐事件载荷、模型/随机流/预测留E/D，聚合发布文件不含身份和预测。
'''
    text('REPORT.md',report)
    text('REPRODUCTION.md',f'''
# R41 执行与复现

实际服务器入口 `{cfg['remote_root']}/linux_supervise.py train_server.py`，Python `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`，Python3.12.3/NumPy1.26.4/torch2.12.0+cu130/scipy1.11.4/cuda:0。先资源/本人argv+cwd预检，再正常源码scp；训练只读复用原R32/R33的缓存，不做迁移一致性测试。

准备 `prepare_stage.py`、`prepare_summary.py`，资源 `resource_check.py`，`deploy_training.py` 冻结源码与配置；`launch_remote.py train_server.py` 唯一启动，`retrieve.py`正常取回。执行源与版本在PRETRAIN_FREEZE/EVALUATION_SOURCE_FREEZE；原model/legacy_model保持R32，`online_training.py`只取消实际NCE与目标并加入审计。cfg中少量remote历史字段来自完整R32配置，以execution/remote_root/实际native为准；evaluation结束日志的R40字样为模板标签，实际域键NO_PATCH_NCE_FINAL/REAL_CONTROL明确。

本机CPU入口 `supervise.py generate_all.py <CPU_python>`、`evaluate_saved.py`、`replay_all.py`、`summarize_and_plot.py`，CPU Python `{cfg['CPU_python']}`；全部TEMP/TMP/cache在本轮E目录，-B/PYTHONDONTWRITEBYTECODE/UTF8，4线程。不得运行旧R40的prepare/summary来覆盖本轮文件。

固定模型NO_PATCH_NCE_FINAL/FINAL_TRANSLATOR.pt（2950）；TRANSLATOR.pt为内部step{t['selected_step']}记录，不是本轮DEV终点。原2000完整输入/ID/z/white/style在ACTUAL_GENERATION_INPUTS，REAL与reference在REAL_DEV_AND_REFERENCE_PRIVATE，joint/probe及CAL_REFERENCE_METADATA仅沿用既有定义。新权重、optimizer/reader/D、未用NCE位置流、投影头初末、实际随机流均保存。BASE_REAL直接引用R39，零再次推理/拟合。

独立4000主完整链已经执行native0、逐值pass，证据REPLAY_RECEIPT和对应NATIVE_RECEIPT，未读保存输出作输入。不同CPU/CUDA库训练不保证跨平台位等，重放只比较本轮同CPU生成。未来恢复在新E/D目录重新登记时间/路径，保存科学config/实际输入/权重/随机量不变；不得删除.used/SSH标记重复提交已封存作业。本文件不触发新的科学计算。
''')
    files=[x for x in ROOT.rglob('*')if x.is_file()];bytes_before=sum(x.stat().st_size for x in files)
    resource={'UTC':utc(),'PID':os.getpid(),'stage':ROOT.name,'registered_UTC':cfg['registered_UTC'],'first_actual_control':load('START_RECEIPT.json'),'executor_deadline':cfg['deadline_UTC'],
        'science_total_s':science,'science_cap_s':1800,'scientific_processes':h,'training_record_s':t['training_s'],'training_native_s':h[0]['elapsed_s'],'training_cap_s':900,
        'CPU_threads':4,'actual_worker_RSS_observed_peak_bytes':max(x['actual_worker_RSS_peak_bytes']for x in h),'RSS_cap_bytes':4*1024**3,'RSS_monitoring_complete':not gaps,'RSS_query_gaps':gaps,'gap_cause':'unknown',
        'GPU_peak_allocated_bytes':tn['GPU_peak_allocated_bytes'],'GPU_cap_bytes':4*1024**3,'new_parameters':0,'G_total_parameters':nce['total_G_parameters'],'participating_G_parameters':nce['participating_G_parameters'],'unused_project_parameters':nce['project_parameter_count'],
        'fits':1,'updates_G_D_reader_each':2950,'actual_training_rows':94400,'unique_update_source_REAL':[3752,18739],'loaded_pool_source_REAL':[6593,21047],
        'NCE_dedicated_forward_calls':0,'NCE_loss_computed':False,'NCE_loss_value':None,'dedicated_encode_calls_avoided':5900,'NCE_position_lists_audit_only':8850,'fresh_direction_groups':8850,'validation_generation_rows':t['validation_generation_rows'],'FLOP':'not measured; no equal-FLOP claim',
        'generation_counters':gen['counters'],'evaluation_counters':ev['counters'],'independent_CPU_fullchain_rows':4000,'old_REFERENCE_newfit_inference':0,
        'science_native_failures':0,'scientific_repeats':0,'display_repair_processes':1,'display_repair_elapsed_s':h[-1]['elapsed_s'],'EVAL_read':0,'CAL_waveform_decode':0,'new_REAL_acquisition':0,'new_G4':0,'installation':0,'agents':0,'Git':0,'old_stage_writes':0,'migration_content_tests':0,
        'before_terminal_file_count':len(files),'before_terminal_stage_bytes':bytes_before,'local_dependency_copy_bytes':prep['dependency_payload_bytes'],'upload_tar_bytes':tr['upload_tar_bytes'],'return_tar_bytes':ret['return_tar_bytes'],
        'packages_are_in_stage_total_not_disjoint_extra':True,'output_cap_bytes':512*1024**2,'final_exact_count_and_bytes':'FINAL_RECEIPT includes terminal controls','decision':gate['decision']}
    put('RESOURCE_RECEIPT.json',resource)
    excluded={'FINAL_MANIFEST.json','FINAL_RECEIPT.json'};items=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file()and p.name not in excluded:
            assert not p.is_symlink();items.append({'path':p.relative_to(ROOT).as_posix(),**digest(p)})
    member_bytes=sum(x['bytes']for x in items);manifest={'stage':ROOT.name,'UTC':utc(),'PID':os.getpid(),'files':items,'file_count':len(items),'member_bytes':member_bytes,'excluded_terminal_controls':sorted(excluded),'self_reference_rule':'manifest excludes itself and final receipt; final receipt anchors manifest; external tool confirms native exit','old_stage_files_modified':False,'scientific_recomputation_in_seal':False}
    raw=encode(manifest);sha=hashlib.sha256(raw).hexdigest();put('FINAL_MANIFEST.json',manifest)
    receipt={'stage':ROOT.name,'status':'sealed','UTC':utc(),'PID':os.getpid(),'decision':gate['decision'],'retain':gate['retain'],'gates_in_order':passes,'science_total_s':science,
        'manifest_sha256':sha,'manifest_bytes':len(raw),'manifest_member_count':len(items),'manifest_member_bytes':member_bytes,'stage_file_count_including_terminal_controls':len(items)+2,'stage_total_bytes_including_terminal_controls':0,'receipt_bytes':0,
        'terminal_hashes':{n:digest(ROOT/n)for n in ['REPORT.md','REPRODUCTION.md','RESOURCE_RECEIPT.json','NATIVE_CHAIN.json','FAILURE_ACCOUNTING.json','FIGURE_QA.json']},'output_cap_bytes':512*1024**2,'deadline_UTC':cfg['deadline_UTC'],'before_deadline':True,'native_exit_evidence':'await actual external tool receipt; self receipt is not native exit','no_further_stage_writes':True,'scientific_job_active':False}
    for _ in range(8):
        n=len(encode(receipt));total=member_bytes+len(raw)+n
        if receipt['receipt_bytes']==n and receipt['stage_total_bytes_including_terminal_controls']==total:break
        receipt['receipt_bytes']=n;receipt['stage_total_bytes_including_terminal_controls']=total
    else:raise RuntimeError('terminal size convergence')
    assert receipt['stage_total_bytes_including_terminal_controls']<=512*1024**2;put('FINAL_RECEIPT.json',receipt)
    final=(ROOT/'FINAL_RECEIPT.json').read_bytes();last=max(p.stat().st_mtime_ns for p in ROOT.rglob('*')if p.is_file())
    print(json.dumps({'PID':os.getpid(),'status':'sealed','FINAL_MANIFEST_sha256':sha,'FINAL_RECEIPT_sha256':hashlib.sha256(final).hexdigest(),'last_stage_write_UTC':datetime.fromtimestamp(last/1e9,timezone.utc).isoformat(),'file_count':len(items)+2,'bytes':receipt['stage_total_bytes_including_terminal_controls'],'science_s':science,'seal_control_elapsed_s':time.monotonic()-START,'no_further_stage_writes':True}),flush=True)
if __name__=='__main__':main()
