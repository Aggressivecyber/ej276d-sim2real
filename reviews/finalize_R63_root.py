from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path(__file__).resolve().parents[1];ID='R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001';S=R/'work'/ID
NOW=datetime.now(timezone.utc).isoformat();ARM='COVERAGE_RESPONSE';CTRL='COVERAGE_REAL_CONTROL'
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,v):(R/p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
def local(n):return read('work/'+ID+'/'+n)
def bind(n):
    p=R/n
    return {'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def outcome(b):return '通过'if b else'未通过'
def value(x,fmt='.8g'):return format(x,fmt)if x is not None else'未定义'
M=local('METRICS_COMPACT.json');gate=local('GATE_METRICS.json');A=M['acceptance_A_C'][ARM]
train=local(ARM+'/TRAINING_RECEIPT.json');audit=local('TRAINING_FROZEN_AUDIT.json');replay=local('REPLAY_RECEIPT.json');qa=local('FIGURE_QA.json')
direction=read('reviews/R63_ROOT_DIRECTION_001.json')
assert direction['scientific_decision']==gate['decision'] and not direction['model_goal_complete']
assert qa['all_figures_opened'] and qa['layout_pass'] and audit['pass'] and train['updates']==50000
assert local('FROZEN_ARTIFACT_REVIEW.json')['pass']
entries=['train_final.py','numerical_check.py','generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py']
native=[local(n+'.NATIVE_RECEIPT.json')for n in entries]
assert all(n['termination_reason']is None and n['handshake_before_imports'] and n['actual_worker_native_exit']==0 and n['worker_self_receipt']['native_exit']==0 for n in native)
assert replay['rows']==4000 and replay['pass'] and replay['portable_loader_executed']
assert not local('LIVE_OBSERVATION.json')['actual_process']['exists']
failed_attempts=[];seen_attempts=set()
for p in sorted((S/'repairs').rglob('*.NATIVE_RECEIPT.json')):
    n=json.loads(p.read_text('utf8'));key=(n.get('actual_worker_PID'),n.get('entry'),n.get('start_UTC'))
    if n.get('actual_worker_native_exit')not in [None,0]and key not in seen_attempts:
        failed_attempts.append(n);seen_attempts.add(key)
all_attempts=native+failed_attempts
science=sum(n['elapsed_s']for n in all_attempts);stepgate=gate['models'][ARM]
views=['full_historical_shape','normalized_u480','full_voltage','Q_PSD_2D','baseline_local']
lines=['# R63：全窗坐标与基线时序联合分布覆盖','',
    '原 A/B/C：'+' / '.join(k+' '+outcome(A[k]['pass'])for k in ['A','B','C'])+'。本轮固定末点已经完成训练、数值检查、评价和独立重放；完整模型目标仍需同一方法满足原 A–F。','',
    '本轮在冻结 R61 七项损失之上增加全坐标经验 W1 的 RMS 池化，以及31维基线RMS/lag1–20相关性/10频带的边缘与切片联合分布损失，两项系数均1。全部6593源和21047 REAL父事件、网络、初始化、六条原随机流、曝光次数、原标准化、噪声及电荷/ADC构造保持。新增TRAIN时序常数在首个更新前按原REAL区域固定；投影方向使用独立第七条随机流。唯一50000步末点，不按评价选参。','',
    '| 共同 ADC 表示下的视图 | SOURCE BA [95% CI] | 同处理 REAL BA [95% CI] |','|---|---:|---:|']
for v in views:
    cells=[]
    for n in [ARM,CTRL]:
        row=M['C2ST']['results'][n][v];ci=row['BA_95CI'];cells.append(f"{row['balanced_accuracy']:.3f} [{ci[0]:.6f}, {ci[1]:.6f}]")
    lines.append('| '+v+' | '+' | '.join(cells)+' |')
lines+=['','A 要求全部十个区间完整位于 [0.40,0.60]，使用未舍入数值判定。RAW REAL–REAL、BASE、R49 和 R61 比较复用已冻结结果；本轮没有重新拟合历史对照。','',
    '| 相对冻结 R61 的配对 BA 差 | SOURCE ΔBA [95% CI] | 同处理 REAL ΔBA [95% CI] |','|---|---:|---:|']
for v in views:
    cells=[]
    for n in [ARM,CTRL]:
        row=M['paired_R63_minus_frozen_R61'][n][v];ci=row['BA_delta_95CI'];cells.append(f"{row['BA_delta_first_minus_second']:+.3f} [{ci[0]:+.6f}, {ci[1]:+.6f}]")
    lines.append('| '+v+' | '+' | '.join(cells)+' |')
lines+=['','配对区间使用相同留出身份、组和 500 次 bootstrap，条件于两个冻结分类器。一致性审计支持同一网络和原随机流下的固定目标扩展比较，不能区分两个新增项各自的因果贡献；一个训练种子不能估计重复训练波动，历史单次采集也不能证明独立采集收益。','',
    '总体 PSD W1='+value(A['B']['overall_PSD_W1'])+'；固定共同 Q1–7 等权平均 W1='+value(A['B']['common_Q1_7_PSD_W1'])+'。两个原阈值均为 0.010。','',
    '| 成分 | 必需 Q2–7 峰位 MAE | raw SD 平均相对误差 | 必需区密度拟合失败 |','|---|---:|---:|---|']
for k in ['0','1']:
    w=A['B']['raw_width_relative_errors'][k]
    lines.append('| '+('低峰'if k=='0'else'高峰')+' | '+value(A['B']['peak_MAE_required_Q2_7'][k])+' | '+value(w['mean_absolute_relative_rawSD_error'],'.3%')+' | '+str(w['failed_required_bins'])+' |')
lines+=['','全部八区及失败密度拟合保留。原始源 purehad Q3/Q4 分母为 213/181；旧 postADC 定义的 207/182 另列，不混用。密度成分不是粒子真值。','',
    '| 原阶段门 | 结果 |','|---|---|']
for k,v in stepgate.items():
    if k.startswith('gate'):lines.append('| '+k+' | '+outcome(v['pass'])+' |')
lines+=['','| 原始源 purehad 区 | N | 输出谷比例 | BASE 谷比例 | 增量 | 结果 |','|---|---:|---:|---:|---:|---|']
for x in stepgate['gate3_source_preservation']['actual_source_pure_had_valleys']:
    lines.append('| '+x['actual_source_bin']+' | '+str(x['N'])+' | '+value(x['candidate'])+' | '+value(x['BASE'])+' | '+value(x['delta'])+' | '+outcome(x['pass'])+' |')
lines+=['','| 全人口源关联 | 输出 Spearman | BASE Spearman | 结果 |','|---|---:|---:|---|']
for x in stepgate['gate3_source_preservation']['whole_population_source_ranks']:
    lines.append('| '+x['measure']+' | '+value(x['candidate'])+' | '+value(x['BASE'])+' | '+outcome(x['pass'])+' |')
lines+=['','| 噪声量 | 候选 W1 | BASE W1 | 相对变化 |','|---|---:|---:|---:|']
for k in ['baseline_RMS_V_overall_W1','ACF1_overall_W1']:
    x=stepgate['gate2_PSD_width_centers_noise']['measures'][k]
    lines.append('| '+k+' | '+value(x['candidate'])+' | '+value(x['BASE'])+' | '+value(x['relative_change'],'+.3%')+' |')
lines+=['','模型内包含预登记的电压范围与原电荷联合欧氏投影。下表明确报告其影响；后续 ADC 的零裁剪不能说明内部没有修改波形。','',
    '| 人口 | 触发范围投影事件 / N | 投影修正 RMS 均值 (V) | 最大单点修正 (V) | 电压上下限采样点总数 |','|---|---:|---:|---:|---:|']
for n,label in [(ARM,'SOURCE'),(CTRL,'同处理 REAL')]:
    x=M['explicit_constraint_projection'][n];f=x['fields']
    lines.append('| '+label+' | '+str(x['range_active_events'])+' / '+str(x['N'])+' | '+value(f['constraint_projection_RMS_V']['mean'])+' | '+value(f['constraint_projection_maxabs_V']['quantiles'][-1])+' | '+str(x['rail_sample_total'])+' |')
lines+=['','支持内外及原电荷区域的投影统计见 `CONSTRAINT_PROJECTION_SUMMARY.json`。TRAIN 的逐更新记录是按原区域质量加权的小批量估计；不是冻结模型对全 TRAIN 的再次推理。固定 TRAIN 数值探针也保留每事件投影记录。','']
precision=audit['coverage_extension']
lines+=['| 冻结设计一致性核验 | 结果 |','|---|---|']
for k,v in precision['checks'].items():lines.append('| '+k+' | '+outcome(v)+' |')
lines+=['','原排序罚 = 保留可分辨罚 + 省略重叠罚，全部50000步分解最大绝对残差 '+value(precision['original_minus_resolved_minus_omitted_maxabs'])+'。下表为逐更新小批量记录均值，不是冻结模型全TRAIN评价。','',
    '| 排序训练记录 | 逐更新均值 |','|---|---:|']
for k,v in precision['precision_trace_summary'].items():lines.append('| '+k+' | '+value(v['mean'])+' |')
lines+=['','R60 三个预设冲突标记均未触发的结论保持；R62 两个描述标记为另行预登记的覆盖扩展提供依据，不能据此声称识别唯一原因。','']
num=local('TRAIN_SOLVER_CHECK.json')['models'][ARM]
lines+=['| 新增覆盖训练记录 | 逐更新均值 |','|---|---:|']
for k,v in precision['coverage_trace_summary'].items():lines.append('| '+k+' | '+value(v['mean'])+' |')
lines+=['','首个本机人工检查因 sqrt(1e-20) 与 Python 常量相差 1.29247e-26 V 而触发精确相等断言；修正为 float64 epsilon 范围比较后，CPU/CUDA 全部人工检查通过。损失、系数、数据、原A–F未变；原失败和源文件已保存。科学 worker 失败尝试数量：'+str(len(failed_attempts))+'，全部纳入资源回执。','']
gen=local('GENERATION_RECEIPT.json')
lines += [
    '网络位移只依赖当前输入、噪声和电荷条件，作为时间/状态常数场执行原 Heun128；缓存一次神经预测不代表省略数值检查。TRAIN 128/256 差 median='+value(num['RMS_median_LSB'])+' LSB，p95='+value(num['RMS_p95_LSB'])+' LSB；preADC 最大相对电荷误差='+value(num['preADC_relative_q_max'])+'，原阈值仍为 0.25 / 1 LSB 和 1e-5。','',
    '主输出和探针合计 5984 条；实际网络行数 '+str(sum(gen['actual_network_rows_by_arm'].values()))+'，数学场求值行数 '+str(sum(gen['field_forward_rows_by_arm'].values()))+'。新 HGB 10 次、密度描述槽 16 个，4000 条独立进程重放通过可移动入口逐位一致。','',
    '训练回执核验：'+outcome(audit['pass'])+'；源/REAL参考/REAL保持实际唯一父事件 '+str(train['actual_unique_SOURCE_REAL_reference_REAL_identity'])+'，消费行数分别为 16000000 / 16000000 / 3200000。它们是重复曝光，不能当作独立新事件。原六条随机流与R61完全匹配；独立重构第七条流的全部消费字节和末状态。22列损失、9列新增覆盖分解、精度排序分解和父事件曝光次数完整保存。','',
    '六个成功阶段均原生退出 0；计入已保留失败尝试，共 '+str(len(all_attempts))+' 个科学 worker 的实际耗时合计 '+value(science,'.6f')+' 秒；worker 峰值 RSS '+str(max(n['actual_worker_RSS_peak_bytes']for n in all_attempts))+' B。模型训练 GPU allocated / reserved 峰 '+str(train['GPU_allocated_peak_bytes'])+' / '+str(train['GPU_reserved_peak_bytes'])+' B。无计算时间配额；机器安全限制保留。','',
    '本轮训练没有解码验证波形，训练结束后按预登记方案读取历史 DEV；最终 EVAL 和新 CAL 波形未解码。单次历史采集与固定评价器的通过不等于独立采集或物理逐事件对应关系得到证明。','',
    direction['next_action'],'',
    '[协议](tasks/'+ID+'.md) · [验收](work/'+ID+'/ACCEPTANCE_A_C.json) · [指标](work/'+ID+'/METRICS_COMPACT.json) · [R61 比较](work/'+ID+'/PAIRED_R63_MINUS_R61.json) · [复现](work/'+ID+'/REPRODUCTION.md) · [数值](work/'+ID+'/TRAIN_SOLVER_CHECK.json) · [投影](work/'+ID+'/CONSTRAINT_PROJECTION_SUMMARY.json) · [重放](work/'+ID+'/REPLAY_RECEIPT.json)','']
figures=['training_and_checks.png','canonical_five_view_response.png','PSD_all_eight_bins.png','PSD_centers_widths_and_W1.png',
    'noise_and_source_association.png','complete_waveform_population.png','fixed_time_boundaries.png','explicit_constraint_projection.png','precision_order_decomposition.png','coverage_loss_decomposition.png']
for f in figures:lines += ['!['+f+'](work/'+ID+'/figures/'+f+')','']
(R/'R63_REPORT.md').write_text('\n'.join(lines),'utf8')
doc=S/'REPRODUCTION.md';text=doc.read_text('utf8');text=text.replace('本文件编写时训练仍在运行，是否完成及科学门是否通过，以之后的原生退出、', '训练与全部预登记评价、独立重放已完成；具体结果以原生退出、')
text=text.replace('仅训练完成后存在','唯一第 50000 步末点');doc.write_text(text,'utf8')
resource={'UTC':NOW,'stage':ID,'science_s':science,'native_entries':entries,'native_receipts':native,
    'failed_native_attempts_preserved':failed_attempts,'scientific_worker_attempts_total':len(all_attempts),
    'local_artificial_assertion_repair':'reviews/R63_ARTIFICIAL_ASSERTION_REPAIR_001.json',
    'response_training_runs_started':1,'new_response_updates':50000,'generation_rows':5984,'new_HGB_fits':10,
    'density_description_slots':16,'density_EM_attempts':M['counters']['density_EM_attempts'],'independent_replay_rows':4000,
    'DEV_read':True,'EVAL_read':0,'CAL_waveform_decode':0,'compute_budget_enforced':False,
    'worker_peak_RSS_bytes':max(n['actual_worker_RSS_peak_bytes']for n in all_attempts),
    'sampled_tree_peak_RSS_bytes':max(n['job_tree_RSS_peak_bytes']for n in all_attempts),
    'memory_probe_failures':sum(n['memory_probe_failures']for n in all_attempts),
    'GPU_query_failure_count':sum(len(n.get('GPU_query_failures',[]))for n in all_attempts)}
write('work/'+ID+'/RESOURCE_RECEIPT.json',resource)
required=['config.json','config_post.json','PROTOCOL.md','TRAINING_FROZEN_AUDIT.json','FROZEN_ARTIFACT_REVIEW.json','TRAIN_SOLVER_CHECK.json',
    'POSTPROCESS_FILE_BINDING.json','TRAIN_TEMPORAL_CONSTANTS.npz','TRAIN_TEMPORAL_CONSTANTS.json',
    'METRICS_COMPACT.json','ACCEPTANCE_A_C.json','GATE_METRICS.json','PAIRED_R63_MINUS_R61.json','CONSTRAINT_PROJECTION_SUMMARY.json',
    'REPLAY_RECEIPT.json','FIGURE_QA.json','REPRODUCTION.md','RESOURCE_RECEIPT.json','infer_saved.py',
    'UNIT_CHARGE_CONSTANTS.npz','NOISE_CALIBRATION.npz','SOURCE_ORDER_PRECISION_DEFINITION.json','SOURCE_PSD_PRECISION_PRIVATE.npz','coverage_audit.py',ARM+'/COVERAGE_BY_UPDATE_PRIVATE.npz',ARM+'/PRECISION_ORDER_BY_UPDATE_PRIVATE.npz',ARM+'/FINAL_FLOW.pt',ARM+'/TRAINING_RECEIPT.json']
write('work/'+ID+'/FINAL_RECEIPT.json',{'UTC':NOW,'stage':ID,'status':'scientific_endpoint_complete_delivery_pending',
    'model_goal_complete':False,'scientific_decision':gate['decision'],'A_C_pass':gate['primary_A_C_pass'],
    'required_files':[bind('work/'+ID+'/'+n)for n in required],'resource_receipt':'RESOURCE_RECEIPT.json',
    'actual_independent_replay_pass':replay['pass'],'private_Git_and_D_delivery':'pending separate receipts'})
b=read('RESOURCE_BUDGET.json');block=b['improvement_block'];assert not any(q['stage']==ID for q in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({'stage':ID,'receipt':'work/'+ID+'/RESOURCE_RECEIPT.json','science_s':science,
    'response_training_runs_started':1,'new_response_updates':50000,'status':'fixed_recipe_science_complete','stage_science_complete':True})
block['actual_science_seconds_recorded']+=science
next(q for q in block['registered_stages']if q['stage']==ID).update(status='fixed_recipe_science_complete',active_scientific_jobs=0,
    actual_science_seconds=science,new_response_updates=50000,stage_science_complete=True)
block['response_training_runs_started']=sum(q.get('response_training_runs_started',0)for q in block['actual_stage_receipts'])
write('RESOURCE_BUDGET.json',b)
job={'stage':ID,'status':'science_complete_delivery_pending','active_scientific_jobs':0,
    'actual_worker_PID':native[-1]['actual_worker_PID'],'supervisor_PID':native[-1]['supervisor_PID'],'actual_native_exit':0}
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    x=read(name);x.update(current_job=job,current_stage_progress='science_complete_delivery_pending',application_goal_status='active',
        active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,next_action=direction['next_action'],
        latest_completed_stage=ID,model_goal_complete=False,latest_completed_report='R63_REPORT.md',latest_scientific_decision=gate['decision'],
        latest_key_result_review='reviews/R63_ROOT_DIRECTION_001.json',status='R63_science_complete_delivery_pending_goal_active')
    if name=='PROGRAM.json':x['updated_at_utc']=NOW
    else:x.update(project_status=x['status'],current_activity='root_R63_delivery',current_direction=direction['next_action'],updated_UTC=NOW)
    if name=='PROJECT_PROGRESS.json':
        evidence=['work/'+ID+'/ACCEPTANCE_A_C.json','work/'+ID+'/GATE_METRICS.json','R63_REPORT.md']
        updates={
            'target_distribution_learning':('R63_A_pass' if A['A']['pass'] else 'R63_A_failed_original_thresholds_retained',evidence),
            'G4_conditioned_full_response':('R63_A_C_pass' if gate['primary_A_C_pass'] else 'R63_joint_A_C_not_passed',evidence),
            'charge_conditional_width_noise_waveform':('R63_B_pass' if A['B']['pass'] else 'R63_B_failed_original_thresholds_retained',evidence),
            'task_utility':('not_evaluated_for_frozen_R63',['PROJECT_ACCEPTANCE.json','reviews/FROZEN_A_F_VALIDATION_PREPARATION_001.md']),
            'charge_gap_generalization':('not_evaluated_for_frozen_R63',['PROJECT_ACCEPTANCE.json','reviews/FROZEN_A_F_VALIDATION_PREPARATION_001.md']),
            'reproducible_final_model':('R63_inference_and_replay_verified_full_A_F_unproven',['work/'+ID+'/FROZEN_ARTIFACT_REVIEW.json','work/'+ID+'/REPLAY_RECEIPT.json']),
            'final_research_delivery':('R63_scientific_endpoint_complete_private_Git_D_pending',['work/'+ID+'/FINAL_RECEIPT.json'])}
        for row in x.get('requirements',[]):
            if row['id']not in updates:continue
            row.setdefault('prior_status_records',[]).append({'status':row['status'],'latest_evidence':row.get('latest_evidence',[])})
            row['status'],row['latest_evidence']=updates[row['id']]
            row['candidate_stage']=ID
        x['full_A_F_proven']=False
    write(name,x)
for name,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    x=read(name);q=next(q for q in x[key]if q[ident]==ID);q.update(status='science_complete_delivery_pending',scientific_decision=gate['decision'],
        science_s=science,native_exit=0,final_receipt='work/'+ID+'/FINAL_RECEIPT.json',model_goal_complete=False,active_scientific_jobs=0,new_training_runs_completed=1);write(name,x)
top='# 当前执行：R63 全窗与基线联合覆盖科学终点完成，长期目标 active\n\nR63 已完成唯一 50000 步完整波形覆盖扩展训练、原数值门、完整响应/源/噪声/边界评价和独立重放。A/B/C：'+' / '.join(outcome(A[k]['pass'])for k in ['A','B','C'])+'。明确报告内部约束投影；原 A–F 不变。私有 Git 与 D 盘交付待完成。'+direction['next_action']+' 主助手统筹并按需使用多智能体；全部执行和规划子智能体须与主助手当前模型及思考强度完全一致，继承配置且不降级覆盖。等待期间主动调研、分析既有证据和准备后续方法；无计算配额，单科学作业、共享资源和本机 E/D 存储限制保留。\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
    p=R/name;old=p.read_text('utf8');assert '## 以下为历史记录'in old;p.write_text(top+'## 以下为历史记录'+old.split('## 以下为历史记录',1)[1],'utf8')
print(json.dumps({'science_s':science,'A_C_pass':gate['primary_A_C_pass'],'model_goal_complete':False,'decision':gate['decision']},indent=2))
