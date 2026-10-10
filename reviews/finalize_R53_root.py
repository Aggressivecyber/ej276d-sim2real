from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path(__file__).resolve().parents[1];ID='R53_PULSE_ONLY_TEACHER_FLOW_001';S=R/'work'/ID;NOW=datetime.now(timezone.utc).isoformat()
def read(path):return json.loads((R/path).read_text('utf8'))
def write(path,value):(R/path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def local(name):return read('work/'+ID+'/'+name)
def bind(path):
 p=R/path
 return {'path':str(p.relative_to(R)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
M=local('METRICS_COMPACT.json');gate=local('GATE_METRICS.json');arm='JOINT_PULSE_RESPONSE';ctrl='JOINT_PULSE_REAL_CONTROL'
A=M['acceptance_A_C'][arm];teacher=local('TEACHER_VALIDATION.json');train=local(arm+'/TRAINING_RECEIPT.json');replay=local('REPLAY_RECEIPT.json');qa=local('FIGURE_QA.json')
direction=read('reviews/R53_ROOT_DIRECTION_001.json')
assert direction['scientific_decision']==gate['decision'] and direction['model_goal_complete'] is False
assert qa['all_figures_opened'] and qa['layout_pass'] and teacher['passed'] and train['updates']==50000
entries=['train_final.py','numerical_check.py','generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py','render_figures_only.py','render_final_figures.py','render_legend_only.py']
native=[local(n+'.NATIVE_RECEIPT.json')for n in entries]
assert all(n['termination_reason']is None and n['handshake_before_imports']for n in native)
assert all(n['actual_worker_native_exit']==0 and n['worker_self_receipt']['native_exit']==0 for n in native[:5]+native[7:])
assert native[5]['actual_worker_native_exit']==1 and native[5]['worker_self_receipt']['error']=="KeyError('histogram_counts')"
assert native[6]['actual_worker_native_exit']==1 and 'TypeError' in native[6]['worker_self_receipt']['error']
assert local('PLOT_RECOVERY_RECEIPT.json')['status']=='complete'
assert replay['rows']==4000 and replay['portable_loader_executed']
live=local('LIVE_OBSERVATION.json');assert not live['actual_process']['exists']
science=sum(n['elapsed_s']for n in native);stepgate=gate['models'][arm]
lines=['# R53：固定平滑脉冲教师与完整响应评估','',
 '本轮只训练一个固定 50000 步模型。教师端采用已登记的电荷/PSD 耦合，并以唯一的脉冲区平滑校正保留实测基线与尾段；训练的 SOURCE、REAL 身份支路和随机暴露流与 R48 对照一致。组合改动的结果不能分别归因给两个组件。','',
 '本轮 A/B/C：'+' / '.join(k+' '+('通过'if A[k]['pass']else'未通过')for k in ['A','B','C'])+'。同一冻结方法尚未完成原 A–F，长期目标继续保持 active。','',
 '| 同一 ADC 表示下的视图 | SOURCE BA [95% CI] | 同处理 REAL BA [95% CI] |','|---|---:|---:|']
for view in ['full_historical_shape','normalized_u480','full_voltage','Q_PSD_2D','baseline_local']:
 vals=[]
 for name in [arm,ctrl]:
  row=M['C2ST']['results'][name][view];ci=row['BA_95CI'];vals.append(f"{row['balanced_accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
 lines.append('| '+view+' | '+' | '.join(vals)+' |')
lines+=['','判据仍为每个区间完整落在 [0.40, 0.60]。BASE、RAW REAL–REAL 与 R49 对照复用 R51 的固定判别器结果；新方法在服务器 GPU 生成，旧 R49 控制结果来自既有运行，不宣称新增跨平台位级验证。','',
 f"总体 PSD W1={A['B']['overall_PSD_W1']:.8f}；原固定 Q1–7 等权平均 W1={A['B']['common_Q1_7_PSD_W1']:.8f}，两项门槛均为 0.010。",'',
 '| 分量 | 必需 Q2–7 峰位 MAE | raw SD 平均相对误差 | 失败必需区 |','|---|---:|---:|---|']
for k in ['0','1']:
 w=A['B']['raw_width_relative_errors'][k];center=A['B']['peak_MAE_required_Q2_7'][k]
 lines.append('| '+('低峰'if k=='0'else'高峰')+' | '+(f'{center:.8f}'if center is not None else'未定义')+' | '+(f"{w['mean_absolute_relative_rawSD_error']:.3%}"if w['mean_absolute_relative_rawSD_error']is not None else'未定义')+' | '+str(w['failed_required_bins'])+' |')
lines+=['','全部八个 Q 区及所有失败拟合均保留；密度分量不是粒子真值。原始 SOURCE 分箱继续采用未量化输入，purehad Q3/Q4 分别保留 213/181 个事件。旧 postADC 参考分箱的 207/182 属于另一种描述，未用来替代主分母。','',
 '| 原六项阶段门 | 结果 |','|---|---|']
for key,row in stepgate.items():
 if key.startswith('gate'):lines.append('| '+key+' | '+('通过'if row['pass']else'未通过')+' |')
num=local('TRAIN_SOLVER_CHECK.json')['models'][arm]
lines+=['',f"TRAIN 128/256 数值差异：median={num['RMS_median_LSB']:.6f} LSB，p95={num['RMS_p95_LSB']:.6f} LSB，电荷相对误差最大值={num['preADC_relative_q_max']:.3e}。阈值仍为 0.25 / 1 LSB 和 1e-5。",'',
 f"独立 GPU 回放实际处理 {replay['rows']} 条，位级一致结果为 {replay['pass']}；回放通过可移动推理入口加载包内权重与常量。主样本及探针合计 5984 条，新 HGB 10 次，密度描述 16 个分区。",'',
 f"训练、数值检查、生成、评估和回放均原生退出 0。汇总指标写出后的两次绘图分别因旧对照缺失直方图及分箱边界为空退出 1；只从已保存特征重绘后的展示作业退出 0，两次失败和原图均保留。累计科学/必要汇总耗时 {science:.6f} 秒，最高 worker RSS {max(n['actual_worker_RSS_peak_bytes']for n in native)} B。RSS 查询缺失共 {sum(n['memory_probe_failures']for n in native)} 次，逐次时间和未知原因保留在原生回执，未触发终止。本轮响应训练 1 次、50000 步；DEV 历史暴露继续披露，未读取 EVAL 或重新解码 CAL。",'',
 direction['next_action'],'',
 '[原协议](tasks/'+ID+'.md) · [完整判据](work/'+ID+'/ACCEPTANCE_A_C.json) · [指标](work/'+ID+'/METRICS_COMPACT.json) · [推理与复现](work/'+ID+'/REPRODUCTION.md) · [训练回执](work/'+ID+'/'+arm+'/TRAINING_RECEIPT.json) · [独立回放](work/'+ID+'/REPLAY_RECEIPT.json) · [资源回执](work/'+ID+'/RESOURCE_RECEIPT.json)','']
for name in ['canonical_five_view_response.png','PSD_all_eight_bins.png','PSD_centers_widths_and_W1.png','noise_and_source_association.png','complete_waveform_population.png','fixed_time_boundaries.png','training_and_checks.png']:
 lines+=['!['+name+'](work/'+ID+'/figures/'+name+')','']
(R/'R53_REPORT.md').write_text('\n'.join(lines),'utf8')
resource={'UTC':NOW,'stage':ID,'science_s':science,'native_entries':entries,'native_receipts':native,
 'response_training_runs_started':1,'new_response_updates':50000,'generation_rows':5984,'new_HGB_fits':10,
 'density_description_slots':16,'density_EM_attempts':M['counters']['density_EM_attempts'],'independent_replay_rows':4000,
 'DEV_read':True,'EVAL_read':0,'CAL_waveform_decode':0,'compute_budget_enforced':False,
 'worker_peak_RSS_bytes':max(n['actual_worker_RSS_peak_bytes']for n in native),
 'sampled_tree_peak_RSS_bytes':max(n['job_tree_RSS_peak_bytes']for n in native),
 'memory_probe_failures':sum(n['memory_probe_failures']for n in native),
 'GPU_query_failure_count':sum(len(n.get('GPU_query_failures',[]))for n in native),
 'display_failures_preserved':[{k:n[k]for k in ['entry','actual_worker_PID','actual_worker_native_exit','elapsed_s','worker_self_receipt']}for n in native[5:7]],
 'display_repair_new_fits_or_neural_inference':0}
write('work/'+ID+'/RESOURCE_RECEIPT.json',resource)
required=['config.json','config_post.json','PROTOCOL.md','TEACHER_VALIDATION.json','TRAIN_SOLVER_CHECK.json',
 'METRICS_COMPACT.json','ACCEPTANCE_A_C.json','GATE_METRICS.json','REPLAY_RECEIPT.json','FIGURE_QA.json',
 'REPRODUCTION.md','RESOURCE_RECEIPT.json','infer_saved.py','UNIT_CHARGE_CONSTANTS.npz','NOISE_CALIBRATION.npz',arm+'/FINAL_FLOW.pt',arm+'/TRAINING_RECEIPT.json']
write('work/'+ID+'/FINAL_RECEIPT.json',{'UTC':NOW,'stage':ID,'status':'scientific_endpoint_complete_delivery_pending',
 'model_goal_complete':False,'scientific_decision':gate['decision'],'A_C_pass':gate['primary_A_C_pass'],
 'required_files':[bind('work/'+ID+'/'+n)for n in required],'resource_receipt':'RESOURCE_RECEIPT.json',
 'actual_independent_replay_pass':replay['pass'],'private_Git_and_D_delivery':'pending separate receipts'})
b=read('RESOURCE_BUDGET.json');block=b['improvement_block'];assert not any(q['stage']==ID for q in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({'stage':ID,'receipt':'work/'+ID+'/RESOURCE_RECEIPT.json','science_s':science,
 'response_training_runs_started':1,'new_response_updates':50000,'status':'fixed_recipe_science_complete','stage_science_complete':True})
block['actual_science_seconds_recorded']+=science
next(q for q in block['registered_stages']if q['stage']==ID).update(status='fixed_recipe_science_complete',active_scientific_jobs=0,actual_science_seconds= science,new_response_updates=50000,stage_science_complete=True)
block['response_training_runs_started']=sum(q.get('response_training_runs_started',0)for q in block['actual_stage_receipts']);write('RESOURCE_BUDGET.json',b)
job={'stage':ID,'status':'science_complete_delivery_pending','active_scientific_jobs':0,
     'actual_worker_PID':native[-1]['actual_worker_PID'],'supervisor_PID':native[-1]['supervisor_PID'],'actual_native_exit':0}
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 x=read(name);x.update(current_job=job,current_stage_progress='science_complete_delivery_pending',application_goal_status='active',
  active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,next_action=direction['next_action'],
  latest_completed_stage=ID,model_goal_complete=False,latest_completed_report='R53_REPORT.md',latest_scientific_decision=gate['decision'],
  latest_key_result_review='reviews/R53_ROOT_DIRECTION_001.json',status='R53_science_complete_delivery_pending_goal_active')
 if name=='PROGRAM.json':x['updated_at_utc']=NOW
 else:x.update(project_status=x['status'],current_activity='root_R53_delivery',current_direction=direction['next_action'],updated_UTC=NOW)
 write(name,x)
for name,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 x=read(name);q=next(q for q in x[key]if q[ident]==ID);q.update(status='science_complete_delivery_pending',scientific_decision=gate['decision'],
  science_s=science,native_exit=0,final_receipt='work/'+ID+'/FINAL_RECEIPT.json',model_goal_complete=False,active_scientific_jobs=0,new_training_runs_completed=1)
 write(name,x)
top='# 当前执行：R53 固定方案评估结束，长期目标 active\n\nR53 已按原阈值完成教师检查、50000 步训练、数值检查、完整响应/源/噪声/边界评价及独立回放。A/B/C：'+ ' / '.join('通过'if A[k]['pass']else'未通过'for k in ['A','B','C'])+'。原 A–F 不变；详细结果见 R53_REPORT.md。当前完成私有 Git 与 D 盘交付。'+direction['next_action']+' 主助手独立执行，不使用子智能体；无计算配额，CPU12/RSS48GiB/GPU10GiB、单科学作业及本机 E/D 存储限制保留。\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
 p=R/name;old=p.read_text('utf8');assert '## 以下为历史记录' in old
 tail=old.split('## 以下为历史记录',1)[1];p.write_text(top+'## 以下为历史记录'+tail,'utf8')
print(json.dumps({'science_s':science,'cumulative_science_s':block['actual_science_seconds_recorded'],'A_C_pass':gate['primary_A_C_pass'],'model_goal_complete':False,'decision':gate['decision']},indent=2))
