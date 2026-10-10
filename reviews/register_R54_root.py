from pathlib import Path
from datetime import datetime,timezone
import json,shutil,hashlib,os
R=Path(__file__).resolve().parents[1];ID='R54_CONTINUOUS_CHARGE_FLOW_001';S=R/'work'/ID;OLD=R/'work/R53_PULSE_ONLY_TEACHER_FLOW_001';REG=R/'reviews/R54_REGISTRATION_001';NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,x):(R/p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert not S.exists() and not REG.exists()
assert read('PROGRAM.json')['status']=='R53_delivered_long_term_goal_active'
assert read('R53_DELIVERY_COMPLETION_RECEIPT.json')['status']=='delivered'
assert not read('work/R53_PULSE_ONLY_TEACHER_FLOW_001/LIVE_OBSERVATION.json')['actual_process']['exists']
S.mkdir();REG.mkdir()
for n in ['AGENTS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','PROJECT_OBJECTIVE.md','RESOURCE_BUDGET.json','jobs.json','experiments.json']:shutil.copy2(R/n,REG/n)
C=read('work/R53_PULSE_ONLY_TEACHER_FLOW_001/config.json')
oldremote=C['remote_root'];remote=oldremote.split('/R53_ROOT_')[0]+'/R54_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
C.update(stage=ID,registered_UTC=NOW,remote_root=remote,protocol='tasks/'+ID+'.md',arm='CONTINUOUS_Q_RESPONSE',arms=['CONTINUOUS_Q_RESPONSE'],
 continuous_condition='within_bin_logQ_centered_by_frozen_TRAIN_edges_no_clip',continuous_condition_embedding_index=1,
 unchanged_zero_embedding_indices=list(range(2,17)),historical_R53_root=oldremote,
 historical_R53_training_receipt=oldremote+'/JOINT_PULSE_RESPONSE/TRAINING_RECEIPT.json',
 sole_change='activate_one_continuous_logQ_input_in_existing_embedding')
write('work/'+ID+'/config.json',C)
names=['network.py','source_flow_core.py','noise_core.py','teacher_core.py','train_final.py','linux_supervise.py','linux_owned_worker.py','Q_DEFINITION.json','NOISE_CALIBRATION.npz','NOISE_CALIBRATION.json']
for name in names:shutil.copy2(OLD/name,S/name)
shutil.copy2(R/C['protocol'],S/'PROTOCOL.md')
write('work/'+ID+'/INPUT_BINDING.json',{'UTC':NOW,'prior_stage':'R53_PULSE_ONLY_TEACHER_FLOW_001','original_TRAIN':C['TRAIN_path'],'original_stream':C['original_stream'],'full_coupling':C['full_coupling_path'],'partial_coupling':C['partial_coupling_path'],'frozen_constants':C['constants_path'],'frozen_noise':C['noise_calibration_path'],'common_initial_weights':C['original_initial_state_path'],'historical_R53_training_receipt':C['historical_R53_training_receipt'],'copied_before_modification':[bind(S/n) for n in names],'trained_weights_loaded':False})
banner='# 当前执行：R54 连续电荷条件固定消融\n\nR53 已作为负结果交付。本轮按 tasks/'+ID+'.md 只增加由原 TRAIN 电荷边界定义的连续 logQ 条件，保持同一教师、初始权重、数据、随机流与唯一 50000 更新终点。原 A–F、数值精度和 SOURCE 人口不变；主助手独立、无子智能体、无计算配额，服务器资源/单科学作业及本机 E/D 约束保持。\n\n## 以下为历史记录\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/name).write_text(banner+(R/name).read_text('utf8'),'utf8')
action='Run one preregistered continuous-logQ conditioning ablation with unchanged R53 teacher and 50000 final updates; then original numerical, response, source, noise and independent replay gates.'
job={'stage':ID,'status':'registered_not_launched','active_scientific_jobs':0}
for fn in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 x=read(fn);x.update(current_stage=ID,current_task=C['protocol'],current_stage_scope='single_continuous_charge_condition_ablation',current_stage_progress='registered_not_launched',current_job=job,next_action=action,current_stage_registration='reviews/R54_REGISTRATION_001/RECEIPT.json',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,subagents_allowed=False,application_goal_status='active',model_goal_complete=False)
 x['execution_location']={**x['execution_location'],'remote_root':remote,'mode':'server_GPU_single_root_neural_training','preflight_required':True}
 if fn=='PROGRAM.json':x.update(status='R54_registered_long_term_goal_active',updated_at_utc=NOW)
 else:x.update(status='R54_registered_long_term_goal_active',project_status='R54_registered_long_term_goal_active',updated_UTC=NOW,current_activity='root_preparing_R54',current_direction=action)
 write(fn,x)
B=read('RESOURCE_BUDGET.json');B['current_registered_stage']=ID;B['improvement_block']['registered_stages'].append({'stage':ID,'kind':'single_continuous_charge_condition_ablation','registered_UTC':NOW,'science_budget':None,'planned_new_training_runs':1,'fixed_updates':50000,'status':'registered'});write('RESOURCE_BUDGET.json',B)
for fn,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 x=read(fn);x[key].append({ident:ID,'status':'registered','owner':'/root','registered_UTC':NOW,'protocol':C['protocol'],'output_root':'work/'+ID,'remote_root':remote,'model_candidate':True,'new_training_runs_started':0});write(fn,x)
write('reviews/R54_REGISTRATION_001/RECEIPT.json',{'UTC':NOW,'stage':ID,'root_control_PID':os.getpid(),'protocol':bind(R/C['protocol']),'config':bind(S/'config.json'),'owner':'/root','application_goal_status':'active','original_A_F_unchanged':True,'scientific_execution_started':False,'response_training_runs_started':0,'remote_root':remote})
print(json.dumps({'stage':ID,'UTC':NOW,'remote_root':remote,'planned_training':1,'fixed_updates':50000},indent=2))
