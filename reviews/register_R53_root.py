from pathlib import Path
from datetime import datetime,timezone
import json,shutil,hashlib,os
R=Path(__file__).resolve().parents[1];STAGE='R53_PULSE_ONLY_TEACHER_FLOW_001';S=R/'work'/STAGE;REG=R/'reviews/R53_REGISTRATION_001';NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,x):(R/p).write_bytes((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert not S.exists() and not REG.exists() and read('PROGRAM.json')['status']=='R52_delivered_long_term_goal_active'
S.mkdir();REG.mkdir()
meta=['AGENTS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','PROJECT_OBJECTIVE.md','RESOURCE_BUDGET.json','jobs.json','experiments.json']
for name in meta:shutil.copy2(R/name,REG/name)
old=read('work/R52_JOINT_CHARGE_PSD_COUPLING_001/config.json');r48=read('work/R48_GUIDED_NOISE_COMPLETION_FLOW_001/config.json');r49=read('work/R49_FROZEN_HEUN128_RESPONSE_001/config.json')
cfg={k:old[k] for k in ['server_python','CPU_threads','RSS_limit_bytes','GPU_limit_bytes','MemAvailable_min_bytes','science_budget','deadline_UTC','output_limit_bytes','compute_budget_enforced','TRAIN_path','original_stream','source_local_TRAIN','source_local_stream']}
cfg.update({k:r48[k] for k in ['model_seed','PCG64_seeds','expected_R33_environment','constants_path','train_solver_input']})
remote='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R53_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
cfg.update(stage=STAGE,owner='/root',subagents_allowed=False,registered_UTC=NOW,remote_root=remote,protocol='tasks/'+STAGE+'.md',
 arm='JOINT_PULSE_RESPONSE',arms=['JOINT_PULSE_RESPONSE'],updates=50000,batch=32,lr=.0002,betas=[.9,.999],gradient_clip=5.,loss_weights=[.5,.5],
 allowed_scientific_entries=['train_final.py','numerical_check.py','generate_all.py','evaluate_saved.py','replay_all.py'],
 GPU_scientific_entries=['train_final.py','numerical_check.py','generate_all.py','replay_all.py'],
 full_coupling_path=old['remote_root']+'/full_cycle_COUPLING_PRIVATE.npz',partial_coupling_path=old['remote_root']+'/final_partial_cycle_COUPLING_PRIVATE.npz',
 noise_calibration_path=r48['remote_root']+'/NOISE_CALIBRATION.npz',noise_calibration_json_path=r48['remote_root']+'/NOISE_CALIBRATION.json',
 original_initial_state_path=r48['remote_root']+'/COMMON_INITIAL_FLOW.pt',original_training_receipt=r48['remote_root']+'/GUIDED_NOISE_COMPLETION/TRAINING_RECEIPT.json',
 historical_control_weights=r48['remote_root']+'/GUIDED_NOISE_COMPLETION/FINAL_FLOW.pt',
 frozen_eval_input_root=r49['frozen_eval_input_root'],historical_R49_local='work/R49_LOCAL_HEUN128_RECOVERY_001',canonical_control_results='work/R51_CANONICAL_SOURCE_TEACHER_DIAGNOSTIC_001/CANONICAL_C2ST.json',
 solver_steps=128,solver_check_steps=256,numeric_median_LSB_max=.25,numeric_p95_LSB_max=1.,preADC_charge_relative_error_limit=1e-5,
 teacher_relative_Q_error_limit=1e-10,teacher_PSD_error_limit=1e-10,teacher_baseline_tail_exact=True,
 generation_chunk=64,planned_response_training_runs=1,full_cycle_repeats=16,full_cycle_rows=94400,partial_cycle_rows=89600,
 teacher_definition='fixed_minimum_second_difference_pulse_correction_preserving_target_PSD_and_baseline',training_wall_s=None,absolute_deadline_UTC=None)
write('work/'+STAGE+'/config.json',cfg)
for name in ['network.py','noise_core.py','source_flow_core.py']:shutil.copy2(R/'work/R48_GUIDED_NOISE_COMPLETION_FLOW_001'/name,S/name)
shutil.copy2(R/'work/R52_JOINT_CHARGE_PSD_COUPLING_001/linux_owned_worker.py',S/'linux_owned_worker.py')
shutil.copy2(R/'work/R52_JOINT_CHARGE_PSD_COUPLING_001/Q_DEFINITION.json',S/'Q_DEFINITION.json')
shutil.copy2(R/cfg['protocol'],S/'PROTOCOL.md')
write('work/'+STAGE+'/INPUT_BINDING.json',{'UTC':NOW,'original_TRAIN':cfg['TRAIN_path'],'original_stream':cfg['original_stream'],'fixed_R52_full_coupling':cfg['full_coupling_path'],'fixed_R52_partial_coupling':cfg['partial_coupling_path'],'frozen_R48_noise':cfg['noise_calibration_path'],'frozen_R43_unit_Q_constants':cfg['constants_path'],'source_code_copies':[bind(S/p) for p in ['network.py','noise_core.py','source_flow_core.py']],'R52_negative_rewritten':False,'new_training_before_registration':0})
banner='# 当前执行：R53平滑脉冲端点与单次神经训练\n\nR52固定配对方案的Q1/Q8噪声失败保留。现按tasks/'+STAGE+'.md实施新端点：保留实测基线与末尾，只在脉冲区加唯一平滑电荷/PSD修正。TRAIN恒等式检查后从零固定50000更新，随后原128/256数值检查与完整响应评价。原A–F及源人口不变，主助手独立、无子智能体、无计算预算，服务器CPU12/RSS48GiB/GPU10GiB和单科学作业、本机E/D约束保持。\n\n## 以下为历史记录\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/name).write_bytes((banner+(R/name).read_text('utf8')).encode())
job={'stage':STAGE,'status':'registered_not_launched','active_scientific_jobs':0};action='Validate the fixed pulse-only TRAIN endpoint; if structural checks pass run the sole 50000-update model, then unchanged numerical and complete response validation.'
for fn in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 x=read(fn);x.update(current_stage=STAGE,current_task=cfg['protocol'],current_stage_scope='fixed_pulse_only_teacher_then_single_neural_training',current_stage_progress='registered_not_launched',current_job=job,next_action=action,current_stage_registration='reviews/R53_REGISTRATION_001/RECEIPT.json',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,subagents_allowed=False,application_goal_status='active')
 x['execution_location']={**x['execution_location'],'remote_root':remote,'mode':'server_GPU_single_root_neural_training'}
 if fn=='PROGRAM.json':x.update(status='R53_registered_long_term_goal_active',updated_at_utc=NOW)
 else:x.update(project_status='R53_registered_long_term_goal_active',updated_UTC=NOW,current_activity='root_preparing_R53',current_direction=action)
 write(fn,x)
b=read('RESOURCE_BUDGET.json');b['current_registered_stage']=STAGE;b['improvement_block']['registered_stages'].append({'stage':STAGE,'kind':'new_joint_pulse_endpoint_neural_method','registered_UTC':NOW,'science_budget':None,'planned_new_training_runs':1,'fixed_updates':50000,'status':'registered'});write('RESOURCE_BUDGET.json',b)
for fn,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 x=read(fn);x[key].append({ident:STAGE,'status':'registered','owner':'/root','registered_UTC':NOW,'protocol':cfg['protocol'],'output_root':'work/'+STAGE,'remote_root':remote,'model_candidate':True,'new_training_runs_started':0});write(fn,x)
write('reviews/R53_REGISTRATION_001/RECEIPT.json',{'UTC':NOW,'stage':STAGE,'root_control_PID':os.getpid(),'protocol':bind(R/cfg['protocol']),'config':bind(S/'config.json'),'owner':'/root','application_goal_status':'active','original_A_F_unchanged':True,'scientific_execution_started':False,'response_training_runs_started':0,'remote_root':remote})
print(json.dumps({'stage':STAGE,'UTC':NOW,'remote_root':remote,'planned_training':1,'fixed_updates':50000},indent=2))
