from pathlib import Path
from datetime import datetime,timezone
import json,shutil,hashlib,os
R=Path(__file__).resolve().parents[1];STAGE='R52_JOINT_CHARGE_PSD_COUPLING_001';S=R/'work'/STAGE;REG=R/'reviews/R52_REGISTRATION_001';NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,x):(R/p).write_bytes((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert not S.exists() and not REG.exists() and read('PROGRAM.json')['status']=='R51_delivered_long_term_goal_active'
S.mkdir();REG.mkdir()
for name in ['AGENTS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','PROJECT_OBJECTIVE.md','RESOURCE_BUDGET.json','jobs.json','experiments.json']:shutil.copy2(R/name,REG/name)
old=read('work/R51_CANONICAL_SOURCE_TEACHER_DIAGNOSTIC_001/config.json')
cfg={k:old[k] for k in ['server_python','CPU_threads','RSS_limit_bytes','GPU_limit_bytes','MemAvailable_min_bytes','science_budget','deadline_UTC','output_limit_bytes','compute_budget_enforced','TRAIN_path','original_stream','source_local_TRAIN','source_local_stream']}
cfg.update(stage=STAGE,owner='/root',subagents_allowed=False,registered_UTC=NOW,
 remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R52_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'),
 protocol='tasks/'+STAGE+'.md',allowed_scientific_entries=['numeric_worker.py'],charge_quantile_groups=16,
 blocks={'full_cycle':94400,'final_partial_cycle':89600},complete_cycle_repeats=16,
 PSD_parent_SD_ratio_max=.5,required_PSD_direction_hits_bins2_7=4,min_repeated_parents=20,
 extra_RMS_W1_tolerance_V=.1/16383,response_training_runs=0,HGB_fits=0,density_fits=0,DEV_read=0,CAL_decode=0,EVAL_read=0)
write('work/'+STAGE+'/config.json',cfg)
for name in ['linux_supervise_post.py','linux_owned_worker.py','Q_DEFINITION.json']:shutil.copy2(R/'work/R51_CANONICAL_SOURCE_TEACHER_DIAGNOSTIC_001'/name,S/name)
p=S/'linux_supervise_post.py';p.write_bytes(p.read_bytes().replace(b'R51 requires',b'R52 requires'))
shutil.copy2(R/cfg['protocol'],S/'PROTOCOL.md')
write('work/'+STAGE+'/INPUT_BINDING.json',{'UTC':NOW,'TRAIN_path':cfg['TRAIN_path'],'original_stream':cfg['original_stream'],'original_Q_definition':bind(S/'Q_DEFINITION.json'),'R51_diagnostic':'work/R51_CANONICAL_SOURCE_TEACHER_DIAGNOSTIC_001/TEACHER_DIAGNOSTIC.json','DEV_read':0,'CAL_decode':0,'EVAL_read':0})
banner='# 当前执行：主助手独立推进R52联合目标诊断\n\nR51已交付，长期目标active。现按tasks/'+STAGE+'.md检查唯一的16电荷分位内PSD排序构造，分别保留94400完整周期与89600末周期的目标次数。通过预定噪声和父事件离散度检查后才另行登记响应训练。本轮0响应训练，原A–F、无计算预算、单科学作业和本机E/D要求保持，无子智能体。\n\n## 以下为历史记录\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/name).write_bytes((banner+(R/name).read_text('utf8')).encode())
job={'stage':STAGE,'status':'registered_not_launched','active_scientific_jobs':0};action='Execute the one registered joint charge/PSD construction on full and partial TRAIN exposure blocks; assess both fixed noise and parent-PSD criteria before new training.'
for fn in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 x=read(fn);x.update(current_stage=STAGE,current_task=cfg['protocol'],current_stage_scope='TRAIN_joint_charge_PSD_coupling_diagnostic',current_stage_progress='registered_not_launched',current_job=job,next_action=action,current_stage_registration='reviews/R52_REGISTRATION_001/RECEIPT.json',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,subagents_allowed=False,application_goal_status='active')
 x['execution_location']={**x['execution_location'],'remote_root':cfg['remote_root'],'mode':'server_CPU_TRAIN_only_joint_coupling_diagnostic'}
 if fn=='PROGRAM.json':x.update(status='R52_registered_long_term_goal_active',updated_at_utc=NOW)
 else:x.update(project_status='R52_registered_long_term_goal_active',updated_UTC=NOW,current_activity='root_preparing_R52',current_direction=action)
 write(fn,x)
b=read('RESOURCE_BUDGET.json');b['current_registered_stage']=STAGE;b['improvement_block']['registered_stages'].append({'stage':STAGE,'kind':'diagnostic_not_response_method','registered_UTC':NOW,'science_budget':None,'new_training_runs':0,'new_HGB_fits':0,'status':'registered'});write('RESOURCE_BUDGET.json',b)
for fn,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 x=read(fn);x[key].append({ident:STAGE,'status':'registered','owner':'/root','registered_UTC':NOW,'protocol':cfg['protocol'],'output_root':'work/'+STAGE,'remote_root':cfg['remote_root'],'model_candidate':False});write(fn,x)
write('reviews/R52_REGISTRATION_001/RECEIPT.json',{'UTC':NOW,'stage':STAGE,'root_control_PID':os.getpid(),'protocol':bind(R/cfg['protocol']),'config':bind(S/'config.json'),'owner':'/root','application_goal_status':'active','original_A_F_unchanged':True,'scientific_execution_started':False,'response_training_runs':0,'remote_root':cfg['remote_root']})
print(json.dumps({'stage':STAGE,'UTC':NOW,'remote_root':cfg['remote_root'],'new_training':0,'new_HGB':0},indent=2))
