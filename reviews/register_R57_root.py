from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,subprocess,os
R=Path(__file__).resolve().parents[1];ID='R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001';S=R/'work'/ID;REG=R/'reviews/R57_REGISTRATION_001';NOW=datetime.now(timezone.utc).isoformat()
def read(n):return json.loads((R/n).read_text('utf8'))
def write(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert not S.exists() and not REG.exists()
assert read('PROGRAM.json')['status']=='R56_delivered_long_term_goal_active' and read('PROGRAM.json')['active_scientific_jobs']==0
assert read('R56_DELIVERY_COMPLETION_RECEIPT.json')['status']=='delivered'
old=read('work/R56_TRAIN_ENDPOINT_LEARNING_DIAGNOSTIC_001/config.json');p55=read('work/R55_PSD_SOURCE_COMPATIBILITY_001/config.json')
cfg={k:old[k] for k in ['server_python','CPU_threads','RSS_limit_bytes','GPU_limit_bytes','MemAvailable_min_bytes','science_budget','deadline_UTC','output_limit_bytes','compute_budget_enforced']}
inputs={k:old['input_paths'][k] for k in ['TRAIN_path','constants_path','basis']};inputs['valley_definition']=p55['input_paths']['valley_definition']
code='from pathlib import Path\nimport json,hashlib\ninputs='+repr(inputs)+'\nprint(json.dumps({k:{"path":p,"bytes":Path(p).stat().st_size,"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest()}for k,p in inputs.items()}))\n'
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=60)
assert p.returncode==0,p.stderr
binding=json.loads(p.stdout)
assert binding['valley_definition']['sha256']==bind(R/'work/R43_SCALAR_TIME_NEURAL_FLOW_001/VALLEY_DEFINITION.json')['sha256']
S.mkdir();REG.mkdir()
meta=['AGENTS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','PROJECT_OBJECTIVE.md','RESOURCE_BUDGET.json','jobs.json','experiments.json']
for n in meta:shutil.copy2(R/n,REG/n)
cfg.update(stage=ID,owner='/root',subagents_allowed=False,registered_UTC=NOW,remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R57_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'),protocol='tasks/'+ID+'.md',input_paths=inputs,
    allowed_scientific_entries=['numeric_worker.py'],GPU_scientific_entries=[],SOURCE_N=6593,REAL_N=21047,charge_quantile_groups=16,
    Q_relative_max=1e-10,PSD_preservation_max=1e-10,PSD_overall_max=.01,PSD_conditional_mean_max=.01,baseline_W1_max=1e-12,source_valley_delta_max=.02,
    response_training_runs=0,HGB_fits=0,density_fits=0,ODE_generation=0,DEV_read=0,CAL_decode=0,EVAL_read=0)
write('work/'+ID+'/config.json',cfg)
for n in ['noise_core.py','teacher_core.py','canonical.py']:(S/n).write_text((R/'work/R56_TRAIN_ENDPOINT_LEARNING_DIAGNOSTIC_001'/n).read_text('utf8').rstrip()+'\n','utf8')
for n in ['linux_supervise_post.py','linux_owned_worker.py']:
    text=(R/'work/R55_PSD_SOURCE_COMPATIBILITY_001'/n).read_text('utf8').replace('R55 requires','R57 requires').rstrip()+'\n';(S/n).write_text(text,'utf8')
shutil.copy2(R/cfg['protocol'],S/'PROTOCOL.md')
write('work/'+ID+'/INPUT_BINDING.json',{'UTC':NOW,'inputs':binding,'binding_only_no_new_statistics':True,'SSH_native_exit':0,'acceptance':bind(R/'PROJECT_ACCEPTANCE.json'),'R56_report':bind(R/'R56_REPORT.md'),'valley_provenance':'frozen R3B REAL_TRAIN_fit2592 middle third between mixture means; no new fit'})
banner='# 当前执行：R57 原TRAIN父事件等权教师预检\n\nR56已交付，原训练采样倾斜与等权源分布不一致。现登记一次全6593源、十区域/固定16电荷质量分位的等权教师构造；未启动科学进程。0神经训练，保留支持外和失败质量；原A–F与验收源权重不变。主助手独立、无计算配额、单科学作业、本机E/D约束保持。\n\n## 以下为历史记录\n\n'
for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/n).write_text(banner+(R/n).read_text('utf8'),'utf8')
action='Execute one fixed equal-parent TRAIN teacher construction and its source/PSD/ADC/noise checks; close on any failed necessary teacher condition before registering neural training.'
for n in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    x=read(n);x.update(status='R57_registered_long_term_goal_active',current_stage=ID,current_task=cfg['protocol'],current_stage_scope='equal_parent_TRAIN_teacher_diagnostic',current_stage_progress='registered_not_launched',current_job={'stage':ID,'status':'registered_not_launched','active_scientific_jobs':0},next_action=action,current_stage_registration='reviews/R57_REGISTRATION_001/RECEIPT.json',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,subagents_allowed=False,application_goal_status='active',model_goal_complete=False)
    x['execution_location'].update(mode='server_CPU_teacher_diagnostic',remote_root=cfg['remote_root'],current_stage_GPU_used=False,preflight_required=True)
    if n=='PROGRAM.json':x['updated_at_utc']=NOW
    else:x.update(project_status=x['status'],updated_UTC=NOW,current_activity='root_preparing_R57',current_direction=action)
    write(n,x)
b=read('RESOURCE_BUDGET.json');b['current_registered_stage']=ID;b['improvement_block']['registered_stages'].append({'stage':ID,'kind':'equal_parent_teacher_diagnostic_not_response_method','registered_UTC':NOW,'science_budget':None,'new_training_runs':0,'new_HGB_fits':0,'status':'registered'});write('RESOURCE_BUDGET.json',b)
for n,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    x=read(n);assert not any(v[ident]==ID for v in x[key]);x[key].append({ident:ID,'status':'registered','owner':'/root','registered_UTC':NOW,'protocol':cfg['protocol'],'output_root':'work/'+ID,'remote_root':cfg['remote_root'],'model_candidate':False});write(n,x)
write('reviews/R57_REGISTRATION_001/RECEIPT.json',{'UTC':NOW,'stage':ID,'root_control_PID':os.getpid(),'protocol':bind(R/cfg['protocol']),'config':bind(S/'config.json'),'input_binding':bind(S/'INPUT_BINDING.json'),'owner':'/root','application_goal_status':'active','original_A_F_unchanged':True,'scientific_execution_started':False,'response_training_runs':0,'remote_root':cfg['remote_root']})
print(json.dumps({'stage':ID,'UTC':NOW,'remote_root':cfg['remote_root'],'scientific_execution_started':False},indent=2))
