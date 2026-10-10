from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil, subprocess, os

R=Path(__file__).resolve().parents[1]
ID='R56_TRAIN_ENDPOINT_LEARNING_DIAGNOSTIC_001'
S=R/'work'/ID
REG=R/'reviews/R56_REGISTRATION_001'
NOW=datetime.now(timezone.utc).isoformat()
def read(n):return json.loads((R/n).read_text('utf8'))
def write(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert not S.exists() and not REG.exists()
assert read('PROGRAM.json')['status']=='R55_delivered_long_term_goal_active'
assert read('PROGRAM.json')['active_scientific_jobs']==0
assert read('R55_DELIVERY_COMPLETION_RECEIPT.json')['status']=='delivered'
old=read('work/R54_CONTINUOUS_CHARGE_FLOW_001/config.json')
keys=['server_python','CPU_threads','RSS_limit_bytes','GPU_limit_bytes','MemAvailable_min_bytes','science_budget','deadline_UTC','output_limit_bytes','compute_budget_enforced']
cfg={k:old[k] for k in keys}
inputs={k:old[k] for k in ['TRAIN_path','original_stream','full_coupling_path','partial_coupling_path','constants_path','noise_calibration_path']}
inputs.update(weights=old['remote_root']+'/CONTINUOUS_Q_RESPONSE/FINAL_FLOW.pt',basis=old['remote_root']+'/PULSE_BASIS.npz',
    original_training_receipt=old['remote_root']+'/CONTINUOUS_Q_RESPONSE/TRAINING_RECEIPT.json',
    random_initial=old['remote_root']+'/CONTINUOUS_Q_RESPONSE/INITIAL_RANDOM_STATES.json',
    teacher_validation=old['remote_root']+'/TEACHER_VALIDATION.json')
code='from pathlib import Path\nimport json,hashlib\ninputs='+repr(inputs)+'\n'
code+='print(json.dumps({k:{"path":p,"bytes":Path(p).stat().st_size,"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest()}for k,p in inputs.items()}))\n'
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=60)
assert p.returncode==0,p.stderr
bound=json.loads(p.stdout)
assert bound['weights']['sha256']==bind(R/'work/R54_CONTINUOUS_CHARGE_FLOW_001/CONTINUOUS_Q_RESPONSE/FINAL_FLOW.pt')['sha256']
S.mkdir();REG.mkdir()
meta=['AGENTS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','PROJECT_OBJECTIVE.md','RESOURCE_BUDGET.json','jobs.json','experiments.json']
for n in meta:shutil.copy2(R/n,REG/n)
cfg.update(stage=ID,owner='/root',subagents_allowed=False,registered_UTC=NOW,
    remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R56_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'),
    protocol='tasks/'+ID+'.md',input_paths=inputs,PCG64_seeds=old['PCG64_seeds'],expected_environment=old['expected_R33_environment'],
    allowed_scientific_entries=['diagnose_train.py'],GPU_scientific_entries=['diagnose_train.py'],
    full_rows=94400,partial_rows=89600,full_cycle_repeats=16,SOURCE_N=6593,REAL_N=21047,
    path_batch=32,path_batches=2950,endpoint_seed=20261010561,solver_steps=128,generation_chunk=64,
    learning_W1_threshold=.01,learning_residual_ratio=.20,learning_required_bins=4,required_bins_onebased=list(range(2,8)),
    response_training_runs=0,HGB_fits=0,density_fits=0,DEV_read=0,EVAL_read=0,CAL_decode=0)
write('work/'+ID+'/config.json',cfg)
files=['network.py','noise_core.py','source_flow_core.py','teacher_core.py','canonical.py','linux_supervise.py','linux_owned_worker.py']
for name in files:
    text=(R/'work/R54_CONTINUOUS_CHARGE_FLOW_001'/name).read_text('utf8').rstrip()+'\n'
    if name=='linux_supervise.py':text=text.replace('R54 requires','R56 requires')
    (S/name).write_text(text,'utf8')
shutil.copy2(R/cfg['protocol'],S/'PROTOCOL.md')
provenance=[R/'work/R23_ONLINE_STANDARDIZED_CRITIC_001/prepare_data.py',R/'work/R26_CONDITIONAL_BATCH_TRANSLATION_001/reorder_core.py',R/'work/R43_SCALAR_TIME_NEURAL_FLOW_001/train_server.py',R/'work/R54_CONTINUOUS_CHARGE_FLOW_001/train_final.py']
write('work/'+ID+'/INPUT_BINDING.json',{'UTC':NOW,'inputs':bound,'binding_only_no_new_statistics':True,'SSH_native_exit':0,'provenance_code':[bind(p) for p in provenance],'acceptance':bind(R/'PROJECT_ACCEPTANCE.json')})
banner='# 当前执行：R56 TRAIN 教师与冻结学习误差诊断\n\nR55 已交付，未证明目标冲突。现登记固定 R54 的 TRAIN 端点与速度残差诊断，保留全部6593源，核查旧支持内采样覆盖；0更新、0新增响应模型、0 DEV/EVAL读取。主助手独立，无计算配额，原A–F与E/D约束保持。\n\n## 以下为历史记录\n\n'
for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/n).write_text(banner+(R/n).read_text('utf8'),'utf8')
action='Complete the fixed TRAIN-only teacher/population/learning diagnostic before choosing one evidence-backed response mechanism; no R54 extension or DEV selection.'
job={'stage':ID,'status':'registered_not_launched','active_scientific_jobs':0}
for n in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    x=read(n);x.update(current_stage=ID,current_task=cfg['protocol'],current_stage_scope='TRAIN_only_frozen_teacher_learning_diagnostic',current_stage_progress='registered_not_launched',
        current_job=job,next_action=action,current_stage_registration='reviews/R56_REGISTRATION_001/RECEIPT.json',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,
        application_goal_status='active',model_goal_complete=False,subagents_allowed=False,status='R56_registered_long_term_goal_active',latest_delivered_stage='R55_PSD_SOURCE_COMPATIBILITY_001',latest_response_model_stage='R54_CONTINUOUS_CHARGE_FLOW_001',latest_diagnostic_stage='R55_PSD_SOURCE_COMPATIBILITY_001',
        goal_status_note='Application get_goal confirmed active; same original A-F objective, root-only, no quotas.',latest_user_steering='主助手独立完成原A–F长期目标，不设计算预算')
    x['execution_location'].update(mode='server_GPU_frozen_TRAIN_diagnostic',remote_root=cfg['remote_root'],current_stage_GPU_used=True,preflight_required=True)
    if n=='PROGRAM.json':x['updated_at_utc']=NOW
    else:x.update(project_status=x['status'],updated_UTC=NOW,current_activity='root_preparing_R56',current_direction=action)
    write(n,x)
b=read('RESOURCE_BUDGET.json');b['current_registered_stage']=ID
b['improvement_block']['registered_stages'].append({'stage':ID,'kind':'TRAIN_frozen_diagnostic_not_response_method','registered_UTC':NOW,'science_budget':None,'new_training_runs':0,'new_HGB_fits':0,'status':'registered'})
write('RESOURCE_BUDGET.json',b)
for n,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    x=read(n);assert not any(v[ident]==ID for v in x[key])
    x[key].append({ident:ID,'status':'registered','owner':'/root','registered_UTC':NOW,'protocol':cfg['protocol'],'output_root':'work/'+ID,'remote_root':cfg['remote_root'],'model_candidate':False});write(n,x)
write('reviews/R56_REGISTRATION_001/RECEIPT.json',{'UTC':NOW,'stage':ID,'root_control_PID':os.getpid(),'protocol':bind(R/cfg['protocol']),'config':bind(S/'config.json'),'input_binding':bind(S/'INPUT_BINDING.json'),'owner':'/root','application_goal_status':'active','original_A_F_unchanged':True,'scientific_execution_started':False,'response_training_runs':0,'remote_root':cfg['remote_root']})
print(json.dumps({'stage':ID,'UTC':NOW,'remote_root':cfg['remote_root'],'scientific_execution_started':False},indent=2))
