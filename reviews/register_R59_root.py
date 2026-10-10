from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,subprocess,os
R=Path(__file__).resolve().parents[1];ID='R59_DIRECT_DISTRIBUTION_RESPONSE_001';S=R/'work'/ID;REG=R/'reviews/R59_REGISTRATION_001';NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,v):
    value=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if p=='RESOURCE_BUDGET.json':value=value.replace('\n','\r\n')
    (R/p).write_bytes(value.encode('utf8'))
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert not S.exists() and not REG.exists()
assert read('PROGRAM.json')['status']=='R58_delivered_long_term_goal_active' and read('R58_DELIVERY_COMPLETION_RECEIPT.json')['status']=='delivered'
assert not read('work/R58_JOINT_DISTRIBUTION_FEASIBILITY_001/LIVE_OBSERVATION.json')['actual_process']['exists']
old=read('work/R54_CONTINUOUS_CHARGE_FLOW_001/config.json');oldpost=read('work/R54_CONTINUOUS_CHARGE_FLOW_001/config_post.json');r58=read('work/R58_JOINT_DISTRIBUTION_FEASIBILITY_001/config.json')
remote=old['remote_root'].split('/R54_ROOT_')[0]+'/R59_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
inputs={k:old[k] for k in ['TRAIN_path','constants_path','noise_calibration_path','train_solver_input']}
inputs['valley_definition']=r58['input_paths']['valley_definition']
code='from pathlib import Path\nimport json,hashlib,subprocess,shutil,os\ninputs='+repr(inputs)+'\n'+'''
base='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305';jobs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace');cwd=str((p/'cwd').resolve())
  if 'python' in argv and (base in argv or cwd.startswith(base)):jobs.append({'PID':int(p.name),'cwd':cwd,'argv':argv})
 except(OSError,PermissionError):pass
assert not jobs,jobs
mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
assert not apps,apps
print(json.dumps({'inputs':{k:{'path':p,'bytes':Path(p).stat().st_size,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}for k,p in inputs.items()},'read_only_preflight':{'own_project_processes':jobs,'available_memory_bytes':mem['MemAvailable'],'GPU':gpu,'GPU_compute_processes':apps,'server_disk_free_bytes':shutil.disk_usage(base).free},'DEV_EVAL_CAL_decoded':False}))
'''
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=60)
assert p.returncode==0,p.stderr
observed=json.loads(p.stdout)
assert observed['read_only_preflight']['available_memory_bytes']>=old['MemAvailable_min_bytes']
assert int(observed['read_only_preflight']['GPU'].split(',')[-1].strip())*1048576>=11811160064
S.mkdir();REG.mkdir()
for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','PROGRAM.json','PROJECT_PROGRESS.json','RESOURCE_BUDGET.json','jobs.json','experiments.json']:shutil.copy2(R/n,REG/n)
C={k:old[k] for k in ['server_python','CPU_threads','RSS_limit_bytes','GPU_limit_bytes','MemAvailable_min_bytes','TRAIN_path','source_local_TRAIN','constants_path','noise_calibration_path','train_solver_input','expected_R33_environment']}
C.update(stage=ID,owner='/root',subagents_allowed=False,registered_UTC=NOW,remote_root=remote,protocol='tasks/'+ID+'.md',arm='DIRECT_DISTRIBUTION_RESPONSE',arms=['DIRECT_DISTRIBUTION_RESPONSE'],
    science_budget=None,deadline_UTC=None,output_limit_bytes=None,compute_budget_enforced=False,model_seed=20261010590,
    PCG64_seeds={k:s for k,s in zip(['source_rows','REAL_reference_rows','REAL_identity_rows','source_epsilon','identity_epsilon','projection_directions'],range(20261010591,20261010597))},
    updates=50000,batch_per_region=32,identity_batch=64,region_categories=[-1,*range(8),8],lr=2e-4,lr_initial=1e-6,lr_final=2e-5,warmup_updates=500,betas=[.9,.999],gradient_clip=5.0,
    projection_directions=64,box_projection_iterations=64,solver_steps=128,solver_check_steps=256,generation_chunk=64,inference_device='cuda:0',
    input_paths=inputs,valley_definition=inputs['valley_definition'],SOURCE_N=6593,REAL_N=21047,PSD_loss_scale=.01,valley_delta=.02,valley_smoothing=.002,
    feature_names=['V480','U480','DU479','D2U478'],feature_scale_floor={'V480':1/16383,'U480':.001,'DU479':.001,'D2U478':.001},
    noise_scale_floor=[1/16383/10,.01],loss_weights={k:1. for k in ['full_window','PSD','source_order','source_valley','noise','REAL_identity','box_correction']},
    allowed_scientific_entries=['train_final.py','numerical_check.py','generate_all.py','evaluate_saved.py','replay_all.py'],GPU_scientific_entries=['train_final.py','numerical_check.py','generate_all.py','replay_all.py'],
    historical_R54_root=old['remote_root'],historical_R54_report='R54_REPORT.md',prior_joint_law_used_as_teacher=False,training_source_global_measure='original equal parent weights via explicit region importance correction')
write('work/'+ID+'/config.json',C)
for n in ['noise_core.py','NOISE_CALIBRATION.npz','NOISE_CALIBRATION.json','UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json','linux_owned_worker.py','HISTORICAL_TRAIN_SOLVER_CHECK.json']:
    shutil.copy2(R/'work/R54_CONTINUOUS_CHARGE_FLOW_001'/n,S/n)
assert hashlib.sha256((S/'NOISE_CALIBRATION.npz').read_bytes()).hexdigest()==observed['inputs']['noise_calibration_path']['sha256']
assert hashlib.sha256((S/'UNIT_CHARGE_CONSTANTS.npz').read_bytes()).hexdigest()==observed['inputs']['constants_path']['sha256']
value=(R/'work/R54_CONTINUOUS_CHARGE_FLOW_001/linux_supervise.py').read_text('utf8').replace('R54 requires explicit null','R59 requires explicit null')
(S/'linux_supervise.py').write_bytes(value.encode('utf8'))
shutil.copy2(R/C['protocol'],S/'PROTOCOL.md')
write('work/'+ID+'/INPUT_BINDING.json',{'UTC':NOW,**observed,'original_acceptance':[bind(R/n) for n in ['PROJECT_ACCEPTANCE.json','PROJECT_ACCEPTANCE.md']],
    'backbone_reference':bind(R/'work/R54_CONTINUOUS_CHARGE_FLOW_001/network.py'),'previous_statistical_evidence':bind(R/'work/R58_JOINT_DISTRIBUTION_FEASIBILITY_001/RESULT.json'),
    'trained_weights_loaded':False,'teacher_waveforms_or_R58_law_targets_loaded':False})
banner='# 当前执行：R59 直接无配对全窗神经分布训练已登记\n\nR58统计约束检查已交付。本轮执行 tasks/'+ID+'.md：完整源与实测分布损失、明确的原父事件总体权重、源保持、同处理REAL和电荷/电压约束；不构造跨域目标波形。唯一50000更新末点，原128/256数值检查及A–F保持。主助手独立，无子智能体、无计算配额；CPU12/RSS48GiB/GPU10GiB、单科学作业与本机E/D限制保持。\n\n## 以下为历史记录\n\n'
for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/n).write_bytes((banner+(R/n).read_text('utf8')).encode('utf8'))
action='Implement and run the single registered direct unpaired full-waveform distribution method with original-parent global weights, source/REAL controls and explicit charge/voltage constraints; freeze update50000 and execute original numerical and full response gates.'
job={'stage':ID,'status':'registered_not_launched','active_scientific_jobs':0}
for n in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    v=read(n);v.update(current_stage=ID,current_task=C['protocol'],current_stage_scope='single_direct_unpaired_fullwaveform_distribution_method',current_stage_progress='registered_not_launched',current_job=job,next_action=action,current_stage_registration='reviews/R59_REGISTRATION_001/RECEIPT.json',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,subagents_allowed=False,application_goal_status='active',model_goal_complete=False)
    v['execution_location']={**v['execution_location'],'remote_root':remote,'mode':'server_GPU_single_root_neural_training','preflight_required':True}
    if n=='PROGRAM.json':v.update(status='R59_registered_long_term_goal_active',updated_at_utc=NOW)
    else:v.update(status='R59_registered_long_term_goal_active',project_status='R59_registered_long_term_goal_active',updated_UTC=NOW,current_activity='root_implementing_R59',current_direction=action)
    write(n,v)
B=read('RESOURCE_BUDGET.json');B['current_registered_stage']=ID;B['improvement_block']['registered_stages'].append({'stage':ID,'kind':'direct_unpaired_distribution_neural_response','registered_UTC':NOW,'science_budget':None,'planned_new_training_runs':1,'fixed_updates':50000,'status':'registered'});write('RESOURCE_BUDGET.json',B)
for n,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    v=read(n);v[key].append({ident:ID,'status':'registered','owner':'/root','registered_UTC':NOW,'protocol':C['protocol'],'output_root':'work/'+ID,'remote_root':remote,'model_candidate':True,'new_training_runs_started':0});write(n,v)
write('reviews/R59_REGISTRATION_001/RECEIPT.json',{'UTC':NOW,'stage':ID,'root_control_PID':os.getpid(),'protocol':bind(R/C['protocol']),'config':bind(S/'config.json'),'owner':'/root','application_goal_status':'active','original_A_F_unchanged':True,'scientific_execution_started':False,'response_training_runs_started':0,'remote_root':remote,'read_only_preflight':observed['read_only_preflight']})
print(json.dumps({'stage':ID,'UTC':NOW,'remote_root':remote,'planned_training':1,'fixed_updates':50000,'scientific_execution_started':False,'preflight':observed['read_only_preflight']},indent=2))
