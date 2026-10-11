from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil, subprocess, os, copy
R=Path(__file__).resolve().parents[1]
ID='R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001';S=R/'work'/ID
H=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001';REG=R/'reviews/R63_REGISTRATION_001'
NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,v):(R/p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert not S.exists() and not REG.exists()
assert read('PROGRAM.json')['status']=='R62_delivered_long_term_goal_active'
assert read('R62_DELIVERY_COMPLETION_RECEIPT.json')['status']=='delivered'
assert not read('work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001/LIVE_OBSERVATION.json')['actual_process']['exists']
old=json.loads((H/'config.json').read_text('utf8'))
historical=json.loads((H/'PRECISION_RESOLVED_RESPONSE/TRAINING_RECEIPT.json').read_text('utf8'))
hr=old['remote_root'];remote=hr.split('/R61_ROOT_')[0]+'/R63_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
inputs={k:v for k,v in old['input_paths'].items() if not k.startswith('R59_')}
inputs.update(distribution_statistics=hr+'/TRAIN_DISTRIBUTION_CONSTANTS.npz',R61_training_receipt=hr+'/PRECISION_RESOLVED_RESPONSE/TRAINING_RECEIPT.json',
              R61_parent_exposures=hr+'/PRECISION_RESOLVED_RESPONSE/PARENT_EXPOSURES_PRIVATE.npz',R61_final_random_states=hr+'/PRECISION_RESOLVED_RESPONSE/FINAL_RANDOM_STATES.json')
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
print(json.dumps({'inputs':{k:{'path':p,'bytes':Path(p).stat().st_size,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}for k,p in inputs.items()},'read_only_preflight':{'own_project_processes':jobs,'available_memory_bytes':mem['MemAvailable'],'GPU':gpu,'GPU_compute_processes':apps,'server_disk_free_bytes':shutil.disk_usage(base).free},'scientific_arrays_decoded':False}))
'''
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=60)
assert p.returncode==0,p.stderr
observed=json.loads(p.stdout)
assert observed['read_only_preflight']['available_memory_bytes']>=old['MemAvailable_min_bytes']
assert int(observed['read_only_preflight']['GPU'].split(',')[-1].strip())*1048576>=11811160064
for key,name in [('distribution_statistics','TRAIN_DISTRIBUTION_CONSTANTS.npz'),('R61_training_receipt','PRECISION_RESOLVED_RESPONSE/TRAINING_RECEIPT.json'),('R61_parent_exposures','PRECISION_RESOLVED_RESPONSE/PARENT_EXPOSURES_PRIVATE.npz'),('R61_final_random_states','PRECISION_RESOLVED_RESPONSE/FINAL_RANDOM_STATES.json')]:
 assert bind(H/name)['sha256']==observed['inputs'][key]['sha256']
dfree=shutil.disk_usage('D:/').free
delivery_estimate=read('R61_BACKUP_RECEIPT.json')['copied_bytes']+10*1024**2
assert dfree>delivery_estimate+20*1024**2,('backup_capacity',dfree,delivery_estimate)
S.mkdir();REG.mkdir()
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','PROGRAM.json','PROJECT_PROGRESS.json','RESOURCE_BUDGET.json','jobs.json','experiments.json']:shutil.copy2(R/name,REG/name)
C=copy.deepcopy(old)
for key in ['expected_R59_initial_state_SHA','expected_R59_stream_hashes','fixed_endpoint_single_factor_ablation','constants_reused_from_frozen_R59']:C.pop(key,None)
C.update(stage=ID,arm='COVERAGE_RESPONSE',arms=['COVERAGE_RESPONSE'],registered_UTC=NOW,remote_root=remote,protocol='tasks/'+ID+'.md',input_paths=inputs,
 historical_R61_root=hr,historical_R61_report='R61_REPORT.md',historical_R62_report='R62_REPORT.md',
 expected_R61_initial_state_SHA=historical['initial_state_SHA'],expected_R61_stream_hashes=historical['stream_hashes'],
 sole_scientific_change='Add nonnegative full-coordinate RMS empirical-W1 and 31-feature baseline temporal marginal/sliced distribution losses to the unchanged original seven losses.',
 fixed_endpoint_objective_extension=True,original_distribution_constants_reused_from_R61=True,new_temporal_statistics_TRAIN_only=True,
 temporal_feature_count=31,temporal_lags=list(range(1,21)),temporal_band_bin_count=5,temporal_band_count=10,temporal_projection_directions=64,
 temporal_scale_floor=[1/16383/10]+[.01]*30,temporal_power_floor=1e-20,coordinate_RMS_pooling='positive_l2_norm_of_coordinate_empirical_W1_div_sqrt_dimension')
C['PCG64_seeds']['baseline_projection_directions']=20261011631
C['loss_weights'].update(coordinate_RMS=1.,temporal_joint=1.)
for k in ['model_seed','updates','batch_per_region','identity_batch','region_categories','lr','lr_initial','lr_final','warmup_updates','betas','gradient_clip','projection_directions','box_projection_iterations','solver_steps','solver_check_steps','generation_chunk','PSD_loss_scale','valley_delta','valley_smoothing','feature_names','feature_scale_floor','noise_scale_floor','source_order_rule']:assert C[k]==old[k],k
assert {k:C['PCG64_seeds'][k]for k in old['PCG64_seeds']}==old['PCG64_seeds']
write('work/'+ID+'/config.json',C)
frozen=['network.py','noise_core.py','source_flow_core.py','precision_order.py','linux_owned_worker.py','NOISE_CALIBRATION.npz','NOISE_CALIBRATION.json','UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json','HISTORICAL_TRAIN_SOLVER_CHECK.json']
for name in frozen+['train_final.py','artificial_check.py','linux_supervise.py']:shutil.copy2(H/name,S/name)
shutil.copy2(H/'distribution_loss.py',S/'R61_distribution_loss.py')
shutil.copy2(R/C['protocol'],S/'PROTOCOL.md')
write('work/'+ID+'/INPUT_BINDING.json',{'UTC':NOW,**observed,'original_acceptance':[bind(R/n)for n in ['PROJECT_ACCEPTANCE.json','PROJECT_ACCEPTANCE.md']],
 'frozen_R61_helpers':{n:bind(H/n)for n in frozen+['distribution_loss.py','train_final.py','artificial_check.py']},
 'R62_evidence':[bind(R/'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001/RESULT.json'),bind(R/'reviews/R62_ROOT_DIRECTION_001.json')],
 'trained_weights_loaded':False,'local_D_free_bytes':dfree,'delivery_bytes_estimate_not_science_quota':delivery_estimate})
action='Implement and check the preregistered two-term distribution coverage extension, train once from the same R61 initialization to update50000, then original numerical gates and complete A-C/replay; D/E only after passing.'
banner='# 当前执行：R63 全窗及基线联合分布覆盖已登记\n\n保留原R61网络、人口权重、噪声、七项损失和六条随机流；增加全坐标RMS经验W1与31维基线时序联合分布两项，外层系数均1，从同一初始化训练至唯一50000步末点。新TRAIN标准化、全部公式和终点已在读取新TRAIN前冻结。原A–F保持；主助手独立、无子智能体/计算配额、CPU12/RSS48GiB/GPU10GiB、单科学作业、本机E/D。当前已登记，尚未启动科学训练。\n\n## 以下为历史记录\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/name).write_bytes((banner+(R/name).read_text('utf8')).encode('utf8'))
job={'stage':ID,'status':'registered_not_launched','active_scientific_jobs':0}
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 v=read(name);v.update(current_stage=ID,current_task=C['protocol'],current_stage_scope='full_waveform_and_temporal_joint_distribution_coverage_extension',current_stage_progress='registered_not_launched',current_job=job,next_action=action,current_stage_registration='reviews/R63_REGISTRATION_001/RECEIPT.json',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,subagents_allowed=False,application_goal_status='active',model_goal_complete=False,status='R63_registered_long_term_goal_active')
 v['execution_location'].update(remote_root=remote,mode='server_GPU_single_root_neural_training',preflight_required=True,current_stage_GPU_used=False)
 if name=='PROGRAM.json':v.update(updated_at_utc=NOW)
 else:v.update(project_status='R63_registered_long_term_goal_active',updated_UTC=NOW,current_activity='root_implementing_R63',current_direction=action)
 write(name,v)
b=read('RESOURCE_BUDGET.json');b['current_registered_stage']=ID
b['improvement_block']['registered_stages'].append({'stage':ID,'kind':'full_waveform_temporal_joint_coverage_extension','registered_UTC':NOW,'science_budget':None,'planned_new_training_runs':1,'fixed_terminal_update':50000,'status':'registered'})
write('RESOURCE_BUDGET.json',b)
for name,key,identity in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 v=read(name);v[key].append({identity:ID,'status':'registered','owner':'/root','registered_UTC':NOW,'protocol':C['protocol'],'output_root':'work/'+ID,'remote_root':remote,'new_training_runs_started':0,'fixed_terminal_update':50000,'model_goal_complete':False});write(name,v)
write('reviews/R63_REGISTRATION_001/RECEIPT.json',{'UTC':NOW,'stage':ID,'root_control_PID':os.getpid(),'protocol':bind(R/C['protocol']),'config':bind(S/'config.json'),'owner':'/root','application_goal_status':'active','original_A_F_unchanged':True,'scientific_execution_started':False,'response_training_runs_started':0,'remote_root':remote,'read_only_preflight':observed['read_only_preflight'],'D_free_bytes':dfree,'same_initialization_and_original_six_streams_required':True,'new_independent_temporal_direction_seed':20261011631})
print(json.dumps({'stage':ID,'UTC':NOW,'remote_root':remote,'new_training_runs_registered':1,'scientific_execution_started':False,'preflight':observed['read_only_preflight'],'D_free_bytes':dfree},indent=2))
