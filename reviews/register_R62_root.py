from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,subprocess,os
R=Path(__file__).resolve().parents[1];ID='R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001';S=R/'work'/ID;REG=R/'reviews/R62_REGISTRATION_001';NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,v):(R/p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert not S.exists() and not REG.exists()
assert read('PROGRAM.json')['status']=='R61_delivered_long_term_goal_active' and read('R61_DELIVERY_COMPLETION_RECEIPT.json')['status']=='delivered'
H=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001';old=json.loads((H/'config.json').read_text('utf8'));hr=old['remote_root'];arm=old['arm']
remote=hr.split('/R61_ROOT_')[0]+'/R62_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
inputs={'TRAIN_path':old['TRAIN_path'],'constants_path':hr+'/UNIT_CHARGE_CONSTANTS.npz','noise_calibration_path':hr+'/NOISE_CALIBRATION.npz','distribution_statistics':hr+'/TRAIN_DISTRIBUTION_CONSTANTS.npz','frozen_weight':hr+'/'+arm+'/FINAL_FLOW.pt','frozen_training_receipt':hr+'/'+arm+'/TRAINING_RECEIPT.json','frozen_training_losses':hr+'/'+arm+'/LOSS_BY_UPDATE_PRIVATE.npz','frozen_training_config':hr+'/config.json'}
code='from pathlib import Path\nimport json,hashlib,subprocess,shutil,os\ninputs='+repr(inputs)+'\n'+'''
base='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305';jobs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit()or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace');cwd=str((p/'cwd').resolve())
  if 'python'in argv and(base in argv or cwd.startswith(base)):jobs.append({'PID':int(p.name),'cwd':cwd,'argv':argv})
 except(OSError,PermissionError):pass
assert not jobs,jobs
mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip();assert not apps,apps
print(json.dumps({'inputs':{k:{'path':p,'bytes':Path(p).stat().st_size,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}for k,p in inputs.items()},'read_only_preflight':{'own_project_processes':jobs,'available_memory_bytes':mem['MemAvailable'],'GPU':gpu,'GPU_compute_processes':apps,'server_disk_free_bytes':shutil.disk_usage(base).free},'scientific_arrays_decoded':False}))
'''
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=60)
assert p.returncode==0,p.stderr;observed=json.loads(p.stdout)
assert observed['read_only_preflight']['available_memory_bytes']>=old['MemAvailable_min_bytes']
assert int(observed['read_only_preflight']['GPU'].split(',')[-1].strip())*1048576>=11811160064
assert shutil.disk_usage('D:/').free>20*1024**2
assert observed['inputs']['frozen_weight']['sha256']==json.loads((H/'FROZEN_ARTIFACT_REVIEW.json').read_text('utf8'))['weight_file_SHA256']
S.mkdir();REG.mkdir()
for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','PROGRAM.json','PROJECT_PROGRESS.json','RESOURCE_BUDGET.json','jobs.json','experiments.json']:shutil.copy2(R/n,REG/n)
C={k:old[k]for k in ['server_python','CPU_threads','RSS_limit_bytes','GPU_limit_bytes','MemAvailable_min_bytes','expected_R33_environment']}
seeds={'full_SOURCE_epsilon':20261011600,'full_REAL_epsilon':20261011601,'full_feature_directions':20261011602,'batch_SOURCE_rows':20261011623,'batch_REAL_A_rows':20261011624,'batch_REAL_B_rows':20261011625,'batch_REAL_control_rows':20261011626,'covariance_reference_rows':20261011627}
r60=read('work/R60_FROZEN_TRAIN_GRADIENT_DIAGNOSTIC_001/RANDOM_STREAM_RECEIPT.json')
C.update(stage=ID,owner='/root',subagents_allowed=False,registered_UTC=NOW,remote_root=remote,protocol='tasks/'+ID+'.md',historical_R61_root=hr,input_paths=inputs,science_budget=None,deadline_UTC=None,output_limit_bytes=None,compute_budget_enforced=False,PCG64_seeds=seeds,expected_full_stream_hashes={k:r60['chains'][k]for k in list(seeds)[:3]},SOURCE_N=6593,REAL_N=21047,full_parent_rows=27640,generation_chunk=64,solver_steps=128,planned_total_network_rows=27640,resampling_batches=32,batch_per_region=32,projection_directions=64,baseline_lags=list(range(1,21)),spectral_band_bins=5,covariance_reference_replicates=32,valley_edge_half_width=0.004,ADC_LSB_V=1/16383,marker_boundary_required=2,marker_covariance_required_regions=6,response_training_runs=0,response_updates=0,HGB_fits=0,density_fits=0,DEV_read=0,EVAL_read=0,CAL_decode=0,allowed_scientific_entries=['diagnose_train.py'],GPU_scientific_entries=['diagnose_train.py'])
write('work/'+ID+'/config.json',C)
frozen=['network.py','noise_core.py','source_flow_core.py','distribution_loss.py','precision_order.py']
for n in frozen+['linux_owned_worker.py']:shutil.copy2(H/n,S/n)
(S/'linux_supervise.py').write_bytes((H/'linux_supervise.py').read_bytes().replace(b'R61 requires explicit null',b'R62 requires explicit null'))
shutil.copy2(R/C['protocol'],S/'PROTOCOL.md')
write('work/'+ID+'/INPUT_BINDING.json',{'UTC':NOW,**observed,'original_acceptance':[bind(R/n)for n in ['PROJECT_ACCEPTANCE.json','PROJECT_ACCEPTANCE.md']],'frozen_R61_helpers':{n:bind(H/n)for n in frozen},'frozen_weight_tensor_SHA':json.loads((H/arm/'TRAINING_RECEIPT.json').read_text('utf8'))['final_state_SHA'],'source_parent_ID_SHA':json.loads((H/'TRAIN_POPULATION_AND_COORDINATES.json').read_text('utf8'))['SOURCE_ids_SHA256'],'R60_full_stream_reference':bind(R/'work/R60_FROZEN_TRAIN_GRADIENT_DIAGNOSTIC_001/RANDOM_STREAM_RECEIPT.json'),'local_D_free_bytes':shutil.disk_usage('D:/').free,'scientific_execution_started':False})
action='Execute the preregistered frozen R61 full-parent waveform-path, baseline-joint and fixed32 saved-endpoint resampling diagnostic on legal TRAIN only; no model updates or validation reads.'
banner='# 当前执行：R62 冻结完整 TRAIN 波形残差诊断已登记\n\nR61固定末点已关闭并实际交付。本轮按 tasks/'+ID+'.md 完成全部27640原TRAIN父事件的一次固定前向，检查波形路径、固定边界、基线联合结构和32次保存端点重采样。0训练/教师/HGB/密度拟合，0 DEV/EVAL/CAL解码。原A–F保持，长期目标active；主助手独立、无计算配额，机器安全/单科学作业/本机E-D限制保持。\n\n## 以下为历史记录\n\n'
for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/n).write_bytes((banner+(R/n).read_text('utf8')).encode('utf8'))
job={'stage':ID,'status':'registered_not_launched','active_scientific_jobs':0}
for n in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 v=read(n);v.update(current_stage=ID,current_task=C['protocol'],current_stage_scope='frozen_R61_TRAIN_full_waveform_residual_diagnostic',current_stage_progress='registered_not_launched',current_job=job,next_action=action,current_stage_registration='reviews/R62_REGISTRATION_001/RECEIPT.json',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,subagents_allowed=False,application_goal_status='active',model_goal_complete=False)
 v['execution_location']={**v['execution_location'],'remote_root':remote,'mode':'server_GPU_single_frozen_diagnostic','preflight_required':True}
 if n=='PROGRAM.json':v.update(status='R62_registered_long_term_goal_active',updated_at_utc=NOW)
 else:v.update(status='R62_registered_long_term_goal_active',project_status='R62_registered_long_term_goal_active',updated_UTC=NOW,current_activity='root_implementing_R62',current_direction=action)
 write(n,v)
B=read('RESOURCE_BUDGET.json');B['current_registered_stage']=ID;B['improvement_block']['registered_stages'].append({'stage':ID,'kind':'frozen_TRAIN_full_waveform_residual_diagnostic','registered_UTC':NOW,'science_budget':None,'planned_new_training_runs':0,'planned_neural_forward_rows':27640,'status':'registered'});write('RESOURCE_BUDGET.json',B)
for n,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 v=read(n);assert not any(q.get(ident)==ID for q in v[key]);v[key].append({ident:ID,'status':'registered','owner':'/root','registered_UTC':NOW,'protocol':C['protocol'],'output_root':'work/'+ID,'remote_root':remote,'model_candidate':False,'new_training_runs_started':0});write(n,v)
write('reviews/R62_REGISTRATION_001/RECEIPT.json',{'UTC':NOW,'stage':ID,'root_control_PID':os.getpid(),'protocol':bind(R/C['protocol']),'config':bind(S/'config.json'),'owner':'/root','application_goal_status':'active','original_A_F_unchanged':True,'scientific_execution_started':False,'response_training_runs_started':0,'remote_root':remote,'read_only_preflight':observed['read_only_preflight']})
print(json.dumps({'stage':ID,'UTC':NOW,'remote_root':remote,'planned_forward_rows':27640,'new_training':0,'D_free_bytes':shutil.disk_usage('D:/').free},indent=2))
