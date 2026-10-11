"""Retain finished artificial receipts and record the observed R65 training start."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os
import shutil
import subprocess

R=Path(__file__).resolve().parents[1]
S=R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
read=lambda p:json.loads(p.read_text('utf8'))
write=lambda p,v:p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(S/'config.json')
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
k.SetProcessWorkingSetSizeEx.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_ulong]
h=k.GetCurrentProcess()
assert k.SetProcessAffinityMask(h,8) and k.SetProcessWorkingSetSizeEx(h,16*1024**2,1024**3,6)
host='rylan@100.82.231.102'
remote=cfg['remote_root']
pre=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8']
assert not (S/'artificial_results').exists()
subprocess.run(pre+['-r',host+':'+remote+'/artificial_results',str(S)],check=True,timeout=60)
names=['ARTIFICIAL_SUITE_RECEIPT.json','TRAIN_STARTED.json','train_final.py.START_RESOURCE.json']
subprocess.run(pre+[host+':'+remote+'/'+name for name in names]+[str(S)],check=True,timeout=60)
suite=read(S/'ARTIFICIAL_SUITE_RECEIPT.json')
start=read(S/'TRAIN_STARTED.json')
assert suite['pass'] and suite['all_8_cpu_cuda_artificial_receipts_passed']
assert suite['scientific_training_updates']==0 and suite['implementation_review_SHA']==sha(S/'ROOT_IMPLEMENTATION_REVIEW.json')
artificial=[]
for item in suite['results']:
    assert sha(S/item['path'])==item['sha256']
    val=read(S/item['path'])
    if item['name'].startswith('R65_JOINT_NETWORK_ADAM'):
        assert val['passed'] and val['caller_state_preserved'] and all(x['passed'] for x in val['checks'])
        detail={'checks':len(val['checks']),'reference_clip_norm':val['reference_clip_norm'],
                'replay_clip_norm':val['replay_clip_norm'],'all_parameter_count':val['all_parameter_count'],
                'synthetic_optimizer_steps':val['compact']['synthetic_optimizer_steps']}
    elif item['name'].startswith('R65_JOINT_OBJECTIVE'):
        assert all(x['passed'] for x in val['tests'].values())
        detail={'checks':val['check_count']}
    elif item['name'].startswith('R64_OBJECTIVE'):
        assert val['compact']['passed'] and val['compact']['checks']==69
        detail=val['compact']
    else:
        assert val['pass'] and val['state_unchanged']
        detail={'pass':True,'state_unchanged':True}
    artificial.append({'name':item['name'],'receipt_SHA256':item['sha256'],**detail})
assert len(artificial)==8 and start['sole_terminal_update']==50000
assert start['frozen_binding']['implementation_review_SHA']==sha(S/'ROOT_IMPLEMENTATION_REVIEW.json')
observed=read(S/'LATEST_REMOTE_OBSERVATION.json')
assert observed['actual_PID_present'] and observed['TRAIN_STARTED.json']==start
now=datetime.now(timezone.utc).isoformat()
J={'stage':cfg['stage'],'status':'full_empirical_training_running','UTC':now,
   'active_scientific_jobs':1,'remote_root':remote,'actual_worker_PID':start['PID'],
   'native_attempt_id':start['native_attempt_id'],'new_training_runs_started':1,
   'artificial_checks_passed':8,'scientific_terminal_update':50000,
   'current_observation':'work/'+cfg['stage']+'/LATEST_REMOTE_OBSERVATION.json',
   'launch_receipt':'work/'+cfg['stage']+'/LAUNCH_RECEIPT.json',
   'new_response_updates_at_observation':observed.get('PHYSICAL_PROGRESS.json',{}).get('last_completed_update'),
   'model_goal_complete':False}
receipt={'UTC':now,'root_PID':os.getpid(),'stage':cfg['stage'],
    'server_artificial_suite_passed':True,'artificial_results':artificial,
    'suite_SHA256':sha(S/'ARTIFICIAL_SUITE_RECEIPT.json'),'suite_elapsed_s':suite['elapsed_s'],
    'suite_GPU_allocated_peak_bytes':suite['GPU_allocated_peak_bytes'],
    'suite_GPU_reserved_peak_bytes':suite['GPU_reserved_peak_bytes'],
    'TRAIN_started_receipt':start,'job':J,'original_A_F_unchanged':True,
    'no_scientific_outcome_claimed':True,'native_training_exit':None}
write(S/'ROOT_LAUNCH_AND_ARTIFICIAL_REVIEW.json',receipt)
B=R/'reviews/R65_LAUNCH_OBSERVATION_001'
B.mkdir(exist_ok=False)
for name in ('PROGRAM.json','PROJECT_PROGRESS.json','jobs.json','experiments.json','RESOURCE_BUDGET.json',
             'AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md'):
    shutil.copy2(R/name,B/name)
for name in ('PROGRAM.json','PROJECT_PROGRESS.json'):
    val=read(R/name)
    val.update(status='R65_training_active_long_term_goal_active',current_stage_progress=J['status'],
       current_job=J,active_scientific_jobs=1,active_remote_scientific_jobs=1,active_local_scientific_jobs=0,
       model_goal_complete=False,application_goal_status='active',
       next_action='Complete fixed update50000, then original numerical gate, original A-C and replay; D/E only if original gates pass.')
    val['execution_location'].update(preflight_required=False,current_stage_GPU_used=True,
       mode='server_GPU_single_root_neural_training',remote_root=remote)
    if name=='PROGRAM.json':val['updated_at_utc']=now
    else:val.update(project_status=val['status'],updated_UTC=now,current_activity='R65_full_empirical_training')
    write(R/name,val)
for name,key,identity in (('jobs.json','jobs','stage'),('experiments.json','experiments','id')):
    val=read(R/name)
    rec=[x for x in val[key] if x.get(identity)==cfg['stage']]
    assert len(rec)==1
    rec[0].update(status=J['status'],updated_UTC=now,current_job=J,new_training_runs_started=1,
       active_scientific_jobs=1,artificial_checks_passed=8)
    val.update(active_scientific_jobs=1,active_remote_scientific_jobs=1,active_local_scientific_jobs=0)
    write(R/name,val)
val=read(R/'RESOURCE_BUDGET.json')
stages=val['improvement_block']['registered_stages']
rec=[x for x in stages if x.get('stage')==cfg['stage']]
assert len(rec)==1
rec[0].update(status=J['status'],new_training_runs_started=1,actual_worker_PID=start['PID'])
val['current_scientific_execution_location']='server_GPU'
write(R/'RESOURCE_BUDGET.json',val)
top=('# 当前执行：R65 全量有限 TRAIN 分布训练运行中\n\n'
     '已绑定112份源码/配置/回执；8项CPU/CUDA人工目标、网络回放与D+C/Adam检查全部通过，'
     '随后开始唯一原初始化50000步训练。每步保留全部6593 SOURCE和21047 REAL，原C微批与A–F不变。'
     '本次人工验证不代表科学验收通过。单科学作业、服务器CPU12/RSS48GiB/GPU10GiB、'
     '本机共享CPU4/RSS4GiB和全部新增本地E/D存储保持；无计算配额。\n\n')
for name in ('AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md'):
    (R/name).write_bytes((top+'## 以下为历史记录\n\n'+(R/name).read_text('utf8')).encode())
write(B/'RECEIPT.json',receipt)
print(json.dumps({'artificial_checks_passed':len(artificial),'native_training_PID':start['PID'],
      'new_training_runs_started':1,'sole_terminal_update':50000,'artificial_details':artificial},indent=2))
