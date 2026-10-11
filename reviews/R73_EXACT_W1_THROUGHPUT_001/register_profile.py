"""One prospective profile registration; no model endpoint admitted."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
O=R/'work/R72_FULL_WAVEFORM_FULL_TRAIN_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'registration_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not(P/'REGISTRATION.json').exists()
assert read(R/'jobs.json')['active_scientific_jobs']==0
assert read(R/'R72_DELIVERY_RECEIPT.json')['remote_main_SHA']==read(R/'R72_DELIVERY_RECEIPT.json')['local_commit']
C=read(S/'config.json');old=read(O/'config.json')
allowed={'stage','remote_root','allowed_scientific_entries','GPU_scientific_entries','profile_warmup_updates','profile_timed_updates','profile_only','prior_R72_remote'}
assert {k for k in C if C[k]!=old.get(k)}.issubset(allowed)
assert C['profile_warmup_updates']==10 and C['profile_timed_updates']==10 and C['profile_only']
assert C['updates']==50000 and C['input_paths']==old['input_paths']
equal=[]
for f in S.glob('*.py'):
    ast.parse(f.read_text('utf8'));compile(f.read_text('utf8'),str(f),'exec')
    if f.name!='train_profile.py':assert sha(f)==sha(O/f.name);equal.append(f.name)
assert sha(S/'INPUT_BINDING.json')==sha(O/'INPUT_BINDING.json')
assert sha(S/'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json')==sha(R/'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json')
now=datetime.now(timezone.utc).isoformat();snap=P/'before_registration';snap.mkdir()
for name in ['AGENTS.md','STATUS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','jobs.json','experiments.json']:shutil.copy2(R/name,snap/name)
files=[dict(name=f.name,path=C['remote_root']+'/'+f.name,sha256=sha(f),bytes=f.stat().st_size)for f in sorted(S.iterdir())if f.is_file()and f.name!='FILE_BINDING.json']
write(S/'FILE_BINDING.json',dict(UTC=now,files=files,scope='R72 unchanged scientific computation, R73 timing-only wrapper and profile endpoint'))
prefix='''# 当前执行：R73 完整更新分项计时已登记

R72 成本关闭并已私有 Git/D 盘交付，主训练 0。R73 只对同一全人口/九损失/参数反向作 10 暖机+10 计时更新；记录各项同步耗时后恢复初始权重、空 Adam 和全部随机流。主训练 0，不使用质量挑选参数，不凭本 profile 启动 50000。数学等价优化须人工值/梯度核验及新完整成本门。v5 长期目标 active。

任务：tasks/R73_EXACT_W1_THROUGHPUT_001.md。其余父事件、50000 科学终点、4 小时判断、R65 停止及资源/E-D/既有辅助政策保持。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:(R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
entry=dict(id=P.name,stage=P.name,kind='engineering_complete_parent_component_profile',status='registered_not_started',defined_UTC=now,
  task='tasks/'+P.name+'.md',registration='reviews/'+P.name+'/REGISTRATION.json',remote_root=C['remote_root'],
  main_updates=0,fixed_engineering_updates=20,SOURCE_N=6593,REAL_N=21047,heldout_data_authorized=False,model_goal_complete=False)
for name,key in [('experiments.json','experiments'),('jobs.json','jobs')]:
    d=read(R/name);assert not any(v.get('stage',v.get('id'))==P.name for v in d[key]);d[key].append(entry.copy())
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);d.update(updated_UTC=now,status='R73_profile_registered',current_stage=P.name,current_task=entry['task'],
      current_stage_registration=entry['registration'],current_job=None,current_activity='complete_parent_component_cost_profile',
      active_scientific_jobs=0,application_goal_status='active',model_goal_complete=False,next_action='Run one20-update timing profile and retain zero main updates',
      latest_completed_backup='R72_DELIVERY_RECEIPT.json',latest_delivery_status='R72_private_Git_D_backup_complete')
    write(R/name,d)
task=R/entry['task'];local=[f for f in S.iterdir()if f.is_file()]+[task,P/'prepare_profile.py',P/'register_profile.py',P/'control.py',P/'collect.py']
write(P/'ROOT_IMPLEMENTATION_REVIEW.json',dict(UTC=now,timing_only_diff_reviewed=True,unchanged_R72_sources=equal,
  original_inputs_model_random_streams_optimizer_loss_and_full_parent_VJP_unchanged=True,profile_count=20,main_updates=0,
  initial_and_rng_restore_required=True,no_quality_selection=True,scientific_arrays_decoded=False,guard={**guard,**memory()}))
write(P/'REGISTRATION.json',dict(UTC=now,status='REGISTERED_BEFORE_SCIENCE',stage=P.name,
  local_files=[dict(path=f.relative_to(R).as_posix(),sha256=sha(f),bytes=f.stat().st_size)for f in local],
  root_review='reviews/'+P.name+'/ROOT_IMPLEMENTATION_REVIEW.json',main_updates=0,fixed_engineering_updates=20,current_acceptance_version=5))
print(json.dumps(dict(stage=P.name,registered=True,main_updates=0,engineering_updates=20,files=len(files),remote_root=C['remote_root'])),flush=True)
