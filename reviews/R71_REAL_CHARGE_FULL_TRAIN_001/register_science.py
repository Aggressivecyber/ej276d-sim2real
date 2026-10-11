"""Root prospective registration; static source reads only."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'registration_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not(P/'REGISTRATION.json').exists()
assert read(R/'PROGRAM.json')['active_scientific_jobs']==0
assert read(P/'GOAL_CREATION_RECEIPT.json')['result']['goal']['status']=='active'
assert sha(S/'public_charge_head.py')=='c99ddf9c83106da23e95b0f9ae805dae9260f30c8489eef0d61ccd3e12922496'
loss=P/'charge_loss_candidate/charge_losses.py'
assert sha(loss)=='e266c6c2882a322eaa5d1d2e5a8527e4d3644432bfd59c659007243613cbf1d5'
shutil.copy2(loss,S/'charge_losses.py')
for path in list(S.glob('*.py'))+[P/'control.py',P/'register_science.py']:
    ast.parse(path.read_text('utf8'));compile(path.read_text('utf8'),str(path),'exec')
snapshot=P/'before_registration';snapshot.mkdir()
names=['AGENTS.md','STATUS.md','PLAN.md','PROJECT_PROGRESS.json','PROGRAM.json','experiments.json','jobs.json','CURRENT_ACCEPTANCE.json']
for name in names:shutil.copy2(R/name,snapshot/name)
current=read(R/'CURRENT_ACCEPTANCE.json');assert current['version']==5
for name,key in [('document','document_sha256'),('registry','registry_sha256')]:assert sha(R/current[name])==current[key]
current.update(application_goal_text_may_still_name_v4=False,active_v5_goal_creation_receipt='reviews/'+P.name+'/GOAL_CREATION_RECEIPT.json')
write(R/'CURRENT_ACCEPTANCE.json',current)
for name in ['CURRENT_ACCEPTANCE.json',current['document'],current['registry']]:shutil.copy2(R/name,S/name)
task=R/'tasks'/(P.name+'.md');shutil.copy2(task,S/'PROTOCOL.md')
C=read(S/'config.json');remote=Path(C['remote_root']);now=datetime.now(timezone.utc).isoformat()
files=[]
for f in sorted(S.iterdir()):
    if f.is_file()and f.name!='FILE_BINDING.json':files.append(dict(name=f.name,path=C['remote_root']+'/'+f.name,sha256=sha(f),bytes=f.stat().st_size))
write(S/'FILE_BINDING.json',dict(UTC=now,files=files,scope='flat immutable submitted scientific source/config; inputs separately bound'))
prefix='''# 当前执行：R71 公共电荷响应全量 TRAIN 已登记

当前 v5 长期目标 active。固定全部6593 SOURCE与21047 REAL的公共Q响应头、三项完整人口目标、50000步唯一科学终点。先做50次完整TRAIN工程更新计时，恢复同初始权重/空Adam/RNG；若保守总耗时<=4小时，同一受控作业直接执行全量终点及固定TRAIN数值/响应检查。否则保留测时证据，不缩人口或改终点。完整波形残差此阶段为零，局部电荷训练不等于A–F完成。

协议：tasks/R71_REAL_CHARGE_FULL_TRAIN_001.md；注册：reviews/R71_REAL_CHARGE_FULL_TRAIN_001/REGISTRATION.json。其余v5门、同处理REAL、全部来源/权重和后续4000 held-out独立重放保留。R65仍停止。CPU12/RSS48GiB/GPU10GiB、单科学作业及本机E/D存储保持。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:
    (R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
entry=dict(id=P.name,stage=P.name,kind='TRAIN_only_complete_parent_charge_response_component',status='registered_not_started',
    defined_UTC=now,task='tasks/'+P.name+'.md',registration='reviews/'+P.name+'/REGISTRATION.json',
    remote_root=C['remote_root'],source_N=6593,REAL_N=21047,unique_main_updates=50000,engineering_updates=50,
    cost_limit_seconds=14400,science_wall_cap_s=None,model_goal_complete=False,heldout_data_authorized=False)
for name,key in [('experiments.json','experiments'),('jobs.json','jobs')]:
    d=read(R/name);assert not any(v.get('id',v.get('stage'))==P.name for v in d[key]);d[key].append(entry.copy())
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);d.update(updated_UTC=now,status='R71_registered_not_started',current_stage=P.name,current_task=entry['task'],
        current_stage_registration=entry['registration'],numeric_execution_released=True,current_activity='registered_full_TRAIN_cost_then_endpoint',
        active_scientific_jobs=0,current_job=None,application_goal_status='active',model_goal_complete=False,
        next_action='Launch one bound supervised R71 worker; measured <=4h automatically runs full endpoint')
    write(R/name,d)
localfiles=list(S.glob('*'))+[task,P/'control.py',P/'register_science.py',R/'CURRENT_ACCEPTANCE.json',R/current['document'],R/current['registry']]
localfiles=[f for f in localfiles if f.is_file()]
review=dict(UTC=now,scope='Root static review before scientific arrays',
    head_and_loss_artificial_CPU_evidence='implementation_candidate and charge_loss_candidate receipts',
    analytic_noise_independent_audit='implementation_candidate common-response audit',
    corrected_before_registration=['inactive sqrt gradient','progress duplicate elapsed_s','independent reference finite/slope rejection'],
    original_data_scope='TRAIN only; no generic inputs loader or heldout file path',
    full_population_every_update=True,initialization_restore_bound=True,single50000_endpoint=True,
    charge_component_not_A_F=True,new_acceptance_thresholds=False,guard={**guard,**memory()},science_arrays_read=False)
write(P/'ROOT_IMPLEMENTATION_REVIEW.json',review)
write(P/'REGISTRATION.json',dict(status='REGISTERED_BEFORE_SCIENCE',UTC=now,stage=P.name,science_arrays_read=False,
    current_acceptance_version=5,local_files=[dict(path=f.relative_to(R).as_posix(),sha256=sha(f),bytes=f.stat().st_size)for f in localfiles],
    root_review='reviews/'+P.name+'/ROOT_IMPLEMENTATION_REVIEW.json',user_authorized_full_if_hours=True))
print(json.dumps(dict(registered=P.name,files=len(localfiles),remote_root=C['remote_root'],science_arrays_read=False,local_guard={**guard,**memory()})))
