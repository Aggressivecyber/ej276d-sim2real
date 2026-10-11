"""Bind the exact-computation candidate before any new TRAIN execution."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
O=R/'work/R72_FULL_WAVEFORM_FULL_TRAIN_001';W=R/'reviews/R73_EXACT_W1_THROUGHPUT_001/w1_candidate'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'registration_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not(P/'REGISTRATION.json').exists()
assert read(R/'jobs.json')['active_scientific_jobs']==0
assert read(R/'R73_DELIVERY_RECEIPT.json')['remote_main_SHA']==read(R/'R73_DELIVERY_RECEIPT.json')['local_commit']
assert read(P/'ROOT_COMBINED_ARTIFICIAL.json')['pass_all']
assert read(W/'ARTIFACT_COMPLETION_RECEIPT.json')['status']=='SINGLE_CANDIDATE_PREPARED_ARTIFICIAL_CPU_PASS'
assert read(W/'ARTIFACT_COMPLETION_RECEIPT.json')['native_exit_code']==0
C=read(S/'config.json');old=read(O/'config.json')
allowed={'stage','remote_root','implementation','engineering_equivalence_updates','equivalence_value_scaled_atol','equivalence_parameter_gradient_relative_L2',
  'equivalence_Adam_parameter_maxabs','original_R72_source_SHA256','original_R72_cost_p90_seconds'}
assert {k for k in C if C[k]!=old.get(k)}.issubset(allowed)
assert C['updates']==50000 and C['neural_replay_chunk']==27640 and C['cost_limit_seconds']==14400
assert C['input_paths']==old['input_paths'] and C['loss_weights']==old['loss_weights'] and C['hidden']==128
assert C['engineering_equivalence_updates']==[1,25,50]
equal=[]
new_names={'train_full.py','combined_loss.py','check_combined_loss.py','exact_uniform_w1.py','check_exact_uniform_w1.py'}
for f in S.glob('*.py'):
    ast.parse(f.read_text('utf8'));compile(f.read_text('utf8'),str(f),'exec')
    if f.name not in new_names:assert sha(f)==sha(O/f.name);equal.append(f.name)
for name in ['combined_loss.py','check_combined_loss.py']:assert sha(S/name)==sha(P/name)
for name in ['exact_uniform_w1.py','check_exact_uniform_w1.py']:assert sha(S/name)==sha(W/name)
assert sha(S/'exact_uniform_w1.py')=='633ab611739cec8da90ec2fc6894c3c5ecff02df97600dd38469b1b0c1b9e18a'
assert sha(S/'check_exact_uniform_w1.py')=='ac22a4ea4c47f67d07d96988928ff1550fad718ffaccdd4a91e6ab38e520580a'
assert sha(S/'INPUT_BINDING.json')==sha(O/'INPUT_BINDING.json')
current=read(R/'CURRENT_ACCEPTANCE.json');assert current['version']==5
for n,k in [('document','document_sha256'),('registry','registry_sha256')]:assert sha(R/current[n])==current[k]
now=datetime.now(timezone.utc).isoformat();snap=P/'before_registration';snap.mkdir()
for name in ['AGENTS.md','STATUS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','jobs.json','experiments.json']:shutil.copy2(R/name,snap/name)
files=[dict(name=f.name,path=C['remote_root']+'/'+f.name,sha256=sha(f),bytes=f.stat().st_size)for f in sorted(S.iterdir())if f.is_file()and f.name!='FILE_BINDING.json']
write(S/'FILE_BINDING.json',dict(UTC=now,files=files,scope='R72 unchanged scientific method with exact computation optimization, independent checks and same50k cost admission'))
prefix='''# 当前执行：R74 精确损失实现已登记，保留全量50000终点

R73分项成本和精确W1人工候选已Git/D交付。R74只把重复索引W1反向改为精确区间长度解析梯度，并共享full_window/coordinate_RMS的边缘图/常量参考投影；九项定义、全6593SOURCE+21047REAL、全部参数VJP、初始化/随机流/学习率、50000终点及v5不变。

先CPU/CUDA人工检查，再50工程更新，在1/25/50步从相同权重/Adam/RNG额外对照原R72完整参数梯度及Adam。全部前瞻数值线通过且保守50k+尾检≤4小时则恢复初态直接全量；否则主训练0。无质量选参，不读取held-out，R65不恢复。长期目标active；主统筹/既有辅助、资源、单作业和E-D保持。

任务：tasks/R74_EXACT_LOSS_FULL_TRAIN_001.md；注册：reviews/R74_EXACT_LOSS_FULL_TRAIN_001/REGISTRATION.json。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:(R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
entry=dict(id=P.name,stage=P.name,kind='exact_computation_optimization_of_R72_complete_waveform',status='registered_not_started',defined_UTC=now,
  task='tasks/'+P.name+'.md',registration='reviews/'+P.name+'/REGISTRATION.json',remote_root=C['remote_root'],SOURCE_N=6593,REAL_N=21047,
  unique_main_updates=50000,engineering_updates=50,extra_same_start_reference_updates=3,cost_limit_seconds=14400,heldout_data_authorized=False,model_goal_complete=False)
for name,key in [('experiments.json','experiments'),('jobs.json','jobs')]:
    d=read(R/name);assert not any(v.get('stage',v.get('id'))==P.name for v in d[key]);d[key].append(entry.copy())
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);d.update(updated_UTC=now,status='R74_registered_not_started',current_stage=P.name,current_task=entry['task'],
      current_stage_registration=entry['registration'],current_job=None,current_activity='exact_loss_equivalence_then_same_full_endpoint_cost_gate',
      active_scientific_jobs=0,application_goal_status='active',model_goal_complete=False,next_action='Run single bound R74; CPU/CUDA and full-parent equivalence,50 timing updates, restored50k only if<=4h',
      latest_completed_backup='R73_DELIVERY_RECEIPT.json',latest_delivery_status='R73_private_Git_D_backup_complete')
    write(R/name,d)
review=dict(UTC=now,root_static_and_artificial_review_pass=True,unchanged_R72_modules=equal,unchanged_scientific_inputs_and_seeds=True,
  all_nine_terms_populations_and_fixed_endpoint_preserved=True,group_sum_backward_roundoff_not_bitwise_trajectory_claim=True,
  artificial_result=read(P/'ROOT_COMBINED_ARTIFICIAL.json')['result'],exact_W1_candidate_receipt='reviews/R73_EXACT_W1_THROUGHPUT_001/w1_candidate/ARTIFACT_COMPLETION_RECEIPT.json',
  representative_full_parent_checks=[1,25,50],prospective_tolerances={k:C[k]for k in allowed if k.startswith('equivalence_')},
  scientific_arrays_decoded=False,guard={**guard,**memory()})
write(P/'ROOT_IMPLEMENTATION_REVIEW.json',review)
task=R/entry['task'];local=[f for f in S.iterdir()if f.is_file()]+[task,P/'prepare_stage.py',P/'register_science.py',P/'control.py',P/'collect.py',P/'ROOT_IMPLEMENTATION_REVIEW.json']
write(P/'REGISTRATION.json',dict(UTC=now,status='REGISTERED_BEFORE_SCIENCE',stage=P.name,
  local_files=[dict(path=f.relative_to(R).as_posix(),sha256=sha(f),bytes=f.stat().st_size)for f in local],
  root_review='reviews/'+P.name+'/ROOT_IMPLEMENTATION_REVIEW.json',unique_main_updates=50000,engineering_updates=50,extra_reference_updates=3,current_acceptance_version=5))
print(json.dumps(dict(stage=P.name,registered=True,files=len(files),remote_root=C['remote_root'],scientific_arrays_decoded=False)),flush=True)
