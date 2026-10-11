"""Prospective R72 registration, before any scientific array decoding."""
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
assert read(R/'jobs.json')['active_scientific_jobs']==0
assert read(R/'PROGRAM.json')['application_goal_status']=='active'
assert read(P/'ROOT_JOINT_ARTIFICIAL.json')['pass_all']
assert read(P/'response_candidate/ARTIFICIAL_CHECK.json')['result']['pass_all']
assert read(P/'response_candidate/NATIVE_RESOURCE_RECEIPT.json')['native_exit_code']==0
loss=read(P/'loss_candidate/ARTIFICIAL_CHECK_RECEIPT.json');assert loss['status']=='PASS_ARTIFICIAL_ONLY'
for name,digest in loss['source_SHA256'].items():assert sha(P/'loss_candidate'/name)==digest,name
reviewed={'response_candidate':['waveform_response.py','waveform_numpy_reference.py','check_waveform_response.py'],
          'loss_candidate':['full_waveform_loss.py','check_full_waveform_loss.py']}
response_manifest={row['path']:row['sha256']for row in read(P/'response_candidate/SHA256_MANIFEST.json')['artifacts']}
for name in reviewed['response_candidate']:assert sha(P/'response_candidate'/name)==response_manifest[name],name
for folder,names in reviewed.items():
    for name in names:shutil.copy2(P/folder/name,S/name)
unchanged=[]
old=R/'work/R71_REAL_CHARGE_FULL_TRAIN_001'
for name in ['public_charge_head.py','physical_operators.py','common_response.py','numpy_reference.py','linux_owned_worker.py']:
    assert ast.dump(ast.parse((S/name).read_text('utf8')))==ast.dump(ast.parse((old/name).read_text('utf8'))),name
    shutil.copy2(old/name,S/name);unchanged.append(dict(name=name,sha256=sha(S/name),byte_equal_R71=True))
for f in [*S.glob('*.py'),P/'control.py',P/'collect.py',P/'register_science.py']:
    ast.parse(f.read_text('utf8'));compile(f.read_text('utf8'),str(f),'exec')
current=read(R/'CURRENT_ACCEPTANCE.json');assert current['version']==5
for name,key in [('document','document_sha256'),('registry','registry_sha256')]:assert sha(R/current[name])==current[key]
for name in ['CURRENT_ACCEPTANCE.json',current['document'],current['registry']]:shutil.copy2(R/name,S/name)
task=R/'tasks'/(P.name+'.md');shutil.copy2(task,S/'PROTOCOL.md')
C=read(S/'config.json');assert C['updates']==50000 and C['neural_replay_chunk']==27640 and C['hidden']==128
assert C['input_paths']['Q_head_weights']==read(old/'config.json')['remote_root']+'/FINAL_HEAD.pt'
now=datetime.now(timezone.utc).isoformat();snapshot=P/'before_registration';snapshot.mkdir()
for name in ['AGENTS.md','STATUS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','jobs.json','experiments.json']:
    shutil.copy2(R/name,snapshot/name)
files=[dict(name=f.name,path=C['remote_root']+'/'+f.name,sha256=sha(f),bytes=f.stat().st_size)
       for f in sorted(S.iterdir())if f.is_file()and f.name!='FILE_BINDING.json']
write(S/'FILE_BINDING.json',dict(UTC=now,files=files,scope='immutable scientific source/config/protocol; actual TRAIN and frozenQ inputs separately bound'))
prefix='''# 当前执行：R72 完整波形全人口方法已登记

冻结R71公共Q头，唯一482→128→128→480全480残差与九项完整人口目标已前瞻绑定。全部6593 SOURCE和21047 REAL每步均参与，按当前canonical输出Q重建条件，各父原权重/来源/谷/REAL配对保留；两遍包含全部父的参数反向。CPU人工模块、精确W1/顺序及完整NN梯度/Adam核验已通过，服务器先做CUDA人工检查。

先10暖机+40完整计时更新，再恢复初始状态/空Adam/全部随机流。若保守50000步加尾检≤4小时，同一作业直接跑唯一末点；否则主训练0步，封存成本证据，不缩人口或改端点。完整A–F和最终held-out4000仍未完成，TRAIN4000仅数值/序列化检查。R65不恢复。v5/资源/E-D保持，主助手统筹，辅助gpt-6.1-sol/max。

协议：tasks/R72_FULL_WAVEFORM_FULL_TRAIN_001.md；注册：reviews/R72_FULL_WAVEFORM_FULL_TRAIN_001/REGISTRATION.json。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:(R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
entry=dict(id=P.name,stage=P.name,kind='complete_parent_waveform_response_after_frozen_charge_head',status='registered_not_started',defined_UTC=now,
    task='tasks/'+P.name+'.md',registration='reviews/'+P.name+'/REGISTRATION.json',remote_root=C['remote_root'],SOURCE_N=6593,REAL_N=21047,
    unique_main_updates=50000,engineering_updates=50,cost_limit_seconds=14400,science_wall_cap_s=None,model_goal_complete=False,heldout_data_authorized=False)
for name,key in [('experiments.json','experiments'),('jobs.json','jobs')]:
    d=read(R/name);assert not any(v.get('id',v.get('stage'))==P.name for v in d[key]);d[key].append(entry.copy())
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);d.update(updated_UTC=now,status='R72_registered_not_started',current_stage=P.name,current_task=entry['task'],
      current_stage_registration=entry['registration'],numeric_execution_released=True,current_activity='registered_complete_parent_cost_then_full_endpoint',
      active_scientific_jobs=0,current_job=None,application_goal_status='active',model_goal_complete=False,
      next_action='Launch one bound R72 process; CUDA artificial pass before TRAIN, measured<=4h admits sole50000 endpoint')
    write(R/name,d)
review=dict(UTC=now,scope='Root static and actual artificial CPU review before science',unchanged_R71_operators=unchanged,
    common_path_and_frozenQ=True,all_parents_and_current_output_bins=True,all_original_source_order_denominator=True,
    independent_NumPy_full_reference=True,full_population_VJP_checked=True,source_h_original_float=True,
    corrected_before_registration=['root NumPy constants passed to Tensor-only public API','root final partial chunk stop bound','iter_terms three-value API'],
    engineered_permutation_claim='math/tolerance on artificial fixture; byte permutation failure retained, no universal integer-code assertion',
    numeric_and_original_order_replay_gates_unchanged=True,loss_CPU=loss['status'],root_joint=read(P/'ROOT_JOINT_ARTIFICIAL.json')['result'],
    artificial_test_code_wrapper_update='run_checks calls loss CPU/CUDA once with required compiled backend; joint calculation unchanged',
    science_arrays_decoded=False,guard={**guard,**memory()})
write(P/'ROOT_IMPLEMENTATION_REVIEW.json',review)
localfiles=[f for f in S.iterdir()if f.is_file()]+[task,P/'control.py',P/'collect.py',P/'register_science.py',R/'CURRENT_ACCEPTANCE.json',R/current['document'],R/current['registry']]
write(P/'REGISTRATION.json',dict(status='REGISTERED_BEFORE_SCIENCE',UTC=now,stage=P.name,science_arrays_read=False,current_acceptance_version=5,
    local_files=[dict(path=f.relative_to(R).as_posix(),sha256=sha(f),bytes=f.stat().st_size)for f in localfiles],
    root_review='reviews/'+P.name+'/ROOT_IMPLEMENTATION_REVIEW.json',user_authorized_full_if_hours=True))
print(json.dumps(dict(registered=P.name,files=len(files),remote_root=C['remote_root'],science_arrays_decoded=False,guard={**guard,**memory()})))
