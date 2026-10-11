"""Prospectively register a full-data endpoint selected solely from R76 cost."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
O=R/'work/R76_RETAINED_GRAPH_FULL_TRAIN_001';OP=R/'reviews/R76_RETAINED_GRAPH_FULL_TRAIN_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'registration_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not S.exists()and not(P/'REGISTRATION.json').exists()
assert read(R/'jobs.json')['active_scientific_jobs']==0
delivery=read(R/'R76_DELIVERY_RECEIPT.json');assert delivery['remote_main_SHA']==delivery['local_commit']and delivery['all_copied_file_SHA256_verified']
decision=read(OP/'NEXT_FIXED_ENDPOINT_DECISION.json');assert decision['selected_new_scheme_updates']==25000 and not decision['quality_metrics_consulted']
assert sha(O/'COST_ADMISSION.json')==decision['input_cost_SHA256']
assert read(OP/'ROOT_RESULT_REVIEW.json')['root_cost_evidence_review_pass']and read(O/'FULL_PARENT_EQUIVALENCE.json')['pass_all']
assert read(O/'RESULT.json')['main_updates']==0
S.mkdir()
for row in read(O/'FILE_BINDING.json')['files']:
    assert sha(O/row['name'])==row['sha256'];shutil.copy2(O/row['name'],S/row['name'])
old=read(S/'config.json');C=dict(old)
C.update(stage=P.name,remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R77_FULL_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'),
    updates=25000,endpoint='sole completed shape update25000, independently registered after R76 cost rejection; frozen R71 Q head update50000; no checkpoint selection',
    endpoint_selection='R76 observed setup+1.25*p90*updates+1800<=14400; largest5000 block; cost only, before new training',
    previous_cost_receipt_SHA256=decision['input_cost_SHA256'],previous_fixed50000_result='R76 cost rejected, main0, original protocol unchanged')
write(S/'config.json',C)
changed={k for k in C if C[k]!=old.get(k)}
assert changed=={'stage','remote_root','updates','endpoint','endpoint_selection','previous_cost_receipt_SHA256','previous_fixed50000_result'}
for name in ['control.py','collect.py']:
    code=(OP/name).read_text('utf8').replace('R76_','R77_')
    if name=='control.py':code=code.replace("artificial=result.get('artificial_checks')","artificial_pass_all=(result.get('artificial_checks')or{}).get('pass_all')")
    (P/name).write_text(code,'utf8')
task=f'''# R77 全部TRAIN人口的独立25000固定科学终点

用户要求数小时可完成时直接全量，否则按实际开销采用可执行固定方案。R76完整50工程更新p90={decision['observed_p90_seconds']:.12f}秒，原50000保守6.884707小时，数值对照通过但主训练0。依据其已封存成本，在4小时内取最大的5000整数步块，得到新方案25000，保守{decision['estimated_new_scheme_hours']:.9f}小时。本选择未使用训练质量或held-out。R72/R74/R76的原50000协议和成本负结果保持，25000不会标为完成原50000。

固定科学端点：从与R76相同的原始模型权重、冻结R71唯一50000 Q-head、空Adam和全部原随机流开始，完成唯一25000 shape更新。恢复检查点每5000步仅作进程恢复，不能按表现选择。LR仍500步线性warmup、相同起始/峰值/末值，之后余弦退火的总长度在本次执行前固定为25000；不得跑到中途再更换端点、人口或损失。

所有6593 SOURCE及21047 REAL每步完整参与，27640参数VJP，不删除父事件/支持外/失败区或重加权。九项定义、current canonical条件、四视图/投影/尺度、同处理REAL、SOURCE约束、模型架构、FP32网络/FP64物理、精确W1/条件VJP/保留图实现、RNG和输入常数与R76逐字节相同；全部生产.py文件直接绑定R76副本，无新算法变体。R71电荷头冻结与原TRAIN参照不变。该固定候选需之后按同一冻结方法完成v5 A–F；本训练本身不能证明缺区适用性或总体完成。

仍先CPU/CUDA人工完整检查，并按第1/25/50步同模型/Adam/RNG对原R72完整更新作数值对照（九值scaled<=1e-11、参数梯度相对L2<=1e-6、Adam参数maxabs<=1e-6）。10暖机+40计时后恢复同原初态、空Adam和所有RNG。只有本次实际setup+1.25*p90*25000+1800<=14400秒且资源/数值全部通过，才在同一作业直接运行唯一25000终点及完整TRAIN NumPy数值/序列化重放/响应描述。成本若再次拒绝保留主训练0，禁止本作业内再缩短。

主训练不读取held-out或评价挑选参数。最终4000 held-out独立重放、五视图HGB、PSD/峰宽/噪声、全部六门及D/E/F均按现行v5保留；所有失败按原条件记录。任何非有限、必需条件丢失、实际资源超限或数值失败终止并保留完整证据，不删除事件继续冒充通过。

服务器CPU0..11/RSS48GiB/GPU10GiB，单科学作业；本机共享CPU4/RSS4GiB，各控制/辅助1CPU/1GiB，新增文件仅E/D；旧R65停止。根负责科学方向/核验/调度/交付，既有gpt-6.1-sol/max辅助静态实现。所有代码/输入/协议/端点/资源在REGISTRATION和FILE/INPUT_BINDING中一次性事前绑定，不另设逐步审批。
'''
(R/'tasks'/(P.name+'.md')).write_text(task,'utf8');(S/'PROTOCOL.md').write_text(task,'utf8')
equal=[]
for f in S.glob('*.py'):
    ast.parse(f.read_text('utf8'));compile(f.read_text('utf8'),str(f),'exec');assert sha(f)==sha(O/f.name);equal.append(f.name)
for f in [P/'control.py',P/'collect.py']:ast.parse(f.read_text('utf8'));compile(f.read_text('utf8'),str(f),'exec')
assert sha(S/'INPUT_BINDING.json')==sha(O/'INPUT_BINDING.json')
assert C['SOURCE_N']==6593 and C['REAL_N']==21047 and C['neural_replay_chunk']==27640 and C['cost_limit_seconds']==14400
assert C['input_paths']==old['input_paths']and C['loss_weights']==old['loss_weights']and C['engineering_equivalence_updates']==[1,25,50]
current=read(R/'CURRENT_ACCEPTANCE.json');assert current['version']==5
for n,k in [('document','document_sha256'),('registry','registry_sha256')]:assert sha(R/current[n])==current[k]
now=datetime.now(timezone.utc).isoformat();snap=P/'before_registration';snap.mkdir()
for name in ['AGENTS.md','STATUS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','jobs.json','experiments.json']:shutil.copy2(R/name,snap/name)
files=[dict(name=f.name,path=C['remote_root']+'/'+f.name,sha256=sha(f),bytes=f.stat().st_size)for f in sorted(S.iterdir())if f.is_file()and f.name!='FILE_BINDING.json']
write(S/'FILE_BINDING.json',dict(UTC=now,files=files,scope='R76 identical production source and full-data population; new fixed25000 scientific endpoint selected only from saved cost'))
prefix='''# 当前执行：R77独立25000全数据方案已登记

R76原50000成本结果已封存并完成Git/D交付。R77根据其p90实测选择4小时内最大5000整数步块25000，保守约3.70小时，生产.py逐字节沿用R76；全部6593SOURCE+21047REAL、九项/权重/精度/公共响应与v5保持。新唯一终点与余弦LR总长前瞻固定，不能按质量改步数；原50000不追改。

先CPU/CUDA/三个原实现完整数值对照，再50工程更新。恢复原初态/空Adam/RNG后，本次实际保守25000总耗时<=4小时便直接完整训练和TRAIN尾检；否则主训练0。当前登记不等于训练开始或A–F通过。旧R65停止，长期目标active，根/既有辅助/资源/单作业/E-D保持。

任务：tasks/R77_FIXED25000_FULL_DATA_TRAIN_001.md；注册：reviews/R77_FIXED25000_FULL_DATA_TRAIN_001/REGISTRATION.json。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:(R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
entry=dict(id=P.name,stage=P.name,kind='fixed25000_full_data_scheme_after_observed_cost',status='registered_not_started',defined_UTC=now,
    task='tasks/'+P.name+'.md',registration='reviews/'+P.name+'/REGISTRATION.json',remote_root=C['remote_root'],SOURCE_N=6593,REAL_N=21047,
    unique_main_updates=25000,engineering_updates=50,extra_same_start_reference_updates=3,cost_limit_seconds=14400,heldout_data_authorized=False,model_goal_complete=False)
for name,key in [('experiments.json','experiments'),('jobs.json','jobs')]:
    d=read(R/name);assert not any(v.get('stage',v.get('id'))==P.name for v in d[key]);d[key].append(entry.copy())
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);d.update(updated_UTC=now,status='R77_fixed25000_registered_not_started',current_stage=P.name,current_task=entry['task'],
        current_stage_registration=entry['registration'],current_job=None,current_activity='full_parent_fixed25000_cost_gate_then_unique_endpoint',
        active_scientific_jobs=0,application_goal_status='active',model_goal_complete=False,next_action='Run bound R77; after fresh cost and numerical admission restore and directly complete full25000 endpoint',
        latest_completed_backup='R76_DELIVERY_RECEIPT.json',latest_delivery_status='R76_private_Git_D_backup_complete')
    write(R/name,d)
review=dict(UTC=now,root_static_review_pass=True,all_production_modules_identical_R76=equal,full_legal_population_unchanged=True,
    config_changes=sorted(changed),new_unique_endpoint=25000,cost_only_decision=decision,quality_selection=False,
    prior_CPU_CUDA_and_full_parent_equivalence_pass=True,new_execution_will_repeat_prospective_checks=True,
    scientific_arrays_decoded=False,guard={**guard,**memory()})
write(P/'ROOT_IMPLEMENTATION_REVIEW.json',review)
local=[f for f in S.iterdir()if f.is_file()]+[R/entry['task'],P/'prepare_register.py',P/'control.py',P/'collect.py',P/'ROOT_IMPLEMENTATION_REVIEW.json']
write(P/'REGISTRATION.json',dict(UTC=now,status='REGISTERED_BEFORE_SCIENCE',stage=P.name,
    local_files=[dict(path=f.relative_to(R).as_posix(),sha256=sha(f),bytes=f.stat().st_size)for f in local],
    root_review='reviews/'+P.name+'/ROOT_IMPLEMENTATION_REVIEW.json',unique_main_updates=25000,engineering_updates=50,extra_reference_updates=3,current_acceptance_version=5))
print(json.dumps(dict(stage=P.name,registered=True,files=len(files),unchanged_production_modules=len(equal),remote_root=C['remote_root'],new_fixed_updates=25000,scientific_arrays_decoded=False)),flush=True)
