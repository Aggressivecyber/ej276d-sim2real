"""Review R74 cost and bind one component/operator profile of its unchanged update."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
O=R/'work/R74_EXACT_LOSS_FULL_TRAIN_001';OP=R/'reviews/R74_EXACT_LOSS_FULL_TRAIN_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'preparation_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not S.exists()and not(P/'REGISTRATION.json').exists();S.mkdir()
assert read(R/'jobs.json')['active_scientific_jobs']==0
cost=read(O/'COST_ADMISSION.json');eq=read(O/'FULL_PARENT_EQUIVALENCE.json');native=read(O/'train_full.py.NATIVE_RECEIPT.json')
assert native['actual_worker_native_exit']==native['worker_self_receipt']['native_exit']==0
assert read(O/'RESULT.json')['main_updates']==0 and not cost['pass_gate']and eq['pass_all']
assert cost['engineering_updates']==50 and len(eq['checks'])==3 and all(j['pass_gate']for j in eq['checks'])
assert cost['restored_state_SHA256']==cost['initial_state_SHA256']and cost['all_RNG_restored']and cost['optimizer_empty']
now=datetime.now(timezone.utc).isoformat()
review=dict(UTC=now,root_cost_evidence_review_pass=True,main_updates=0,engineering_candidate_updates=50,extra_original_reference_updates=3,
  native_exit=0,median_seconds=cost['median_update_seconds'],p90_seconds=cost['p90_update_seconds'],estimated_hours=cost['estimate_full_run_seconds']/3600,
  full_parent_equivalence=eq,CPU_CUDA_artificial_pass=read(O/'ARTIFICIAL_CHECKS.json')['pass_all'],initial_state_optimizer_and_rng_restored=True,
  RSS_peak_bytes=native['actual_worker_RSS_peak_bytes'],GPU_peak_bytes=native['GPU_process_memory_peak_bytes'],
  RSS_probe_failures=native['memory_probe_failure_events'],no_A_F_quality_verdict=True,model_goal_complete=False,
  next_action='Single unchanged R74 component and operator profile; test exact local conditional VJP accumulation without repeated full-array scatter',guard={**guard,**memory()})
assert not(OP/'ROOT_RESULT_REVIEW.json').exists();write(OP/'ROOT_RESULT_REVIEW.json',review)
(R/'R74_REPORT.md').write_text(f'''# R74 精确W1反向与共享边缘图：成本仍不准入

CPU/CUDA人工核验全部通过；实际全部6593SOURCE+21047REAL在工程1/25/50步与同模型/Adam/RNG的原R72完整更新对照，九项值、完整参数梯度、Adam更新的观察差异均0。50次候选工程更新加3次原实现对照，原生exit0，主训练0；初态/空Adam/全部RNG恢复。

候选median {cost['median_update_seconds']:.9f}秒、p90 {cost['p90_update_seconds']:.9f}秒；相对R72 median减少 {(1-cost['median_update_seconds']/read(R/'work/R72_FULL_WAVEFORM_FULL_TRAIN_001/COST_ADMISSION.json')['median_update_seconds']):.2%}。相同50000步+尾检保守估计{cost['estimate_full_run_seconds']/3600:.6f}小时，高于4小时。精确计算优化有实际吞吐收益，完整响应质量仍未评价。数据人口、九项定义、50000唯一终点和v5保持。

下一步一次固定分项/算子profile定位剩余计算成本，验证局部条件梯度回填的等价实现。失败不会触发删除父事件或改变验收；R65不恢复。实际Git/D交付另见R74_DELIVERY_RECEIPT.json。
''','utf8')
for row in read(O/'FILE_BINDING.json')['files']:
    assert sha(O/row['name'])==row['sha256'];shutil.copy2(O/row['name'],S/row['name'])
C=read(S/'config.json');C.update(stage=P.name,remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R75_PROFILE_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'),
   allowed_scientific_entries=['train_profile.py'],GPU_scientific_entries=['train_profile.py'],profile_warmup_updates=10,profile_timed_updates=10,profile_operator_updates=1,profile_only=True)
write(S/'config.json',C)
core=(O/'train_full.py').read_text('utf8').split('    opt=optimizer_new();timings=[];bench=')[0]
core=core.replace('    def update(opt,step,use_original=False,capture=False):\n', '''    def update(opt,step,use_original=False,capture=False):
        torch.cuda.synchronize();mark_time=time.perf_counter();profile={}
        def tick(label):
            nonlocal mark_time
            torch.cuda.synchronize();now=time.perf_counter();profile[label]=now-mark_time;mark_time=now
''')
core=core.replace('        first=[[],[],[]];max_q64=0.;max_q32=0.;projection_n=0',"        tick('RNG_and_learning_rate')\n        first=[[],[],[]];max_q64=0.;max_q32=0.;projection_n=0")
core=core.replace('        # Six independent leaves',"        tick('neural_physical_firstpass')\n        # Six independent leaves")
core=core.replace('        if use_original:\n',"        tick('endpoint_leaves_setup')\n        if use_original:\n")
core=core.replace('        for group_names,term,terms,_ in groups:\n',"        for group_names,term,terms,_ in groups:\n            label='+'.join(group_names);tick('forward_'+label)\n")
core=core.replace('            gradient=torch.autograd.grad(term,leaves,allow_unused=True)',"            gradient=torch.autograd.grad(term,leaves,allow_unused=True)\n            tick('backward_'+label)")
core=core.replace('            values.update({name:v.detach()for name,v in terms.items()});del gradient,term',"            values.update({name:v.detach()for name,v in terms.items()});del gradient,term\n            tick('accumulate_'+label)")
core=core.replace('        parameter_gradient=[torch.zeros_like(p,dtype=torch.float64)for p in params]',"        tick('endpoint_gradient_concatenation')\n        parameter_gradient=[torch.zeros_like(p,dtype=torch.float64)for p in params]")
core=core.replace('        for p,g in zip(params,parameter_gradient):p.grad=g.to(p.dtype)',"        tick('complete_neural_parameter_VJP')\n        for p,g in zip(params,parameter_gradient):p.grad=g.to(p.dtype)")
core=core.replace('        result=dict(total=sum(values.values()),**values,gradient_norm=norm.detach(),learning_rate=lr,',"        tick('clip_and_Adam')\n        result=dict(profile=profile,total=sum(values.values()),**values,gradient_norm=norm.detach(),learning_rate=lr,")
tail=(R/'work/R73_EXACT_W1_THROUGHPUT_001/train_profile.py').read_text('utf8').split("    opt=optimizer_new();rows=[];warm=C['profile_warmup_updates']")[1]
tail="    opt=optimizer_new();rows=[];warm=C['profile_warmup_updates']"+tail
insert='''    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=False,with_stack=False)as profiler:
        operator_row=update(opt,warm+timed+1);torch.cuda.synchronize()
    operators=[]
    for event in profiler.key_averages(group_by_input_shape=True):
        operators.append(dict(name=event.key,count=event.count,input_shapes=event.input_shapes,
            self_cpu_us=event.self_cpu_time_total,cpu_total_us=event.cpu_time_total,
            self_device_us=getattr(event,'self_device_time_total',0.),device_total_us=getattr(event,'device_time_total',0.)))
    write('OPERATOR_PROFILE.json',dict(UTC=utc(),engineering_update=warm+timed+1,main_updates=0,record_shapes=True,profile_memory=False,
       additional_instrumentation_overhead_not_cost_forecast=True,CUDA_times_present=any(v['self_device_us']>0 for v in operators),operators=operators))
    del profiler,operator_row
'''
tail=tail.replace('    del opt,row\n',insert+'    del opt,row\n')
tail=tail.replace('engineering_updates=warm+timed,','engineering_updates=warm+timed+1,operator_updates=1,')
tail=tail.replace("scope='R72 exact complete-parent update", "scope='R74 exact complete-parent update")
(S/'train_profile.py').write_text(core+tail,'utf8')
for name in ['control.py','collect.py']:(P/name).write_text((OP/name).read_text('utf8').replace('R74_','R75_').replace('train_full.py','train_profile.py'),'utf8')
task='''# R75 R74 同计算分项与算子工程 profile

R74已完成50工程候选+3同初态原实现对照，值/参数梯度/Adam差0，但全量50k保守7.47小时仍未准入。主训练0。现对同一R74实现、全6593SOURCE+21047REAL和完整参数VJP作一次工程profile，不改任何科学定义/模型/输入/随机流/学习率。

固定10暖机+10同步分项计时，再追加第21步torch CPU/CUDA算子profile（record_shapes，不保存科学值或波形）。21步全部是丢弃的工程更新，之后恢复R72相同初始权重、空Adam和全部RNG。记录前向/反向/累积与算子总/自身耗时；profiling开销不能当未仪器化成本预测。不进入50k、没有质量选择、held-out读取0。

优先验证剩余成本是否来自按bin取样反向的重复全矩阵散射。人工候选只允许精确局部条件梯度和唯一父索引回填；必须CPU/CUDA及实际全人口值/梯度/Adam核验，新登记成本门后才可正式训练。九损失、人口和v5 A–F保持。服务器CPU0..11/RSS48GiB/GPU10GiB、单科学作业；本机共享CPU4/RSS4GiB，控制器/辅助各1CPU1GiB、新本地文件E/D。R65停止。
'''
(R/'tasks'/(P.name+'.md')).write_text(task,'utf8');(S/'PROTOCOL.md').write_text(task,'utf8')
for f in [*S.glob('*.py'),P/'control.py',P/'collect.py']:ast.parse(f.read_text('utf8'));compile(f.read_text('utf8'),str(f),'exec')
files=[dict(name=f.name,path=C['remote_root']+'/'+f.name,sha256=sha(f),bytes=f.stat().st_size)for f in sorted(S.iterdir())if f.is_file()and f.name!='FILE_BINDING.json']
write(S/'FILE_BINDING.json',dict(UTC=now,files=files,scope='unchanged R74 computation, timing boundaries and fixed21 engineering endpoint'))
entry=dict(stage=P.name,id=P.name,status='registered_not_started',task='tasks/'+P.name+'.md',registration='reviews/'+P.name+'/REGISTRATION.json',remote_root=C['remote_root'],
  engineering_updates=21,main_updates=0,heldout_data_authorized=False,model_goal_complete=False)
for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
    d=read(R/name);old_entry=next(j for j in d[key]if j.get('stage',j.get('id'))==OP.name)
    old_entry.update(status='COST_TOO_HIGH_FULL_MAIN_NOT_STARTED',root_review='reviews/'+OP.name+'/ROOT_RESULT_REVIEW.json',engineering_updates=50,extra_reference_updates=3,main_updates=0,native_exit=0)
    assert not any(j.get('stage',j.get('id'))==P.name for j in d[key]);d[key].append(entry.copy())
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);d.update(updated_UTC=now,status='R75_operator_profile_registered',current_stage=P.name,current_task=entry['task'],current_stage_registration=entry['registration'],
      latest_completed_stage=OP.name,latest_completed_report='R74_REPORT.md',latest_completed_review='reviews/'+OP.name+'/ROOT_RESULT_REVIEW.json',current_job=None,
      active_scientific_jobs=0,application_goal_status='active',model_goal_complete=False,next_action='Run one fixed21 engineering component/operator profile of unchanged R74 computation')
    write(R/name,d)
prefix='''# 当前执行：R74精确优化成本关闭，R75单次算子profile已登记

R74 CPU/CUDA和全人口1/25/50步的值/参数梯度/Adam均通过（观察差0）；完整步median0.39644秒，较R72减少24.1%，50k保守7.47小时>4，主训练0。现R75固定10暖机+10计时+1算子profile，只定位同一实现剩余开销。全部工程状态恢复，不进入主训练或held-out；v5目标active，原人口/九项/50k/4小时判断不改。

任务：tasks/R75_CONDITIONAL_VJP_OPTIMIZATION_001.md。精确局部条件梯度人工候选另行核验；主助手与既有辅助、资源/单作业/E-D/R65停止保持。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:(R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
task_path=R/entry['task'];local=[f for f in S.iterdir()if f.is_file()]+[P/'prepare_register_profile.py',P/'control.py',P/'collect.py',task_path]
write(P/'REGISTRATION.json',dict(UTC=now,status='REGISTERED_BEFORE_SCIENCE',stage=P.name,fixed_engineering_updates=21,main_updates=0,
  local_files=[dict(path=f.relative_to(R).as_posix(),sha256=sha(f),bytes=f.stat().st_size)for f in local],current_acceptance_version=5,profile_diff_review='timing-only wrapper built from unchanged R74 update; CPU/CUDA tests retained'))
print(json.dumps(dict(registered=True,stage=P.name,files=len(files),remote_root=C['remote_root'],engineering_updates=21,main_updates=0)),flush=True)
