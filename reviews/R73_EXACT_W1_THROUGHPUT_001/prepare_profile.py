"""Freeze one R72-equivalent component profile; no candidate quality experiment."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
O=R/'work/R72_FULL_WAVEFORM_FULL_TRAIN_001';OP=R/'reviews/R72_FULL_WAVEFORM_FULL_TRAIN_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'preparation_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not S.exists();S.mkdir()
assert read(R/'jobs.json')['active_scientific_jobs']==0
assert read(OP/'ROOT_RESULT_REVIEW.json')['root_cost_evidence_review_pass']
for row in read(O/'FILE_BINDING.json')['files']:
    assert sha(O/row['name'])==row['sha256']
    shutil.copy2(O/row['name'],S/row['name'])
C=read(S/'config.json');C.update(stage=P.name,remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R73_PROFILE_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'),
   allowed_scientific_entries=['train_profile.py'],GPU_scientific_entries=['train_profile.py'],
   profile_warmup_updates=10,profile_timed_updates=10,profile_only=True,prior_R72_remote=read(O/'config.json')['remote_root'])
write(S/'config.json',C)
base=(O/'train_full.py').read_text('utf8')
core=base.split('    opt=optimizer_new();timings=[];bench=')[0]
core=core.replace('    def update(opt,step):\n', '''    def update(opt,step):
        torch.cuda.synchronize();mark_time=time.perf_counter();profile={}
        def tick(label):
            nonlocal mark_time
            torch.cuda.synchronize();now=time.perf_counter();profile[label]=now-mark_time;mark_time=now
''')
core=core.replace("        first=[[],[],[]];max_q64=0.;max_q32=0.;projection_n=0", "        tick('RNG_and_learning_rate')\n        first=[[],[],[]];max_q64=0.;max_q32=0.;projection_n=0")
core=core.replace('        # Six independent leaves', "        tick('neural_physical_firstpass')\n        # Six independent leaves")
core=core.replace('        for name,term,_ in objective.iter_terms(*leaves,*[ds,dt]):', "        tick('endpoint_leaves_setup')\n        iterator=iter(objective.iter_terms(*leaves,*[ds,dt]))\n        for expected_name in names:\n            name,term,_=next(iterator);assert name==expected_name\n            tick('forward_'+name)")
core=core.replace('            gradient=torch.autograd.grad(term,leaves,allow_unused=True)', "            gradient=torch.autograd.grad(term,leaves,allow_unused=True)\n            tick('backward_'+name)")
core=core.replace('            values[name]=term.detach();del gradient,term', "            values[name]=term.detach();del gradient,term\n            tick('accumulate_'+name)")
core=core.replace('        parameter_gradient=[torch.zeros_like(p,dtype=torch.float64)for p in params]', "        tick('endpoint_gradient_concatenation')\n        parameter_gradient=[torch.zeros_like(p,dtype=torch.float64)for p in params]")
core=core.replace('        for p,g in zip(params,parameter_gradient):p.grad=g.to(p.dtype)', "        tick('complete_neural_parameter_VJP')\n        for p,g in zip(params,parameter_gradient):p.grad=g.to(p.dtype)")
core=core.replace("        return dict(total=sum(values.values()),**values,gradient_norm=norm.detach(),learning_rate=lr,", "        tick('clip_and_Adam')\n        return dict(profile=profile,total=sum(values.values()),**values,gradient_norm=norm.detach(),learning_rate=lr,")
tail='''    opt=optimizer_new();rows=[];warm=C['profile_warmup_updates'];timed=C['profile_timed_updates']
    progress('complete_parent_component_profile',main_updates=0)
    for i in range(1,warm+timed+1):
        torch.cuda.synchronize();start=time.perf_counter();row=update(opt,i);torch.cuda.synchronize();elapsed=time.perf_counter()-start
        rows.append(dict(engineering_update=i,warmup=i<=warm,seconds=elapsed,components=row['profile']))
        if i%5==0:progress('complete_parent_component_profile',engineering_updates=i,main_updates=0,last_seconds=elapsed)
    del opt,row
    model.load_state_dict(initial);model.zero_grad(set_to_none=True);torch.set_rng_state(rngcpu);torch.cuda.set_rng_state_all(rnggpu)
    source_rng.set_state(rngs);real_rng.set_state(rngr);directions_rng.bit_generator.state=copy.deepcopy(direction_initial);temporal_rng.bit_generator.state=copy.deepcopy(temporal_initial)
    opt=optimizer_new();assert state_hash(model.state_dict())==init_sha and not opt.state
    assert torch.equal(source_rng.get_state(),rngs)and torch.equal(real_rng.get_state(),rngr)
    assert torch.equal(torch.get_rng_state(),rngcpu)and all(torch.equal(a,b)for a,b in zip(torch.cuda.get_rng_state_all(),rnggpu))
    measured=rows[warm:];keys=list(measured[0]['components'])
    receipt=dict(UTC=utc(),scope='R72 exact complete-parent update with synchronized component boundaries; engineering only',
      engineering_updates=warm+timed,main_updates=0,warmup=warm,timed=timed,rows=rows,
      component_median_seconds={k:float(np.median([r['components'][k]for r in measured]))for k in keys},
      median_seconds=float(np.median([r['seconds']for r in measured])),p90_seconds=float(np.quantile([r['seconds']for r in measured],.9)),
      SOURCE_N=nS,REAL_N=nR,all_VJP_N=N,quality_metrics_recorded_or_used=False,
      initial_state_SHA256=init_sha,restored_state_SHA256=state_hash(model.state_dict()),all_RNG_restored=True,optimizer_empty=True,
      synchronization_changes_timing_only=True,instrumented_cost_not_direct_full_admission=True,model_goal_complete=False)
    write('COMPONENT_PROFILE.json',receipt)
    write('RESULT.json',dict(status='COMPLETE_PARENT_COMPONENT_PROFILE_COMPLETE',engineering_updates=warm+timed,main_updates=0,model_goal_complete=False))
    progress('complete_parent_component_profile_complete',engineering_updates=warm+timed,main_updates=0)
try:main()
except BaseException as exc:
    write('FAILURE.json',dict(UTC=utc(),error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-START,
      scientific_conclusion='engineering profile execution invalid; not an A-F negative',model_goal_complete=False));raise
'''
(S/'train_profile.py').write_text(core+tail,'utf8')
# Unchanged train_full.py is included only as a bound source reference, not an allowed entry.
for name in ['control.py','collect.py']:
    code=(OP/name).read_text('utf8').replace('R72_','R73_').replace('train_full.py','train_profile.py')
    (P/name).write_text(code,'utf8')
task='''# R73 完整更新分项计时

问题：R72 相同完整人口更新约 0.52 秒、保守 50000 步约 9.63 小时。一次同步分项 profile 判断计算耗时来源，供数学等价实现优化。仅工程诊断，不比较科学质量、不产生正式模型。

输入、R71 冻结 Q 头、482→128→128→480 网络、九项完整分布损失、原父事件与权重、初始化/随机流/学习率/Adam 均按 R72 修复后 FILE_BINDING 和 INPUT_BINDING 保持。全部 6593 SOURCE 与 21047 REAL 每步参与，全部 27640 参数反向。合法 TRAIN 范围；DEV/EVAL/CAL 数组读取 0。

固定 10 次暖机加 10 次计时更新，从 R72 相同初始状态和 RNG 开始。记录随机数、神经/物理前向、九项损失各自前向/反向/累积、完整参数 VJP 和 Adam 的同步耗时。同步边界只改变计时开销，不改变运算和科学对象。保留最后的权重恢复、空 Adam 和全部 RNG 恢复核验。20 步后停止，正式主训练 0；禁止由 profile 直接进入 50000 步。质量分数不保存或用于决策。

运行之前重做已冻结人工 CPU/CUDA 检查。数据规模、每项完整父人口、原始初态 SHA 和资源实际回执检查；仪器化耗时不能直接冒充未仪器化吞吐。等价候选另有人工值/梯度核验及新登记的完整成本门后才可进入主训练。v5 A–F、50000 科学终点和 4 小时判断不变。

服务器 CPU0..11/RSS48GiB/单GPU10GiB、一科学作业；本机共享CPU4/RSS4GiB，控制器1CPU1GiB、全部新增本地E/D。不改共享环境；不恢复 R65。主助手统筹，既有 gpt-6.1-sol/max 仅人工实现与静态核验。
'''
(R/'tasks'/(P.name+'.md')).write_text(task,'utf8');(S/'PROTOCOL.md').write_text(task,'utf8')
for f in [*S.glob('*.py'),P/'control.py',P/'collect.py']:ast.parse(f.read_text('utf8'));compile(f.read_text('utf8'),str(f),'exec')
write(P/'PREPARATION_RECEIPT.json',dict(UTC=datetime.now(timezone.utc).isoformat(),scope='static profile preparation',
  original_R72_train_SHA256=sha(O/'train_full.py'),profile_SHA256=sha(S/'train_profile.py'),CPU_CUDA_artificial_unmodified=True,
  candidate_W1_not_installed=True,main_training_not_authorized_by_this_profile=True,scientific_arrays_decoded=False,guard={**guard,**memory()}))
print(json.dumps(dict(stage=P.name,prepared=True,scientific_arrays_decoded=False,remote_root=C['remote_root'])),flush=True)
