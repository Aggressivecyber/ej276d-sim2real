"""Prepare the single exact-computation optimization and same fixed R72 endpoint."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
O=R/'work/R72_FULL_WAVEFORM_FULL_TRAIN_001';OP=R/'reviews/R72_FULL_WAVEFORM_FULL_TRAIN_001'
W=R/'reviews/R73_EXACT_W1_THROUGHPUT_001/w1_candidate'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'preparation_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not S.exists();S.mkdir()
assert read(R/'jobs.json')['active_scientific_jobs']==0
assert read(P/'ROOT_COMBINED_ARTIFICIAL.json')['pass_all']
for row in read(O/'FILE_BINDING.json')['files']:
    assert sha(O/row['name'])==row['sha256'];shutil.copy2(O/row['name'],S/row['name'])
for f in [P/'combined_loss.py',P/'check_combined_loss.py',W/'exact_uniform_w1.py',W/'check_exact_uniform_w1.py']:shutil.copy2(f,S/f.name)
C=read(S/'config.json');C.update(stage=P.name,remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R74_FULL_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'),
    implementation='R72 nine unchanged laws; exact rational W1 analytical backward; shared full_window/coordinate_RMS marginal graph',
    engineering_equivalence_updates=[1,25,50],equivalence_value_scaled_atol=1e-11,equivalence_parameter_gradient_relative_L2=1e-6,
    equivalence_Adam_parameter_maxabs=1e-6,original_R72_source_SHA256=sha(O/'train_full.py'),
    original_R72_cost_p90_seconds=read(O/'COST_ADMISSION.json')['p90_update_seconds'])
write(S/'config.json',C)
code=(O/'train_full.py').read_text('utf8')
code=code.replace('    from full_waveform_loss import FullWaveformLoss', '''    from combined_loss import FullWaveformLoss
    from full_waveform_loss import FullWaveformLoss as OriginalFullWaveformLoss,UniformW1Cache as OriginalW1Cache
    from check_exact_uniform_w1 import run_checks as check_exact_w1
    from check_combined_loss import run as check_combined
''')
code=code.replace("    checks=run_checks(C['artificial_seed']);write('ARTIFICIAL_CHECKS.json',checks);assert checks['pass_all']", '''    checks=run_checks(C['artificial_seed']);assert checks['pass_all']
    checks['exact_W1']={device:check_exact_w1(device,original_source=R/'full_waveform_loss.py')for device in ['cpu','cuda']}
    checks['combined']=check_combined(C['artificial_seed'],require_cuda=True)
    assert all(value['pass_all']for value in checks['exact_W1'].values())and checks['combined']['pass_all']
    write('ARTIFICIAL_CHECKS.json',checks)
''')
code=code.replace("    initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};init_sha=state_hash(initial)", '''    reference_objective=copy.copy(objective);reference_objective.__class__=OriginalFullWaveformLoss
    reference_objective.w1=OriginalW1Cache(objective.device)
    initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};init_sha=state_hash(initial)
''')
code=code.replace('    def update(opt,step):','    def update(opt,step,use_original=False,capture=False):')
old='''        for name,term,_ in objective.iter_terms(*leaves,*[ds,dt]):
            assert name in names and name not in values
            if not bool(torch.isfinite(term)):raise FloatingPointError('nonfinite loss '+name)
            gradient=torch.autograd.grad(term,leaves,allow_unused=True)
            for j,g in enumerate(gradient):
                if g is not None:accum[j].add_(g.detach().to(torch.float64))
            values[name]=term.detach();del gradient,term'''
new='''        if use_original:
            groups=(((name,),term,{name:term},diag)for name,term,diag in reference_objective.iter_terms(*leaves,ds,dt))
        else:groups=objective.iter_gradient_groups(*leaves,ds,dt)
        for group_names,term,terms,_ in groups:
            assert all(name in names and name not in values for name in group_names)
            if not bool(torch.isfinite(term)):raise FloatingPointError('nonfinite loss '+str(group_names))
            gradient=torch.autograd.grad(term,leaves,allow_unused=True)
            for j,g in enumerate(gradient):
                if g is not None:accum[j].add_(g.detach().to(torch.float64))
            values.update({name:v.detach()for name,v in terms.items()});del gradient,term'''
assert old in code;code=code.replace(old,new)
code=code.replace("        norm=torch.nn.utils.clip_grad_norm_(params,C['gradient_clip'],error_if_nonfinite=True);opt.step()", "        captured_gradient=torch.cat([p.grad.detach().flatten().to(torch.float64)for p in params]).cpu()if capture else None\n        norm=torch.nn.utils.clip_grad_norm_(params,C['gradient_clip'],error_if_nonfinite=True);opt.step()")
code=code.replace('        return dict(total=sum(values.values()),**values,gradient_norm=norm.detach(),learning_rate=lr,','        result=dict(total=sum(values.values()),**values,gradient_norm=norm.detach(),learning_rate=lr,')
code=code.replace('            firstpass_rows=N,replay_rows=N,replay_chunks=replay_chunks)\n', '''            firstpass_rows=N,replay_rows=N,replay_chunks=replay_chunks)
        if capture:result['capture']=dict(parameter_gradient=captured_gradient,after_parameters=torch.cat([p.detach().flatten().to(torch.float64)for p in params]).cpu())
        return result
''')
oldloop='''    for i in range(1,bench+1):
        torch.cuda.synchronize();t=time.perf_counter();row=update(opt,i);torch.cuda.synchronize();elapsed=time.perf_counter()-t
        if i>C['benchmark_warmup_updates']:timings.append(elapsed)
        if i==1 or i%5==0:progress('engineering_complete_parent_update_timing',engineering_updates=i,main_updates=0,last_update_seconds=elapsed)'''
newloop='''    equivalence=[]
    def save_update_start():
        return dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),cpu=torch.get_rng_state().clone(),
            gpu=[v.clone()for v in torch.cuda.get_rng_state_all()],source=source_rng.get_state().clone(),real=real_rng.get_state().clone(),
            directions=copy.deepcopy(directions_rng.bit_generator.state),temporal=copy.deepcopy(temporal_rng.bit_generator.state))
    def restore_update_start(state):
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);model.zero_grad(set_to_none=True)
        torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['gpu']);source_rng.set_state(state['source']);real_rng.set_state(state['real'])
        directions_rng.bit_generator.state=copy.deepcopy(state['directions']);temporal_rng.bit_generator.state=copy.deepcopy(state['temporal'])
    for i in range(1,bench+1):
        verify=i in C['engineering_equivalence_updates']
        if verify:
            same_start=save_update_start();reference_row=update(opt,i,use_original=True,capture=True);restore_update_start(same_start)
        torch.cuda.synchronize();t=time.perf_counter();row=update(opt,i,capture=verify);torch.cuda.synchronize();elapsed=time.perf_counter()-t
        if i>C['benchmark_warmup_updates']:timings.append(elapsed)
        if verify:
            scaled_values={name:abs(float(row[name])-float(reference_row[name]))/max(abs(float(reference_row[name])),1.)for name in names}
            a,b=reference_row['capture']['parameter_gradient'],row['capture']['parameter_gradient']
            relative=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-30));absolute=float((a-b).abs().max())
            weight_error=float((reference_row['capture']['after_parameters']-row['capture']['after_parameters']).abs().max())
            item=dict(update=i,nine_values_scaled_errors=scaled_values,gradient_relative_L2=relative,gradient_maxabs=absolute,
                Adam_parameter_maxabs=weight_error,same_model_optimizer_rng_and_full_parents=True,quality_selection=False,
                pass_gate=max(scaled_values.values())<=C['equivalence_value_scaled_atol']and relative<=C['equivalence_parameter_gradient_relative_L2']and weight_error<=C['equivalence_Adam_parameter_maxabs'])
            equivalence.append(item);write('FULL_PARENT_EQUIVALENCE.json',dict(checks=equivalence,pass_all=all(v['pass_gate']for v in equivalence),main_updates=0))
            assert item['pass_gate'],item
            del same_start,reference_row,a,b,row['capture']
        if i==1 or i%5==0:progress('engineering_complete_parent_update_timing',engineering_updates=i,main_updates=0,last_update_seconds=elapsed)
    assert [v['update']for v in equivalence]==C['engineering_equivalence_updates']
'''
assert oldloop in code;code=code.replace(oldloop,newloop)
(S/'train_full.py').write_text(code,'utf8')
for name in ['control.py','collect.py']:(P/name).write_text((OP/name).read_text('utf8').replace('R72_','R74_'),'utf8')
task='''# R74 相同完整目标的精确计算优化和 50000 唯一终点

依据：R73 已按同一 R72 计算作完整人口分项计时。全窗/逐坐标项前反向合计约 72%，神经+物理前向及完整参数 VJP 约 16%。本项只优化计算共享与精确有限均匀 W1 的一阶反向；不改模型、九项科学定义、人口、尺度、权重、分区、噪声或验收。

W1 前向仍按 R72 stable sort 和精确 LCM quantile cells 的原 float64 表达式逐字节计算。对排序后 a_i 的 quantile 区间，导数是该区间与 b<x 的质量区间交长减去与 b>x 的交长；等值贡献零。实际人口的 LCM/int64 长度先精确求差，再除以原分母。通过稳定排序的唯一逆索引 scatter 返回梯度，不进行重复索引浮点原子加法。两侧梯度、重复值、任意上游及 b_sorted 由独立 Fraction 单元枚举和旧实现核验。只支持本项一阶优化，不宣称高阶导数或长轨迹逐位相同。

full_window 与 coordinate_RMS 使用同一四视图、同一实际 canonical Q 分区和同一 marginal W1 向量；每项按原顺序/尺度分别算值，再对两项之和作一次反向。两域对同一冻结 REAL 投影排序复用一次。其余七项和六端点累积保持。每步仍为 6593 SOURCE+21047 REAL 及全部 27640 参数 VJP；不删除支持外、失败区、父事件或来源权重。

模型、初始化、种子、50k 学习率轨迹、Adam、Q头、常数、独立参考、尾检及 SOURCE/REAL 相同公共输入路径均继承 R72 注册后固定方案。50000 是唯一正式权重终点；不选恢复检查点，不用 held-out 选参。v5 A–F 和 held-out4000 完整重放仍须在同一冻结方法之后评价。TRAIN 检查不代替最终验收。

执行前先重做 R72 人工 CPU/CUDA 全套及新 W1/合并目标人工核验。CPU 已通过9值差0、六端点差<=5.69e-14、参数梯度及 Adam 差0的50行人工例。服务器工程更新仍固定10暖机+40计时；在第1/25/50步各从同一模型/Adam/RNG开始额外做一次原 R72 全人口更新作数值对照，随后恢复并做候选更新。比较九值 abs(diff)/max(1,abs(ref))<=1e-11，完整 float32 参数梯度相对L2<=1e-6，Adam 后参数最大绝对差<=1e-6；任一失败就终止且主训练0。只记录差异，不使用质量改善挑选实现。

额外原实现对照不计入候选步时；复制参数比较的候选开销保留在测时中，因此略保守。50次工程更新后恢复同一初始权重、空Adam和所有RNG。相同规则 `setup +1.25*p90*50000+1800 <=14400秒` 才在同一作业直接跑唯一50000正式终点；否则主训练0。所有工程模型丢弃。登记后不得改人口/损失/端点；本实现失败须保留证据后另行决定。

服务器CPU0..11/RSS48GiB/GPU10GiB，单科学作业；本机共享CPU4/RSS4GiB，辅助和控制器各1CPU/1GiB，全部新增本地文件E/D。原R65保持停止。根统筹/审查/交付，既有gpt-6.1-sol/max辅助。详细代码/输入/本协议一次性绑定于REGISTRATION及FILE/INPUT_BINDING，无逐步再次审批。
'''
(R/'tasks'/(P.name+'.md')).write_text(task,'utf8');(S/'PROTOCOL.md').write_text(task,'utf8')
for f in [*S.glob('*.py'),P/'control.py',P/'collect.py']:ast.parse(f.read_text('utf8'));compile(f.read_text('utf8'),str(f),'exec')
write(P/'PREPARATION_RECEIPT.json',dict(UTC=datetime.now(timezone.utc).isoformat(),stage=P.name,scientific_arrays_decoded=False,
  full_population_equivalence_updates=C['engineering_equivalence_updates'],updates=50000,engineering_updates=50,
  numerical_CPU_artificial=read(P/'ROOT_COMBINED_ARTIFICIAL.json'),guard={**guard,**memory()}))
print(json.dumps(dict(prepared=True,stage=P.name,remote_root=C['remote_root'],main_updates=50000,first_cost_gate_seconds=14400,scientific_arrays_decoded=False)),flush=True)
