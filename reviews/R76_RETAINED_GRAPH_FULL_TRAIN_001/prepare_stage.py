"""Prepare one same-law conditional VJP and retained neural graph candidate."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
O=R/'work/R74_EXACT_LOSS_FULL_TRAIN_001';OP=R/'reviews/R74_EXACT_LOSS_FULL_TRAIN_001'
W=R/'reviews/R75_CONDITIONAL_VJP_OPTIMIZATION_001/conditional_candidate'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'preparation_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not S.exists()
assert read(R/'jobs.json')['active_scientific_jobs']==0
assert read(P/'ROOT_RETAINED_GRAPH_ARTIFICIAL.json')['pass_all']
completion=read(W/'COMPLETION_RECEIPT.json')
assert completion['status']=='SEALED_CPU_ARTIFICIAL_PASS'and completion['native_artificial_exit']==0
assert sha(W/'SOURCE_MANIFEST.json')==completion['manifest_SHA256']
manifest=read(W/'SOURCE_MANIFEST.json')
for name,digest in manifest['artifact_SHA256'].items():assert sha(W/name)==digest,name
for name,digest in manifest['input_source_SHA256'].items():assert sha(Path(name))==digest,name
S.mkdir()
for row in read(O/'FILE_BINDING.json')['files']:
    assert sha(O/row['name'])==row['sha256'];shutil.copy2(O/row['name'],S/row['name'])
for f in [P/'check_retained_graph.py',W/'conditional_vjp_loss.py',W/'check_conditional_vjp.py']:shutil.copy2(f,S/f.name)
C=read(S/'config.json');C.update(stage=P.name,remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R76_FULL_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'),
    implementation='R72 nine unchanged laws; R74 exact W1; per-condition unique feature gradient writes; retained full neural/physical graph',
    forward_schedule='one full forward retained until complete endpoint-to-parameter VJP; no neural replay in candidate',
    loss_parameter_gradient='same nine exact full-population endpoint gradients in float64; complete retained graph parameter VJP, cast at original output and parameter dtype')
write(S/'config.json',C)
code=(O/'train_full.py').read_text('utf8')
def replace(old,new):
    global code
    assert old in code,old[:120]
    code=code.replace(old,new)
replace('    from combined_loss import FullWaveformLoss','    from conditional_vjp_loss import FullWaveformLoss\n    from check_conditional_vjp import run_checks as check_conditional\n    from check_retained_graph import run as check_retained')
replace("    write('ARTIFICIAL_CHECKS.json',checks)","    checks['conditional_VJP']={device:check_conditional(device,C['artificial_seed'])for device in ['cpu','cuda']}\n    checks['retained_graph']=check_retained(C['artificial_seed'],require_cuda=True)\n    assert all(v['pass_all']for v in checks['conditional_VJP'].values())and checks['retained_graph']['pass_all']\n    write('ARTIFICIAL_CHECKS.json',checks)")
replace('''        with torch.no_grad():
            for start in range(0,N,chunk):''','''        with (torch.no_grad() if use_original else torch.enable_grad()):
            for start in range(0,N,chunk):''')
replace('for j,v in enumerate([adc,pre,raw]):first[j].append(v.detach())','for j,v in enumerate([adc,pre,raw]):first[j].append(v.detach() if use_original else v)')
replace("float(meta['Q_relative_error_f64'].max())","float(meta['Q_relative_error_f64'].detach().max())")
replace("float(meta['Q_relative_error_f32'].max())","float(meta['Q_relative_error_f32'].detach().max())")
replace('# Six independent leaves avoid retaining the neural graph for full-population losses.','# Six detached leaves separate loss graphs from the retained endpoint-to-parameter graph.')
old='''        # Every row is replayed in the same forward chunks and stochastic draw.
        parameter_gradient=[torch.zeros_like(p,dtype=torch.float64)for p in params]
        replay_chunks=0
        for start in range(0,N,chunk):
            end=min(start+chunk,N)
            pre,adc,raw,meta=forward_prepared(model,slice_prepared(prepared,start,end),eps[start:end],straight_through=True)
            outputs=[adc,pre,raw]
            if not all(torch.equal(v.detach(),saved[j][start:end])for j,v in enumerate(outputs)):
                raise FloatingPointError('same-update complete-parent endpoint replay not bitwise')
            gradients=torch.autograd.grad(outputs,params,grad_outputs=[g[start:end].to(v.dtype)for g,v in zip(all_gradient,outputs)])
            for acc,g in zip(parameter_gradient,gradients):acc.add_(g.to(torch.float64))
            replay_chunks+=1
        for p,g in zip(params,parameter_gradient):p.grad=g.to(p.dtype)'''
new='''        replay_chunks=0
        if use_original:
            # The original R72 comparison retains its complete replay verbatim.
            parameter_gradient=[torch.zeros_like(p,dtype=torch.float64)for p in params]
            for start in range(0,N,chunk):
                end=min(start+chunk,N)
                pre,adc,raw,meta=forward_prepared(model,slice_prepared(prepared,start,end),eps[start:end],straight_through=True)
                outputs=[adc,pre,raw]
                if not all(torch.equal(v.detach(),saved[j][start:end])for j,v in enumerate(outputs)):
                    raise FloatingPointError('same-update complete-parent endpoint replay not bitwise')
                gradients=torch.autograd.grad(outputs,params,grad_outputs=[g[start:end].to(v.dtype)for g,v in zip(all_gradient,outputs)])
                for acc,g in zip(parameter_gradient,gradients):acc.add_(g.to(torch.float64))
                replay_chunks+=1
            for p,g in zip(params,parameter_gradient):p.grad=g.to(p.dtype)
        else:
            # All rows, outputs and upstream gradients are unchanged. Use the
            # first full forward graph once, avoiding a second identical forward.
            gradients=torch.autograd.grad(saved,params,grad_outputs=[g.to(v.dtype)for g,v in zip(all_gradient,saved)])
            for p,g in zip(params,gradients):p.grad=g.to(p.dtype)
        del saved,all_gradient,gradients'''
replace(old,new)
replace('firstpass_rows=N,replay_rows=N,replay_chunks=replay_chunks)','firstpass_rows=N,replay_rows=N if use_original else 0,replay_chunks=replay_chunks,parameter_VJP_rows=N,\n            retained_forward_graph=not use_original,neural_forward_calls=math.ceil(N/chunk)*(2 if use_original else 1))')
replace("complete_parent_per_update=True,heldout_arrays_read=False,model_goal_complete=False)","complete_parent_per_update=True,retained_first_forward_graph=True,neural_replay_rows=0,\n      neural_forward_calls=C['updates']*math.ceil(N/chunk),heldout_arrays_read=False,model_goal_complete=False)")
(S/'train_full.py').write_text(code,'utf8')
for name in ['control.py','collect.py']:(P/name).write_text((OP/name).read_text('utf8').replace('R74_','R76_'),'utf8')
task='''# R76 原完整目标的条件梯度唯一回填和单次神经前向

R74精确W1和共享两项计算通过CPU/CUDA及完整父事件1/25/50同状态数值对照，但50k保守7.47小时。R75固定算子profile观察到full_window/coordinate_RMS反向约0.156秒及重复全矩阵scatter/add/fill。此项只改变一阶计算调度，完整九项科学定义、全人口、同一公共响应、固定50000终点与4小时判断不变。

首组合并分布项沿R74同一四视图和current canonical Q分区计算逐条件精确W1。条件局部normalized features计算局部一阶导数，按原域/视图/条件权重除以原尺度后，向完整feature gradient每行唯一写入一次。四视图一起作一次canonical VJP，避免每个条件区对完整矩阵进行scatter累加。两域梯度分别缓存，再乘任意标量上游；两项标量按原求和顺序报告，其余七项继承不变。所有本项权重等于原1；不宣称高阶梯度或长训练轨迹逐位相同。

完整6593 SOURCE+21047 REAL的首次同一神经/物理前向保留图，六端点叶子仍detach独立计算九项梯度，float64累积后一次沿首次图对全部模型参数求VJP。全部27640父事件和噪声向量保留，未删参数反传。候选训练神经replay行数准确为0，完整参数VJP仍27640；旧R72数值对照保持两次前向及逐位endpoint replay检查。原始模型、初始化、LR轨迹、Adam、噪声和投影随机流、Q头、常数、特征/物理精度均沿用；推理/NumPy/BoxQ/ADC模块逐字节不变。保留图内存须满足原GPU10GiB上限。

先运行既有R72/R74 CPU和CUDA人工检查，再运行条件区梯度检查（九值、每组/六叶、网络梯度、Adam、tie、norm零、当前Q迁移、任意上游与规定失败）及保留图检查（全50和chunk13）。任一失败不能加载科学数组。之后仍10暖机+40完整更新计时，第1/25/50步从相同模型、Adam和所有RNG额外做原R72全人口更新作对照；九值scaled差<=1e-11，完整参数梯度相对L2<=1e-6，Adam参数maxabs<=1e-6，任一失败主训练0。候选真实计时包括比较步capture开销，无质量选参。

全部工程参数丢弃，恢复原初始权重/空Adam/所有RNG，按同一setup +1.25*p90*50000+1800<=14400秒判断。通过才在同一个作业直接完成唯一50000正式终点和原TRAIN数值/响应检查；不通过则主训练0，不截短原方案或改变父事件/评价人口。结果仅为工程成本或训练阶段，完整v5 A–F、固定任务效用、REAL-only FULL/GAP和4000独立重放仍须以后续冻结方法完整验证。

新来源/代码/协议在REGISTRATION与FILE/INPUT_BINDING中事前绑定。单科学作业、服务器CPU0..11/RSS48GiB/GPU10GiB、本机共享CPU4/RSS4GiB、根及辅助1CPU/1GiB和E/D新增存储保持。R65不恢复，旧负结果不改。主助手统筹/审核/交付，既有gpt-6.1-sol/max辅助。
'''
(R/'tasks'/(P.name+'.md')).write_text(task,'utf8');(S/'PROTOCOL.md').write_text(task,'utf8')
for f in [*S.glob('*.py'),P/'control.py',P/'collect.py']:ast.parse(f.read_text('utf8'));compile(f.read_text('utf8'),str(f),'exec')
write(P/'PREPARATION_RECEIPT.json',dict(UTC=datetime.now(timezone.utc).isoformat(),stage=P.name,scientific_arrays_decoded=False,
    ROOT_retained_CPU_artificial=read(P/'ROOT_RETAINED_GRAPH_ARTIFICIAL.json'),conditional_candidate_completion_SHA256=sha(W/'COMPLETION_RECEIPT.json'),
    updates=50000,engineering_updates=50,extra_original_updates=3,guard={**guard,**memory()}))
print(json.dumps(dict(prepared=True,stage=P.name,remote_root=C['remote_root'],main_updates=50000,scientific_arrays_decoded=False)),flush=True)
