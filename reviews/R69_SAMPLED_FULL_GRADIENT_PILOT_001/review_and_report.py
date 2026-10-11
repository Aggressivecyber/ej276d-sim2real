"""Independently recompute gates from saved scalars; never run or fit a model."""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction
import sys,json,hashlib,math,shutil
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R69_SAMPLED_FULL_GRADIENT_PILOT_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'report_runtime')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(x,y):return math.isclose(x,y,rel_tol=2e-12,abs_tol=1e-12)
C=read(S/'config.json');v=read(S/'RESULT.json');col=read(P/'COLLECTION_RECEIPT.json');native=col['native'];a=read(S/'GRADIENT_ADMISSION.json')
for row in read(P/'REGISTRATION.json')['local_files']:assert sha(R/row['path'])==row['sha256'],row['path']
assert sha(R/'PROJECT_ACCEPTANCE.json')=='b0e4c13302dc67b0eec0799e77fdbab603e29a293cb32bbc26fa0734f025ac9e'
assert sha(R/'PROJECT_ACCEPTANCE.md')=='d6f68a1e902d901bcceafaa1d23f97b1a8ff5ad9f52acf46b00b82d7a6bdb04d'
assert col['old_checkpoint_unchanged']and col['downloaded_and_reopened']and not col['GPU_processes']
assert read(S/'ENVIRONMENT.json')['CPU_affinity']==list(range(12))
assert native['elapsed_s']<=601 and native['job_tree_RSS_peak_bytes']<=C['RSS_limit_bytes'] and native['actual_worker_RSS_peak_bytes']<=C['RSS_limit_bytes']
assert native['GPU_process_memory_peak_bytes']<=C['GPU_limit_bytes']
assert v['original_A_F_complete']==v['original_R65_50000_complete']==v['automatic_long_resume']==False
restores=read(S/'COMMON_START_RESTORATION.json');assert all(r['state_hashes']==restores[0]['state_hashes']for r in restores)
assert read(S/'ARTIFICIAL_CHECK.json')['checks_passed']==35
M,k=a['original_block_count'],a['sampled_block_count'];assert(M,k)==(104,4)
assert len(a['block_squared_norms'])==M and close(math.fsum(a['block_squared_norms']),a['sum_block_squared_norm'])
F=Fraction.from_float
exact=Fraction(M*(M-k),k*(M-1))*(F(a['sum_block_squared_norm'])-F(a['D_squared_norm'])/M)
assert exact>=0 and close(float(exact),a['conditional_MSE_trace'])
assert a['pass_gate']==(float(exact)<=a['D_squared_norm']and float(exact)<=a['D_plus_C_squared_norm'])
assert a['model_and_Adam_unchanged']and a['no_optimizer_update']and a['new_response_updates']==0
assert a['source_parents']==6593 and a['REAL_parents']==21047
counts=a['physical_counts']
for key,value in {'neural_forward_rows':13570,'neural_forward_calls':209,'D_parameter_VJP_calls':104,'C_parameter_reverse_calls':1,'optimizer_step_calls':0,'gradient_clip_calls':0,'parameter_cast_calls':0,'endpoint_replay_checked_chunks':104}.items():assert counts[key]==value,(key,counts[key])
completed=v['status']in['POSITIVE_LOCAL_ONLY','NEGATIVE_LOCAL_SUPPORT']
if completed or v['status']=='NEGATIVE_GRADIENT_ADMISSION':
 assert native['actual_worker_native_exit']==0 and native['termination_reason']is None
 assert v['elapsed_s']<C['worker_cap_s']
 assert v['GPU_allocated_peak_bytes']<=C['GPU_limit_bytes']and v['GPU_reserved_peak_bytes']<=C['GPU_limit_bytes']
root=dict(UTC=datetime.now(timezone.utc).isoformat(),stage=S.name,result_status=v['status'],native_exit=native['actual_worker_native_exit'],native_wall_s=native['elapsed_s'],
 completed_comparison=completed,gradient_formula_recomputed_from_saved_norms=True,gradient_admission=a,common_start_restoration_verified=True,
 original_acceptance_hashes_verified=True,all_frozen_local_hashes_verified=True,old_R65_checkpoint_unchanged=True,
 original_A_F_complete=False,model_goal_complete=False,independent_training_or_acquisition_replicates=False,
 new_model_runs_in_root_review=0,DEV_read=0,EVAL_read=0,CAL_decode=0,R65_automatic_resume=False)
report=['# R69 当前完整分布的抽块反传小试','',
 '这是一项从原第500步共同起点出发的固定 TRAIN 诊断。先测梯度近似误差，再按预登记条件决定是否比较响应质量。原 A–F 和完整模型目标仍未完成。','',
 f"梯度准入通过：{a['pass_gate']}。固定一次模型、噪声及方向，先完整反传104个原块，再由各块平方范数独立复算抽4块估计器的条件方差。相对完整 D 梯度平方范数为 {a['relative_MSE_to_D']:.9g}，相对 D+C 为 {a['relative_MSE_to_D_plus_C']:.9g}，预设上限均为1。模型和 Adam 原状态未变；该步没有更新、clip或Adam step。",'',
 '无偏性只针对 clip 前、给定当前端点和方向的程序梯度。没有证明 clip/Adam、后续训练轨迹或响应质量无偏。候选每步仍使用全部6593 SOURCE、21047 REAL计算当前五项分布目标，只对4/104原块做参数反传；原四项 C 每步照常计算。','']
if completed:
 assert a['pass_gate'] and read(S/'BASELINE_BUDGET_GATE.json')['pass_gate']
 names=['baseline','candidate','mini'];ev=[read(S/n/'EVALUATION.json')for n in names]
 ft=read(S/'candidate/TRAINING_RECEIPT.json');mt=read(S/'mini/TRAINING_RECEIPT.json')
 assert ft['completed_updates']==100 and ft['training_elapsed_s']<=200 and mt['training_elapsed_s']<=200
 assert mt['cost_match_pass']and abs(ft['training_elapsed_s']-mt['training_elapsed_s'])<=max(.5,.01*ft['training_elapsed_s'])
 assert ft['fixed_learning_rate']==mt['fixed_learning_rate']==C['fixed_learning_rate']
 for e,n in zip(ev,names):
  assert e['arm']==n and e['probe_count']==8 and e['SOURCE_N']==6593 and e['REAL_N']==21047 and e['train_streams_and_model_unchanged']
  assert [r['seed']for r in e['probes']]==C['probe_seeds']
  assert set(e['REAL_control']['pieces'])=={'full_window','coordinate_RMS','PSD','noise','temporal_joint'}
  assert e['REAL_control']['numeric']['q64']<=1e-5 and e['REAL_control']['numeric']['q32']<=1e-5
  for r in e['probes']:
   assert set(r['pieces'])=={'full_window','coordinate_RMS','PSD','noise','temporal_joint','source_order','source_valley','REAL_identity','box_correction'}
   assert close(r['score'],math.fsum(r['pieces'].values()))
   assert r['numeric']['q64']<=1e-5 and r['numeric']['q32']<=1e-5
   assert all('PSD_W1_region_'+str(b)in r['nonnegative']for b in [-1,*range(9)])
 with np.load(S/'candidate/BLOCK_SAMPLING_PRIVATE.npz',allow_pickle=False)as z:
  selected=z['selected_blocks'].copy();backward=z['backward_parent_counts'].copy()
 rng=np.random.Generator(np.random.PCG64(C['block_sampler_seed']))
 expected=np.asarray([sorted(rng.choice(104,4,replace=False).tolist())for _ in range(100)])
 assert np.array_equal(selected,expected)
 recounted=np.zeros(6593,dtype='i8')
 for b in selected.ravel():recounted[64*b:min(64*b+64,6593)]+=1
 assert np.array_equal(recounted,backward)
 assert int(backward.sum())==ft['actual_D_backward_parent_occurrences']
 assert ft['D_parameter_VJP_calls']==400 and ft['neural_forward_rows']==100*(6593+384)+int(backward.sum())
 assert ft['counter_delta']['scientific_neural_forward_rows']==ft['neural_forward_rows']
 assert mt['counter_delta']['scientific_neural_forward_rows']==384*mt['completed_updates']
 assert mt['joint_microbatch_reverse_calls']==mt['completed_updates']
 scores=np.asarray([[r['score']for r in e['probes']]for e in ev]);b,f,m=scores
 primary=bool(f.mean()<=.9*m.mean()and f.mean()<=.9*b.mean()and int((f<m).sum())>=6)
 assert primary==v['primary']['pass_gate']
 for key,value in {'baseline_mean':b.mean(),'candidate_mean':f.mean(),'mini_mean':m.mean(),'candidate_vs_mini_relative_improvement':1-f.mean()/m.mean(),'candidate_vs_baseline_relative_improvement':1-f.mean()/b.mean()}.items():assert close(value,v['primary'][key])
 boot=np.random.Generator(np.random.PCG64(C['bootstrap_seed'])).integers(0,8,(1000,8))
 ci=np.quantile((f-m)[boot].mean(1),[.025,.975]);assert np.allclose(ci,v['primary']['descriptive_paired_noise_MC_candidate_minus_mini_95_interval'],rtol=1e-12,atol=1e-12)
 harms=[]
 for group in ['pieces','nonnegative','correlations','valleys','REAL_pieces','REAL_nonnegative']:
  real_group=group.startswith('REAL_');g=group[5:]if real_group else group
  first=ev[0]['REAL_control'][g]if real_group else ev[0]['probes'][0][g]
  for key in first:
   seq=[[e['REAL_control'][g][key]]if real_group else[r[g][key]for r in e['probes']]for e in ev]
   if any(x is None for s in seq for x in s):row=dict(group=group,key=key,status='unknown',pass_gate=False)
   else:
    bv,fv,mv=[float(np.mean(s))for s in seq]
    if g=='correlations':bound=max(bv,mv)-.02;ok=fv>=bound
    elif g=='valleys':bound=min(bv,mv)+.01;ok=fv<=bound
    else:bound=min(bv,mv)+.05*max(bv,mv)+1e-10;ok=fv<=bound
    row=dict(group=group,key=key,baseline=bv,candidate=fv,mini=mv,bound=bound,pass_gate=bool(ok))
   harms.append(row)
 assert len(harms)==len(v['harm_checks'])==102
 for r,old in zip(harms,v['harm_checks']):
  assert(r['group'],r['key'],r['pass_gate'])==(old['group'],old['key'],old['pass_gate'])
  if 'bound'in r:
   for key,oldkey in [('baseline','baseline'),('candidate','full'),('mini','mini'),('bound','bound')]:assert close(r[key],old[oldkey])
 allharms=all(x['pass_gate']for x in harms);assert allharms==v['harm_gate_pass']
 assert bool(primary and allharms)==(v['status']=='POSITIVE_LOCAL_ONLY')
 means=lambda e,g,key:float(np.mean([r[g][key]for r in e['probes']]))
 scopes=['overall',*[f'region_{i}'for i in range(8)],'region_-1','region_8']
 psd=[dict(scope=sc,**{n:means(e,'nonnegative','PSD_W1_'+sc)for n,e in zip(names,ev)})for sc in scopes]
 failed=[x for x in harms if not x['pass_gate']]
 root.update(primary=v['primary'],primary_recomputed=True,all102_harms_recomputed_from_saved_evaluations=True,paired_MC_interval_recomputed=True,
  candidate_training=ft,mini_training=mt,sampling_trace_and_backward_exposure_verified=True,harms=harms,failed_harm_checks=failed,PSD_means=psd)
 decision='仅支持另行登记的有上限确认，尚无长跑或原A–F通过证据。'if v['status']=='POSITIVE_LOCAL_ONLY'else'联合局部收益门未通过，关闭固定K4候选；不恢复R65、不改k或随机种子重试。'
 report.extend([f'**结论：{decision}**','',
  f"K4完成100步，用时 {ft['training_elapsed_s']:.3f} 秒；原微批完成 {mt['completed_updates']} 步，用时 {mt['training_elapsed_s']:.3f} 秒，实际差值 {mt['cost_match_absolute_s']:.6f} 秒。两臂同模型、Adam、原随机流与固定学习率；根复核确认抽块序列和实际父反传曝光吻合。",'',
  '| 共同指标 | 起点 | K4 | 原微批 |','| --- | ---: | ---: | ---: |',
  f'| 九项共同损失均值 | {b.mean():.9g} | {f.mean():.9g} | {m.mean():.9g} |',
  f"| 总体 PSD W1 | {psd[0]['baseline']:.9g} | {psd[0]['candidate']:.9g} | {psd[0]['mini']:.9g} |",'',
  f"K4相对微批的损失降低 {100*(1-f.mean()/m.mean()):.3f}%；相对起点降低 {100*(1-f.mean()/b.mean()):.3f}%。共同8探针赢 {int((f<m).sum())}/8；主门通过 {primary}，完整102项无实质退化门通过 {allharms}。失败 {len(failed)}/102，全部保留。",'',
  f'固定模型噪声/方向配对的K4减微批均值差描述性95%区间为 [{ci[0]:.9g}, {ci[1]:.9g}]；不是独立训练或独立采集置信区间。','',
  '| 未通过项 | 起点 | K4 | 原微批 |','| --- | ---: | ---: | ---: |'])
 for r in failed:
  report.append(f"| {r['group']} / {r['key']} | {r['baseline']:.8g} | {r['candidate']:.8g} | {r['mini']:.8g} |"if'bound'in r else f"| {r['group']} / {r['key']} | unknown | unknown | unknown |")
 report.extend(['','| 全部原区域 PSD W1 | 起点 | K4 | 原微批 |','| --- | ---: | ---: | ---: |'])
 for r in psd:report.append(f"| {r['scope']} | {r['baseline']:.8g} | {r['candidate']:.8g} | {r['mini']:.8g} |")
 report.append('')
 fig,axes=plt.subplots(1,2,figsize=(12.2,4.9),layout='constrained')
 colors=['#8b949e','#d85d42','#2368a0'];labels=['Shared start','K4 (100 steps)',f"Mini ({mt['completed_updates']} steps)"]
 ax=axes[0];ax.bar(np.arange(3),scores.mean(1),color=colors,width=.62)
 for j,s in enumerate(scores):
  ax.scatter(j+np.linspace(-.13,.13,8),s,color='#233344',s=15,zorder=3)
  ax.text(j,max(s)+.03*scores.max(),f'{s.mean():.3f}',ha='center',fontsize=10)
 ax.set_xticks(range(3),labels,fontsize=9);ax.set_ylim(0,scores.max()*1.22);ax.set_ylabel('Common nine-term score (lower is better)')
 ax.set_title('Eight shared full-parent TRAIN probes',fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 ax=axes[1];xx=np.arange(len(psd))
 for n,color,label in zip(names,colors,labels):ax.plot(xx,[r[n]for r in psd],marker='o',ms=3.5,color=color,label=label)
 ax.axhline(.01,color='#444444',ls='--',lw=1,label='Original PSD target; not an A-F verdict')
 ax.set_xticks(xx,['Overall',*[f'Q{i}'for i in range(1,9)],'Below','Above'],rotation=40,ha='right',fontsize=9)
 ax.set_ylabel('Canonical PSD W1');ax.set_title('All original charge regions retained',fontsize=11);ax.grid(alpha=.2);ax.legend(fontsize=7.4)
 for ax in axes:ax.spines[['top','right']].set_visible(False)
 fig.suptitle('R69: gradient admission passed; '+('joint local gate passed'if v['status']=='POSITIVE_LOCAL_ONLY'else'joint local gate not passed'),fontsize=13)
 fig.savefig(S/'R69_COMPARISON.png',dpi=170);fig.savefig(S/'R69_COMPARISON.svg');plt.close(fig)
 root['figure']='work/'+S.name+'/R69_COMPARISON.png'
else:
 decision='固定K4未通过梯度准入，零响应更新及零响应探针后关闭。'if v['status']=='NEGATIVE_GRADIENT_ADMISSION'else'未获得完整有效比较，封存本次执行；不追加重试或恢复昂贵R65。'
 if v['status']=='NEGATIVE_GRADIENT_ADMISSION':assert not a['pass_gate']and v['new_response_updates']==v['new_response_probes']==0
 report.extend([f'**结论：{decision}**','',f"执行状态：{v['status']}。说明：{v.get('error',v.get('reason',v.get('decision','unknown')))}。",''])
 fig,ax=plt.subplots(figsize=(6.5,4),layout='constrained')
 values=[a['relative_MSE_to_D'],a['relative_MSE_to_D_plus_C']];ax.bar(['Relative to D','Relative to D+C'],values,color=['#d85d42','#2368a0'])
 ax.axhline(1,color='#444444',ls='--',label='Prospective limit');ax.set_ylabel('Conditional gradient MSE / squared norm');ax.set_title('R69 fixed-draw gradient admission');ax.legend()
 fig.savefig(S/'R69_COMPARISON.png',dpi=170);fig.savefig(S/'R69_COMPARISON.svg');plt.close(fig);root['figure']='work/'+S.name+'/R69_COMPARISON.png'
root['final_decision']=decision
known_completed_training_arms=sum((S/n/'TRAINING_RECEIPT.json').exists()for n in ['candidate','mini'])
known_started_training_runs=2 if completed else 0 if v['status']in['NEGATIVE_GRADIENT_ADMISSION','INCONCLUSIVE_BUDGET']else None
root.update(completed_training_arm_receipts=known_completed_training_arms,training_runs_started=known_started_training_runs)
report.extend([f"实际原生计算 {native['elapsed_s']:.3f} 秒，登记上限600秒；worker已退出，GPU计算进程为空。CPU亲和性0–11，观测worker峰值RSS {native['actual_worker_RSS_peak_bytes']/1024**3:.3f} GiB，进程GPU峰值 {native['GPU_process_memory_peak_bytes']/1024**3:.3f} GiB。",'',
 '该结果只适用于原第500步共同起点、固定LR和这一次短程试验。起点由旧全量D训练产生；8组探针是固定模型噪声重复。所有原父、区域和支持外事件保持；未访问DEV/EVAL/CAL，未运行新HGB、峰位/峰宽拟合或D/E任务评价。不能据此宣称原A–F通过或不可行。','',
 '复核入口：work/R69_SAMPLED_FULL_GRADIENT_PILOT_001/RESULT.json；reviews/R69_SAMPLED_FULL_GRADIENT_PILOT_001/ROOT_RESULT_REVIEW.json、COLLECTION_RECEIPT.json。协议和冻结源代码同目录保留；完整复现需原R65依赖与第500步原检查点。','',
 '后续决策依据已封存的联合证据，遵守当前候选停止边界。总目标保持active，原验收、源人口和源电荷不变。',''])
(R/'R69_REPORT.md').write_text('\n'.join(report),encoding='utf8')
root.update(report_SHA256=sha(R/'R69_REPORT.md'),figure_SHA256=sha(S/'R69_COMPARISON.png'),resource=dict(guard,**memory()))
write(P/'ROOT_RESULT_REVIEW.json',root)
snapshot=P/'before_closure_metadata';snapshot.mkdir()
for n in ['PROJECT_PROGRESS.json','PROGRAM.json','AGENTS.md','STATUS.md','PLAN.md','PROJECT_OBJECTIVE.md','jobs.json','experiments.json']:shutil.copy2(R/n,snapshot/n)
for n in ['PROJECT_PROGRESS.json','PROGRAM.json']:
 d=read(R/n);d.update(updated_UTC=root['UTC'],status='R69_fixed_candidate_closed',project_status='R69_fixed_candidate_closed',
  current_stage=S.name,current_stage_progress=v['status'],current_activity='R69_root_review_complete_delivery_pending',
  active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,latest_completed_stage=S.name,
  latest_diagnostic_stage=S.name,latest_completed_report='R69_REPORT.md',latest_completed_review='reviews/'+S.name+'/ROOT_RESULT_REVIEW.json',
  latest_scientific_decision=decision,next_action='Deliver preserved R69 evidence, then choose the next independent mechanism from joint results; no K4 scan/retry or R65 restart',
  original_A_F_complete=False,model_goal_complete=False,project_model_goal_complete=False,application_goal_status='active',
  current_stage_response_training_runs_started=known_started_training_runs,remote_job_state='terminal_native_receipt_verified',
  global_single_physical_job_confirmed=True,paused_remote_scientific_jobs=0)
 write(R/n,d)
for n,key,field in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 d=read(R/n);e=next(x for x in d[key]if x[field]==S.name)
 e.update(status='closed_'+v['status'].lower(),active_scientific_jobs=0,model_goal_complete=False,root_review='reviews/'+S.name+'/ROOT_RESULT_REVIEW.json',result='work/'+S.name+'/RESULT.json')
 d.update(updated_UTC=root['UTC'],active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/n,d)
header='# Current: R69 fixed K4 gradient and equal-cost evidence closed\n\nResult: '+v['status']+'. The no-update gradient error gate '+('passed'if a['pass_gate']else'failed')+'. Root recomputed the saved evidence, original hashes and native resource receipt. No active scientific worker; no K4/seed scan or R65 restart. Original full A-F remains incomplete. See R69_REPORT.md and reviews/'+S.name+'/ROOT_RESULT_REVIEW.json.\n\n## Earlier records\n\n'
for n in ['AGENTS.md','STATUS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/n).write_text(header+(R/n).read_text('utf8'),encoding='utf8')
print(json.dumps(dict(status=v['status'],native_wall_s=native['elapsed_s'],primary=root.get('primary'),failed_harms=len(root.get('failed_harm_checks',[])),decision=decision,report='R69_REPORT.md'),ensure_ascii=False,indent=2))
