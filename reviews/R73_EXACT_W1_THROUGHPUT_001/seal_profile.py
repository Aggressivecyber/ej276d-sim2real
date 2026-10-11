"""Review saved profile times and unchanged bindings; no scientific array reads."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,statistics,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'seal_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not(P/'ROOT_RESULT_REVIEW.json').exists()
v=read(S/'COMPONENT_PROFILE.json');native=read(S/'train_profile.py.NATIVE_RECEIPT.json');cfg=read(S/'config.json')
assert native['actual_worker_native_exit']==native['launcher_native_exit']==native['worker_self_receipt']['native_exit']==0
assert v['engineering_updates']==20 and v['main_updates']==0 and len(v['rows'])==20 and v['warmup']==v['timed']==10
assert v['initial_state_SHA256']==v['restored_state_SHA256']==read(R/'work/R72_FULL_WAVEFORM_FULL_TRAIN_001/COST_ADMISSION.json')['initial_state_SHA256']
assert v['all_RNG_restored']and v['optimizer_empty']and not v['quality_metrics_recorded_or_used']
assert v['SOURCE_N']==6593 and v['REAL_N']==21047 and v['all_VJP_N']==27640
assert not(S/'FINAL_SHAPE.pt').exists()
measured=v['rows'][10:];median=statistics.median(row['seconds']for row in measured)
assert median==v['median_seconds']
for k,value in v['component_median_seconds'].items():assert statistics.median(row['components'][k]for row in measured)==value
for row in read(S/'FILE_BINDING.json')['files']:assert sha(S/row['name'])==row['sha256'],row['name']
times=v['component_median_seconds'];major=sum(times[k]for k in ['forward_full_window','backward_full_window','forward_coordinate_RMS','backward_coordinate_RMS'])
now=datetime.now(timezone.utc).isoformat()
review=dict(UTC=now,root_profile_evidence_review_pass=True,engineering_updates=20,main_updates=0,median_update_seconds=median,
  p90_update_seconds=v['p90_seconds'],full_and_coordinate_seconds=major,full_and_coordinate_fraction_of_median=major/median,
  component_median_seconds=times,initial_state_and_all_rng_restored=True,no_quality_selection=True,
  native_exit=0,actual_worker_PID=native['actual_worker_PID'],native_elapsed_s=native['elapsed_s'],
  RSS_peak_bytes=native['actual_worker_RSS_peak_bytes'],GPU_peak_bytes=native['GPU_process_memory_peak_bytes'],
  RSS_probe_failures=native['memory_probe_failure_events'],instrumented_cost_not_admission=True,
  scientific_decision='Engineering bottleneck in repeated full-window/coordinate distributions, no A-F verdict',
  next_action='R74 preserves nine laws and50000 endpoint; exact W1 analytical backward and shared marginal graph require CPU/CUDA and full-parent equivalence before cost gate',
  application_goal_status='active',model_goal_complete=False,guard={**guard,**memory()})
assert review['RSS_peak_bytes']<=cfg['RSS_limit_bytes'] and review['GPU_peak_bytes']<=cfg['GPU_limit_bytes']
write(P/'ROOT_RESULT_REVIEW.json',review)
for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
    d=read(R/name);entry=next(j for j in d[key]if j.get('stage',j.get('id'))==P.name)
    entry.update(status='COMPLETE_PARENT_COMPONENT_PROFILE_COMPLETE',root_review='reviews/'+P.name+'/ROOT_RESULT_REVIEW.json',native_exit=0,engineering_updates=20,main_updates=0,updated_UTC=now)
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);d.update(updated_UTC=now,status='R73_profile_complete_R74_exact_implementation_preparing',latest_completed_stage=P.name,
      latest_completed_report='R73_REPORT.md',latest_completed_review='reviews/'+P.name+'/ROOT_RESULT_REVIEW.json',next_action=review['next_action'],active_scientific_jobs=0,
      application_goal_status='active',model_goal_complete=False);write(R/name,d)
prefix='''# 当前执行：R73 分项计时已完成，准备 R74 精确计算优化

相同完整更新20次、exit0，median0.52466秒。全窗/逐坐标项的前反向约占72%，神经/物理前向及完整参数VJP约16%；主训练0且初始权重/空Adam/RNG已恢复。只定位工程开销，无A–F质量结论。R74准备精确W1解析梯度与共享两项边缘图；九定义/全父人口/50k/4小时规则及v5保持，需人工与全父梯度对照后重测成本。

根核验和报告：reviews/R73_EXACT_W1_THROUGHPUT_001/ROOT_RESULT_REVIEW.json、R73_REPORT.md。长期目标active；既有辅助/资源/E-D/R65停止保持。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:(R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
(R/'R73_REPORT.md').write_text(f'''# R73 完整更新分项耗时

R72 同一完整人口计算，10 暖机+10 计时更新，原生 exit0。同步分项后的整步 median {median:.9f} 秒、p90 {v['p90_seconds']:.9f} 秒。主训练0，初始状态/空Adam/全部RNG恢复；这是工程定位，不是科学质量试验。

| 部分 | 前向秒 | 反向秒 |
|---|---:|---:|
| 全窗分布 | {times['forward_full_window']:.6f} | {times['backward_full_window']:.6f} |
| 逐坐标 RMS | {times['forward_coordinate_RMS']:.6f} | {times['backward_coordinate_RMS']:.6f} |
| 时间联合 | {times['forward_temporal_joint']:.6f} | {times['backward_temporal_joint']:.6f} |
| 神经与物理整链 | {times['neural_physical_firstpass']:.6f} | {times['complete_neural_parameter_VJP']:.6f} |

全窗/逐坐标四部分合计约 {major/median:.2%}。R74 据此只准备精确反向及相同边缘计算的复用；实际加速待完整步测量，不能从局部算子预测宣称。所有6593 SOURCE、21047 REAL、完整27640 VJP和九项原目标保持。数值等价前瞻阈值、CPU/CUDA与实际全人口对照、新成本门后才可执行50k；原v5 A–F仍未完成。

实际stage交付见 R73_DELIVERY_RECEIPT.json。原生尾部RSS探针不可用记录仍保留，终态由监督器与worker回执共同确认。详细每步分项和来源：work/{P.name}/COMPONENT_PROFILE.json。
''','utf8')
delivery=(R/'reviews/R72_FULL_WAVEFORM_FULL_TRAIN_001/deliver_cost.py').read_text('utf8')
delivery=delivery.replace('R72','R73').replace('r72','r73').replace("root_cost_evidence_review_pass","root_profile_evidence_review_pass")
delivery=delivery.replace("'R71_DELIVERY_RECEIPT.json','tasks/'","'R72_DELIVERY_RECEIPT.json','tasks/'")
delivery=delivery.replace('engineering_updates=50','engineering_updates=20').replace('50 discarded engineering updates','20 discarded engineering updates')
delivery=delivery.replace('Close R73 full-waveform cost admission with zero main updates','Profile complete-parent loss costs and preserve exact W1 candidate')
delivery=delivery.replace('r73_cost_gate_','r73_component_profile_').replace('cost_gate_negative_delivery','engineering_component_profile_delivery')
(P/'deliver_profile.py').write_text(delivery,'utf8')
print(json.dumps({k:v for k,v in review.items()if k not in ['component_median_seconds','RSS_probe_failures']},ensure_ascii=False),flush=True)
