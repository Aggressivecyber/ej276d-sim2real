"""Review the bounded saved component/operator profile, without scientific arrays."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,statistics,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'seal_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not(P/'ROOT_RESULT_REVIEW.json').exists()
v=read(S/'COMPONENT_PROFILE.json');op=read(S/'OPERATOR_PROFILE.json');native=read(S/'train_profile.py.NATIVE_RECEIPT.json');C=read(S/'config.json')
assert v['engineering_updates']==21 and v['operator_updates']==1 and len(v['rows'])==20 and v['main_updates']==0
assert native['actual_worker_native_exit']==native['launcher_native_exit']==native['worker_self_receipt']['native_exit']==0
assert op['engineering_update']==21 and op['CUDA_times_present']and op['main_updates']==0
assert v['initial_state_SHA256']==v['restored_state_SHA256']==read(R/'work/R72_FULL_WAVEFORM_FULL_TRAIN_001/COST_ADMISSION.json')['initial_state_SHA256']
assert v['all_RNG_restored']and v['optimizer_empty']and not v['quality_metrics_recorded_or_used']
assert (v['SOURCE_N'],v['REAL_N'],v['all_VJP_N'])==(6593,21047,27640)
assert statistics.median(row['seconds']for row in v['rows'][10:])==v['median_seconds']
for k,value in v['component_median_seconds'].items():assert statistics.median(row['components'][k]for row in v['rows'][10:])==value
for row in read(S/'FILE_BINDING.json')['files']:assert sha(S/row['name'])==row['sha256'],row['name']
assert native['actual_worker_RSS_peak_bytes']<=C['RSS_limit_bytes']and native['GPU_process_memory_peak_bytes']<=C['GPU_limit_bytes']
now=datetime.now(timezone.utc).isoformat();times=v['component_median_seconds']
selected=[j for j in op['operators']if j['name']=='aten::add_'and j['input_shapes']and j['input_shapes'][0]==[21047,480]
  or j['name']=='aten::fill_'and j['input_shapes']and j['input_shapes'][0]==[21047,480]
  or 'indexing_backward_kernel_stride_1<double>'in j['name']]
review=dict(UTC=now,root_profile_evidence_review_pass=True,engineering_updates=21,main_updates=0,native_exit=0,
  median_seconds=v['median_seconds'],p90_seconds=v['p90_seconds'],component_median_seconds=times,
  selected_operator_records=selected,operator_attribution_note='CUDA kernel rows and aten rows overlap attribution; do not sum as disjoint costs or predict speedup',
  CPU_CUDA_artificial_pass=read(S/'ARTIFICIAL_CHECKS.json')['pass_all'],initial_state_and_all_rng_restored=True,
  RSS_peak_bytes=native['actual_worker_RSS_peak_bytes'],GPU_peak_bytes=native['GPU_process_memory_peak_bytes'],RSS_probe_failures=native['memory_probe_failure_events'],
  scientific_decision='Repeated full-array gradient scatter/add/fill observed; exact conditional VJP candidate needs direct verification, no quality verdict',
  next_action='R76 preparation: exact local conditional feature gradient assembly plus one retained neural forward; same nine laws and50k cost gate',
  application_goal_status='active',model_goal_complete=False,guard={**guard,**memory()})
write(P/'ROOT_RESULT_REVIEW.json',review)
(R/'R75_REPORT.md').write_text(f'''# R75 剩余分布反向开销

相同R74完整更新，10暖机+10计时+1算子profile，原生exit0。主训练0，初态/空Adam/全部RNG恢复。同步整步median {v['median_seconds']:.9f}秒，首组合并分布项前向 {times['forward_full_window+coordinate_RMS']:.6f}秒/反向 {times['backward_full_window+coordinate_RMS']:.6f}秒；时间联合反向 {times['backward_temporal_joint']:.6f}秒，神经/物理前向 {times['neural_physical_firstpass']:.6f}秒，完整参数VJP {times['complete_neural_parameter_VJP']:.6f}秒。

算子profile观察到重复全矩阵add/fill和indexing_backward。CUDA核和aten记录存在归属重叠，不能把其时间相加或当理论加速上界；仪器化单步也不能替代新候选成本门。原始记录在work/{P.name}/OPERATOR_PROFILE.json。

下一候选：按现有条件区计算精确局部梯度并唯一回填，四个完整特征视图仅一次反向到canonical；保留一次神经前向图后用完整端点梯度做参数VJP。所有父事件、九项定义、50000终点及v5保持；必须CPU/CUDA与全人口值/梯度/Adam对照通过后重新测时。10GiB GPU限额不放宽。若实际开销仍不准入，保留原固定方案成本失败，再依据实测成本前瞻决定独立可执行方案。

R74/R75属于工程证据，完整响应A–F未验证。交付以实际R75_DELIVERY_RECEIPT.json为准。
''','utf8')
for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
    d=read(R/name);entry=next(j for j in d[key]if j.get('stage',j.get('id'))==P.name)
    entry.update(status='COMPLETE_PARENT_COMPONENT_PROFILE_COMPLETE',native_exit=0,engineering_updates=21,main_updates=0,root_review='reviews/'+P.name+'/ROOT_RESULT_REVIEW.json',updated_UTC=now)
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);d.update(updated_UTC=now,status='R75_profile_complete_R76_exact_VJP_preparing',latest_completed_stage=P.name,
      latest_completed_report='R75_REPORT.md',latest_completed_review='reviews/'+P.name+'/ROOT_RESULT_REVIEW.json',next_action=review['next_action'],
      active_scientific_jobs=0,application_goal_status='active',model_goal_complete=False);write(R/name,d)
prefix='''# 当前执行：R75算子profile已完成，准备R76精确梯度回填与单次前向

21次工程更新正常exit0、主训练0；原始状态/空Adam/RNG恢复。实际首组合并分布反向约0.156秒，重复全矩阵scatter/add/fill可见。下一候选按条件区算同一精确梯度并唯一回填，神经图只前向一次后完整VJP；九项/全父/50000/4小时/v5不变，数值与资源/成本门未通过前不启动主训练。长期目标active。

根核验：reviews/R75_CONDITIONAL_VJP_OPTIMIZATION_001/ROOT_RESULT_REVIEW.json；报告R75_REPORT.md。R65停止，主助手/既有辅助/资源/单作业/E-D保持。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:(R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
print(json.dumps({k:v for k,v in review.items()if k not in ['selected_operator_records','component_median_seconds','RSS_probe_failures']},ensure_ascii=False),flush=True)
