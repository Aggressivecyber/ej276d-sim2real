"""Close the observed R72 cost gate without fitting or reading scientific arrays."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, shutil, sys
P=Path(__file__).resolve().parent; R=P.parents[1]; S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard=apply(P/'seal_runtime')
def read(p): return json.loads(p.read_text('utf8'))
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (P/'ROOT_RESULT_REVIEW.json').exists()
C=read(S/'config.json'); cost=read(S/'COST_ADMISSION.json'); native=read(S/'train_full.py.NATIVE_RECEIPT.json')
result=read(S/'RESULT.json'); checks=read(S/'ARTIFICIAL_CHECKS.json'); data=read(S/'DATA_RECEIPT.json')
assert native['actual_worker_native_exit']==native['launcher_native_exit']==native['worker_self_receipt']['native_exit']==0
assert native['actual_worker_PID']==native['worker_self_receipt']['PID'] and native['handshake_before_imports']
assert checks['pass_all'] and read(P/'COLLECTION_RECEIPT.json')['all_file_SHA_verified']
assert result['status']=='COST_TOO_HIGH_FULL_MAIN_NOT_STARTED' and result['main_updates']==0
assert cost['engineering_updates']==50 and len(cost['timed_update_seconds'])==40
t=sorted(cost['timed_update_seconds']); q=.9*(len(t)-1); p90=t[int(q)]*(1-(q%1))+t[math.ceil(q)]*(q%1)
assert abs(p90-cost['p90_update_seconds'])<1e-15
assert (t[19]+t[20])/2==cost['median_update_seconds']
assert cost['estimate_full_run_seconds']>C['cost_limit_seconds']==14400 and not cost['pass_gate']
assert cost['restored_state_SHA256']==cost['initial_state_SHA256']==data['initial_state_SHA256']
assert all(cost[k] for k in ['optimizer_empty','all_RNG_restored','engineering_parameters_discarded'])
assert not cost['quality_metrics_used_for_cost_decision'] and not data['heldout_arrays_read']
assert cost['per_update_complete_SOURCE_rows']==6593 and cost['per_update_complete_REAL_rows']==21047
assert cost['all_parameter_VJP_rows']==27640
assert not (S/'FINAL_SHAPE.pt').exists() and not (S/'TRAINING_RECEIPT.json').exists()
assert max(native['actual_worker_RSS_peak_bytes'],native['job_tree_RSS_peak_bytes'])<=C['RSS_limit_bytes']
assert native['GPU_process_memory_peak_bytes']<=C['GPU_limit_bytes']
source_checks=[]
for row in read(S/'FILE_BINDING.json')['files']:
    f=S/row['name']; assert sha(f)==row['sha256'],row['name']
    source_checks.append(dict(name=f.name,sha256=row['sha256']))
current=read(R/'CURRENT_ACCEPTANCE.json'); assert current['version']==5
for n,k in [('document','document_sha256'),('registry','registry_sha256')]: assert sha(R/current[n])==current[k]
now=datetime.now(timezone.utc).isoformat()
review=dict(UTC=now,root_cost_evidence_review_pass=True,scientific_decision='COST_TOO_HIGH_FULL_MAIN_NOT_STARTED',
  complete_engineering_updates=50,complete_main_updates=0,median_seconds=cost['median_update_seconds'],p90_seconds=p90,
  conservative_full_endpoint_seconds=cost['estimate_full_run_seconds'],conservative_hours=cost['estimate_full_run_seconds']/3600,
  threshold_seconds=14400,initial_state_restored=True,empty_Adam_and_all_RNG_restored=True,artificial_CPU_CUDA_pass=True,
  SOURCE_N=6593,REAL_N=21047,heldout_arrays_read=False,quality_used_for_cost_decision=False,
  native_exit=0,actual_worker_PID=native['actual_worker_PID'],native_elapsed_s=native['elapsed_s'],
  actual_worker_RSS_peak_bytes=native['actual_worker_RSS_peak_bytes'],GPU_process_peak_bytes=native['GPU_process_memory_peak_bytes'],
  terminal_RSS_probe_failures=native['memory_probe_failure_events'],probe_failure_cause='unknown; two terminal-time samples unavailable, native self receipt retained',
  immutable_source_checks=source_checks,model_goal_complete=False,application_goal_status='active',
  next_action='One bounded complete-update component profile; exact computational optimization must pass value/gradient equivalence and fresh same-rule cost admission',
  local_guard={**guard,**memory()})
write(P/'ROOT_RESULT_REVIEW.json',review)
snap=P/'before_cost_closure'; snap.mkdir()
for name in ['AGENTS.md','STATUS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','jobs.json','experiments.json']: shutil.copy2(R/name,snap/name)
prefix='''# 当前执行：R72 全量测时已关闭，主训练 0 步，v5 目标 active

R72 修复后的 CPU/CUDA 人工检查通过，50 次完整 TRAIN 工程更新正常退出。每步 median 0.522197 秒、p90 0.525428 秒；50000 步加预留保守估计 9.6307 小时，超过本次 4 小时条件。初始权重、空 Adam 和全部随机流恢复；主训练 0 步，没有最终波形权重，也没有 A–F 科学结论。首轮 CUDA 设备别名错误和原始证据保留。

下一项先对同一完整更新作有界分项计时，再验证等价计算优化。全部父事件、九项损失、50000 终点和 v5 验收保持；不恢复 R65。主助手统筹、既有 gpt-6.1-sol/max 辅助；服务器 CPU12/RSS48GiB/GPU10GiB、单科学作业、本机共享 CPU4/RSS4GiB、新文件 E/D 保持。

根核验：reviews/R72_FULL_WAVEFORM_FULL_TRAIN_001/ROOT_RESULT_REVIEW.json；报告：R72_REPORT.md。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']: (R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
    d=read(R/name); e=next(v for v in d[key] if v.get('stage',v.get('id'))==P.name)
    e.update(status=result['status'],root_review='reviews/'+P.name+'/ROOT_RESULT_REVIEW.json',engineering_updates=50,main_updates=0,
      median_update_seconds=cost['median_update_seconds'],p90_update_seconds=p90,conservative_hours=cost['estimate_full_run_seconds']/3600,native_exit=0,updated_UTC=now)
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0); write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name); d.update(updated_UTC=now,status='R72_cost_gate_closed_main_zero',current_stage_progress=result['status'],
      latest_completed_stage=P.name,latest_completed_report='R72_REPORT.md',latest_completed_review='reviews/'+P.name+'/ROOT_RESULT_REVIEW.json',
      latest_scientific_decision='Only cost inadmissibility established; no complete-waveform scientific endpoint or A-F verdict',
      current_model_fit_role='R71 frozen learned Q component complete; R72 full-waveform main has zero updates',
      next_action=review['next_action'],active_scientific_jobs=0,application_goal_status='active',model_goal_complete=False)
    write(R/name,d)
(R/'R72_REPORT.md').write_text(f'''# R72 全量完整波形训练：成本门关闭

50 次完整人口工程更新完成，主训练 **0 步**。实际 median {cost['median_update_seconds']:.9f} 秒，p90 {p90:.9f} 秒。按预登记 `setup + 1.25 × p90 × 50000 + 1800` 估计 {cost['estimate_full_run_seconds']/3600:.6f} 小时，高于 4 小时执行条件。CPU/CUDA 人工值、梯度及 Adam 检查通过；这不构成完整响应科学成功或失败。

每步含 6593 SOURCE、21047 REAL、完整九项目标和所有 27640 父事件参数反向。工程权重全部丢弃，初始状态 SHA `{cost['initial_state_SHA256']}` 恢复，Adam 为空且所有随机流恢复。未读取 held-out，未按质量选参；没有 FINAL_SHAPE 或主训练回执。

原生 worker `{native['actual_worker_PID']}` exit 0，过程 {native['elapsed_s']:.3f} 秒。记录峰值 RSS {native['actual_worker_RSS_peak_bytes']/2**30:.3f} GiB、GPU {native['GPU_process_memory_peak_bytes']/2**30:.3f} GiB。退出附近两个 RSS 采样不可用，原因仍为 unknown；监督器和 worker 自回执的正常退出一致。首轮 cuda/cuda:0 人工设备检查错误发生在科学数组读取前；原源码/日志/绑定完整留存于 attempt_01_predata_device_alias。

下一步对相同完整更新分项计时，验证数学等价的分布损失实现。人口、损失、50000 终点和 v5 A–F 不变。长期目标 active；未恢复 R65。当前阶段交付以 R72_DELIVERY_RECEIPT.json 的实际回执为准。
''','utf8')
print(json.dumps({k:v for k,v in review.items() if k not in ['immutable_source_checks','terminal_RSS_probe_failures']},ensure_ascii=False),flush=True)
