"""Reconcile the user's cost objection and the verified pause without new science."""
from pathlib import Path
from datetime import datetime,timezone
import ctypes
import hashlib
import json
import shutil
import sys
import tomllib

P=Path(__file__).resolve().parent
R=P.parents[1]
S=R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),8)
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
pause=read(P/'BENEFIT_EVIDENCE_PAUSE_RECEIPT.json')
verification=read(P/'LATEST_RESUME_VERIFICATION.json')
assert pause['status']=='PAUSED_FOR_USER_REQUESTED_BENEFIT_EVIDENCE'
assert pause['physical_progress']['last_completed_update']==613 and pause['in_memory_state_retained']
assert verification['last_completed_update']==613 and verification['status']=='PAUSED_NATIVE_STATE_VERIFIED'
assert len(verification['progress_comparisons'])==1 and verification['progress_comparisons'][0]['global_step']==600
assert verification['progress_comparisons'][0]['all_compared_fields_equal']
stopped=read(S/'controller_runtime/PIPELINE_WATCH_STOPPED.json')
assert stopped['state']=='LOCAL_WATCH_STOPPED'
pids=(ctypes.c_ulong*32768)();needed=ctypes.c_ulong()
assert ctypes.WinDLL('psapi',use_last_error=True).EnumProcesses(pids,ctypes.sizeof(pids),ctypes.byref(needed))
assert 56812 not in pids[:needed.value//ctypes.sizeof(ctypes.c_ulong)]
automation_path=Path('C:/Users/10629/.codex/automations/r65/automation.toml')
automation=tomllib.loads(automation_path.read_text('utf8'))
assert automation['id']=='r65' and automation['kind']=='heartbeat' and automation['status']=='PAUSED'
assert automation['target_thread_id']=='01a0ffc1-e68d-7123-b09b-0c86a7253724'
write(P/'AUTOMATION_PAUSED_RECEIPT.json',{'UTC':datetime.now(timezone.utc).isoformat(),
 'automation_id':'r65','status':'PAUSED','kind':'heartbeat','target_thread_id':automation['target_thread_id'],
 'verified_native_app_configuration_SHA256':sha(automation_path),
 'reason':'The user requests evidence of benefit before further expensive training. Former completion estimate no longer applies.'})
backup=P/'before_benefit_pause_metadata'
names=['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','PROJECT_PROGRESS.json','PROGRAM.json']
if backup.exists():
    assert '--repair-partial-metadata' in sys.argv and all((backup/name).is_file() for name in names)
    assert read(R/'PROJECT_PROGRESS.json')['status']=='R65_paused_pending_low_cost_benefit_evidence'
else:
    backup.mkdir()
    for name in names:shutil.copy2(R/name,backup/name)
utc=datetime.now(timezone.utc).isoformat()
policy={'UTC':utc,'user_request':'你能先证明这样能带来提高吗 成本太大了',
 'benefit_of_full_population_training_proven':False,'R65_continuation':'paused_pending_low_cost_benefit_evidence',
 'automatic_resume_authorized':False,'new_scientific_pilot_registered':False,'new_scientific_pilot_executed':False,
 'root_leads_and_reviews':True,'analysis_model':'gpt-6.1-sol','analysis_reasoning_effort':'max',
 'acceptance_original_A_F_unchanged':True,'new_local_files':'E/D only',
 'old_50000_endpoint_role':'Original R65 protocol retained as an incomplete paused experiment; never relabel step613 as final scientific endpoint.',
 'evidence_needed_before_expensive_continuation':'A preregistered, tightly capped same-start/equal-compute comparison on fixed TRAIN probes with fresh noise and common metrics; no DEV/EVAL selection and no full-run proof from one pilot.'}
header=('# Current user instruction: prove benefit before further full-distribution cost\n\n'
 'R65 is PAUSED at completed update613, in update614; the worker and its supervisor are stopped with their memory retained. '
 'The durable checkpoint is update500. The local post-controller has stopped and the completion heartbeat is PAUSED. '
 'The user asked for evidence that the full-population change improves results before paying its large cost. '
 'No such quality or equal-compute benefit has been established. The original R65 50000-step endpoint and original A-F remain unchanged but incomplete. '
 'Do not resume the expensive run automatically. No new pilot has been registered or executed. '
 'Any proposed pilot must first declare a small fixed compute cap, common starting state, equal-compute comparison, fixed TRAIN probes and stopping rules; '
 'TRAIN proxy improvements cannot establish final A-F success. Root leads and reviews; auxiliary analysis, when useful, uses gpt-6.1-sol/max. '
 'All new local files remain on E/D. See R65_COST_BENEFIT_REVIEW.md and reviews/R65_THROUGHPUT_OPTIMIZATION_001/BENEFIT_EVIDENCE_PAUSE_RECEIPT.json.\n\n'
 '## Earlier records\n\n')
for name in names[:3]:(R/name).write_text(header+(backup/name).read_text('utf8'),encoding='utf8')
job={'stage':S.name,'status':'paused_pending_benefit_evidence','UTC':pause['UTC'],
 'actual_worker_PID':330411,'supervisor_PID':330406,'native_attempt_id':pause['native_attempt_id'],
 'process_state':'T','native_exit':None,'completed_logical_updates':613,'current_inflight_update':614,
 'last_committed_checkpoint_update':500,'in_memory_state_retained':True,'GPU_context_memory_retained':True,
 'active_computing_scientific_jobs':0,'paused_scientific_jobs':1,'logical_response_training_runs':1,
 'scientific_terminal_update':50000,'model_goal_complete':False,
 'pause_receipt':'reviews/R65_THROUGHPUT_OPTIMIZATION_001/BENEFIT_EVIDENCE_PAUSE_RECEIPT.json'}
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
    data=read(backup/name)
    data.update(updated_UTC=utc,updated_at_utc=utc,status='R65_paused_pending_low_cost_benefit_evidence',
      current_stage_progress='paused_pending_low_cost_benefit_evidence',current_activity='cost_benefit_evidence_review',
      current_job=job,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,
      paused_remote_scientific_jobs=1,remote_job_state='worker_and_supervisor_SIGSTOP_native_exit_unknown',
      application_goal_status='paused',goal_status_note='Application goal was already paused; expensive R65 now separately paused after user cost objection. Model goal remains unmet.',
      latest_user_steering=policy['user_request'],latest_user_execution_policy=policy,
      latest_user_revision='reviews/R65_THROUGHPUT_OPTIMIZATION_001/BENEFIT_EVIDENCE_REVIEW.json',
      latest_user_agent_policy='Scoped auxiliary analysis uses gpt-6.1-sol/max with root leading and reviewing; no auxiliary agents are currently running.',
      latest_permission_status='Prior full-run continuation is superseded by the request to establish benefit before further high cost.',
      next_action='Keep R65 paused; assess only low-cost preregistered evidence of equal-compute benefit before any long continuation.',
      current_engineering_addendum='Equivalent gradient-pack optimization applied and verified; it improves throughput, not established scientific quality.',
      scheduled_wakeup={'automation_id':'r65','status':'PAUSED','reason':'training cost-benefit hold'},
      post_controller={'PID':56812,'status':'stopped','native_stop_receipt':'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/controller_runtime/PIPELINE_WATCH_STOPPED.json'},
      model_goal_complete=False)
    if 'project_status' in data:data['project_status']=data['status']
    if isinstance(data.get('execution_coordination'),dict):
        data['execution_coordination'].update(subagents_allowed=True,parallelism='optional_scoped_analysis_only',
          subagents_same_model_as_root_required=False,subagents_same_reasoning_effort_as_root_required=False,
          agent_model_policy='gpt-6.1-sol/max_for_auxiliary_analysis_root_leads_reviews',waiting_work_required=[])
    elif 'execution_coordination' in data:
        data['prior_execution_coordination_text']=data['execution_coordination']
        data['execution_coordination']='Root leads and reviews; optional gpt-6.1-sol/max analysis; no active auxiliary agents; expensive training remains paused.'
    raw=read(Path(verification['raw_observation_path']))
    data['latest_training_progress']=[r for r in raw['current_trace'] if r['native_attempt_id']==pause['native_attempt_id']]
    write(R/name,data)
old=read(R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/COVERAGE_RESPONSE/TRAINING_RECEIPT.json')
old_s=old['elapsed_s']/old['updates'];new_s=verification['speed']['resume_first_100_full_updates_seconds_per_update']
result={'UTC':utc,'status':'NO_EVIDENCE_TO_JUSTIFY_FULL_R65_COST_CONTINUATION','policy':policy,
 'previous_R63':{'updates':50000,'elapsed_seconds':old['elapsed_s'],'mean_seconds_per_update':old_s,'neural_rows_per_update':384},
 'R65':{'neural_rows_per_update':13570,'neural_row_ratio':13570/384,
   'measured_new_seconds_per_update':new_s,'measured_original_seconds_per_update':verification['speed']['original_step_500_to_600_seconds_per_update'],
   'estimated_full_50000_hours_at_current_speed':50000*new_s/3600,'seconds_per_update_ratio_to_R63':new_s/old_s,
   'old_R63_average_vs_short_R65_interval_not_controlled_benchmark':True},
 'evidence_boundaries':[
  'R64 only compared sampled gradient estimators against one frozen finite empirical reference, excluded C and did not train either candidate.',
  'R64 joint128 had worse total-D parameter squared error than four32 average; this does not prove full-D long-run quality or equal-cost benefit.',
  'Artificial CPU/CUDA gradient byte equivalence and matching step600 recorded endpoint/100-update aggregate metrics only validate the speed optimization.',
  'The current 613-update TRAIN monitoring cannot establish A-F; minibatch and full-population training loss values are different estimators and are not a fair cross-method test.'
 ],'low_cost_comparison_proposal':{'status':'PROPOSED_NOT_REGISTERED_OR_EXECUTED',
   'total_server_GPU_wall_seconds_cap':600,'arms':['original_minibatch','full_empirical_D'],
   'requirements':['same_start_and_optimizer_state','equal_measured_compute','common_fixed_TRAIN_parent_measure_and_fresh_noise_probes',
                   'report_PSD_full_waveform_noise_and_source_preservation_together','no_DEV_EVAL_or_intermediate_final_model_selection'],
   'stop_rule':'No consistent benefit at equal compute, or material regression in source/noise/charge constraints, means do not restore the expensive run.',
   'positive_result_limit':'A small pilot supports only a staged follow-up with a cost cap, never an automatic 70-hour commitment or a proof of A-F.'},
 'native_pause_receipt_SHA256':sha(P/'BENEFIT_EVIDENCE_PAUSE_RECEIPT.json'),
 'controller_stopped':True,'completion_automation_status':'PAUSED','new_scientific_computation':False,
 'original_A_F_changed':False,'model_goal_complete':False}
write(P/'BENEFIT_EVIDENCE_REVIEW.json',result)
report=f'''# R65 成本与收益证据复核

目前没有证据证明“每步完整人口计算并训练50000步”能够带来值得成本的质量提升，因此暂停本方案。用户于2026-10-11要求先证明收益，再承担高成本。

| 项目 | R63 原小批量 | R65 每步完整分布 |
|---|---:|---:|
| 每步神经前向处理行数 | 384 | 13,570（35.34倍） |
| 每步耗时 | 全程平均 {old_s:.4f} 秒 | 优化后100步实测 {new_s:.4f} 秒 |
| 50000步训练时间 | 已实跑 {old['elapsed_s']/60:.2f} 分钟 | 按当前步速估算 {50000*new_s/3600:.2f} 小时 |

R65 每步对6593个SOURCE执行两次网络前向，再加原384行C微批；21047个REAL还参加完整分布计算。神经行数为2×6593+384=13570，另有104次分块参数VJP及CPU float64梯度累加。保持相同更新步数并不等于保持相同计算成本。表中旧全程均值与新100步测量仅用于成本估计，不是硬件受控性能基准。

R64只在冻结模型和固定噪声端点上比较分布梯度估计，没有训练候选且不含四项C。其联合128总D参数均方误差比四个32批平均增加206.362353，原95%配对区间[99.051001,343.486026]。它不证明R65能改善最终波形、PSD或原A-F，也不证明每GPU小时收益。直接据此投入相同50000步，缺少低成本对照支持。

已应用的梯度打包优化通过CPU/CUDA人工逐位检查，恢复后第600步的记录值和前100步聚合值与归档旧实现完全一致，完整步耗时由6.4788秒降至{new_s:.4f}秒。这证明工程加速和已检查范围内的一致性，不证明完整分布训练优于原小批量。

当前已完成613步，在614步中途暂停；worker330411及supervisor330406均为T状态，确认CPU计数停止增长。第500步磁盘检查点和当前内存状态保留；GPU上下文及显存仍占用。原生退出未知，不能记作训练完成。后台控制器已停止，原定10月14日检查的heartbeat也已暂停。

可考虑的下一项验证是总GPU墙钟最多10分钟的同一起点、等算力两臂对照，在共同固定TRAIN人口及新噪声探针上联合检查PSD、全窗、噪声和源保真。需先登记具体预算分配、指标及停止规则。无稳定优势或关键约束退化就不恢复长跑；小试阳性也只支持下一次有限投入，不能直接批准70小时。本小试尚未登记或运行，不读取DEV/EVAL选参。

[暂停回执](reviews/R65_THROUGHPUT_OPTIMIZATION_001/BENEFIT_EVIDENCE_PAUSE_RECEIPT.json) · [证据复核](reviews/R65_THROUGHPUT_OPTIMIZATION_001/BENEFIT_EVIDENCE_REVIEW.json) · [R64结果](R64_REPORT.md)
'''
(R/'R65_COST_BENEFIT_REVIEW.md').write_text(report,encoding='utf8')
(R/'R65_OPTIMIZATION_RUNNING.md').write_text('# R65 已暂停，等待收益证据\n\n原预计完成时间已失效；训练、后台接续和完成检查均暂停。当前状态和成本判断见 [成本与收益复核](R65_COST_BENEFIT_REVIEW.md)。模型目标未完成。\n',encoding='utf8')
print(json.dumps({'status':result['status'],'training_paused':True,'controller_stopped':True,
 'automation_paused':True,'neural_row_ratio':13570/384,'R63_minutes':old['elapsed_s']/60,
 'R65_estimated_hours_if_resumed':50000*new_s/3600,'new_scientific_computation':False},indent=2))
