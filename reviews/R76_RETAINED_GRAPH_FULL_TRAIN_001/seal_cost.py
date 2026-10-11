"""Seal R76 cost-only evidence and calculate a prospective new fixed endpoint."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,statistics,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'seal_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not(P/'ROOT_RESULT_REVIEW.json').exists()
C=read(S/'config.json');cost=read(S/'COST_ADMISSION.json');eq=read(S/'FULL_PARENT_EQUIVALENCE.json');native=read(S/'train_full.py.NATIVE_RECEIPT.json')
assert native['actual_worker_native_exit']==native['launcher_native_exit']==native['worker_self_receipt']['native_exit']==0
assert read(S/'RESULT.json')['main_updates']==0 and not cost['pass_gate']and C['updates']==50000
assert cost['engineering_updates']==50 and len(cost['timed_update_seconds'])==40
assert cost['median_update_seconds']==statistics.median(cost['timed_update_seconds'])
assert cost['initial_state_SHA256']==cost['restored_state_SHA256']==read(R/'work/R72_FULL_WAVEFORM_FULL_TRAIN_001/COST_ADMISSION.json')['initial_state_SHA256']
assert cost['optimizer_empty']and cost['all_RNG_restored']and cost['engineering_parameters_discarded']and not cost['quality_metrics_used_for_cost_decision']
assert eq['pass_all']and [v['update']for v in eq['checks']]==[1,25,50]
for v in eq['checks']:
    assert v['pass_gate']and max(v['nine_values_scaled_errors'].values())<=C['equivalence_value_scaled_atol']
    assert v['gradient_relative_L2']<=C['equivalence_parameter_gradient_relative_L2']and v['Adam_parameter_maxabs']<=C['equivalence_Adam_parameter_maxabs']
checks=read(S/'ARTIFICIAL_CHECKS.json');assert checks['pass_all']and checks['retained_graph']['pass_all']
assert all(v['pass_all']for v in checks['conditional_VJP'].values())
for row in read(S/'FILE_BINDING.json')['files']:assert sha(S/row['name'])==row['sha256'],row['name']
assert native['actual_worker_RSS_peak_bytes']<=C['RSS_limit_bytes']and native['GPU_process_memory_peak_bytes']<=C['GPU_limit_bytes']
setup=cost['estimate_full_run_seconds']-C['cost_safety_multiplier']*cost['p90_update_seconds']*50000-C['postprocess_reserve_seconds']
continuous=(C['cost_limit_seconds']-setup-C['postprocess_reserve_seconds'])/(C['cost_safety_multiplier']*cost['p90_update_seconds'])
prospective=5000*math.floor(continuous/5000);assert prospective==25000
new_estimate=setup+C['cost_safety_multiplier']*cost['p90_update_seconds']*prospective+C['postprocess_reserve_seconds']
now=datetime.now(timezone.utc).isoformat()
review=dict(UTC=now,root_cost_evidence_review_pass=True,main_updates=0,engineering_candidate_updates=50,extra_original_reference_updates=3,native_exit=0,
    median_seconds=cost['median_update_seconds'],p90_seconds=cost['p90_update_seconds'],estimated_hours=cost['estimate_full_run_seconds']/3600,
    full_parent_equivalence=eq,CPU_CUDA_artificial_pass=True,initial_state_optimizer_and_rng_restored=True,
    RSS_peak_bytes=native['actual_worker_RSS_peak_bytes'],GPU_peak_bytes=native['GPU_process_memory_peak_bytes'],RSS_probe_failures=native['memory_probe_failure_events'],
    no_A_F_quality_verdict=True,model_goal_complete=False,application_goal_status='active',
    next_action='Register independent R77 full-data fixed25000 scheme from cost only; original50k cost rejection preserved',guard={**guard,**memory()})
write(P/'ROOT_RESULT_REVIEW.json',review)
decision=dict(UTC=now,reason='User authorized an executable fixed full-data scheme if full50k cannot complete within several hours',
    input_cost_receipt='work/'+P.name+'/COST_ADMISSION.json',input_cost_SHA256=sha(S/'COST_ADMISSION.json'),
    original_updates=50000,original_main_updates=0,original_result='cost_gate_rejected_unchanged',
    rule='largest complete 5000-update block satisfying observed setup+1.25*p90*updates+1800<=14400; fixed prospectively before new training',
    observed_setup_seconds=setup,observed_p90_seconds=cost['p90_update_seconds'],continuous_max_updates=continuous,
    selected_new_scheme_updates=prospective,estimated_new_scheme_seconds=new_estimate,estimated_new_scheme_hours=new_estimate/3600,
    quality_metrics_consulted=False,heldout_access=False,all_6593_SOURCE_and_21047_REAL_per_update=True,
    new_LR_schedule='same500 warmup and cosine endpoints, new denominator25000-500 prospectively',
    thresholds_or_populations_changed=False,requires_new_registration_and_full_restore=True,root_user_commentary_disclosed=True)
write(P/'NEXT_FIXED_ENDPOINT_DECISION.json',decision)
(R/'R76_REPORT.md').write_text(f'''# R76 条件梯度与保留图成本结果

原生exit0，CPU/CUDA人工检查和完整人口第1/25/50步原R72数值对照通过：九值、参数梯度及一次Adam观察差均0。50候选+3原实现工程更新，主训练0；初始权重、空Adam和全部随机流恢复。每步median {cost['median_update_seconds']:.9f}秒，p90 {cost['p90_update_seconds']:.9f}秒；原50000终点保守预计{review['estimated_hours']:.6f}小时>4，原方案按成本关闭。GPU峰值{native['GPU_process_memory_peak_bytes']}字节，保留终止附近两次RSS未知探测记录，不从探测缺失推断失败。

下一项R77是独立固定25000方案，按本次实测成本取4小时内最大的完整5000步块，保守预计{new_estimate/3600:.6f}小时。全部6593 SOURCE+21047 REAL仍每步完整参与；九项定义/权重、随机流、架构、精度、原实测参照、v5门不改。唯一端点和余弦学习率总长度在新注册前固定，不读取质量结果或held-out选择。原R72/R74/R76的50000未运行记录保持，不把25000说成完成原50000。

当前只有工程证据，没有完整波形科学权重或A–F通过结论。交付以实际R76_DELIVERY_RECEIPT.json为准。
''','utf8')
for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
    d=read(R/name);entry=next(j for j in d[key]if j.get('stage',j.get('id'))==P.name)
    entry.update(status='COST_TOO_HIGH_FULL_MAIN_NOT_STARTED',native_exit=0,engineering_updates=50,main_updates=0,root_review='reviews/'+P.name+'/ROOT_RESULT_REVIEW.json',updated_UTC=now)
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);d.update(updated_UTC=now,status='R76_cost_closed_R77_fixed25000_preparing',latest_completed_stage=P.name,
        latest_completed_report='R76_REPORT.md',latest_completed_review='reviews/'+P.name+'/ROOT_RESULT_REVIEW.json',next_action=review['next_action'],
        active_scientific_jobs=0,application_goal_status='active',model_goal_complete=False);write(R/name,d)
prefix='''# 当前执行：R76原50000成本关闭，准备独立R77全数据25000固定方案

完整CPU/CUDA和原R72同状态1/25/50数值对照通过，九值/参数梯度/Adam观察差0。50工程更新exit0后初态/空Adam/RNG恢复；median0.36084秒、p900.36722秒、50000保守6.8847小时，主训练0。R77按实测成本选择4小时内最大5000整数步块25000，保守约3.70小时。全部6593SOURCE+21047REAL每步保留，损失/架构/精度/v5不变；新端点与余弦LR总长前瞻独立注册，不追改旧50000或用评价选步数。

证据：reviews/R76_RETAINED_GRAPH_FULL_TRAIN_001/ROOT_RESULT_REVIEW.json与NEXT_FIXED_ENDPOINT_DECISION.json。长期目标active，旧R65停止，根/既有辅助/资源/单作业/E-D保持。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:(R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
print(json.dumps(dict(main_updates=0,old50k_estimated_hours=review['estimated_hours'],new_fixed_updates=prospective,new_estimated_hours=new_estimate/3600,root_review_pass=True)),flush=True)
