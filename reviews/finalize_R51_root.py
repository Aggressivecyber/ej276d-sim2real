from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path(__file__).resolve().parents[1];STAGE='R51_CANONICAL_SOURCE_TEACHER_DIAGNOSTIC_001';S=R/'work'/STAGE;NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,x):(R/p).write_bytes((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def bind(p):
 p=R/p;return {'path':str(p.relative_to(R)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
n=read('work/'+STAGE+'/numeric_worker.py.NATIVE_RECEIPT.json');c=read('work/'+STAGE+'/CANONICAL_C2ST.json');t=read('work/'+STAGE+'/TEACHER_DIAGNOSTIC.json');m=read('work/'+STAGE+'/SOURCE_METRICS.json')
assert n['actual_worker_native_exit']==0 and n['worker_self_receipt']['native_exit']==0 and n['termination_reason'] is None
assert c['new_fits']==32 and c['reused_results']==3 and len(c['results'])==35 and n['memory_probe_failures']==0
assert t['direction_hits_bins2_7']==6 and all(x['target_occurrence_multiset_equal'] for x in t['preserved_per_condition_multisets'])
assert all(c['results']['R49_COMPLETION_REAL__'+v]['CI_inside_original_band'] for v in read('work/'+STAGE+'/config.json')['canonical_views'])
figs=[bind(str(p.relative_to(R))) for p in sorted((S/'figures').glob('*.png'))];assert len(figs)==3
write('work/'+STAGE+'/FIGURE_QA.json',{'UTC':NOW,'reviewer':'/root','actual_image_opened':True,'figures':figs,
 'observation':'All three rendered PNGs opened. All 35 heatmap values and labels, all eight Q bins, axes, legends and teacher curves readable with no clipping. Low global-rank curve near zero is consistent with saved numeric values.','acceptance_from_layout':False})
next_action='Register one TRAIN-only joint charge-and-PSD coupling construction that preserves exact source/target occurrence marginals and reduces source-Q noise rescaling; require fixed target-noise and within-parent PSD checks before any new response training. Do not train global-PSD-only coupling while its noise conflict remains.'
decision={'UTC':NOW,'stage':STAGE,'scientific_decision':'canonical_REAL_five_views_pass_SOURCE_fail_coupling_dispersion_and_target_noise_conflict_confirmed',
 'global_rank_direction_hits_bins2_7':6,'target_occurrence_multiset_preserved':True,'original_R49_judgment_rewritten':False,
 'same_treated_REAL_canonical_five_view_intervals_inside_band':True,'SOURCE_canonical_five_view_intervals_inside_band':False,
 'noise_conflict':'Whole-waveform source-Q rescaling increases TRAIN baseline-RMS W1 by approximately one order of magnitude or more; global PSD rank alone retains it.',
 'new_response_training':0,'model_goal_complete':False,'application_goal_status':'active','next_action':next_action}
write('reviews/R51_ROOT_DIRECTION_001.json',decision)
lines=['# R51：SOURCE 偏差与训练目标的两项冲突','',
 '32个新HGB及3个复用结果已完成。共同ADC整数码重建后，两臂同处理REAL的五视图95%区间全部落入[0.40,0.60]；SOURCE五视图仍全部未达到联合要求。原R49历史判定保留，本轮未重拟合条件峰宽、未验收新的响应模型。','',
 '| 人口 | Shape493 BA | u480 BA | V480 BA | Q/PSD BA | 基线 BA |',
 '|---|---:|---:|---:|---:|---:|']
cfg=read('work/'+STAGE+'/config.json')
for name in cfg['domains']:
 vals=[f"{c['results'][name+'__'+v]['BA']:.3f}"+(' *' if c['results'][name+'__'+v]['CI_inside_original_band'] else '') for v in cfg['canonical_views']]
 lines.append('| '+name+' | '+' | '.join(vals)+' |')
lines+=['','星号表示整个95%分组区间位于容差带内。完整区间和配对差见链接指标。SOURCE主候选总体PSD W1=0.028896，固定Q1–Q7平均0.039321；原要求为0.01。Q3密度未收敛仍保留。源电荷人口差异、pure-had谷比例及所有八区均保留，没有重加权或删区。','',
 '原TRAIN载入6593个SOURCE与21047个REAL，其中实际94400次出现涉及3752个SOURCE与18739个REAL。按原Q区进行全局PSD同序配对，逐区及全局的目标出现次数完全保留。Q2–Q7的父事件内目标PSD标准差中位数均降至原来的0.39%–1.42%，满足预登记至少4区降低50%的机制判据。此关联是统计构造，不是物理逐事件真值，也不证明神经转换会改善。','',
 '| Q区 | 原父事件目标PSD SD中位数 | 全局排序SD中位数 | 原目标RMS W1 / mV | 电荷缩放后 / mV | 全局排序后再缩放 / mV |','|---|---:|---:|---:|---:|---:|']
for row in t['by_bin']:
 a=row['original'];b=row['global_rank'];lines.append(f"| {row['bin']} | {row['original_parent_target_PSD_std']['quantiles'][2]:.6f} | {row['global_parent_target_PSD_std']['quantiles'][2]:.6f} | {a['unscaled_target_RMS_W1_vs_REAL_TRAIN_own_Q']*1000:.6f} | {a['rescaled_teacher_RMS_W1_vs_REAL_TRAIN_own_Q']*1000:.6f} | {b['rescaled_teacher_RMS_W1_vs_REAL_TRAIN_own_Q']*1000:.6f} |")
lines+=['',
 '现有目标为 q_SOURCE × REAL/q_REAL。全部94400次出现中，缩放PSD变化最大6.66e-16，噪声RMS缩放恒等式误差1.30e-18 V。目标PSD对REAL TRAIN本区的W1为0.00037–0.00177，但目标噪声RMS距离由约0.001–0.002 mV增至0.031–0.034 mV；仅全局PSD排序仍保留这一偏差。噪声目标构造与父事件关联离散是两个不同问题。','',
 '下一步固定一种联合电荷与PSD的统计耦合，保持原出现次数、源人口和模型验收线，用TRAIN目标检查同时约束噪声缩放和父事件目标离散度，之后才决定响应训练。不会只换PSD排序就重复训练，也不会将模型目标标记完成。','',
 f"原生退出0；worker {n['actual_worker_PID']} / supervisor {n['supervisor_PID']}。科学计时{n['elapsed_s']:.6f}秒，worker自报峰值{n['actual_worker_RSS_peak_bytes']} B，轮询作业树峰值{n['job_tree_RSS_peak_bytes']} B；两者采样方式不同，观测失败0。新增响应训练/生成/密度拟合、EVAL读取和CAL解码均为0。",'',
 '这些结果条件于原固定开发人口及评价器，且未作多重比较校正；不能代表独立采集泛化。','',
 '[协议](tasks/'+STAGE+'.md) · [35项判别及配对差](work/'+STAGE+'/CANONICAL_C2ST.json) · [SOURCE八区指标](work/'+STAGE+'/SOURCE_METRICS.json) · [训练目标诊断](work/'+STAGE+'/TEACHER_DIAGNOSTIC.json) · [原生回执](work/'+STAGE+'/numeric_worker.py.NATIVE_RECEIPT.json) · [下一步](reviews/R51_ROOT_DIRECTION_001.json)','',
 '![五视图](work/'+STAGE+'/figures/canonical_five_views.png)','',
 '![SOURCE条件PSD与噪声](work/'+STAGE+'/figures/source_conditional_PSD_noise.png)','',
 '![目标关联与噪声](work/'+STAGE+'/figures/teacher_coupling_and_noise.png)','']
(R/'R51_REPORT.md').write_bytes(('\n'.join(lines)).encode())
(S/'REPRODUCTION.md').write_bytes(('# R51 复现\n\n在独立目录放置本轮脚本、配置、SOURCE_INPUT_PRIVATE.npz与固定定义。config引用原TRAIN、R43关联和R50输入/预测/指标，按INPUT_BINDING及R50备份恢复这些依赖并更新路径。运行 `python -B linux_supervise_post.py numeric_worker.py`，得到固定35项结果、目标诊断、私有预测和关联、三图及原生回执。已运行目录不可清除.used标记重跑。新增响应训练0；服务器CPU12/RSS48GiB。复现是新的科学执行，需另记实际资源及原生结果；不得覆盖本轮。\n').encode())
resource={'UTC':NOW,'stage':STAGE,'science_s':n['elapsed_s'],'actual_native_exit':0,'actual_worker_PID':n['actual_worker_PID'],'supervisor_PID':n['supervisor_PID'],
 'worker_self_and_sampled_peak_RSS_bytes':n['actual_worker_RSS_peak_bytes'],'sampled_job_tree_RSS_peak_bytes':n['job_tree_RSS_peak_bytes'],
 'memory_probe_failures':0,'response_training_runs':0,'new_generation_rows':0,'HGB_fits':32,'reused_HGB_results':3,'density_fits':0,'EVAL_read':0,'CAL_decode':0,'compute_budget_enforced':False}
write('work/'+STAGE+'/RESOURCE_RECEIPT.json',resource)
write('work/'+STAGE+'/FINAL_RECEIPT.json',{'UTC':NOW,'stage':STAGE,'status':'scientific_diagnostic_complete','science':resource,
 'scientific_decision':decision['scientific_decision'],'model_goal_complete':False,'required_files':[bind('work/'+STAGE+'/'+f) for f in ['CANONICAL_C2ST.json','TEACHER_DIAGNOSTIC.json','SOURCE_METRICS.json','REPRESENTATION_BINDING.json','INPUT_BINDING.json','NUMERIC_RECEIPT.json','RESOURCE_RECEIPT.json','FIGURE_QA.json','REPRODUCTION.md']],
 'publication_backup':'pending separate delivery receipts','new_response_candidate':False})
b=read('RESOURCE_BUDGET.json');block=b['improvement_block'];assert not any(x['stage']==STAGE for x in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({'stage':STAGE,'receipt':'work/'+STAGE+'/RESOURCE_RECEIPT.json','science_s':n['elapsed_s'],'response_training_runs_started':0,'new_response_updates':0,'status':'sealed_diagnostic','stage_science_complete':True})
block['actual_science_seconds_recorded']+=n['elapsed_s'];next(x for x in block['registered_stages'] if x['stage']==STAGE)['status']='sealed_diagnostic';write('RESOURCE_BUDGET.json',b)
job={'stage':STAGE,'status':'scientific_complete_delivery_pending','active_scientific_jobs':0,'actual_worker_PID':n['actual_worker_PID'],'supervisor_PID':n['supervisor_PID'],'actual_native_exit':0}
for fn in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 x=read(fn);x.update(current_job=job,current_stage_progress='scientific_complete_delivery_pending',application_goal_status='active',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,next_action=next_action,latest_completed_stage=STAGE,model_goal_complete=False,latest_completed_report='R51_REPORT.md',latest_scientific_decision=decision['scientific_decision'],latest_key_result_review='reviews/R51_ROOT_DIRECTION_001.json')
 if fn=='PROGRAM.json':x.update(status='R51_scientific_complete_delivery_pending_goal_active',updated_at_utc=NOW)
 else:x.update(project_status='R51_scientific_complete_goal_active',current_activity='root_delivery_then_joint_teacher_protocol',current_direction=next_action,updated_UTC=NOW)
 write(fn,x)
for fn,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 x=read(fn);a=next(v for v in x[key] if v[ident]==STAGE);a.update(status='scientific_complete_delivery_pending',scientific_decision=decision['scientific_decision'],science_s=n['elapsed_s'],native_exit=0,final_receipt='work/'+STAGE+'/FINAL_RECEIPT.json',model_goal_complete=False);write(fn,x)
for fn in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
 p=R/fn;txt=p.read_text('utf8');first,tail=txt.split('## 以下为历史记录',1)
 first='# 当前执行：R51诊断完成，长期目标active\n\n共同ADC表示下同处理REAL五视图通过区间要求，SOURCE仍失败。原94400行关联的离散度可降低，但目标按源电荷缩放的噪声冲突仍存在；下一轮先固定联合电荷/PSD构造并通过TRAIN目标检查。原A–F不变，主助手独立执行、无子智能体、无计算预算，服务器CPU12/RSS48GiB/GPU10GiB及单科学作业、本机E/D限制保持。本轮科学终点已完成，当前完成私有Git和D盘交付。详见R51_REPORT.md与reviews/R51_ROOT_DIRECTION_001.json。\n\n'
 p.write_bytes((first+'## 以下为历史记录'+tail).encode())
print(json.dumps({'science_s':n['elapsed_s'],'cumulative_science_s':block['actual_science_seconds_recorded'],'model_goal_complete':False,'direction':decision['scientific_decision']},indent=2))
