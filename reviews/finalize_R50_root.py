from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
ROOT=Path(__file__).resolve().parents[1];STAGE='R50_FROZEN_VOLTAGE_ATTRIBUTION_001';S=ROOT/'work'/STAGE
NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((ROOT/p).read_text('utf8'))
def write(p,x):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def bind(p):
    p=ROOT/p;return {'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
native=read('work/'+STAGE+'/numeric_worker.py.NATIVE_RECEIPT.json');m=read('work/'+STAGE+'/METRICS.json');d=read('work/'+STAGE+'/DESCRIPTIVE_METRICS.json')
assert native['actual_worker_native_exit']==0 and native['worker_self_receipt']['native_exit']==0
assert native['termination_reason'] is None and m['HGB_fits']==24
assert native['actual_worker_PID']==10615 and native['supervisor_PID']==10614
assert native['memory_probe_failures']==0 and len(list((S/'figures').glob('*.png')))==3
figures=[bind(str(p.relative_to(ROOT))) for p in sorted((S/'figures').glob('*.png'))]
write('work/'+STAGE+'/FIGURE_QA.json',{'UTC':NOW,'reviewer':'/root','actual_image_opened':True,
    'observation':'All three images visually opened; titles, legends, axes and confidence bars readable, no clipping or overlap. Full fixed population CDF tails retained.',
    'figures':figures,'scientific_acceptance_from_layout':False})
decision={'UTC':NOW,'stage':STAGE,'scientific_decision':'ADC_float_representation_confound_supported_frozen_diagnostic_complete',
    'RAW_ADC_code_changes':0,'RAW_ADC_voltage_RMS_mean_V':d['paired_changes']['RAW_LEGACY_ADC']['RMS_vs_same_RAW_V']['mean'],
    'RAW_native_BA':m['results']['RAW_REFERENCE__native_voltage']['BA'],
    'RAW_ADC_native_BA':m['results']['RAW_LEGACY_ADC__native_voltage']['BA'],
    'RAW_ADC_canonical_BA':m['results']['RAW_LEGACY_ADC__canonical_ADC_voltage']['BA'],
    'completion_post_native':m['results']['COMPLETION_POST__native_voltage'],
    'completion_post_canonical':m['results']['COMPLETION_POST__canonical_ADC_voltage'],
    'ADC_pre_post_code_disagreements':d['ADC_identity'],
    'original_R49_acceptance_rewritten':False,'new_accepted_response_candidate':False,'model_goal_complete':False,
    'remaining':'SOURCE full shape, PSD/Q-conditional width, absolute source noise and source/temporal constraints remain unresolved.',
    'next_action':'Register common ADC code representation prospectively for future candidates and controls; then isolate the SOURCE conditional PSD/width and noise mismatch before choosing a new response-training mechanism. Keep original raw-voltage results alongside canonical diagnostics; no historical threshold or result rewriting.'}
write('reviews/R50_ROOT_DIRECTION_001.json',decision)
lines=['# R50：ADC 表示干扰已得到受控验证','',
 '24个预登记HGB比较已在服务器完成，新增响应训练、生成和密度拟合均为0。原始REAL参考只经过旧float32 ADC后，全部960000个样点的整数码保持不变，但BA从0.546升到0.838；平均波形RMS变化仅1.5624e-8 V。共同按整数码以float64重建后，BA恢复0.546，特征和预测与相同重建的RAW完全一致。这支持数字表示产生了强判别信号。','',
 '主候选REAL输出原电压BA=0.828，共同重建后0.526，95%区间[0.4745,0.5797]；配对变化为-0.302，95%区间[-0.3502,-0.2379]。固定噪声对照0.830降为0.538。基线均值、基线RMS、去基线电压及Q/PSD/峰值的固定诊断区间均落在原工程容差内；这不等于全部响应验收通过。','',
 '| 人口 | 原电压 BA [95% CI] | 共同 ADC 重建 BA [95% CI] |','|---|---|---|']
for name in ['RAW_REFERENCE','RAW_LEGACY_ADC','FIXED_PRE','FIXED_POST','COMPLETION_PRE','COMPLETION_POST']:
    a=m['results'][name+'__native_voltage'];b=m['results'][name+'__canonical_ADC_voltage']
    fmt=lambda r:f"{r['BA']:.3f} [{r['BA_95CI'][0]:.4f}, {r['BA_95CI'][1]:.4f}]"
    lines.append(f'| {name} | {fmt(a)} | {fmt(b)} |')
lines += ['', '旧float32 ADC在模型量化前后还造成少量边界码差异：固定对照161/960000、主候选67/960000样点。因而不能笼统声称所有模型数组都只是无损换表示；RAW ADC-only对照的码变化确实为0。两臂原ADC重算与保存输出逐位一致。', '',
 '原R49失败结论保持；R50是原因诊断，不是重新验收。SOURCE形状BA 0.780、PSD W1 0.02890、必需Q3宽度未定义及源条件约束失败仍需解决。未打开EVAL或新增CAL解码。分组区间条件于固定评价器与当前采集，24个固定比较属于探索性机制诊断，不宣称多重比较校正或独立采集泛化。','',
 '下一步将ADC整数码及共同浮点重建作为前瞻性处理定义写入新协议，保留旧原电压指标；再定位SOURCE条件PSD/宽度与噪声失配，按证据选择新响应机制。','',
 f"服务器原生退出0；worker {native['actual_worker_PID']} / supervisor {native['supervisor_PID']}，科学作业树计时{native['elapsed_s']:.6f}秒，RSS峰值{native['job_tree_RSS_peak_bytes']} B，观测缺口0。本轮固定科学终点完成，长期目标继续active。",'',
 '[长期目标与结束条件](LONG_TERM_GOAL_20261010.md) · [预登记协议](tasks/'+STAGE+'.md) · [完整24项指标](work/'+STAGE+'/METRICS.json) · [归因描述](work/'+STAGE+'/DESCRIPTIVE_METRICS.json) · [原生回执](work/'+STAGE+'/numeric_worker.py.NATIVE_RECEIPT.json) · [后续方向](reviews/R50_ROOT_DIRECTION_001.json)','',
 '![电压归因](work/'+STAGE+'/figures/voltage_attribution.png)','',
 '![诊断分解](work/'+STAGE+'/figures/baseline_pulse_decomposition.png)','',
 '![全人口描述](work/'+STAGE+'/figures/paired_population_descriptions.png)','']
(ROOT/'R50_REPORT.md').write_text('\n'.join(lines),'utf8')
(S/'REPRODUCTION.md').write_text('# R50 复现\n\n在既有服务器Python环境下，把本目录的INPUT_PRIVATE.npz、config.json和三个Python脚本放在新的独立目录，运行 `python -B linux_supervise_post.py numeric_worker.py`。原已执行目录含.used标记，不能删除标记后重跑。输入来自INPUT_BINDING.json，原R49文件保持。固定24个HGB、原分组索引和种子，0响应训练。输出包含METRICS、DESCRIPTIVE_METRICS、三个图及原生进程回执。服务器CPU12/RSS48GiB；每次重跑须独立记录，不以重跑替代失败历史。\n','utf8')
resource={'UTC':NOW,'stage':STAGE,'science_s':native['elapsed_s'],'actual_native_exit':0,
          'actual_worker_PID':native['actual_worker_PID'],'supervisor_PID':native['supervisor_PID'],
          'job_tree_RSS_peak_bytes':native['job_tree_RSS_peak_bytes'],'memory_probe_failures':0,
          'response_training_runs':0,'new_generation_rows':0,'HGB_fits':24,'density_fits':0,'EVAL_read':0,'CAL_decode':0,
          'compute_budget_enforced':False,'elapsed_source':'supervisor native completed job, not SSH wall time'}
write('work/'+STAGE+'/RESOURCE_RECEIPT.json',resource)
write('work/'+STAGE+'/FINAL_RECEIPT.json',{'UTC':NOW,'stage':STAGE,'status':'scientific_diagnostic_complete',
       'science':resource,'scientific_decision':decision['scientific_decision'],'model_goal_complete':False,
       'required_files':[bind('work/'+STAGE+'/'+n) for n in ['METRICS.json','DESCRIPTIVE_METRICS.json','SELECTION_BINDING.json','INPUT_BINDING.json','NUMERIC_RECEIPT.json','FIGURE_QA.json','RESOURCE_RECEIPT.json','REPRODUCTION.md']],
       'publication_backup':'pending separate delivery receipts','new_response_candidate':False})
budget=read('RESOURCE_BUDGET.json');block=budget['improvement_block']
assert not any(x['stage']==STAGE for x in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({'stage':STAGE,'receipt':'work/'+STAGE+'/RESOURCE_RECEIPT.json','science_s':native['elapsed_s'],
    'response_training_runs_started':0,'new_response_updates':0,'status':'sealed_diagnostic','stage_science_complete':True})
block['actual_science_seconds_recorded']+=native['elapsed_s']
next(x for x in block['registered_stages'] if x['stage']==STAGE)['status']='sealed_diagnostic'
write('RESOURCE_BUDGET.json',budget)
job={'stage':STAGE,'status':'scientific_complete_delivery_pending','active_scientific_jobs':0,
     'actual_worker_PID':10615,'supervisor_PID':10614,'actual_native_exit':0}
for fn in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    obj=read(fn);obj['current_job']=job;obj['current_stage_progress']='scientific_complete_delivery_pending'
    obj['application_goal_status']='active';obj['active_scientific_jobs']=0;obj['active_remote_scientific_jobs']=0;obj['active_local_scientific_jobs']=0
    obj['next_action']=decision['next_action'];obj['latest_completed_stage']=STAGE;obj['model_goal_complete']=False
    if fn=='PROGRAM.json':obj.update(status='R50_scientific_complete_delivery_pending_goal_active',latest_completed_report='R50_REPORT.md',latest_scientific_decision=decision['scientific_decision'],latest_key_result_review='reviews/R50_ROOT_DIRECTION_001.json',updated_at_utc=NOW)
    else:obj.update(project_status='R50_scientific_complete_goal_active',current_activity='root_delivery_then_next_registered_protocol',current_direction=decision['next_action'],updated_UTC=NOW)
    write(fn,obj)
for fn,key,identity in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    obj=read(fn);row=next(x for x in obj[key] if x[identity]==STAGE)
    row.update(status='scientific_complete_delivery_pending',scientific_decision=decision['scientific_decision'],science_s=native['elapsed_s'],native_exit=0,
               final_receipt='work/'+STAGE+'/FINAL_RECEIPT.json',model_goal_complete=False);write(fn,obj)
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','SERVER_RESOURCE_POLICY.md']:
    p=ROOT/name;t=p.read_text('utf8');t=t.replace('# 当前执行：主助手独立推进长期目标与 R50','# 当前执行：长期目标 active，R50 诊断已完成',1)
    t=t.replace('当前按 tasks/R50_FROZEN_VOLTAGE_ATTRIBUTION_001.md 登记冻结输出的24个诊断比较，0响应训练。','R50 的24个冻结诊断已完成，确认旧ADC浮点表示干扰；模型仍未达A–F。实际结果见R50_REPORT.md，后续按reviews/R50_ROOT_DIRECTION_001.json推进，当前完成本轮交付。',1)
    p.write_text(t,'utf8')
budget=read('RESOURCE_BUDGET.json');budget['policy_sha256']=bind('SERVER_RESOURCE_POLICY.md')['sha256'];write('RESOURCE_BUDGET.json',budget)
print(json.dumps({'stage':STAGE,'science_s':native['elapsed_s'],'recorded_science_total_s':block['actual_science_seconds_recorded'],'actual_native_exit':0,'goal':'active','model_goal_complete':False,'figures_reviewed':3},indent=2))
