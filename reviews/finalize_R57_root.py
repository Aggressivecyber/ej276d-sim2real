from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path(__file__).resolve().parents[1];ID='R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001';S=R/'work'/ID;NOW=datetime.now(timezone.utc).isoformat()
def read(n):return json.loads((R/n).read_text('utf8'))
def local(n):return json.loads((S/n).read_text('utf8'))
def write(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def bind(p):return {'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def label(b):return '总体' if b is None else '下方支持外' if b==-1 else '上方支持外' if b==8 else 'Q'+str(b+1)
v=local('RESULT.json');native=local('numeric_worker.py.NATIVE_RECEIPT.json');coupling=local('COUPLING_RECEIPT.json');qa=local('FIGURE_QA.json');d=read('reviews/R57_ROOT_DIRECTION_001.json')
assert v['status']==local('NUMERIC_RECEIPT.json')['status']=='complete'
assert native['actual_worker_native_exit']==native['worker_self_receipt']['native_exit']==0 and native['termination_reason'] is None and native['handshake_before_imports']
assert not local('LIVE_OBSERVATION.json')['actual_process']['exists'] and qa['all_figures_opened'] and qa['layout_pass']
assert v['decision']==d['scientific_decision'] and not d['model_goal_complete']
assert v['counter']=={'coupling_constructions':1,'teacher_edges':27724,'new_response_updates':0,'HGB_fits':0,'density_fits':0,'ODE_generation':0}
science=native['elapsed_s'];p=v['PSD_summary']
lines=['# R57：原 TRAIN 父事件等权教师预检','',d['conclusion_zh'],'',
    '本轮是一次固定TRAIN统计教师诊断，不是已训练的神经变换或A–F评价。原2000个验收SOURCE的身份、组成、顺序和权重保持；没有读取DEV/EVAL/CAL或TRAIN包的source_val/target_val。','',
    '每个原电荷区域按实际Q划为16个等质量组，再按signed PSD单调耦合。每个区域的整数总质量为Ns×Nr，每个源父事件有Nr单位，每个REAL父事件有Ns单位；换算到全体源后，每个源精确为1/6593。REAL仅在区间内等权，其总体区间质量随SOURCE。分数边是条件教师概率，不是物理逐事件真值，也不改变原源权重。','',
    '| 原区域 | SOURCE父事件 | REAL父事件 | 耦合边 | 源/教师总体质量 | 原REAL总体质量 |','|---|---:|---:|---:|---:|---:|']
for row in coupling['per_bin']:lines.append(f"| {label(row['raw_bin0'])} | {row['SOURCE_N']} | {row['REAL_N']} | {row['edges']} | {row['global_bin_mass']:.8f} | {row['REAL_original_bin_mass']:.8f} |")
lines+=['',f"十区整数边缘全部严格成立；浮点源边缘最大误差{coupling['source_float_marginal_error_max']:.8g}。旧训练exp(lambda*h)曝光倾斜已从本轮教师测度去除；这属于训练采样政策改变，不能改变最终评价人口。",'',
    '| 预登记教师条件 | 结果 |','|---|---|']
for key,value in v['gates'].items():lines.append(f"| {key} | {'通过' if value else '未通过'} |")
lines+=['',f"总体PSD W1：preADC {p['preADC_overall_W1']:.9f}，共同规范ADC {p['canonical_overall_W1']:.9f}，均超过0.01。共同Q1–7等权平均分别为{p['preADC_common_Q1_Q7_mean_W1']:.9g}和{p['canonical_common_Q1_Q7_mean_W1']:.9g}。",'',
    '| 原区域 | preADC PSD W1（源分箱） | ADC PSD W1（输出自身分箱） |','|---|---:|---:|']
post={row['raw_bin0']:row for row in v['canonical_vs_REAL_own_bins']}
for row in v['preADC_vs_REAL_source_bins']:
    b=row['raw_bin0'];lines.append(f"| {label(b)} | {row['features']['PSD']['W1']:.9g} | {post[b]['features']['PSD']['W1']:.9g} |")
lines+=['',f"教师与实际匹配REAL的总体PSD W1为{v['teacher_vs_actual_matched_REAL'][0]['features']['PSD']['W1']:.9g}；实际匹配REAL与全量REAL的W1为{v['actual_matched_REAL_vs_full_REAL'][0]['features']['PSD']['W1']:.9f}，与教师preADC总体差一致到舍入精度。此构造的总体差来自保留的区间人口比例；精确条件匹配本身不消除这个差。",'',
    '| h==1源组 | 父事件数 | 原源谷概率 | 教师preADC | 教师规范ADC | 允许增量 |','|---|---:|---:|---:|---:|---:|']
for row in v['valleys']:lines.append(f"| Q{row['bin']} | {row['h_one_source_parents']} | {row['original_source_fraction']:.8f} | {row['teacher_preADC_fraction']:.8f} | {row['teacher_canonical_fraction']:.8f} | {row['limit']:.2f} |")
lines+=['','谷区来自冻结R3B REAL TRAIN双峰定义，h==1是原源沉积组成代理。该TRAIN预检以原源为参照，比部分原DEV对BASE的规则更严格；失败只否定本构造，不能证明原A–F不可能，也不能将密度峰当粒子真值。','',
    '| 原区域 | 原源越界父事件 | 教师越界边 | 有越界教师的源父事件 | 教师越界总质量 |','|---|---:|---:|---:|---:|']
for row in v['ADC_support']:lines.append(f"| {label(row['raw_bin0'])} | {row['source_input_range_exceeded_parents']} | {row['teacher_range_exceeded_edges']} | {row['source_parents_with_any_teacher_range_excess']} | {row['teacher_range_exceeded_mass']:.8f} |")
lines+=['','ADC范围为[-0.5,0.5] V。全部1217条越界边来自高端支持外，关联575个源父事件；加权总质量为0.0869073，该区域内占0.6241608。原高端源已有816个输入越界，所以这些计数不能解释成1217个新裁剪事件；每条边和原父事件的分母不同。所有越界与支持外质量保留，preADC及规范ADC没有无效事件。','',
    f"冻结脉冲基底的最大相对Q误差{v['relative_Q_error_max']:.9g}，PSD保持误差{v['PSD_preservation_error_max']:.9g}；0:100基线和400:480尾部与所匹配REAL逐元素相同。八区baseline RMS/ACF1 W1均在1e-12以内。",'',
    '| 区域 | 源Q→教师Q加权秩相关 | 源PSD→教师PSD加权秩相关 | ADC后PSD秩相关 |','|---|---:|---:|---:|']
for b in [None,-1,*range(8),8]:
    a=v['source_associations'][str(b)];lines.append(f"| {label(b)} | {a['Q_preADC']:.8f} | {a['PSD_preADC']:.8f} | {a['PSD_canonical']:.8f} |")
lines+=['','这些秩相关是有限耦合分布的加权midrank描述，重复边不是独立源事件。总体PSD关联明显弱于多数区内关联，不能将区内排序当作原C源保真验收。完整480点电压/峰值归一化形状mean与SD、峰值、logQ、基线、九个固定边界及逐边特征均已保存；支持外图中可见宽的正负修正波瓣，条件PSD相等并不建立完整波形等效。','',
    f"一次CPU科学作业原生退出0，耗时{science:.6f}秒；worker峰RSS {native['actual_worker_RSS_peak_bytes']} B，作业树采样峰值{native['job_tree_RSS_peak_bytes']} B。RSS缺测{native['memory_probe_failures']}次，原因保留unknown；无资源终止。0响应训练/ODE/HGB/密度拟合，6张最终PNG已实际打开核验。",'',
    d['next_action'],'',
    '[协议](tasks/'+ID+'.md) · [结果](work/'+ID+'/RESULT.json) · [整数边缘](work/'+ID+'/COUPLING_RECEIPT.json) · [复现](work/'+ID+'/REPRODUCTION.md) · [科学回执](work/'+ID+'/FINAL_RECEIPT.json)','']
figures=['population_and_margins','TRAIN_PSD_CDF','ADC_and_valley_conditions','TRAIN_full_waveform_V','TRAIN_full_waveform_U','fixed_teacher_examples']
for name in figures:lines+=['!['+name+'](work/'+ID+'/'+name+'.png)','']
(R/'R57_REPORT.md').write_text('\n'.join(lines),'utf8')
resource={'UTC':NOW,'stage':ID,'science_s':science,'native_receipts':[native],'response_training_runs_started':0,'new_response_updates':0,'new_HGB_fits':0,'new_density_fits':0,'counter':v['counter'],'DEV_read':0,'EVAL_read':0,'CAL_decode':0,'compute_budget_enforced':False,'worker_peak_RSS_bytes':native['actual_worker_RSS_peak_bytes'],'sampled_tree_peak_RSS_bytes':native['job_tree_RSS_peak_bytes'],'memory_probe_failures':native['memory_probe_failures']}
write('work/'+ID+'/RESOURCE_RECEIPT.json',resource)
required=['config.json','PROTOCOL.md','INPUT_BINDING.json','numeric_worker.py','render_teacher.py','coupling_core.py','weighted_metrics.py','noise_core.py','teacher_core.py','canonical.py','RESULT.json','COUPLING_RECEIPT.json','ARTIFICIAL_CONSTRUCTION_CHECK.json','COUPLING_PRIVATE.npz','FEATURES_PRIVATE.npz','FIXED_EXAMPLES_PRIVATE.npz','FIGURE_QA.json','RESOURCE_RECEIPT.json','REPRODUCTION.md']+[n+'.png' for n in figures]
write('work/'+ID+'/FINAL_RECEIPT.json',{'UTC':NOW,'stage':ID,'status':'science_complete_delivery_pending','model_goal_complete':False,'scientific_decision':v['decision'],'required_files':[bind(S/n) for n in required],'private_Git_D_delivery':'pending separate receipts'})
b=read('RESOURCE_BUDGET.json');block=b['improvement_block'];assert not any(x['stage']==ID for x in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({'stage':ID,'receipt':'work/'+ID+'/RESOURCE_RECEIPT.json','science_s':science,'response_training_runs_started':0,'new_response_updates':0,'status':'fixed_teacher_closed','stage_science_complete':True})
block['actual_science_seconds_recorded']+=science
next(x for x in block['registered_stages'] if x['stage']==ID).update(status='fixed_teacher_closed',active_scientific_jobs=0,actual_science_seconds=science,stage_science_complete=True);write('RESOURCE_BUDGET.json',b)
job={'stage':ID,'status':'science_complete_delivery_pending','actual_worker_PID':native['actual_worker_PID'],'supervisor_PID':native['supervisor_PID'],'actual_native_exit':0,'active_scientific_jobs':0}
for n in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    x=read(n);x.update(status='R57_science_complete_delivery_pending_goal_active',current_job=job,current_stage_progress='science_complete_delivery_pending',latest_completed_stage=ID,latest_completed_report='R57_REPORT.md',latest_scientific_decision=v['decision'],latest_key_result_review='reviews/R57_ROOT_DIRECTION_001.json',next_action=d['next_action'],application_goal_status='active',model_goal_complete=False,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0)
    if n=='PROGRAM.json':x['updated_at_utc']=NOW
    else:x.update(project_status=x['status'],current_activity='root_R57_delivery',current_direction=d['next_action'],updated_UTC=NOW)
    write(n,x)
for n,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    x=read(n);next(a for a in x[key] if a[ident]==ID).update(status='science_complete_delivery_pending',scientific_decision=v['decision'],science_s=science,native_exit=0,final_receipt='work/'+ID+'/FINAL_RECEIPT.json',model_goal_complete=False,active_scientific_jobs=0);write(n,x)
top='# 当前执行：R57 等权教师未通过，长期目标 active\n\n'+d['conclusion_zh']+' 原A–F保持，模型目标未完成。当前完成私有 Git 与 D 盘交付。'+d['next_action']+' 主助手独立，无子智能体，无计算配额；机器安全、单作业及本机E/D限制保持。\n\n'
for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
    path=R/n;tail=path.read_text('utf8').split('## 以下为历史记录',1)[1];path.write_text(top+'## 以下为历史记录'+tail,'utf8')
print(json.dumps({'decision':v['decision'],'science_s':science,'cumulative_science_s':block['actual_science_seconds_recorded'],'model_goal_complete':False},indent=2))
