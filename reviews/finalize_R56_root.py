from pathlib import Path
from datetime import datetime, timezone
import json,hashlib
R=Path(__file__).resolve().parents[1]
ID='R56_TRAIN_ENDPOINT_LEARNING_DIAGNOSTIC_001';S=R/'work'/ID
NOW=datetime.now(timezone.utc).isoformat()
def read(n):return json.loads((R/n).read_text('utf8'))
def local(n):return json.loads((S/n).read_text('utf8'))
def write(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def bind(p):return {'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
v=local('RESULT.json');native=local('diagnose_train.py.NATIVE_RECEIPT.json');cov=local('COVERAGE.json');paths=local('PATH_SUMMARY.json');teachers=local('TEACHER_POPULATIONS.json');qa=local('FIGURE_QA.json');d=read('reviews/R56_ROOT_DIRECTION_001.json')
assert v['status']==local('NUMERIC_RECEIPT.json')['status']=='complete'
assert native['actual_worker_native_exit']==native['worker_self_receipt']['native_exit']==0 and native['termination_reason'] is None and native['handshake_before_imports']
assert not local('LIVE_OBSERVATION.json')['actual_process']['exists']
assert qa['all_figures_opened'] and qa['layout_pass']
assert v['decision']==d['scientific_decision'] and not d['model_goal_complete']
assert v['counter']['updates']==0 and v['counter']['SOURCE_path_rows']==v['counter']['REAL_identity_path_rows']==94400 and v['counter']['SOURCE_ODE_rows']==6593
science=native['elapsed_s']
lines=['# R56：TRAIN 教师、覆盖与冻结 R54 学习误差','',d['conclusion_zh'],'',
    '本轮未训练或选择新模型；唯一 R54 权重保持，原 A–F 不变。没有读取 DEV/EVAL、重新解码 CAL，TRAIN 包的 source_val/target_val 字段也未解码。以下是训练人口的诊断，不能充当最终响应验收。','',
    '| 项目 | SOURCE | REAL |','|---|---:|---:|',f"| 全量合法 TRAIN | {cov['SOURCE_total']} | {cov['REAL_total']} |",f"| 实际见过的唯一父事件 | {cov['SOURCE_seen']} | {cov['REAL_seen']} |",f"| 原训练曝光次数 | {cov['SOURCE_total_occurrences']} | {cov['REAL_total_occurrences']} |",'',
    '旧 R23 采样器按八区间等概率取样，区间内 SOURCE 使用既有 exp(lambda*h)，REAL 均匀抽样。后续排序/训练沿用曝光流。2840 个源事件位于原电荷支持外，全部未被训练采样；支持内3753个中仅1个未见。原字段与本轮电荷重算的分箱完全一致。全量源没有在本轮被删除或重加权；h 只作为已有沉积组成代理，不解释为独立粒子真值。','',
    '| 原区间 | 源父事件 | 已见源 | REAL父事件 | 已见REAL |','|---|---:|---:|---:|---:|']
for row in cov['by_raw_bin']:
    b=row['raw_bin0'];label='下方支持外' if b==-1 else '上方支持外' if b==8 else 'Q'+str(b+1)
    lines.append(f"| {label} | {row['SOURCE_parents']} | {row['SOURCE_seen']} | {row['REAL_parents']} | {row['REAL_seen']} |")
lines+=['','全周期94400和末周期89600条脉冲教师均保留原目标PSD与基线/尾部，且未增加新的目标人口。下表采用原全周期曝光：R54每个源父事件只生成一次，重复曝光共享同一噪声实现，因此这些重复不是独立样本，也没有逐事件物理配对真值。教师与REAL的W1来自同一原电荷区间，模型比较与路径误差仍是不同对象。','',
    '| 区间 | 教师→REAL W1 | R54→教师 W1 | R54→REAL W1 | B | G | abs(B)/D | 标记 |','|---|---:|---:|---:|---:|---:|---:|---|']
for m in v['markers']:
    lines.append('| Q'+str(m['bin'])+' | '+' | '.join(f"{m[k]:.6f}" if m[k] is not None else '未定义' for k in ['teacher_REAL_W1','R54_teacher_W1','R54_REAL_W1','B','G','abs_B_over_D'])+' | '+('命中' if m['learning_marker'] else '未命中')+' |')
lines+=['','同一组固定输出按原源父事件等权统计，与旧训练曝光统计的结果如下。此处只是报告两种既有测度，没有改变验收源权重。','',
    '| 区间 | 原源父事件数 | 父事件等权 PSD W1 | 旧曝光 PSD W1 |','|---|---:|---:|---:|']
for m in v['markers']:
    a=v['output_groups']['all']['preADC_vs_REAL_source_bins'][m['bin']]
    lines.append(f"| Q{m['bin']} | {a['N']} | {a['features']['PSD']['W1']:.8f} | {m['R54_REAL_W1']:.8f} |")
lines+=['',f"预登记学习标记命中 {v['learning_markers']}/6；教师偏差标记 {v['teacher_bias_markers']}/6。学习规则同时要求教师W1≤0.01、模型对教师和REAL的W1都>0.01、\u007cB\u007c/D≥0.20、B与G同号；至少4/6命中才支持预登记的显式PSD切向机制方向。不会把某一子条件改称规则通过。",'',
    '| 冻结路径 | PSD速度RMSE | 带符号残差均值 | 原480维MSE | PSD方向残差能量占比 |','|---|---:|---:|---:|---:|']
for key,label in [('SOURCE','SOURCE'),('REAL_identity','REAL identity')]:
    p=paths[key]['overall'];lines.append(f"| {label} | {p['PSD_RMSE']:.8f} | {p['B_signed_mean']:.8f} | {p['MSE480']:.8f} | {100*p['PSD_mode_energy_fraction']:.5f}% |")
lines+=['','PSD速度是相同单位电荷坐标中的线性泛函；能量占比按原480维MSE的欧氏切向几何定义。最终场在原首周期训练路径上的残差不能等同于实际积分轨迹误差，更不能由占比小推断PSD不重要。','',
    '| 全部源父事件分组 | 父事件数 | preADC越ADC范围数 | preADC无效数 | 对全量REAL总体PSD W1 |','|---|---:|---:|---:|---:|']
for name in ['all','seen','unseen','inside','outside']:
    a=v['output_groups'][name];w=a['preADC_vs_REAL_own_bins'][0]['features']['PSD']['W1']
    lines.append(f"| {name} | {a['parents']} | {a['range_exceeded_preADC']} | {a['invalid_preADC']} | "+(f'{w:.8f}' if w is not None else '未定义')+' |')
lines+=['','这些分组各自的电荷人口不同，不能把总体W1差异解释成单独的泛化因果效应。范围外从未有训练教师，其误差只能报告冻结场外推和与全量REAL的描述；不能虚构范围外教师。按原源分箱、实际输出分箱以及规范ADC后输出分箱的完整统计均在RESULT.json，未删除失败或越界事件。','',
    f"本轮固定 Heun128 共生成6593个TRAIN源端点，最大float32相对电荷误差 {v['q_relative_error_f32_max']:.9g}；这不是新候选的数值门或DEV成功声明。原权重在前后哈希绑定相同。",'',
    f"科学原生退出0，{science:.6f}秒；188800次路径行前向、6593次源ODE、{v['counter']['field_forward_rows']}次总场行前向、184000条教师描述。worker峰RSS {native['actual_worker_RSS_peak_bytes']} B，作业树采样峰值 {native['job_tree_RSS_peak_bytes']} B，RSS缺测 {native['memory_probe_failures']} 次。无训练更新、HGB或密度拟合。",'',
    d['next_action'],'',
    '[协议](tasks/'+ID+'.md) · [结果](work/'+ID+'/RESULT.json) · [覆盖](work/'+ID+'/COVERAGE.json) · [路径残差](work/'+ID+'/PATH_SUMMARY.json) · [教师人口](work/'+ID+'/TEACHER_POPULATIONS.json) · [复现](work/'+ID+'/REPRODUCTION.md)','']
for name in ['coverage','TRAIN_PSD_CDF','TRAIN_path_residual','TRAIN_full_waveform','direction_markers']:lines+=['!['+name+'](work/'+ID+'/'+name+'.png)','']
(R/'R56_REPORT.md').write_text('\n'.join(lines),'utf8')
resource={'UTC':NOW,'stage':ID,'science_s':science,'native_receipts':[native],'response_training_runs_started':0,'new_response_updates':0,'new_HGB_fits':0,'new_density_fits':0,'counter':v['counter'],'DEV_read':0,'EVAL_read':0,'CAL_decode':0,'compute_budget_enforced':False,'worker_peak_RSS_bytes':native['actual_worker_RSS_peak_bytes'],'sampled_tree_peak_RSS_bytes':native['job_tree_RSS_peak_bytes'],'memory_probe_failures':native['memory_probe_failures']}
write('work/'+ID+'/RESOURCE_RECEIPT.json',resource)
required=['config.json','PROTOCOL.md','INPUT_BINDING.json','diagnose_train.py','render_diagnostic.py','network.py','noise_core.py','source_flow_core.py','teacher_core.py','canonical.py','RESULT.json','COVERAGE.json','PATH_SUMMARY.json','TEACHER_POPULATIONS.json','PATH_ROWS_PRIVATE.npz','TRAIN_ENDPOINT_PRIVATE.npz','FEATURES_PRIVATE.npz','FIGURE_QA.json','RESOURCE_RECEIPT.json','REPRODUCTION.md']+[n+'.png' for n in ['coverage','TRAIN_PSD_CDF','TRAIN_path_residual','TRAIN_full_waveform','direction_markers']]
write('work/'+ID+'/FINAL_RECEIPT.json',{'UTC':NOW,'stage':ID,'status':'science_complete_delivery_pending','model_goal_complete':False,'scientific_decision':v['decision'],'required_files':[bind(S/n) for n in required],'private_Git_D_delivery':'pending separate receipts'})
b=read('RESOURCE_BUDGET.json');block=b['improvement_block'];assert not any(x['stage']==ID for x in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({'stage':ID,'receipt':'work/'+ID+'/RESOURCE_RECEIPT.json','science_s':science,'response_training_runs_started':0,'new_response_updates':0,'status':'fixed_diagnostic_complete','stage_science_complete':True})
block['actual_science_seconds_recorded']+=science
next(x for x in block['registered_stages'] if x['stage']==ID).update(status='fixed_diagnostic_complete',active_scientific_jobs=0,actual_science_seconds=science,stage_science_complete=True)
write('RESOURCE_BUDGET.json',b)
job={'stage':ID,'status':'science_complete_delivery_pending','actual_worker_PID':native['actual_worker_PID'],'supervisor_PID':native['supervisor_PID'],'actual_native_exit':0,'active_scientific_jobs':0}
for n in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    x=read(n);x.update(status='R56_science_complete_delivery_pending_goal_active',current_job=job,current_stage_progress='science_complete_delivery_pending',latest_completed_stage=ID,latest_completed_report='R56_REPORT.md',latest_scientific_decision=v['decision'],latest_key_result_review='reviews/R56_ROOT_DIRECTION_001.json',next_action=d['next_action'],application_goal_status='active',model_goal_complete=False,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0)
    if n=='PROGRAM.json':x['updated_at_utc']=NOW
    else:x.update(project_status=x['status'],current_activity='root_R56_delivery',current_direction=d['next_action'],updated_UTC=NOW)
    write(n,x)
for n,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    x=read(n);next(a for a in x[key] if a[ident]==ID).update(status='science_complete_delivery_pending',scientific_decision=v['decision'],science_s=science,native_exit=0,final_receipt='work/'+ID+'/FINAL_RECEIPT.json',model_goal_complete=False,active_scientific_jobs=0);write(n,x)
top='# 当前执行：R56 TRAIN诊断完成，长期目标 active\n\n'+d['conclusion_zh']+' 原A–F保持，模型目标未完成。当前完成私有 Git 与 D 盘交付。'+d['next_action']+' 主助手独立，无子智能体，无计算配额；机器安全、单作业及本机E/D限制保持。\n\n'
for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
    p=R/n;tail=p.read_text('utf8').split('## 以下为历史记录',1)[1];p.write_text(top+'## 以下为历史记录'+tail,'utf8')
print(json.dumps({'decision':v['decision'],'science_s':science,'cumulative_science_s':block['actual_science_seconds_recorded'],'model_goal_complete':False},indent=2))
