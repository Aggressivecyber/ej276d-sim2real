from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path(__file__).resolve().parents[1];ID='R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001';S=R/'work'/ID;NOW=datetime.now(timezone.utc).isoformat()
read=lambda p:json.loads((R/p).read_text('utf8'))
local=lambda p:read('work/'+ID+'/'+p)
write=lambda p,v:(R/p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def bind(p):
    file=R/p;return {'path':file.relative_to(R).as_posix(),'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
result=local('RESULT.json');path=local('PATH_SUMMARY.json');baseline=local('BASELINE_JOINT_SUMMARY.json');loss=local('LOSS_COVERAGE_SUMMARY.json');psd=local('PSD_SUMMARY.json');projection=local('PROJECTION_SUMMARY.json')
native=local('diagnose_train.py.NATIVE_RECEIPT.json');failed=local('repairs/ADC_CHECK_001/diagnose_train.py.NATIVE_RECEIPT.json');review=local('FROZEN_ARTIFACT_REVIEW.json');qa=local('FIGURE_QA.json');direction=read('reviews/R62_ROOT_DIRECTION_001.json')
assert native['actual_worker_native_exit']==native['worker_self_receipt']['native_exit']==0 and native['termination_reason']is None
assert failed['actual_worker_native_exit']==1 and not local('LIVE_OBSERVATION.json')['actual_process']['exists']
assert review['pass']and qa['all_figures_opened']and qa['layout_pass']and qa['figure_count']==8
assert result['diagnostic_markers']==direction['diagnostic_markers']
science=native['elapsed_s']+failed['elapsed_s'];fmt=lambda x:format(x,'.8g')
label=lambda b:'below'if b==-1 else'above'if b==8 else'Q'+str(b+1)
find=lambda stage:next(v for v in path['rows']if v['domain']=='SOURCE'and v['stage']==stage and v['scope']=='all')
src=loss['comparisons']['SOURCE_REAL'];rr=loss['comparisons']['REAL_REAL'];ctrl=loss['comparisons']['REAL_control_REAL']
lines=['# R62：冻结完整 TRAIN 波形残差诊断','',
 '固定诊断完成。两个预设描述标记均触发：神经映射后150点附近有2/3项边界W1增量超过1 LSB；支持内7/8区的SOURCE基线协方差残差超过同处理REAL及全部32次经验参照。这些是TRAIN描述，不是因果检验或新模型验收。R61原A/B/C失败保持，长期目标仍需同一冻结方法满足原A–F。','',
 '冻结R61唯一50000步权重、五个执行模块及八个输入文件。全6593 SOURCE与21047 REAL保持原父顺序、等父权重和十个原始电荷区域，各执行一次固定噪声Heun128；共27640条实际神经前向、433次网络调用、7075840个数学场求值行。另完成32批已保存端点的重采样，无新增神经前向。0训练、求导、教师、HGB或密度拟合，0 DEV/EVAL/CAL解码。','',
 '逐父保留原输入、加噪初态、投影前神经端点、原约束投影后float32与共同ADC五个阶段。完整波形仅用于本次内存统计；保存全480坐标和差分、固定边界、均值/标准差、逐父紧凑量及实际输出SHA。中间非量化状态只用于路径描述，不替代原验收表示。','',
 '| 全TRAIN SOURCE固定边界 | 加噪初态W1 / LSB | 神经投影前W1 / LSB | 增量 / LSB | 最终共同ADC W1 / LSB |','|---|---:|---:|---:|---:|']
for j,name in enumerate(path['boundary_names']):
    a=find('noise_initial')['boundary_W1_V'][j]*16383;b=find('neural_raw')['boundary_W1_V'][j]*16383;c=find('canonical_ADC')['boundary_W1_V'][j]*16383
    lines.append('| '+name+' | '+fmt(a)+' | '+fmt(b)+' | '+fmt(b-a)+' | '+fmt(c)+' |')
lines+=['','预定判定只使用d1_149_150、d2_149、d2_150三项，结果依次为是/否/是。全部九项和所有时间坐标照列。全TRAIN含支持外与范围投影事件，不能把此表数值直接替代R61固定DEV的边界门。','',
 '| 原范围 | SOURCE N | REAL N | SOURCE协方差相对Frobenius误差 | 同处理REAL误差 | 32次经验参照[min,max] | 超出全部参照及REAL控制 |','|---|---:|---:|---:|---:|---:|---|']
for row in baseline['rows']:
    q=row['empirical_reference_quantiles'];name='all'if row['original_region']is None else label(row['original_region'])
    lines.append('| '+name+' | '+str(row['SOURCE_N'])+' | '+str(row['REAL_N'])+' | '+fmt(row['SOURCE_covariance_relative_Frobenius'])+' | '+fmt(row['REAL_control_covariance_relative_Frobenius'])+' | ['+fmt(q[0])+', '+fmt(q[-1])+'] | '+('是'if row['SOURCE_exceeds_all_32_references_and_REAL_control']else'否')+' |')
lines+=['','基线协方差由每父前100点先去自身均值后计算。经验参照从本TRAIN原REAL有放回抽取，样本数与相应SOURCE一致；它不是总体95%置信区间、独立采集或粒子真值。Q8未超过全部32次参照，保留该否定结果。','',
 '基线RMS的总体经验CDF接近，但完整ACF1–20与频谱仍显示差异：SOURCE在短滞后处和中间频带偏离REAL，同处理REAL较接近。已保存100×100协方差、全部31个逐父描述量和十个含正确Nyquist权重的频带。频率只以cycles/sample报告。不能从图中的周期外观直接宣称某个网络层是唯一原因。','',
 '| 原标准化全窗目标 | 完整经验分布损失 | 32批均值 | 32批SD |','|---|---:|---:|---:|']
for name,row in [('SOURCE–REAL',src),('REAL–REAL',rr),('同处理REAL–REAL',ctrl)]:lines.append('| '+name+' | '+fmt(row['full_empirical_loss'])+' | '+fmt(row['batch_loss_summary']['mean'])+' | '+fmt(row['batch_loss_summary']['SD'])+' |')
lines+=['','各批十个原区域分别抽32父事件，使用固定64方向和原SOURCE区域质量；完整分布使用精确不等样本经验W1。完整REAL–REAL为0，而有限批均值为'+fmt(rr['batch_loss_summary']['mean'])+'，显示明显抽样底噪。SOURCE的损失仍明显高于此描述性参照，不能把当前残差都解释成小批量误差。此处没有重新估计置信区间或训练收敛判据。','',
 '高端支持外区贡献完整SOURCE目标的'+format(direction['high_support_out_fraction_of_full_objective'],'.2%')+'。该标量贡献不证明内外梯度冲突；R60三个预定冲突标记未触发的结论保持。全部原父事件、区域质量及支持外区继续保留。','',
 '完整TRAIN SOURCE总体PSD W1='+fmt(psd['overall_SOURCE_W1'])+'，共同Q1–7等区均值='+fmt(psd['common_Q1_Q7_mean_W1'])+'；同处理REAL总体W1='+fmt(psd['overall_REAL_control_W1'])+'。这不覆盖原DEV验收。','',
 '| 原电荷区域 | SOURCE N | REAL N | SOURCE PSD W1 | 同处理REAL PSD W1 |','|---|---:|---:|---:|---:|']
for row in psd['regions']:lines.append('| '+label(row['region'])+' | '+str(row['SOURCE_N'])+' | '+str(row['REAL_N'])+' | '+fmt(row['SOURCE_W1'])+' | '+fmt(row['REAL_control_W1'])+' |')
lines+=['','源PSD与Q的Spearman分别为'+fmt(psd['source_PSD_rank'])+' / '+fmt(psd['source_Q_rank'])+'。Q3/Q4 h==1分母为482/435，两区硬谷比例仍为0；谷高边界±0.004邻域的输出质量为'+format(psd['valleys'][0]['output_high_edge_mass'],'.4%')+' / '+format(psd['valleys'][1]['output_high_edge_mass'],'.4%')+'。全十区和完整直方图范围保留，本轮未拟合峰位或峰宽。','',
 '| 全TRAIN投影 | 范围活跃事件/N | 电压端点采样总数 | 每事件修正RMS均值/V | 最大修正RMS/V | 最大preADC相对Q误差 |','|---|---:|---:|---:|---:|---:|']
for name in ['SOURCE','REAL_control']:
    v=projection[name];lines.append('| '+name+' | '+str(v['range_active_N'])+'/'+str(v['rows'])+' | '+str(v['rail_samples_total'])+' | '+fmt(v['projection_RMS_V']['mean'])+' | '+fmt(v['projection_RMS_V']['max'])+' | '+fmt(v['preADC_relative_Q_error_f32']['max'])+' |')
lines+=['','CPU/CUDA人工测试核验观测版与原solve的float32输出逐位一致。首个服务器尝试在解码TRAIN前因跨NumPy/CUDA的float64 ADC电压逐位断言退出1；实际整数ADC码完全一致，最大电压差为1.1102230246251565e−16 V。只将该人工跨后端断言改为整数码严格一致且电压差不超过float64机器epsilon；模型、原求解器逐位要求、物理精度门和A–F均未改。原代码、日志和原生失败回执保存在repairs/ADC_CHECK_001。成功重试之后才首次读取TRAIN。','',
 '三条完整人口随机流与R60实际消费SHA一致。本机从保存索引和原种子重建全部八条随机流、末状态与链式SHA，并独立重算坐标归约、协方差残差、逐父描述W1、损失权重、PSD/谷区和投影汇总。该复核没有新的模型前向或原TRAIN波形读取；它是保存产物审计，不冒充额外完整模型重放。权重张量SHA前后均为`'+result['frozen_weight_before_SHA']+'`。','',
 '成功及失败两个科学worker累计'+fmt(science)+'秒；成功worker峰值RSS '+str(native['actual_worker_RSS_peak_bytes'])+' B，GPU allocated/reserved峰 '+str(result['GPU_allocated_peak_bytes'])+' / '+str(result['GPU_reserved_peak_bytes'])+' B。RSS查询失败'+str(native['memory_probe_failures']+failed['memory_probe_failures'])+'次，原因保持原回执；GPU查询失败'+str(len(native['GPU_query_failures'])+len(failed['GPU_query_failures']))+'次。实际PID已不存在，无活跃科学作业。CPU12/RSS48GiB/GPU10GiB安全限制及无计算配额政策保持。','',
 '八张原图已全部打开核对。第一张W1图自动坐标扩展到了无意义的负值区域，新增独立显示版本仅把y轴下限设为0；原图、原脚本、统计与科学RESULT不变。显示版本再次打开确认，无裁切或标签重叠。','',
 direction['next_action'],'',
 '[协议](tasks/'+ID+'.md) · [完整结果](work/'+ID+'/RESULT.json) · [路径统计](work/'+ID+'/PATH_SUMMARY.json) · [基线联合结构](work/'+ID+'/BASELINE_JOINT_SUMMARY.json) · [目标覆盖](work/'+ID+'/LOSS_COVERAGE_SUMMARY.json) · [复核](work/'+ID+'/FROZEN_ARTIFACT_REVIEW.json) · [复现](work/'+ID+'/REPRODUCTION.md) · [后续判断](reviews/R62_ROOT_DIRECTION_001.json)','']
for fig in qa['figures']:lines+=['!['+Path(fig['final_display_path']).name+'](work/'+ID+'/'+fig['final_display_path']+')','']
(R/'R62_REPORT.md').write_bytes('\n'.join(lines).encode())
resource={'UTC':NOW,'stage':ID,'science_s':science,'scientific_body_s':result['elapsed_s'],'native_entries':['diagnose_train.py'],
 'native_receipts':[native],'failed_native_attempts_preserved':[failed],'scientific_worker_attempts_total':2,
 'response_training_runs_started':0,'new_response_updates':0,'actual_counters':result['counter'],'compute_budget_enforced':False,
 'worker_peak_RSS_bytes':max(v['actual_worker_RSS_peak_bytes']for v in [native,failed]),'sampled_tree_peak_RSS_bytes':max(v['job_tree_RSS_peak_bytes']for v in [native,failed]),
 'memory_probe_failures':sum(v['memory_probe_failures']for v in [native,failed]),'GPU_query_failure_count':sum(len(v['GPU_query_failures'])for v in [native,failed]),
 'GPU_allocated_peak_bytes':result['GPU_allocated_peak_bytes'],'GPU_reserved_peak_bytes':result['GPU_reserved_peak_bytes'],
 'root_saved_artifact_review_elapsed_s':review['elapsed_s'],'root_saved_artifact_review_peak_RSS_bytes':review['peak_RSS_bytes'],'root_saved_artifact_review_new_model_forwards':0}
write('work/'+ID+'/RESOURCE_RECEIPT.json',resource)
required=['config.json','INPUT_BINDING.json','PROTOCOL.md','ROOT_IMPLEMENTATION_REVIEW.json','REPRODUCTION.md','RESULT.json','FORWARD_RECEIPT.json','DATA_ACCESS_RECEIPT.json',
 'PATH_SUMMARY.json','BASELINE_JOINT_SUMMARY.json','LOSS_COVERAGE_SUMMARY.json','PSD_SUMMARY.json','PROJECTION_SUMMARY.json',
 'PATH_STATISTICS_PRIVATE.npz','BASELINE_JOINT_PRIVATE.npz','LOSS_COVERAGE_PRIVATE.npz','PER_PARENT_PRIVATE.npz',
 'INITIAL_RANDOM_STATES.json','FINAL_RANDOM_STATES.json','RANDOM_STREAM_RECEIPT.json','FROZEN_ARTIFACT_REVIEW.json','FIGURE_QA.json','DISPLAY_ONLY_RECEIPT.json','RESOURCE_RECEIPT.json',
 'diagnose_train.py.NATIVE_RECEIPT.json','repairs/ADC_CHECK_001/REPAIR_REVIEW.json','repairs/ADC_CHECK_001/diagnose_train.py.NATIVE_RECEIPT.json']
write('work/'+ID+'/FINAL_RECEIPT.json',{'UTC':NOW,'stage':ID,'status':'scientific_endpoint_complete_delivery_pending','model_goal_complete':False,
 'scientific_decision':direction['scientific_decision'],'original_A_F_changed':False,'diagnostic_markers':result['diagnostic_markers'],
 'original_A_C_reassessed':False,'new_model_candidate':False,'required_files':[bind('work/'+ID+'/'+n)for n in required],
 'private_Git_and_D_delivery':'pending separate project-root receipts','reuses_historical_R61_weights_and_inputs':True})
b=read('RESOURCE_BUDGET.json');block=b['improvement_block'];assert not any(v['stage']==ID for v in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({'stage':ID,'receipt':'work/'+ID+'/RESOURCE_RECEIPT.json','science_s':science,'response_training_runs_started':0,'new_response_updates':0,'status':'fixed_diagnostic_science_complete','stage_science_complete':True})
block['actual_science_seconds_recorded']+=science
next(v for v in block['registered_stages']if v['stage']==ID).update(status='fixed_diagnostic_science_complete',active_scientific_jobs=0,actual_science_seconds=science,new_response_updates=0,stage_science_complete=True)
write('RESOURCE_BUDGET.json',b)
job={'stage':ID,'status':'science_complete_delivery_pending','active_scientific_jobs':0,'actual_worker_PID':native['actual_worker_PID'],'supervisor_PID':native['supervisor_PID'],'actual_native_exit':0}
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    v=read(name);v.update(current_job=job,current_stage_progress='science_complete_delivery_pending',application_goal_status='active',active_scientific_jobs=0,
     active_remote_scientific_jobs=0,active_local_scientific_jobs=0,next_action=direction['next_action'],latest_completed_stage=ID,model_goal_complete=False,
     latest_completed_report='R62_REPORT.md',latest_scientific_decision=direction['scientific_decision'],latest_key_result_review='reviews/R62_ROOT_DIRECTION_001.json',status='R62_science_complete_delivery_pending_goal_active')
    v['execution_location'].update(current_stage_GPU_used=True,preflight_required=False)
    if name=='PROGRAM.json':v['updated_at_utc']=NOW
    else:v.update(project_status=v['status'],current_activity='root_R62_delivery',current_direction=direction['next_action'],updated_UTC=NOW,full_A_F_proven=False)
    write(name,v)
for name,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    v=read(name);q=next(q for q in v[key]if q[ident]==ID);q.update(status='science_complete_delivery_pending',scientific_decision=direction['scientific_decision'],science_s=science,
     native_exit=0,final_receipt='work/'+ID+'/FINAL_RECEIPT.json',model_goal_complete=False,active_scientific_jobs=0,new_training_runs_completed=0);write(name,v)
top='# 当前执行：R62 冻结完整波形诊断完成，长期目标 active\n\n全部27640原TRAIN父事件的一次前向和32批端点重采样完成。两个预设描述标记触发：神经边界增量2/3，基线协方差残差7/8区；它们不是因果或A–F通过。原R61 A/B/C失败保持，0新训练和DEV/EVAL/CAL读取，原始源人口及权重不变。当前完成私有 Git 与 D 盘交付。'+direction['next_action']+' 主助手独立，无子智能体、无计算配额；机器安全、单科学作业及本机E/D限制保持。\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
    file=R/name;text=file.read_text('utf8');assert '## 以下为历史记录'in text
    file.write_bytes((top+'## 以下为历史记录'+text.split('## 以下为历史记录',1)[1]).encode())
print(json.dumps({'science_s':science,'diagnostic_markers':result['diagnostic_markers'],'report':'R62_REPORT.md','model_goal_complete':False},indent=2))
