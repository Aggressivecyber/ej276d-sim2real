from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path(__file__).resolve().parents[1];ID='R58_JOINT_DISTRIBUTION_FEASIBILITY_001';S=R/'work'/ID;NOW=datetime.now(timezone.utc).isoformat()
def read(n):return json.loads((R/n).read_text('utf8'))
def local(n):return json.loads((S/n).read_text('utf8'))
def write(n,v):
    value=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if n=='RESOURCE_BUDGET.json':value=value.replace('\n','\r\n')
    (R/n).write_bytes(value.encode('utf8'))
def bind(p):return {'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def label(b):return '总体' if b is None else '下方支持外' if b==-1 else '上方支持外' if b==8 else 'Q'+str(b+1)
v=local('RESULT.json');native=local('numeric_worker.py.NATIVE_RECEIPT.json');qa=local('FIGURE_QA.json');d=read('reviews/R58_ROOT_DIRECTION_001.json')
assert v['status']==local('NUMERIC_RECEIPT.json')['status']=='complete'
assert native['actual_worker_native_exit']==native['worker_self_receipt']['native_exit']==0 and native['termination_reason'] is None and native['handshake_before_imports']
assert not local('LIVE_OBSERVATION.json')['actual_process']['exists'] and qa['all_figures_opened'] and qa['layout_pass']
assert v['decision']==d['scientific_decision'] and not d['model_goal_complete']
assert v['counter']=={'LP_attempts':1,'new_response_updates':0,'teacher_waveforms_constructed':0,'HGB_fits':0,'density_fits':0,'ODE_generation':0}
science=native['elapsed_s'];solver=v['solver'];construction=v['construction'];verification=v['verification']
lines=['# R58：联合分布约束检查','',d['conclusion_zh'],'',
    '此前“教师”指流匹配路线中人工构造的目标波形，用于定义从仿真波形到实测域端点的训练方向；它不是另一已训练网络，也不是仿真事件的真实逐事件实测答案。无配对转换并不必须使用这种教师。本轮去掉固定目标波形假设，仅检查联合统计约束，不生成教师波形，不据分数组合进行逐事件真值监督。','',
    'R57全TRAIN零越界预检严于原冻结DEV的零裁剪检查。本轮保留该负结果，同时将ADC积分可行性单列。原A–F、原2000个验收SOURCE的身份/顺序/组成/权重及数据用途全部不变；没有新DEV/EVAL/CAL或TRAIN验证字段解码。','',
    f"全部6593源保持原等权边缘，按十个原电荷区域、h==1标志与64个全局PSD midrank分段得到{construction['SOURCE_groups']}组。全部21047 REAL以精确整数质量参与512个等质量分位段，段均值形成{construction['target_atoms']}个PSD原子。压缩原子对原始完整REAL的总体W1为{v['target_quantization_W1']:.9g}；未删除尾部或原事件。",'',
    '变量x[g,k]采用源父事件计数单位，源组行和及固定REAL原子列和共同约束。每个组的条件概率律均匀归还该组原父事件，各父事件权重仍为1/6593；h与身份始终保留，REAL未被附加粒子标签。唯一目标最大化源/输出PSD的midrank协方差；边缘固定后它与Spearman成正比。这个数值最优仅针对已登记有限族，不能充当所有映射的上界。','',
    f"一次HiGHS dual-simplex：{construction['variables']}个变量、{construction['equalities']}个等式、{construction['inequalities']}个不等式，{solver['nit']}次迭代。求解器状态{solver['status']}：{solver['message']}。求解耗时{solver['elapsed_s']:.6f}秒。原选项及警告保存在LP_SOLVER_RECEIPT.json；没有第二次求解、组数/原子数/约束扫描。",'']
if verification is not None:
    a=verification['validation'];lines+=['| 从保存解独立核验 | 值 |','|---|---:|']
    for key in ['equality_residual_max','inequality_violation_max','lower_bound_violation_max','CDF_slack_upper_bound_violation_max','group_row_error_max','target_column_error_max','per_parent_probability_sum_error_max','SOURCE_global_weight_error_max']:
        lines.append(f"| {key} | {a[key]:.9g} |")
    lines+=['',f"概率非负：{a['strict_nonnegative_probability_law']}；原登记1e-7数值容差内可行：{a['numeric_feasible']}。负值不会被截零修复；此处没有严格有理数可行性证书。",'']
    if verification['PSD'] is not None:
        ps=verification['PSD'];lines+=['| 原区域 | 源父事件 | REAL父事件 | 精确PSD W1 |','|---|---:|---:|---:|']
        for row in ps['by_bin']:lines.append(f"| {label(row['raw_bin0'])} | {row['source_parents']} | {row['REAL_parents']} | {row['W1']:.9g} |")
        lines+=['',f"总体W1为{ps['overall_W1']:.9g}；原共同Q1–Q7等权平均为{ps['common_Q1_Q7_equal_mean_W1']:.12g}。模型中使用原子与全部原REAL PSD值的并集积分，核验用SciPy原始样本精确W1；没有以直方图或几个CDF点代替。处于阈值附近的值按原登记数值容差报告，不能据舍入声称严格小于阈值。",'',
            '| h==1源组 | 父事件数 | 原源谷概率 | 联合律谷概率 | 增量 |','|---|---:|---:|---:|---:|']
        for row in verification['valleys']:lines.append(f"| Q{row['bin']} | {row['source_parents']} | {row['original_fraction']:.9g} | {row['candidate_fraction']:.12g} | {row['delta']:.12g} |")
        lines+=['',f"本有限族的数值最优加权SOURCE PSD关联为{verification['weighted_SOURCE_PSD_rho']:.9g}。冻结原BASE DEV参考为{v['original_DEV_BASE_PSD_rho_context']:.9g}，允许下降0.05后的值为{v['original_DEV_rho_floor_context']:.9g}；这里的TRAIN人口不同，比较仅作上下文，不能把它称为原C门通过。组内共享概率律是本有限族的限制，不能凭该最优值排除更细或不同的映射。",'']
else:lines+=['求解器未提供成功解，保留原状态，不代入替代分布或重新求解。此固定离散问题没有成功结果不等于原A–F不可能。','']
base=v['input_ADC_bounds'];lines+=[f"REAL TRAIN基线均值范围为[{base['observed_REAL_baseline_mean_min_V']:.9g}, {base['observed_REAL_baseline_mean_max_V']:.9g}] V；原源Q范围[{base['source_Q_min']:.9g}, {base['source_Q_max']:.9g}]。在该已观测基线范围内，自由PSD的最大可容纳Q为{base['observed_baseline_free_PSD_max_Q']:.9g}；只受全窗ADC电压范围约束时绝对上界为300。",'',
    '对给定(q,p)，短/长脉冲段电压和分别为q(1−p)+50b与qp+250b。它们必须分别落在[−25,25]与[−125,125]内。没有共同可用b只证明在已观测基线范围下积分不可实现；有共同b也不能证明真实形状、噪声、峰宽、边界或神经变换成立。','']
adc=v['ADC_integral_bounds']
if adc is not None:
    lines+=['| 原区域 | 源父事件 | 原输入越界父事件 | 有不可实现概率的父事件 | 不可实现总质量 | 原输入未越界部分质量 |','|---|---:|---:|---:|---:|---:|']
    for row in adc['rows']:lines.append(f"| {label(row['raw_bin0'])} | {row['source_parents']} | {row['original_input_overrange_parents']} | {row['parents_with_positive_impossible_probability']} | {row['impossible_global_mass']:.9g} | {row['originally_in_range_impossible_global_mass']:.9g} |")
    lines+=['','原输入越界父事件计数与联合律不可实现概率的分母不同；所有原源均保留。TRAIN原始电荷和范围不同于冻结DEV，因此这些数值不能当DEV失败率或放宽其零裁剪门的许可。','']
else:
    lines+=['没有可用的非负联合律，未虚构其ADC概率。源输入本身的自由PSD硬界仍完整保存在ADC_INPUT_BOUNDS.json。','']
lines+=[f"一次CPU科学作业原生退出0，总耗时{science:.6f}秒；worker峰RSS {native['actual_worker_RSS_peak_bytes']} B，作业树采样峰值{native['job_tree_RSS_peak_bytes']} B，RSS缺测{native['memory_probe_failures']}次。0教师波形、响应训练、ODE、HGB或密度拟合。四张最终PNG均已实际打开。",'',
    d['next_action'],'',
    '[协议](tasks/'+ID+'.md) · [结果](work/'+ID+'/RESULT.json) · [LP状态](work/'+ID+'/LP_SOLVER_RECEIPT.json) · [复现](work/'+ID+'/REPRODUCTION.md) · [科学回执](work/'+ID+'/FINAL_RECEIPT.json)','']
figures=['population_and_solver','joint_PSD_CDF','source_association_and_valleys','ADC_integral_bounds']
for name in figures:lines+=['!['+name+'](work/'+ID+'/'+name+'.png)','']
(R/'R58_REPORT.md').write_bytes('\n'.join(lines).encode('utf8'))
resource={'UTC':NOW,'stage':ID,'science_s':science,'native_receipts':[native],'response_training_runs_started':0,'new_response_updates':0,'new_HGB_fits':0,'new_density_fits':0,'counter':v['counter'],'DEV_read':0,'EVAL_read':0,'CAL_decode':0,'compute_budget_enforced':False,'worker_peak_RSS_bytes':native['actual_worker_RSS_peak_bytes'],'sampled_tree_peak_RSS_bytes':native['job_tree_RSS_peak_bytes'],'memory_probe_failures':native['memory_probe_failures']}
write('work/'+ID+'/RESOURCE_RECEIPT.json',resource)
required=['config.json','PROTOCOL.md','INPUT_BINDING.json','numeric_worker.py','joint_core.py','verify_joint.py','render_joint.py','noise_core.py','RESULT.json','LP_CONSTRUCTION.json','LP_SOLVER_RECEIPT.json','LP_PRIVATE.npz','TARGET_ATOMS_PRIVATE.npz','POPULATION.json','ADC_INPUT_BOUNDS.json','FROZEN_VALLEY_DEFINITION.json','ARTIFICIAL_CHECK.json','FIGURE_QA.json','RESOURCE_RECEIPT.json','REPRODUCTION.md']+[n+'.png' for n in figures]
required += [n for n in ['INDEPENDENT_VERIFICATION.json','JOINT_LAW_PRIVATE.npz','POPULATION_PRIVATE.npz','ADC_BOUND.json','ADC_BOUND_PRIVATE.npz'] if (S/n).is_file()]
write('work/'+ID+'/FINAL_RECEIPT.json',{'UTC':NOW,'stage':ID,'status':'science_complete_delivery_pending','model_goal_complete':False,'scientific_decision':v['decision'],'required_files':[bind(S/n) for n in required],'private_Git_D_delivery':'pending separate receipts'})
b=read('RESOURCE_BUDGET.json');block=b['improvement_block'];assert not any(x['stage']==ID for x in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({'stage':ID,'receipt':'work/'+ID+'/RESOURCE_RECEIPT.json','science_s':science,'response_training_runs_started':0,'new_response_updates':0,'status':'fixed_diagnostic_complete','stage_science_complete':True})
block['actual_science_seconds_recorded']+=science
next(x for x in block['registered_stages'] if x['stage']==ID).update(status='fixed_diagnostic_complete',active_scientific_jobs=0,actual_science_seconds=science,stage_science_complete=True);write('RESOURCE_BUDGET.json',b)
job={'stage':ID,'status':'science_complete_delivery_pending','actual_worker_PID':native['actual_worker_PID'],'supervisor_PID':native['supervisor_PID'],'actual_native_exit':0,'active_scientific_jobs':0}
for n in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    x=read(n);x.update(status='R58_science_complete_delivery_pending_goal_active',current_job=job,current_stage_progress='science_complete_delivery_pending',latest_completed_stage=ID,latest_completed_report='R58_REPORT.md',latest_scientific_decision=v['decision'],latest_key_result_review='reviews/R58_ROOT_DIRECTION_001.json',next_action=d['next_action'],application_goal_status='active',model_goal_complete=False,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0)
    if n=='PROGRAM.json':x['updated_at_utc']=NOW
    else:x.update(project_status=x['status'],current_activity='root_R58_delivery',current_direction=d['next_action'],updated_UTC=NOW)
    write(n,x)
for n,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    x=read(n);next(a for a in x[key] if a[ident]==ID).update(status='science_complete_delivery_pending',scientific_decision=v['decision'],science_s=science,native_exit=0,final_receipt='work/'+ID+'/FINAL_RECEIPT.json',model_goal_complete=False,active_scientific_jobs=0);write(n,x)
top='# 当前执行：R58 联合分布检查完成，长期目标 active\n\n'+d['conclusion_zh']+' 原A–F保持，模型目标未完成。当前完成私有 Git 与 D 盘交付。'+d['next_action']+' 主助手独立，无子智能体，无计算配额；机器安全、单作业及本机E/D限制保持。\n\n'
for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
    path=R/n;tail=path.read_text('utf8').split('## 以下为历史记录',1)[1];path.write_bytes((top+'## 以下为历史记录'+tail).encode('utf8'))
print(json.dumps({'decision':v['decision'],'science_s':science,'cumulative_science_s':block['actual_science_seconds_recorded'],'model_goal_complete':False},indent=2))
