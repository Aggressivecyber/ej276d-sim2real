"""Close the fixed diagnostic only after native, arithmetic and visual receipts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import ctypes
import os

R = Path(__file__).resolve().parents[1]
ID = 'R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
S = R/'work'/ID
NOW = datetime.now(timezone.utc).isoformat()
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
handle = kernel.GetCurrentProcess()
mask, system = ctypes.c_size_t(), ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle, ctypes.byref(mask), ctypes.byref(system))
allowed = [i for i in range(64) if mask.value & (1 << i)]
chosen_CPU = allowed[min(3, len(allowed)-1)]
assert kernel.SetProcessAffinityMask(handle, 1 << chosen_CPU)
read = lambda p: json.loads((R/p).read_text('utf8'))
local = lambda p: read('work/'+ID+'/'+p)


def write(p, value):
    (R/p).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def bind(p):
    file = R/p
    h = hashlib.sha256()
    with file.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return dict(path=file.relative_to(R).as_posix(), bytes=file.stat().st_size, sha256=h.hexdigest())


def fmt(value):
    return '未定义' if value is None else format(value, '.9g')


def label(scope):
    if scope == 'all':
        return 'all'
    b = int(scope[6:])
    return 'below' if b == -1 else 'above' if b == 8 else 'Q'+str(b+1)


result, geometry, support = local('RESULT.json'), local('ENDPOINT_GEOMETRY_SUMMARY.json'), local('SUPPORTING_SUMMARY.json')
parameter, endpoint = local('PARAMETER_GRADIENT_SUMMARY.json'), local('ENDPOINT_GRADIENT_SUMMARY.json')
native, review, qa = local('diagnose_train.py.NATIVE_RECEIPT.json'), local('FROZEN_ARTIFACT_REVIEW.json'), local('FIGURE_QA.json')
direction = read('reviews/R64_ROOT_DIRECTION_001.json')
assert native['actual_worker_native_exit'] == native['worker_self_receipt']['native_exit'] == 0
assert native['termination_reason'] is None and local('FETCH_RECEIPT.json')['remote_completion']['actual_PID_gone']
assert local('DIAGNOSTIC_PROGRESS.json')['phase'] == 'registered_diagnostic_endpoint_complete'
assert review['pass_review'] and qa['all_figures_opened'] and qa['layout_pass'] and qa['figure_count'] == 4
assert local('ROOT_ARTIFACT_REVIEW.NATIVE_RECEIPT.json')['native_exit'] == 0
assert direction['model_goal_complete'] is False and direction['original_A_F_changed'] is False
assert direction['primary_parameter_gradient_comparison'] == result['primary_parameter_gradient_comparison']
assert isinstance(direction['next_action'], str) and direction['next_action'].strip()
terms = ['full_window', 'coordinate_RMS', 'temporal_joint']
mass = geometry['population']['source_mass']
bound = geometry['charge_bound_aggregation']
v_observed = math.fsum(m*row['charge_mean_bound']['observed_V480_two_coordinate_contributions']
                      for m, row in zip(mass, geometry['rows'][1:]))
v_bound = bound['output_specific_V480_lower_bound_full_window_plus_RMS']
science = native['elapsed_s']
lines = ['# R64：冻结 TRAIN 完整参照与分布梯度诊断', '', direction['scientific_decision'], '',
 '本次诊断已到预登记终点，没有新增候选模型或重判原 A–F。R63 的原 A/B/C 失败保持，应用长期目标仍为 active。', '',
 '冻结 R63 唯一权重，完整保留6593个 SOURCE、21047个 REAL 原父事件、顺序及各自区域质量。'
 '先按原直接训练端点和固定每父噪声采集全人口，再计算五项 D 的完整有限经验参照。'
 '32组比较共享每区128次有放回抽样：A为四个32槽损失的平均，B为同一128槽联合损失。'
 '原 source_order、source_valley、REAL_identity、box_correction 四项 C 不进入本次梯度比较。', '',
 '五项 D 包括 full_window、PSD、noise、coordinate_RMS、temporal_joint，外权重均为1。'
 'SOURCE及REAL独立使用原区域质量；重复出现的父事件保留其发生质量和梯度累加。'
 '总体训练分布的质量实现先按 binary64 构造 count/total/K，再将其精确整数比归一化；'
 '没有把它称为理想实数分数的位级实现。32到128的二次幂比例保持两臂相同的已表示区域质量。', '',
 '| 完整 CUDA 五项 D 参照 | 数值 |', '|---|---:|']
for name, value in result['full_reference_loss'].items():
    lines.append('| '+name+' | '+fmt(value)+' |')
lines += ['| total_D | '+fmt(sum(result['full_reference_loss'].values()))+' |', '',
 '下列误差使用原参数向量或原父端点坐标的平方欧氏范数，不除以坐标维数。'
 'total_D 是先将五个梯度向量相加再计算误差，不是五项平方误差的和。'
 '唯一主比较是参数空间 total_D 的配对 B−A 平均平方误差；其1000次百分位95%区间以宏组为重采样单位。'
 '端点空间及五个分量是次要描述，区间没有多重比较校正。', '']
for space, summary in [('参数空间', parameter), ('端点空间', endpoint)]:
    lines += ['| '+space+'：分量 | A相对均方误差 | B相对均方误差 | A平均cos | B平均cos | 原始B−A均方误差 | 配对95%区间 |',
              '|---|---:|---:|---:|---:|---:|---|']
    for name in summary['reported_order']:
        a, b, p = summary['arms']['A'][name], summary['arms']['B'][name], summary['paired'][name]
        lines.append('| '+name+' | '+' | '.join(fmt(v) for v in [a['relative_mean_squared_l2_error'],
          b['relative_mean_squared_l2_error'], a['mean_cosine_reference'], b['mean_cosine_reference'],
          p['mean_B_minus_A_squared_l2_error']])+' | ['+', '.join(fmt(v) for v in p['bootstrap_percentile_95_interval'])+'] |')
    lines += ['']
lines += ['| total_D 有限样本分解 | 均方误差 | 均值梯度偏离参照的平方 | 经验离散度 |', '|---|---:|---:|---:|']
for space, summary in [('参数', parameter), ('端点', endpoint)]:
    for arm in ('A', 'B'):
        x = summary['arms'][arm]['total_D']
        lines.append('| '+space+' '+arm+' | '+' | '.join(fmt(v) for v in [x['mean_squared_l2_error'],
          x['mean_gradient']['squared_l2_error'], x['empirical_dispersion_mean_squared_l2']])+' |')
lines += ['', '有限样本恒等式“均方误差=均值梯度偏离平方+经验离散度”在两空间全部分量及总向量均通过原数值容差。'
 '这是当前固定参照下的32组经验分解，不是总体无偏 bias/variance 估计。'
 '端点和参数误差的范数不同；神经网络的链式映射不保序，不能跨空间直接比较误差大小。', '',
 '| 原SOURCE质量加权几何项 | SOURCE | 同处理REAL | 32对REAL均值 | 32对REAL[min,max] |', '|---|---:|---:|---:|---|']
for name in terms:
    f = geometry['matched_N_REAL_pair_weighted_regional_terms'][name]
    lines.append('| '+name+' | '+fmt(geometry['original_SOURCE_mass_weighted_regional_terms'][name])+' | '+
      fmt(geometry['same_treated_REAL_original_SOURCE_mass_weighted_regional_terms'][name])+' | '+fmt(f['mean'])+
      ' | ['+fmt(f['min'])+', '+fmt(f['max'])+'] |')
lines += ['', '每个 REAL 参照从相应原区域有放回抽取两份与 SOURCE 数量相同的样本。'
 '32个数值的范围不是置信区间；两份有限REAL样本与SOURCE对完整REAL的设计也不同。'
 '该参照仅用于三项几何量，不扩展为 PSD 或 noise 的抽样底噪。'
 '完整几何使用冻结 NumPy 实现，梯度参照使用 CUDA；按协议不要求两个后端逐位一致。', '',
 '| 原区域 | SOURCE/REAL N | full_window | coordinate_RMS | temporal_joint | V坐标名义下界/观测贡献 |',
 '|---|---:|---:|---:|---:|---:|']
for row in geometry['rows']:
    x = row['SOURCE_vs_complete_REAL']['terms']
    ratio = None if row['scope'] == 'all' else row['charge_mean_bound']['output_LB_fraction_of_V480_two_coordinate_contributions']
    lines.append('| '+label(row['scope'])+' | '+str(row['SOURCE_N'])+'/'+str(row['REAL_N'])+' | '+
      ' | '.join(fmt(x[name]) for name in terms)+' | '+fmt(ratio)+' |')
lines += ['', 'all 行把各父所在区域的标准化坐标汇成总体分布，与按原质量加权的十区目标分开解释。'
 '高端支持外人口保留918/6593，贡献比例及逐坐标定位见独立几何报告；它不能作为删事件或改权重的依据，标量损失贡献也不是梯度冲突证据。', '',
 '输出特定电荷均值名义下界 B_V='+fmt(v_bound)+'，V-only坐标贡献分母='+fmt(v_observed)+'，比值='+fmt(v_bound/v_observed)+
 '；全部四视图 full_window+coordinate_RMS 分母='+fmt(bound['observed_full_window_plus_RMS'])+'，比值='+
 fmt(bound['output_lower_bound_fraction_of_observed_full_window_plus_RMS'])+'。'
 '两个分母明确区分。全部原名义不等式通过，但这只是当前浮点端点的下界紧致程度，'
 '不是因果归属比例、区间证书或模型无关的 A–F 不可行证明。', '',
 '| 原范围 | raw SOURCE–REAL PSD W1 | 端点preADC–REAL W1 | 共同ADC–REAL W1 | 同处理REAL W1 | 原PSD–输出PSD秩 |',
 '|---|---:|---:|---:|---:|---:|']
for row in support['rows']:
    lines.append('| '+label(row['scope'])+' | '+' | '.join(fmt(v) for v in [
      row['SOURCE_raw_vs_REAL_raw_PSD_W1'], row['SOURCE_preADC_vs_REAL_canonical_PSD_W1'],
      row['SOURCE_canonical_vs_REAL_canonical_PSD_W1'], row['REAL_control_vs_REAL_canonical_PSD_W1'],
      row['source_rank']['original_PSD_vs_output_PSD']])+' |')
common = math.fsum(row['SOURCE_canonical_vs_REAL_canonical_PSD_W1'] for row in support['rows']
                  if row['scope'] in ['region'+str(k) for k in range(7)])/7
lines += ['', '共同Q1–Q7等区 PSD W1均值='+fmt(common)+'。'
 '以上是冻结直接 TRAIN 端点的支持描述，不能替代 R63 历史 DEV 密度拟合或原 A–C；本轮没有新密度拟合。'
 '全部九个边界、全窗均值和标准差、原电荷秩及每区支持均已保留。', '',
 '| 原人口投影 | 范围活跃事件/N | 上下限采样点数 | 修正RMS均值/V | 最大preADC相对Q误差f32 |', '|---|---:|---:|---:|---:|']
for name, v in support['projection'].items():
    lines.append('| '+name+' | '+str(v['raw_endpoint_outside_box_N'])+'/'+str(v['rows'])+' | '+str(v['active_rail_samples'])+
                 ' | '+fmt(v['projection_RMS']['mean'])+' | '+fmt(v['preADC_relative_Q_error_f32']['max'])+' |')
lines += ['', 'CPU/CUDA 各69项人工分布值、端点梯度与重复父检查通过，真实 ResponseNetwork 人工两遍 VJP 检查也通过。'
 '科学计算的33次完整SOURCE重放、3432个原64行/最终1行分块均逐位核对投影端点、整数ADC码和严格Box自由掩码后才注入梯度。'
 '实际神经行数245209、调用3865次、参数VJP33800次、分布目标161次；0训练、更新、教师、HGB、密度拟合及DEV/EVAL/CAL解码。', '',
 '独立本机核验从原种子重建九条随机流及父索引，重算所有保存参数向量的范数、点积、误差、中心化离散度及配对bootstrap；'
 '同时复核端点完整/均值向量和各宏组标量的代数关系，以及保存几何坐标归约和名义下界。'
 '未保存宏端点梯度逐坐标数组，本机没有执行第二轮网络或损失VJP；这些向量可用保存端点及固定索引复现。'
 '唯一冻结权重前后张量SHA均为 `'+result['frozen_weight_before_SHA']+'`，八个输入文件及原A–F字节绑定保持。', '',
 '第一次部署在启动前因Windows清单分隔符与POSIX tar成员名不同而退出；当时没有科学worker或TRAIN读取。'
 '只规范包内路径并验证34份源码字节不变后启动唯一科学worker。原失败包与修复回执完整保留。'
 '科学worker原生退出0，耗时'+fmt(science)+'秒，worker峰值RSS '+str(native['actual_worker_RSS_peak_bytes'])+
 ' B，作业树峰值RSS '+str(native['job_tree_RSS_peak_bytes'])+' B；GPU allocated/reserved峰 '+
 str(result['GPU_allocated_peak_bytes'])+' / '+str(result['GPU_reserved_peak_bytes'])+' B。'
 'RSS查询失败'+str(native['memory_probe_failures'])+'次，原事件和未知原因完整保留；本轮两次均晚于worker自身完成回执，'
 '未达到原监护程序“多于5次即终止”的规则，不补写为0次。GPU查询失败'+str(len(native['GPU_query_failures']))+'次。'
 '实际PID已不存在。服务器CPU12/RSS48GiB/GPU10GiB、本机共同CPU4/RSS4GiB以及不设计算配额的政策保持。', '',
 '四张图已实际打开复核，显示层更改若有则见 FIGURE_QA.json。本机新增源文件、数据、日志、缓存和临时文件保存在E/D；服务器产物位于已登记的阶段目录。', '',
 direction['next_action'], '',
 '[协议](tasks/'+ID+'.md) · [结果](work/'+ID+'/RESULT.json) · [参数梯度](work/'+ID+'/PARAMETER_GRADIENT_SUMMARY.json) · '
 '[端点梯度](work/'+ID+'/ENDPOINT_GRADIENT_SUMMARY.json) · [几何](work/'+ID+'/ENDPOINT_GEOMETRY_SUMMARY.json) · '
 '[独立几何解释](reviews/R64_RESULT_ANALYSIS_001/geometry/REPORT.md) · [独立梯度解释](reviews/R64_RESULT_ANALYSIS_001/gradient/REPORT.md) · [复核](work/'+ID+'/FROZEN_ARTIFACT_REVIEW.json) · '
 '[复现](work/'+ID+'/REPRODUCTION.md) · [后续判断](reviews/R64_ROOT_DIRECTION_001.json)', '']
for fig in qa['figures']:
    lines += ['!['+Path(fig['final_display_path']).name+'](work/'+ID+'/'+fig['final_display_path']+')', '']
(R/'R64_REPORT.md').write_bytes('\n'.join(lines).encode())
resource = dict(UTC=NOW, stage=ID, science_s=science, scientific_body_s=result['elapsed_s'],
  native_entries=['diagnose_train.py'], native_receipts=[native], scientific_worker_attempts_total=1,
  failed_prelaunch_packaging_preserved='reviews/R64_DEPLOYMENT_REPAIR_001', response_training_runs_started=0,
  new_response_updates=0, actual_counters=result['counter'], compute_budget_enforced=False,
  worker_peak_RSS_bytes=native['actual_worker_RSS_peak_bytes'], sampled_tree_peak_RSS_bytes=native['job_tree_RSS_peak_bytes'],
  memory_probe_failures=native['memory_probe_failures'], GPU_query_failure_count=len(native['GPU_query_failures']),
  GPU_allocated_peak_bytes=result['GPU_allocated_peak_bytes'], GPU_reserved_peak_bytes=result['GPU_reserved_peak_bytes'],
  root_saved_artifact_review_elapsed_s=review['elapsed_s'], root_saved_artifact_review_resource=review['resource'],
  root_saved_artifact_review_new_model_forwards=0,
  finalization_process=dict(PID=os.getpid(), CPU_affinity=[chosen_CPU], large_file_SHA_block_bytes=8*1024**2))
write('work/'+ID+'/RESOURCE_RECEIPT.json', resource)
required = ['config.json', 'INPUT_BINDING.json', 'PROTOCOL.md', 'ROOT_IMPLEMENTATION_REVIEW.json', 'REPRODUCTION.md',
  'RESULT.json', 'FORWARD_RECEIPT.json', 'DATA_ACCESS_RECEIPT.json', 'ENDPOINT_REPLAY_RECEIPT.json', 'FULL_REFERENCE_LOSS.json',
  'ENDPOINT_GEOMETRY_SUMMARY.json', 'ENDPOINT_GEOMETRY_PRIVATE.npz', 'SUPPORTING_SUMMARY.json', 'WAVEFORM_STATISTICS_PRIVATE.npz',
  'SOURCE_ENDPOINTS_PRIVATE.npz', 'REAL_control_ENDPOINTS_PRIVATE.npz', 'PARENT_IDENTITIES_PRIVATE.npz',
  'FROZEN_RANDOM_DESIGN_PRIVATE.npz', 'PARAMETER_MANIFEST.json', 'PARAMETER_GRADIENT_SUMMARY.json', 'ENDPOINT_GRADIENT_SUMMARY.json',
  'FULL_REFERENCE_ENDPOINT_GRADIENTS_PRIVATE.npz', 'FULL_REFERENCE_PARAMETER_GRADIENTS_PRIVATE.npz',
  'PARAMETER_MEAN_GRADIENTS_PRIVATE.npz', 'ENDPOINT_MEAN_GRADIENTS_PRIVATE.npz', 'MACRO_LOSS_AND_GRADIENT_RECORDS.json',
  'INITIAL_RANDOM_STATES.json', 'FINAL_RANDOM_STATES.json', 'RANDOM_STREAM_RECEIPT.json', 'FROZEN_ARTIFACT_REVIEW.json',
  'FIGURE_QA.json', 'DISPLAY_ONLY_RECEIPT.json', 'ROOT_ARTIFACT_REVIEW.NATIVE_RECEIPT.json',
  'RESOURCE_RECEIPT.json', 'diagnose_train.py.NATIVE_RECEIPT.json', 'LAUNCH_RECEIPT.json', 'FETCH_RECEIPT.json']
required += ['parameter_gradients_private/macro_'+str(k).zfill(2)+'.npz' for k in range(32)]
write('work/'+ID+'/FINAL_RECEIPT.json', dict(UTC=NOW, stage=ID, status='scientific_endpoint_complete_delivery_pending',
  model_goal_complete=False, scientific_decision=direction['scientific_decision'], original_A_F_changed=False,
  original_A_C_reassessed=False, new_model_candidate=False, required_files=[bind('work/'+ID+'/'+name) for name in required],
  private_Git_and_D_delivery='pending separate project-root receipts', reuses_historical_R63_weights_and_inputs=True))
b = read('RESOURCE_BUDGET.json')
block = b['improvement_block']
assert not any(v['stage'] == ID for v in block['actual_stage_receipts'])
block['actual_stage_receipts'].append(dict(stage=ID, receipt='work/'+ID+'/RESOURCE_RECEIPT.json', science_s=science,
  response_training_runs_started=0, new_response_updates=0, status='fixed_diagnostic_science_complete', stage_science_complete=True))
block['actual_science_seconds_recorded'] += science
next(v for v in block['registered_stages'] if v['stage'] == ID).update(status='fixed_diagnostic_science_complete',
  active_scientific_jobs=0, actual_science_seconds=science, new_response_updates=0, stage_science_complete=True)
write('RESOURCE_BUDGET.json', b)
job = dict(stage=ID, status='science_complete_delivery_pending', active_scientific_jobs=0,
  actual_worker_PID=native['actual_worker_PID'], supervisor_PID=native['supervisor_PID'], actual_native_exit=0)
for name in ('PROGRAM.json', 'PROJECT_PROGRESS.json'):
    v = read(name)
    v.update(current_job=job, current_stage_progress='science_complete_delivery_pending', application_goal_status='active',
      active_scientific_jobs=0, active_remote_scientific_jobs=0, active_local_scientific_jobs=0, next_action=direction['next_action'],
      latest_completed_stage=ID, model_goal_complete=False, latest_completed_report='R64_REPORT.md',
      latest_scientific_decision=direction['scientific_decision'], latest_key_result_review='reviews/R64_ROOT_DIRECTION_001.json',
      status='R64_science_complete_delivery_pending_goal_active')
    v['execution_location'].update(current_stage_GPU_used=True, preflight_required=False)
    if name == 'PROGRAM.json':
        v['updated_at_utc'] = NOW
    else:
        v.update(project_status=v['status'], current_activity='root_R64_delivery', current_direction=direction['next_action'],
                 updated_UTC=NOW, full_A_F_proven=False)
    write(name, v)
for name, key, ident in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    v = read(name)
    q = next(q for q in v[key] if q[ident] == ID)
    q.update(status='science_complete_delivery_pending', scientific_decision=direction['scientific_decision'], science_s=science,
      native_exit=0, final_receipt='work/'+ID+'/FINAL_RECEIPT.json', model_goal_complete=False, active_scientific_jobs=0,
      new_training_runs_completed=0)
    write(name, v)
top = '# 当前执行：R64 冻结参照与配对分布梯度诊断完成，长期目标 active\n\n'+direction['scientific_decision']+\
  ' 原R63 A/B/C失败保持，0新训练及DEV/EVAL/CAL读取。私有 Git 与 D 盘交付待完成。'+direction['next_action']+\
  ' 全部执行和规划代理继承主助手相同模型与思考强度，主助手负责科学方向和Git；无计算配额，单科学作业及共享资源/E-D要求保持。\n\n'
for name in ('AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md'):
    file = R/name
    text = file.read_text('utf8')
    assert '## 以下为历史记录' in text
    file.write_bytes((top+'## 以下为历史记录'+text.split('## 以下为历史记录', 1)[1]).encode())
readme = R/'README.md'
readme.write_bytes(('# 当前入口：R64冻结参照诊断完成，长期目标仍 active\n\n'+direction['scientific_decision']+
    ' 原A–F与R63失败结论保持；实际交付状态以回执为准。\n\n'+
    '[R64报告](R64_REPORT.md) · [科学回执](work/'+ID+'/FINAL_RECEIPT.json) · '+
    '[后续方向](reviews/R64_ROOT_DIRECTION_001.json) · [验收标准](PROJECT_ACCEPTANCE.md)\n\n'+
    '## 以下为历史入口与持续有效的执行要求\n\n'+readme.read_text('utf8')).encode())
print(json.dumps(dict(science_s=science, report='R64_REPORT.md', model_goal_complete=False,
                     primary_parameter=result['primary_parameter_gradient_comparison']), ensure_ascii=False, indent=2))
