from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
ID = 'R54_CONTINUOUS_CHARGE_FLOW_001'
S = R / 'work' / ID
NOW = datetime.now(timezone.utc).isoformat()
ARM = 'CONTINUOUS_Q_RESPONSE'
CTRL = 'CONTINUOUS_Q_REAL_CONTROL'


def read(path):
    return json.loads((R / path).read_text('utf8'))


def write(path, value):
    (R / path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def local(name):
    return read('work/' + ID + '/' + name)


def bind(path):
    p = R / path
    return {'path': p.relative_to(R).as_posix(), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


M = local('METRICS_COMPACT.json')
gate = local('GATE_METRICS.json')
A = M['acceptance_A_C'][ARM]
teacher = local('TEACHER_VALIDATION.json')
train = local(ARM + '/TRAINING_RECEIPT.json')
replay = local('REPLAY_RECEIPT.json')
qa = local('FIGURE_QA.json')
direction = read('reviews/R54_ROOT_DIRECTION_001.json')
assert direction['scientific_decision'] == gate['decision']
assert direction['model_goal_complete'] is False
assert qa['all_figures_opened'] and qa['layout_pass'] and teacher['passed']
assert train['updates'] == 50000
assert all(train['same_R48_source_identity_and_random_streams'].values())
assert all(train['same_R53_all_original_and_teacher_streams'].values())
assert train['continuous_Q_nonzero_gradient_steps'] > 0
entries = ['train_final.py', 'numerical_check.py', 'generate_all.py',
           'evaluate_saved.py', 'replay_all.py', 'summarize_and_plot.py']
native = [local(n + '.NATIVE_RECEIPT.json') for n in entries]
assert all(n['termination_reason'] is None and n['handshake_before_imports'] for n in native)
assert all(n['actual_worker_native_exit'] == 0 and n['worker_self_receipt']['native_exit'] == 0 for n in native)
assert replay['rows'] == 4000 and replay['portable_loader_executed']
live = local('LIVE_OBSERVATION.json')
assert not live['actual_process']['exists']
science = sum(n['elapsed_s'] for n in native)
stepgate = gate['models'][ARM]
views = ['full_historical_shape', 'normalized_u480', 'full_voltage', 'Q_PSD_2D', 'baseline_local']


def result(passed):
    return '通过' if passed else '未通过'


def value_or_undefined(value, pattern='.8f'):
    return format(value, pattern) if value is not None else '未定义'


lines = [
    '# R54：连续电荷条件的固定消融结果', '',
    '同一初始权重、平滑脉冲教师、样本、噪声、随机流和 50000 次更新下，本轮只把原 embedding 的一个零输入替换为连续 logQ 坐标。分区、坐标公式和单一终点在训练前登记；未选择检查点或连续尺度。', '',
    '原 A/B/C：' + ' / '.join(k + ' ' + result(A[k]['pass']) for k in ['A', 'B', 'C']) +
    '。同一冻结方法尚未完成原 A–F，长期目标仍为 active。', '',
    '| 共同 ADC 表示下的视图 | SOURCE BA [95% CI] | 同处理 REAL BA [95% CI] |',
    '|---|---:|---:|',
]
for view in views:
    cells = []
    for name in [ARM, CTRL]:
        row = M['C2ST']['results'][name][view]
        ci = row['BA_95CI']
        cells.append(f"{row['balanced_accuracy']:.3f} [{ci[0]:.6f}, {ci[1]:.6f}]")
    lines.append('| ' + view + ' | ' + ' | '.join(cells) + ' |')
lines += [
    '', 'A 要求每个区间完整位于 [0.40, 0.60]，按未舍入值判定。SOURCE baseline 的区间上界为 0.600091818，仍记未通过。RAW REAL–REAL、BASE 和旧 R49 对照复用已冻结的共同 ADC 结果；R53 对照复用原 GPU 结果。没有新增对照拟合。', '',
    '| 相对 R53 的配对 BA 差 | SOURCE ΔBA [95% CI] | 同处理 REAL ΔBA [95% CI] |',
    '|---|---:|---:|',
]
for view in views:
    cells = []
    for name in [ARM, CTRL]:
        row = M['paired_R54_minus_frozen_R53'][name][view]
        ci = row['BA_delta_95CI']
        cells.append(f"{row['BA_delta_first_minus_second']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    lines.append('| ' + view + ' | ' + ' | '.join(cells) + ' |')
lines += [
    '', '差异使用完全相同的固定留出身份、组和 500 次配对 bootstrap；区间条件于这一对已拟合分类器和一次响应训练，不包含跨采集或重复训练不确定性。', '',
    f"总体 PSD W1={A['B']['overall_PSD_W1']:.8f}；原共同 Q1–7 区等权平均 W1={A['B']['common_Q1_7_PSD_W1']:.8f}。两个原门槛均为 0.010。", '',
    '| 成分 | 必需 Q2–7 峰位 MAE | raw SD 平均相对误差 | 必需区失败 |',
    '|---|---:|---:|---|',
]
for key in ['0', '1']:
    row = A['B']['raw_width_relative_errors'][key]
    center = A['B']['peak_MAE_required_Q2_7'][key]
    lines.append('| ' + ('低峰' if key == '0' else '高峰') + ' | ' + value_or_undefined(center) +
                 ' | ' + value_or_undefined(row['mean_absolute_relative_rawSD_error'], '.3%') +
                 ' | ' + str(row['failed_required_bins']) + ' |')
lines += [
    '', '全部八个区间及失败密度拟合均保留；密度成分没有粒子真值标签。原始 SOURCE purehad Q3/Q4 为 213/181 条，继续使用未量化输入电荷定义；旧 postADC 参考 207/182 单列保存，不能互换分母。', '',
    '| 原阶段门 | 结果 |', '|---|---|',
]
for key, row in stepgate.items():
    if key.startswith('gate'):
        lines.append('| ' + key + ' | ' + result(row['pass']) + ' |')
lines += ['', '| 原始 SOURCE purehad 区间 | N | 输出谷比例 | BASE 谷比例 | 增量上限 | 结果 |',
          '|---|---:|---:|---:|---:|---|']
for row in stepgate['gate3_source_preservation']['actual_source_pure_had_valleys']:
    lines.append(f"| {row['actual_source_bin']} | {row['N']} | {row['candidate']:.6f} | {row['BASE']:.6f} | 0.020000 | {result(row['pass'])} |")
noise = stepgate['gate2_PSD_width_centers_noise']['measures']
lines += ['', '| 噪声量 | 候选 W1 | BASE W1 | 相对变化 |', '|---|---:|---:|---:|']
for key in ['baseline_RMS_V_overall_W1', 'ACF1_overall_W1']:
    row = noise[key]
    lines.append('| ' + key + ' | ' + value_or_undefined(row['candidate'], '.9g') + ' | ' +
                 value_or_undefined(row['BASE'], '.9g') + ' | ' + value_or_undefined(row['relative_change'], '+.3%') + ' |')
num = local('TRAIN_SOLVER_CHECK.json')['models'][ARM]
lines += [
    '', f"TRAIN 128/256 数值差：median={num['RMS_median_LSB']:.6f} LSB、p95={num['RMS_p95_LSB']:.6f} LSB；preADC 相对电荷最大误差={num['preADC_relative_q_max']:.3e}。原门槛为 0.25 / 1 LSB 和 1e-5。", '',
    f"全部原 R48 随机流及 R53 教师匹配流一致性在训练进程内通过；连续电荷权重收到非零梯度的更新数为 {train['continuous_Q_nonzero_gradient_steps']}。该实现记录不能证明响应正确。", '',
    f"独立 GPU 进程通过可移动推理入口实际重放 {replay['rows']} 条，位级一致结果为 {replay['pass']}。全部生成及探针合计 5984 条；新 HGB 拟合 10 次、密度槽 16 个。", '',
    f"六个原生作业均退出 0，累计科学及必要汇总耗时 {science:.6f} 秒；最大 worker RSS {max(n['actual_worker_RSS_peak_bytes'] for n in native)} B，RSS 查询缺失 {sum(n['memory_probe_failures'] for n in native)} 次。缺失查询的具体原因若无日志证据则保持未知。新响应训练 1 次、50000 更新；DEV 已有多轮历史暴露，本轮未打开最终 EVAL 或新解码 CAL。", '',
    direction['next_action'], '',
    '[原协议](tasks/' + ID + '.md) · [原始判据](work/' + ID + '/ACCEPTANCE_A_C.json) · [指标](work/' + ID + '/METRICS_COMPACT.json) · [R53 配对比较](work/' + ID + '/PAIRED_R54_MINUS_R53.json) · [推理与复现](work/' + ID + '/REPRODUCTION.md) · [训练回执](work/' + ID + '/' + ARM + '/TRAINING_RECEIPT.json) · [独立重放](work/' + ID + '/REPLAY_RECEIPT.json) · [资源回执](work/' + ID + '/RESOURCE_RECEIPT.json)', '',
]
for name in ['canonical_five_view_response.png', 'PSD_all_eight_bins.png', 'PSD_centers_widths_and_W1.png',
             'noise_and_source_association.png', 'complete_waveform_population.png',
             'fixed_time_boundaries.png', 'training_and_checks.png']:
    lines += ['![' + name + '](work/' + ID + '/figures/' + name + ')', '']
(R / 'R54_REPORT.md').write_text('\n'.join(lines), 'utf8')
resource = {
    'UTC': NOW, 'stage': ID, 'science_s': science, 'native_entries': entries, 'native_receipts': native,
    'response_training_runs_started': 1, 'new_response_updates': 50000, 'generation_rows': 5984,
    'new_HGB_fits': 10, 'density_description_slots': 16, 'density_EM_attempts': M['counters']['density_EM_attempts'],
    'independent_replay_rows': 4000, 'DEV_read': True, 'EVAL_read': 0, 'CAL_waveform_decode': 0,
    'compute_budget_enforced': False, 'worker_peak_RSS_bytes': max(n['actual_worker_RSS_peak_bytes'] for n in native),
    'sampled_tree_peak_RSS_bytes': max(n['job_tree_RSS_peak_bytes'] for n in native),
    'memory_probe_failures': sum(n['memory_probe_failures'] for n in native),
    'GPU_query_failure_count': sum(len(n.get('GPU_query_failures', [])) for n in native),
}
write('work/' + ID + '/RESOURCE_RECEIPT.json', resource)
required = ['config.json', 'config_post.json', 'PROTOCOL.md', 'TEACHER_VALIDATION.json', 'TRAIN_SOLVER_CHECK.json',
            'METRICS_COMPACT.json', 'ACCEPTANCE_A_C.json', 'GATE_METRICS.json', 'PAIRED_R54_MINUS_R53.json',
            'REPLAY_RECEIPT.json', 'FIGURE_QA.json', 'REPRODUCTION.md', 'RESOURCE_RECEIPT.json',
            'infer_saved.py', 'UNIT_CHARGE_CONSTANTS.npz', 'NOISE_CALIBRATION.npz',
            ARM + '/FINAL_FLOW.pt', ARM + '/TRAINING_RECEIPT.json', ARM + '/CONTINUOUS_CONDITION.json']
write('work/' + ID + '/FINAL_RECEIPT.json', {
    'UTC': NOW, 'stage': ID, 'status': 'scientific_endpoint_complete_delivery_pending',
    'model_goal_complete': False, 'scientific_decision': gate['decision'], 'A_C_pass': gate['primary_A_C_pass'],
    'required_files': [bind('work/' + ID + '/' + name) for name in required],
    'resource_receipt': 'RESOURCE_RECEIPT.json', 'actual_independent_replay_pass': replay['pass'],
    'private_Git_and_D_delivery': 'pending separate receipts',
})
b = read('RESOURCE_BUDGET.json')
block = b['improvement_block']
assert not any(q['stage'] == ID for q in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({
    'stage': ID, 'receipt': 'work/' + ID + '/RESOURCE_RECEIPT.json', 'science_s': science,
    'response_training_runs_started': 1, 'new_response_updates': 50000,
    'status': 'fixed_recipe_science_complete', 'stage_science_complete': True,
})
block['actual_science_seconds_recorded'] += science
next(q for q in block['registered_stages'] if q['stage'] == ID).update(
    status='fixed_recipe_science_complete', active_scientific_jobs=0,
    actual_science_seconds=science, new_response_updates=50000, stage_science_complete=True)
block['response_training_runs_started'] = sum(q.get('response_training_runs_started', 0) for q in block['actual_stage_receipts'])
write('RESOURCE_BUDGET.json', b)
job = {'stage': ID, 'status': 'science_complete_delivery_pending', 'active_scientific_jobs': 0,
       'actual_worker_PID': native[-1]['actual_worker_PID'], 'supervisor_PID': native[-1]['supervisor_PID'],
       'actual_native_exit': 0}
for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    x = read(name)
    x.update(current_job=job, current_stage_progress='science_complete_delivery_pending', application_goal_status='active',
             active_scientific_jobs=0, active_remote_scientific_jobs=0, active_local_scientific_jobs=0,
             next_action=direction['next_action'], latest_completed_stage=ID, model_goal_complete=False,
             latest_completed_report='R54_REPORT.md', latest_scientific_decision=gate['decision'],
             latest_key_result_review='reviews/R54_ROOT_DIRECTION_001.json', status='R54_science_complete_delivery_pending_goal_active')
    if name == 'PROGRAM.json':
        x['updated_at_utc'] = NOW
    else:
        x.update(project_status=x['status'], current_activity='root_R54_delivery', current_direction=direction['next_action'], updated_UTC=NOW)
    write(name, x)
for name, key, ident in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    x = read(name)
    q = next(q for q in x[key] if q[ident] == ID)
    q.update(status='science_complete_delivery_pending', scientific_decision=gate['decision'], science_s=science,
             native_exit=0, final_receipt='work/' + ID + '/FINAL_RECEIPT.json', model_goal_complete=False,
             active_scientific_jobs=0, new_training_runs_completed=1)
    write(name, x)
top = '# 当前执行：R54 固定方案科学终点完成，长期目标 active\n\n' + (
    'R54 单一连续电荷条件消融已完成 50000 更新、原数值门、完整响应/源/噪声/边界评价和独立重放。A/B/C：' +
    ' / '.join(result(A[k]['pass']) for k in ['A', 'B', 'C']) +
    '。原 A–F 不变，详见 R54_REPORT.md。当前完成私有 Git 与 D 盘交付。' + direction['next_action'] +
    ' 主助手独立执行，不使用子智能体；无计算配额，CPU12/RSS48GiB/GPU10GiB、单科学作业及本机 E/D 存储限制保留。\n\n')
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    p = R / name
    old = p.read_text('utf8')
    assert '## 以下为历史记录' in old
    tail = old.split('## 以下为历史记录', 1)[1]
    p.write_text(top + '## 以下为历史记录' + tail, 'utf8')
print(json.dumps({'science_s': science, 'cumulative_science_s': block['actual_science_seconds_recorded'],
                  'A_C_pass': gate['primary_A_C_pass'], 'model_goal_complete': False, 'decision': gate['decision']}, indent=2))
