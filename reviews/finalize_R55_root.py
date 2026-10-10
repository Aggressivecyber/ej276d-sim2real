from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
ID = 'R55_PSD_SOURCE_COMPATIBILITY_001'
S = R / 'work' / ID
NOW = datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((R / name).read_text('utf8'))


def local(name):
    return read('work/' + ID + '/' + name)


def write(name, value):
    (R / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def bind(name):
    p = R / name
    return {'path': p.relative_to(R).as_posix(), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


result = local('RESULT.json')
native = local('numeric_worker.py.NATIVE_RECEIPT.json')
qa = local('FIGURE_QA.json')
direction = read('reviews/R55_ROOT_DIRECTION_001.json')
assert result['status'] == 'complete' and local('NUMERIC_RECEIPT.json')['status'] == 'complete'
assert native['actual_worker_native_exit'] == native['worker_self_receipt']['native_exit'] == 0
assert native['termination_reason'] is None and native['handshake_before_imports']
assert not local('LIVE_OBSERVATION.json')['actual_process']['exists']
assert qa['all_figures_opened'] and qa['layout_pass']
assert result['scientific_decision'] == direction['scientific_decision'] and direction['model_goal_complete'] is False
c = result['construction']
certificate = result['certificate']
bound = certificate['lower_bound']['float_rounded_down'] if certificate else None
science = native['elapsed_s']
counter = result['counter']
assert counter['LP_attempts'] == 1 and all(counter[k] == 0 for k in ['response_updates', 'HGB_fits', 'density_fits', 'ODE_generation', 'EVAL_read', 'CAL_decode'])
lines = ['# R55：固定源人口的必要条件兼容性结果', '', direction['conclusion_zh'], '',
    '这是关于同一固定经验人口的保守分布下界诊断。它没有训练响应模型、生成新候选或修改原 A–F；R54 的失败结论保持。', '',
    '| 项目 | 实际结果 |', '|---|---|',
    '| 原 SOURCE / REAL | 2000 / 2000 |',
    '| 原 purehad Q3 / Q4 | 213 / 181 |',
    '| 可能跨输出分箱的源事件 | ' + str(c['migration_uncertain_SOURCE_rows']) + ' |',
    '| 变量 / 约束 / 非零系数 | ' + str(c['variables']) + ' / ' + str(c['rows']) + ' / ' + str(c['nnz']) + ' |',
    '| 固定 PSD 网格点 | ' + str(c['grid_points']) + ' |',
    '| LP 求解次数 | 1 |',
    '| 求解器状态 | ' + str(result['solver']['status']) + '：' + result['solver']['message'] + ' |',
    '| 数值 primal objective（不能单独证明下界） | ' + str(result['solver']['numerical_primal_objective_not_a_lower_bound']) + ' |',
    '| 精确证书下界的向下浮点表示 | ' + (format(bound, '.12g') if bound is not None else '无有效证书') + ' |',
    '| 诊断冲突门槛 | 严格大于 0.010001 |', '',
    '原总体 W1 验收仍为 0.010；额外 1e-6 只用于让本诊断更保守。共同 Q1–7 的等权条件均值和原始源谷比例也采用向宽松方向的微小余量。LP 中的分数质量变量是数学放宽，未用于重加权原 SOURCE、删源事件或输出新候选。', '',
    '| 原始源区间 | 事件数 | 放宽后的输出区质量范围 |', '|---|---:|---:|']
for key in map(str, range(-1, 9)):
    lo, hi = c['allowed_bin_mass_bounds'][key]
    label = '下方支持外' if key == '-1' else '上方支持外' if key == '8' else 'Q' + str(int(key) + 1)
    lines.append(f"| {label} | {c['raw_source_bin_counts'][key]} | {lo:.6f}–{hi:.6f} |")
lines += ['', '每个采样点预留一整 ADC LSB，再加入原 preADC 相对电荷容差及浮点余量。常数电荷半径为 ' +
          format(c['q_post_radius_constant_Vns'], '.12g') + ' V ns；实际输入最大绝对幅值为 ' +
          format(c['source_abs_input_max_V'], '.12g') + ' V，保存特征与运行时 q 定义差为 ' +
          format(c['source_runtime_vs_feature_q_max_Vns'], '.3g') + ' V ns。支持外事件全部保留。', '',
    '区间平均 CDF 的 Jensen 下界、分箱质量乘积的 McCormick 放宽、源谷概率的下界以及网格外积分的省略，都只会降低所求下界。矩阵向 float64 转换时各行向外放宽；最大右端项余量为 ' +
    format(c['max_rhs_outward_relaxation'], '.3e') + '。实际保存的矩阵与对偶随后用于精确有理数证书。', '']
if certificate:
    lines += [
        '证书计算 `yᵀb + Σ min(0,c−Aᵀy)`，包含 ' + str(certificate['negative_dual_residual_columns']) +
        ' 个负残差列的保守修正。分子、分母和全部矩阵/对偶保存在回执及私有 NPZ 中；求解器 success 标记本身不承担不可能性证明。', '']
lines += [
    '本轮没有为完整波形、固定 HGB、密度峰位/峰宽、源秩、噪声、REAL 控制或 D/E 建立可行性证书；这些仍须由同一冻结方法实际通过。历史 DEV 暴露保持披露，没有读取最终 EVAL 或新解码 CAL。', '',
    f"原生科学进程退出 0，耗时 {science:.6f} 秒，worker 峰值 RSS {native['actual_worker_RSS_peak_bytes']} B，作业树采样峰值 {native['job_tree_RSS_peak_bytes']} B。缺失 RSS 查询 {native['memory_probe_failures']} 次，保留原回执，不补造原因。", '',
    direction['next_action'], '',
    '[协议](tasks/' + ID + '.md) · [结果](work/' + ID + '/RESULT.json) · [构造说明](work/' + ID +
    '/LP_CONSTRUCTION.json) · [求解回执](work/' + ID + '/LP_SOLVER_RECEIPT.json) · [证书](work/' + ID +
    '/LOWER_BOUND_CERTIFICATE.json) · [复现](work/' + ID + '/REPRODUCTION.md)', '',
    '![固定人口兼容性下界](work/' + ID + '/compatibility_bounds.png)', '']
(R / 'R55_REPORT.md').write_text('\n'.join(lines), 'utf8')
resource = {'UTC': NOW, 'stage': ID, 'science_s': science, 'native_receipts': [native],
            'response_training_runs_started': 0, 'new_response_updates': 0, 'new_HGB_fits': 0, 'new_density_fits': 0,
            'LP_attempts': 1, 'certificate_calculations': counter['certificate_calculations'],
            'DEV_read': True, 'EVAL_read': 0, 'CAL_decode': 0, 'compute_budget_enforced': False,
            'worker_peak_RSS_bytes': native['actual_worker_RSS_peak_bytes'], 'sampled_tree_peak_RSS_bytes': native['job_tree_RSS_peak_bytes'],
            'memory_probe_failures': native['memory_probe_failures']}
write('work/' + ID + '/RESOURCE_RECEIPT.json', resource)
required = ['config.json', 'PROTOCOL.md', 'INPUT_BINDING.json', 'numeric_worker.py', 'certificate_core.py',
            'verify_certificate.py', 'LP_PRIVATE.npz', 'LP_CONSTRUCTION.json', 'LP_SOLVER_RECEIPT.json',
            'RESULT.json', 'FIGURE_QA.json', 'RESOURCE_RECEIPT.json', 'REPRODUCTION.md', 'compatibility_bounds.png']
if certificate:
    required.append('LOWER_BOUND_CERTIFICATE.json')
write('work/' + ID + '/FINAL_RECEIPT.json', {'UTC': NOW, 'stage': ID, 'status': 'science_complete_delivery_pending',
    'model_goal_complete': False, 'scientific_decision': result['scientific_decision'],
    'required_files': [bind('work/' + ID + '/' + n) for n in required], 'private_Git_D_delivery': 'pending separate receipts'})
b = read('RESOURCE_BUDGET.json')
block = b['improvement_block']
assert not any(v['stage'] == ID for v in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({'stage': ID, 'receipt': 'work/' + ID + '/RESOURCE_RECEIPT.json',
    'science_s': science, 'response_training_runs_started': 0, 'new_response_updates': 0, 'LP_attempts': 1,
    'status': 'fixed_diagnostic_complete', 'stage_science_complete': True})
block['actual_science_seconds_recorded'] += science
next(v for v in block['registered_stages'] if v['stage'] == ID).update(status='fixed_diagnostic_complete',
    active_scientific_jobs=0, actual_science_seconds=science, stage_science_complete=True)
write('RESOURCE_BUDGET.json', b)
job = {'stage': ID, 'status': 'science_complete_delivery_pending', 'actual_worker_PID': native['actual_worker_PID'],
       'supervisor_PID': native['supervisor_PID'], 'actual_native_exit': 0, 'active_scientific_jobs': 0}
for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    x = read(name)
    x.update(status='R55_science_complete_delivery_pending_goal_active', current_job=job, current_stage_progress='science_complete_delivery_pending',
        latest_completed_stage=ID, latest_completed_report='R55_REPORT.md', latest_scientific_decision=result['scientific_decision'],
        latest_key_result_review='reviews/R55_ROOT_DIRECTION_001.json', next_action=direction['next_action'],
        application_goal_status='active', model_goal_complete=False, active_scientific_jobs=0,
        active_remote_scientific_jobs=0, active_local_scientific_jobs=0)
    if name == 'PROGRAM.json':
        x['updated_at_utc'] = NOW
    else:
        x.update(project_status=x['status'], current_activity='root_R55_delivery', current_direction=direction['next_action'], updated_UTC=NOW)
    write(name, x)
for name, key, ident in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    x = read(name)
    next(v for v in x[key] if v[ident] == ID).update(status='science_complete_delivery_pending',
        scientific_decision=result['scientific_decision'], science_s=science, native_exit=0,
        final_receipt='work/' + ID + '/FINAL_RECEIPT.json', model_goal_complete=False, active_scientific_jobs=0)
    write(name, x)
top = '# 当前执行：R55 必要条件诊断完成，长期目标 active\n\n' + direction['conclusion_zh'] + (
    ' 原 A–F 保持，模型目标未完成。当前完成私有 Git 与 D 盘交付。' + direction['next_action'] +
    ' 主助手独立、无子智能体、无计算配额；CPU12/RSS48GiB、单科学作业及本机 E/D 限制保持。\n\n')
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    p = R / name
    tail = p.read_text('utf8').split('## 以下为历史记录', 1)[1]
    p.write_text(top + '## 以下为历史记录' + tail, 'utf8')
print(json.dumps({'decision': result['scientific_decision'], 'science_s': science,
                  'cumulative_science_s': block['actual_science_seconds_recorded'], 'model_goal_complete': False}, indent=2))
