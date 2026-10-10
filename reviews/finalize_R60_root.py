from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
R = Path(__file__).resolve().parents[1]
ID = 'R60_FROZEN_TRAIN_GRADIENT_DIAGNOSTIC_001'
S = R / 'work' / ID
NOW = datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads((R / p).read_text('utf8'))

def write(p, value):
    (R / p).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8'))

def local(p):
    return read('work/' + ID + '/' + p)

def bind(p):
    path = R / p
    return {'path': path.relative_to(R).as_posix(), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

result = local('RESULT.json')
full = local('FULL_TRAIN_SUMMARY.json')
trace = local('SAVED_R59_TRAINING_TRACE_SUMMARY.json')
native = local('diagnose_train.py.NATIVE_RECEIPT.json')
review = local('FROZEN_ARTIFACT_REVIEW.json')
qa = local('FIGURE_QA.json')
direction = read('reviews/R60_ROOT_DIRECTION_001.json')
assert review['pass'] and qa['all_figures_opened'] and qa['layout_pass'] and qa['figure_count'] == 7
assert native['actual_worker_native_exit'] == native['worker_self_receipt']['native_exit'] == 0
assert native['termination_reason'] is None and not local('LIVE_OBSERVATION.json')['actual_process']['exists']
assert result['numerical_decomposition_pass'] and not any(v['activated'] for v in result['diagnostic_markers'].values())
science = native['elapsed_s']
pair = full['source_order']
groups = pair['groups']
fmt = lambda x: format(x, '.8g')
rows = [
    '# R60：冻结 TRAIN 分布与梯度诊断', '',
    '固定诊断已完成，三个预设冲突标记均未触发：存储排序 17/32、支持外全窗 0/32、谷区与 PSD 4/32，均低于预定的 24/32。保留这些否定结果，不将候选解释写成已确认机制。R59 原 A/B/C 失败保持；本轮没有新模型，长期目标仍需同一冻结方法满足原 A–F。', '',
    '对全部 6593 SOURCE 和 21047 REAL 原 TRAIN 父事件，各用一个固定噪声执行原 Heun128。另完成固定 32 批、17 项梯度分解；合计 39928 行实际神经前向、465 次网络调用、544 次求导。0 训练更新、教师、HGB 或密度拟合；0 DEV/EVAL/CAL 波形读取。所有原区域、支持外事件与源父权重保留。', '',
    '| 冻结 TRAIN 描述 | 数值 |', '|---|---:|',
    '| SOURCE 总体 PSD W1 | ' + fmt(full['overall_SOURCE_PSD_W1']) + ' |',
    '| SOURCE 共同 Q1–7 PSD W1 等区均值 | ' + fmt(full['common_Q1_Q7_mean_SOURCE_PSD_W1']) + ' |',
    '| 同处理 REAL 总体 PSD W1 | ' + fmt(full['overall_same_treated_REAL_PSD_W1']) + ' |',
    '| 原输入/输出 SOURCE PSD Spearman | ' + fmt(full['source_PSD_rank_rho']) + ' |',
    '| 原输入/输出 SOURCE Q Spearman | ' + fmt(full['source_Q_rank_rho']) + ' |', '',
    '总体距离较小并未解决条件响应。Q3/Q4 中 SOURCE 中位 PSD 约为 0.18862/0.18905，95% 分位约为 0.19189/0.19141，仍聚集在窄峰；同处理 REAL 对应中位数为 0.20599/0.21991、95% 分位为 0.33882/0.33130。以下是 TRAIN 描述，不替代原 DEV 验收，也未重新做密度拟合。', '',
    '| 原电荷区域 | SOURCE N | REAL N | SOURCE PSD W1 | 同处理 REAL PSD W1 | SOURCE 范围投影比例 |',
    '|---|---:|---:|---:|---:|---:|']
for item in full['per_original_region']:
    region = item['raw_region']
    label = 'below' if region == -1 else 'above' if region == 8 else 'Q' + str(region + 1)
    rows.append('| ' + label + ' | ' + str(item['SOURCE_N']) + ' | ' + str(item['REAL_N']) + ' | ' + fmt(item['SOURCE_PSD_W1']) + ' | ' + fmt(item['same_treated_REAL_PSD_W1']) + ' | ' + format(item['SOURCE_projection_range_fraction'], '.4%') + ' |')
rows += ['',
    'Q3/Q4 中 h==1 的源父事件分别为 482/435，固定硬谷区输出比例均为 0，平滑谷概率分别为 0.01810694/0.01663660。原谷定义和来源标签边界保持；低谷比例与真实响应匹配必须一起判断。', '',
    '| 全 TRAIN 原非零有序父对 | 对数 | 占比 | 对原排序损失的贡献 |', '|---|---:|---:|---:|']
for key, label in [('roundoff_compatible', 'float32 PSD 区间重叠'), ('resolved', '存储区间可分辨')]:
    item = groups[key]
    rows.append('| ' + label + ' | ' + str(item['pairs']) + ' | ' + format(item['nonzero_pair_fraction'], '.4%') + ' | ' + fmt(item['contribution_to_original_order_loss']) + ' |')
rows += ['',
    '完整有序父对共 ' + str(pair['ordered_parent_pairs_total']) + '；严格非零对 ' + str(pair['strict_nonzero_pairs']) + '，精确相等对 ' + str(pair['exact_equal_pairs']) + '（其中 6593 为同父对角线）。区间重叠者贡献原排序罚的 ' + format(direction['roundoff_compatible_share_of_original_order_penalty'], '.4%') + '。PSD 舍入区间宽度约 9.60e−8 至 1.36e−7；这是 float32 存储包络，不是仪器误差，也不证明真实输入完全等 PSD。', '',
    '存储排序标记中，“质量至少一半”和“梯度范数占比至少一半”均在 32/32 批满足，但与全窗+PSD+噪声梯度的余弦低于 −0.25 仅有 17/32。支持外梯度范数占比条件也为 32/32，但内外余弦冲突为 0/32。谷梯度仅 7 批达到 PSD 梯度范数的一半，其中 4 批满足余弦条件；另外 25 批谷梯度为零，余弦保留未定义。没有修改判据或将单个子条件替代整个标记。', '',
    '全部分解值和参数梯度重组通过预登记数值容差，冻结参数逐张量 SHA 前后一致。保存的 Gram 矩阵、父索引、全部区域 W1、源关联和历史损失轨迹已由本机独立核算；本机复核没有执行新的模型前向。欧氏梯度不是实际 Adam 更新方向，局部反向也不是因果结论。', '',
    '全 TRAIN 有 811/6593 个 SOURCE 触发内部范围投影，其中 810 个位于高端支持外，1 个位于 Q8；同处理 REAL 为 0/21047。高端支持外原输入本身已有 816/918 越界。SOURCE 投影修正 RMS 均值 ' + fmt(full['projection']['SOURCE']['RMS_correction_V_mean']) + ' V、最大事件 RMS ' + fmt(full['projection']['SOURCE']['RMS_correction_V_max']) + ' V，落在电压端点的采样数为 24479。源/REAL 最大 preADC 相对电荷误差为 ' + fmt(full['projection']['SOURCE']['preADC_relative_Q_error_max']) + ' / ' + fmt(full['projection']['same_treated_REAL']['preADC_relative_Q_error_max']) + '。这些全 TRAIN 结果与 R59 主 DEV 中投影为零的范围不同。', '',
    'R59 既有 50000 步记录中，' + format(trace['gradient_clipped_fraction'], '.2%') + ' 的裁剪前梯度范数超过 5；范数中位数为 ' + fmt(trace['gradient_norm_quantiles'][2]) + '。这描述保存的训练轨迹，没有据此挑选另一个检查点。', '',
    '原生作业退出 0，科学执行（监督器计时）' + fmt(science) + ' 秒；worker 峰值 RSS ' + str(native['actual_worker_RSS_peak_bytes']) + ' B。GPU allocated/reserved 峰为 ' + str(result['GPU_allocated_peak_bytes']) + ' / ' + str(result['GPU_reserved_peak_bytes']) + ' B。有 1 次实际 PID 的 RSS 查询失败，回执原因保留 unknown；未把失败补成成功采样。真实 PID 随后的 /proc 查询确认不存在。所有七张图实际打开核对通过。', '',
    direction['next_action'], '',
    '下一轮是另行登记的假设检验，不是 R60 标记触发后的机制确认；当前尚未登记或开始下一轮训练。', '',
    '[协议](tasks/' + ID + '.md) · [结果](work/' + ID + '/RESULT.json) · [逐批记录](work/' + ID + '/GRADIENT_BATCH_RECORDS.json) · [产物复核](work/' + ID + '/FROZEN_ARTIFACT_REVIEW.json) · [复现](work/' + ID + '/REPRODUCTION.md) · [后续判断](reviews/R60_ROOT_DIRECTION_001.json)', '']
for figure in result['figures']:
    rows += ['![' + Path(figure).name + '](work/' + ID + '/' + figure + ')', '']
(R / 'R60_REPORT.md').write_bytes('\n'.join(rows).encode('utf8'))
write('reviews/R60_ROOT_REVIEW_ENVIRONMENT_REPAIR_001.json', {
    'UTC': NOW, 'initial_attempt_native_exit': 1, 'error': "ModuleNotFoundError: No module named 'psutil'",
    'failure_before_saved_data_load_or_model_import': True, 'new_model_forwards_or_training_on_failed_attempt': 0,
    'repair': 'Use native Windows process affinity and memory APIs through ctypes; no package installation.',
    'repaired_checker_native_exit': 0, 'checker': bind('reviews/review_R60_results_root.py')})
resource = {'UTC': NOW, 'stage': ID, 'science_s': science, 'scientific_body_s': result['elapsed_s'],
            'native_entries': ['diagnose_train.py'], 'native_receipts': [native], 'response_training_runs_started': 0,
            'new_response_updates': 0, 'actual_counters': result['counter'], 'compute_budget_enforced': False,
            'worker_peak_RSS_bytes': native['actual_worker_RSS_peak_bytes'], 'sampled_tree_peak_RSS_bytes': native['job_tree_RSS_peak_bytes'],
            'memory_probe_failures': native['memory_probe_failures'], 'GPU_query_failure_count': len(native['GPU_query_failures']),
            'GPU_allocated_peak_bytes': result['GPU_allocated_peak_bytes'], 'GPU_reserved_peak_bytes': result['GPU_reserved_peak_bytes'],
            'root_saved_artifact_review_elapsed_s': review['elapsed_s'], 'root_saved_artifact_review_peak_RSS_bytes': review['peak_RSS_bytes'],
            'root_saved_artifact_review_new_model_forwards': 0}
write('work/' + ID + '/RESOURCE_RECEIPT.json', resource)
required = ['config.json', 'INPUT_BINDING.json', 'PROTOCOL.md', 'ROOT_IMPLEMENTATION_REVIEW.json', 'RESULT.json', 'FULL_TRAIN_SUMMARY.json',
            'GRADIENT_SUMMARY.json', 'GRADIENT_BATCH_RECORDS.json', 'FULL_TRAIN_DIAGNOSTICS_PRIVATE.npz', 'GRADIENT_DIAGNOSTICS_PRIVATE.npz',
            'SAVED_R59_TRAINING_TRACE_SUMMARY.json', 'INITIAL_RANDOM_STATES.json', 'FINAL_RANDOM_STATES.json', 'RANDOM_STREAM_RECEIPT.json',
            'FROZEN_ARTIFACT_REVIEW.json', 'FIGURE_QA.json', 'REPRODUCTION.md', 'RESOURCE_RECEIPT.json', 'diagnose_train.py.NATIVE_RECEIPT.json']
write('work/' + ID + '/FINAL_RECEIPT.json', {'UTC': NOW, 'stage': ID, 'status': 'scientific_endpoint_complete_delivery_pending',
      'model_goal_complete': False, 'scientific_decision': direction['scientific_decision'], 'original_A_F_changed': False,
      'diagnostic_markers': result['diagnostic_markers'], 'original_A_C_reassessed': False, 'new_model_candidate': False,
      'required_files': [bind('work/' + ID + '/' + name) for name in required],
      'private_Git_and_D_delivery': 'pending separate project-root receipts', 'reuses_historical_R59_weights_and_inputs': True})
b = read('RESOURCE_BUDGET.json')
block = b['improvement_block']
assert not any(q['stage'] == ID for q in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({'stage': ID, 'receipt': 'work/' + ID + '/RESOURCE_RECEIPT.json', 'science_s': science,
                                    'response_training_runs_started': 0, 'new_response_updates': 0,
                                    'status': 'fixed_diagnostic_science_complete', 'stage_science_complete': True})
block['actual_science_seconds_recorded'] += science
next(q for q in block['registered_stages'] if q['stage'] == ID).update(status='fixed_diagnostic_science_complete', active_scientific_jobs=0,
       actual_science_seconds=science, new_response_updates=0, stage_science_complete=True)
write('RESOURCE_BUDGET.json', b)
job = {'stage': ID, 'status': 'science_complete_delivery_pending', 'active_scientific_jobs': 0,
       'actual_worker_PID': native['actual_worker_PID'], 'supervisor_PID': native['supervisor_PID'], 'actual_native_exit': 0}
for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    value = read(name)
    value.update(current_job=job, current_stage_progress='science_complete_delivery_pending', application_goal_status='active',
                 active_scientific_jobs=0, active_remote_scientific_jobs=0, active_local_scientific_jobs=0, next_action=direction['next_action'],
                 latest_completed_stage=ID, model_goal_complete=False, latest_completed_report='R60_REPORT.md',
                 latest_scientific_decision=direction['scientific_decision'], latest_key_result_review='reviews/R60_ROOT_DIRECTION_001.json',
                 status='R60_science_complete_delivery_pending_goal_active')
    value['execution_location'].update(current_stage_GPU_used=True, preflight_required=False)
    if name == 'PROGRAM.json':
        value['updated_at_utc'] = NOW
    else:
        value.update(project_status=value['status'], current_activity='root_R60_delivery', current_direction=direction['next_action'], updated_UTC=NOW)
    write(name, value)
for name, key, ident in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    value = read(name)
    q = next(q for q in value[key] if q[ident] == ID)
    q.update(status='science_complete_delivery_pending', scientific_decision=direction['scientific_decision'], science_s=science,
             native_exit=0, final_receipt='work/' + ID + '/FINAL_RECEIPT.json', model_goal_complete=False, active_scientific_jobs=0,
             new_training_runs_completed=0)
    write(name, value)
top = '# 当前执行：R60 冻结 TRAIN 诊断完成，长期目标 active\n\n全部27640原TRAIN父事件和32批17项梯度分解已完成。三个预设冲突标记17/0/4批，均未达24/32，不能宣称机制已定位。float32区间重叠占54.61%非零源父对、贡献82.34%排序罚；它仅支持下一轮另行预登记独立消融。0训练及DEV/EVAL/CAL读取，原A–F保持。当前完成私有 Git 与 D 盘交付。' + direction['next_action'] + ' 主助手独立，无子智能体、无计算配额；机器安全、单科学作业及本机E/D限制保持。\n\n'
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    path = R / name
    text = path.read_text('utf8')
    assert '## 以下为历史记录' in text
    path.write_bytes((top + '## 以下为历史记录' + text.split('## 以下为历史记录', 1)[1]).encode('utf8'))
print(json.dumps({'science_s': science, 'new_training_updates': 0, 'diagnostic_markers': result['diagnostic_markers'],
                  'report': 'R60_REPORT.md', 'model_goal_complete': False}, ensure_ascii=False, indent=2))
