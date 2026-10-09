from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, os

ROOT = Path(__file__).resolve().parent
def now(): return datetime.now(timezone.utc).isoformat()
def read(name): return json.loads((ROOT / name).read_text('utf8'))
cfg = read('config.json')
def write(name, value):
    if datetime.now(timezone.utc) >= datetime.fromisoformat(cfg['deadline_UTC']):
        raise TimeoutError('delivery deadline')
    data = value.encode('utf8') if isinstance(value, str) else (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf8')
    if len(data) > 128 * 1024: raise ValueError('control document cap')
    if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) + len(data) > cfg['output_limit_bytes']:
        raise ValueError('stage output cap')
    (ROOT / name).write_bytes(data)

c, d, g = read('CLASSIFIER_METRICS.json'), read('DECODER_METRICS.json'), read('GEOMETRY_METRICS.json')
history, receipt, env = read('RUN_HISTORY.json'), read('NUMERIC_RECEIPT.json'), read('NUMERIC_STARTED.json')
views = cfg['views']
names = ['FRESH_FINAL', 'ENERGY_FINAL', 'TRAIN_REFERENCE']
labels = {'FRESH_FINAL': 'R28 Fresh2950', 'ENERGY_FINAL': 'R29 Energy2950', 'TRAIN_REFERENCE': 'REAL–REAL'}
vl = {'RAW': 'RAW', 'GLOBAL_CENTER': 'Global center', 'Q_CENTER': 'Q center', 'Q_CENTER_SCALE': 'Q center/scale'}
def ci(x): return '[' + ', '.join(f'{v:.6f}' for v in x) + ']'
lines = [
    '# R31 固定归一化锚点诊断', '',
    '结论：GLOBAL_CENTER 的域身份可由已知峰归一化约束无拟合恢复。去除峰附近固定 50 点后，三种矩变换的 REAL–REAL 控制仍未通过预设容差。关闭“域别矩变换后 C2ST 唯一归因于物理协方差/涨落”的路线，保留完整原始指标。R28/R29 的剩余 430 点仍与实测明显可分，不能据此唯一定位物理原因。', '',
    '输入仅为 R30 已读过的四个保存域，各 2000 行；分类器沿用原每域 600 训练/250 留出行与人工组。先应用 R30 已保存的完整 480 维矩，再唯一保留 0:100 与 150:480。未重估矩、重新归一化或试其他掩码。六域条件支持仍按 R30 bins 1–7；本次逐区留出描述只在双方 N≥10 时报告，共同支持 bins 1–6。', '',
    '以下完整 480 维结果直接复用，430 维为本次 12 个固定 HGB。所有区组 CI 均为 500 次、条件于已拟合分类器，不涵盖响应训练或分类器训练不确定性。同采集人工组不称独立采集批次。', '',
    '| 比较 | 视图 | 原480 BA | 本次430 BA | 430−480 BA / 配对95%CI |',
    '|---|---|---:|---:|---|',
]
for name in names:
    for view in views:
        q = c['comparisons'][name][view]
        p = q['paired430_minus480']
        lines.append(f"| {labels[name]} | {vl[view]} | {q['full480_same_view_reused']['balanced_accuracy']:.3f} | {q['balanced_accuracy']:.3f} | {p['BA_delta_first_minus_second']:+.6f} / {ci(p['BA_delta_95CI'])} |")
lines += ['', 'RAW430 REAL–REAL BA=0.536，95%CI ' + ci(c['comparisons']['TRAIN_REFERENCE']['RAW']['heldout_CI']['balanced_accuracy_95CI']) + '，包含 0.5。对每种变换，预设控制还要求相对 RAW430 的配对 ΔBA CI 完全位于 [−0.10,+0.10]；这是操作性容差，不能解释成物理等效阈值。', '',
          '| 比较 | 430视图−RAW430 | ΔBA | 配对95%CI |', '|---|---|---:|---|']
for name in names:
    for view in views[1:]:
        q = c['comparisons'][name][view]['paired430_transformed_minus_RAW430']
        lines.append(f"| {labels[name]} | {vl[view]} | {q['BA_delta_first_minus_second']:+.6f} | {ci(q['BA_delta_95CI'])} |")
lines += ['', 'REAL–REAL 的 Global/Q/Q-scale 三项控制均失败。Global center 去峰后 BA 从 0.954 降至 0.662，说明排除区域对人工可分性有重要影响；仍不能把完整变换后 BA 分解成唯一的物理因素。RAW430 生成域 BA=0.988/0.984，相比 RAW430 REAL–REAL 0.536，说明剩余坐标也有明显差异；这只是固定掩码诊断，不是完整响应验收。', '',
          '无拟合解码器使用 w=u−μ_domain，分别检验 |max(w+μ_REAL)−1| 与 |max(w+μ_candidate)−1|，取误差较小者，差≤1e−12弃权。真实标签只用于评分。三组留出各 500/500 行覆盖、BA=1、弃权0；真实假设约束误差全部0。错误假设误差分位数如下（0/50/90/99/100%）：', '',
          '| 比较 | 错误假设误差分位数 |', '|---|---|']
for name in names:
    lines.append(f"| {labels[name]} | " + ', '.join(f'{v:.8g}' for v in d[name]['wrong_hypothesis_constraint_error_quantiles']) + ' |')
lines += ['', '这证明已中心化表示中存在可恢复的归一化几何域信息；不证明各 HGB 都仅使用该规则。低峰 floor 适用性独立记录，主分类与解码评分未删除任何行。', '',
          '| 域 | 全体/训练/留出 N | max(u)−1最大绝对误差 | floor活跃 | argmax在100:150 | 留出全局回退 |', '|---|---|---:|---:|---:|---:|']
for name in cfg['domains']:
    pop, train, hold = (g[name][s] for s in ['population', 'train', 'holdout'])
    lines.append(f"| {name} | {pop['N']}/{train['N']}/{hold['N']} | {pop['applicable_max_constraint_error']:.6g} | {pop['peak_floor_active_n']} | {pop['argmax_inside_prompt100_150_n']}/{pop['N']} | {hold['fallback_n']} |")
science = sum(x['elapsed_s'] for x in history)
peak = max(x['actual_worker_RSS_peak_bytes'] for x in history)
lines += ['', '真实执行为本机 CPU4，Python ' + env['python'] + '、NumPy ' + env['numpy'] + '、sklearn ' + env['sklearn'] + '。12/12 HGB 完成；统计、绘图和一次纯布局修复的累计科学墙钟 ' + f'{science:.6f}' + ' s / 300 s，实际 worker RSS 峰 ' + str(peak) + ' B / 4 GiB；全部 native exit0、RSS 查询失败0。没有响应训练/推理/生成、矩重估、EM、新 REAL/EVAL/CAL/G4、SSH 或安装。', '',
          '6 张最终图均已实际打开检查；paired_prompt_exclusion 的原拥挤标签图另名保留，只修布局，未重算指标。各末点仍按旧 TRAIN-val 选择规则与预指定2950末点身份报告，未据 DEV 改选。', '',
          '复现入口、输入路径和保存私有预测见 REPRODUCTION.md/config.json。原始 full480 与本次430及条件描述均保留；矩变换依赖域身份，只能作诊断，不是可部署输出，不计算新 PSD。下一步应回到原始波形残差和受控响应训练问题，本轮不追加任何实验。', '']
write('REPORT.md', '\n'.join(lines))
resource = {
    'stage': cfg['stage'], 'assembled_UTC': now(), 'assembly_PID': os.getpid(),
    'registered_UTC': cfg['registered_UTC'], 'first_actual_command': cfg['first_actual_command'],
    'executor_deadline_UTC': cfg['deadline_UTC'], 'root_deadline_UTC': cfg['root_deadline_UTC'],
    'execution': cfg['execution'], 'versions': env, 'scientific_jobs': history,
    'native_tool_chain': [
        {'entry': 'numeric_worker.py', 'launch_tool': '78b61c', 'terminal_tool': '116817', 'native_exit': 0},
        {'entry': 'render_figures.py', 'tool': 'fabe0f', 'native_exit': 0},
        {'entry': 'render_layout_fix.py', 'tool': '2fa7a1', 'native_exit': 0}],
    'science_elapsed_s': science, 'science_budget_s': cfg['science_budget'],
    'actual_worker_RSS_peak_bytes': peak, 'RSS_limit_bytes': cfg['RSS_limit_bytes'],
    'GPU_allocated_peak_bytes': 0, 'CPU_threads': 4, 'memory_probe_failures': sum(x['memory_probe_failures'] for x in history),
    'counters': receipt['counters'], 'saved_waveform_rows_read': receipt['saved_waveform_rows'],
    'new_scientific_failures': [], 'pure_control_failures': [],
    'layout_original_retained': 'figures/paired_prompt_exclusion.layout_initial.png',
    'layout_fix_scope': 'saved metrics only; no fit/bootstrap/statistic recomputation',
    'output_limit_bytes': cfg['output_limit_bytes'], 'output_final_bytes_recorded_by_final_receipt': True,
    'old_stage_writes': 0, 'domain_transform_physical_unique_attribution_route': 'closed_control_failed',
    'system_dependency_load_IO_not_in_project_data_ledger': True,
    'control_assembly_is_stdlib_documentation_not_scientific_computation': True,
}
write('RESOURCE_RECEIPT.json', resource)
figs = read('PLOT_RECEIPT.json')['figures']
write('FIGURE_QA.json', {'UTC': now(), 'reviewer': 'executor_actual_image_views; ROOT independently viewed all final six',
    'figures': [{'path': 'figures/' + f, 'actually_opened': True, 'labels_and_layout_readable': True} for f in figs],
    'initial_nonfinal_retained': 'figures/paired_prompt_exclusion.layout_initial.png',
    'layout_repair_final_view_tool': 'ac201a', 'final_view_tool_note': 'view_image in same functions exec; exec_command chunk ac201a is the concurrent control inventory',
    'other_initial_final_views': 'actual image views before layout repair; no subsequent changes to other five',
    'metrics_changed_for_QA': False})
text = '# R31 reproduction\n\n'
text += '本轮在 E/D 保存目录运行；旧阶段只读。实际 Python：`' + cfg['execution']['python'] + '`。环境版本见 RESOURCE_RECEIPT.json；真实 science PID 在 RUN_HISTORY.json 与每个 NATIVE_RECEIPT 中。\n\n'
text += '输入路径如下。四域保存输出及 REAL/reference 已由本次唯一 numeric_worker 读取；报告封存不再解码数组。\n\n'
for key, val in cfg['inputs'].items(): text += f'- {key}: `{val}`\n'
for key, val in cfg['generated'].items(): text += f'- {key}: `{val}`\n'
text += '\n配置、原列索引与矩参数引用在 config.json；未重新估计矩。固定600/250行、人工组、QC、原Qbin与N≥20矩支持均复用R30，条件描述N≥10。新私有预测/索引/解码误差在 results，原full480预测路径在配置。\n\n'
text += '运行顺序为 prepare.py → supervise.py numeric_worker.py → supervise.py render_figures.py → supervise.py render_layout_fix.py → assemble_delivery.py → seal.py；实际三次科学 native 链记录在资源回执。脚本的旧期限/单次标记禁止原目录再运行。未来复现须在获授权的新E/D目录复制源码与配置并重新登记期限，保留冻结数学/选序/参数，不覆盖本轮或旧阶段。\n\n'
text += '唯一掩码是0:100与150:480，先完整480变换再掩码。解码器只检验两套反变换均值后的max距1，1e−12差内弃权。12 HGB与500次固定分类器配对区组bootstrap仅执行一次。变换需要域身份，不能部署或推导新PSD。\n'
write('REPRODUCTION.md', text)
bindings = []
for p in sorted(ROOT.glob('*.py')):
    b = p.read_bytes(); ast.parse(b.decode('utf8'), filename=str(p))
    bindings.append({'path': p.name, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(), 'AST_parse': 'passed'})
for name in ['config.json', 'PROTOCOL.md']:
    b = (ROOT / name).read_bytes(); bindings.append({'path': name, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()})
write('SOURCE_BINDINGS.json', {'stage': cfg['stage'], 'UTC': now(), 'files': bindings,
    'executed_numeric_source_unchanged': True, 'delivery_scripts_added_after_science': ['assemble_delivery.py', 'seal.py'],
    'task_sha256': '096bd12182c728ee9929e5eae54f8f66d22a3e4c2a15ce7c6d616cddf8d45412'})
print(json.dumps({'PID': os.getpid(), 'UTC': now(), 'science_s': science, 'HGB': receipt['counters']['HGB_completed'],
    'reports': ['REPORT.md', 'REPRODUCTION.md', 'RESOURCE_RECEIPT.json', 'FIGURE_QA.json', 'SOURCE_BINDINGS.json'], 'science_rerun': False}), flush=True)
