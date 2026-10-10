"""Control-only report, resource summary and one final seal. No scientific arrays."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, time

ROOT = Path(__file__).resolve().parent
START = time.monotonic()
def utc():
    return datetime.now(timezone.utc).isoformat()
def read(name):
    return json.loads((ROOT/name).read_text('utf-8-sig'))
def check_time():
    if datetime.now(timezone.utc) >= datetime.fromisoformat(cfg['deadline_UTC']):
        raise TimeoutError('executor deadline reached; preserve partial files')
def put(name, data):
    check_time()
    (ROOT/name).write_bytes((json.dumps(data, indent=2, ensure_ascii=False,
                                      allow_nan=False)+'\n').encode('utf-8'))
def text(name, value):
    check_time()
    (ROOT/name).write_bytes(value.encode('utf-8'))
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

if (ROOT/'FINAL_RECEIPT.json').exists():
    raise RuntimeError('already sealed; no repeated finalization')
cfg = read('config.json')
check_time()
m = read('METRICS_COMPACT.json')
remote = read('REMOTE_RUN_HISTORY.json')
local = read('RUN_HISTORY.json')
jobs = remote + local
counts = read('NUMERIC_RECEIPT.json')['counts']
assert counts['HGB_attempts'] == counts['HGB_completed'] == 5
assert all(j['actual_worker_native_exit'] == 0 and j['termination_reason'] is None
           for j in jobs)
science_s = sum(j['elapsed_s'] for j in jobs)
assert science_s <= cfg['science_budget']
failures = read('ENGINEERING_FAILURES.json')
assert not failures['scientific_failures'] and not failures['control_failures']

lines = [
    '# R47 人口与初始噪声诊断', '',
    '本轮完成了固定的 5 次诊断 HGB，没有创建新响应候选。电荷分类不能把人口差异确立为硬阻断；实测输入经加噪初态和学习流增量后，基线 RMS 与自相关失配分别扩大。R46 与旧家族的关闭结论、项目原 A–F 验收标准保持。', '',
    '比较沿用原 2000 条 SOURCE、2000 条 REAL TRAIN-reference 和 2000 条 REAL DEV，不删支持外或 unknown。三个池均 2000 条 QC 有效。SOURCE 的 pure-EE / pure-had / mixed 分别为 417 / 1518 / 65，是沉积组成标签；谷区或组成不能改称真实粒子误分类。', '',
    '## 电荷人口', '',
    '| 固定一列 charge-only 检验 | TRAIN BA | 留出 BA | 留出 BA 95% 区组 CI | 留出 AUC | 预定结论 |',
    '|---|---:|---:|---|---:|---|']
for key in ['CHARGE_SOURCE', 'CHARGE_LONG', 'CHARGE_RAW_REFERENCE']:
    r = m['classifiers'][key]
    ci = r['heldout_CI']['balanced_accuracy_95CI']
    conclusion = ('uncertain' if r['population_diagnostic'] == 'uncertain'
                  else '本诊断未检测到额外差异')
    lines.append(f"| {r['candidate']} | {r['training_balanced_accuracy']:.4f} | "
                 f"{r['balanced_accuracy']:.3f} | [{ci[0]:.6f}, {ci[1]:.6f}] | "
                 f"{r['AUC']:.6f} | {conclusion} |")
lines += ['',
    'SOURCE 与 LONG 的区间都跨 0.60，不能称强人口差异、整体验收不可达或固定 Q–PSD HGB 的理论下界。全人口 CDF 同时保留：这与固定 600/250 子样本、有限分类器及区组区间的判断尺度不同。', '',
    '| 全部 2000 条，实际观测 Q | 平均 Q [mV ns] | 对 REAL DEV 的 W1 [mV ns] | KS | 原支持下方 / 上方 |',
    '|---|---:|---:|---:|---|']
for name, r in m['population'].items():
    d = r['W1_KS_vs_REAL_DEV_all_finite']
    lines.append(f"| {name} | {r['Qtotal_mVns_all_rows']['mean']:.6f} | "
                 f"{d['W1']:.6f} | {d['KS']:.4f} | "
                 f"{r['counts']['below_lower_support']} / {r['counts']['above_upper_support']} |")
lines += ['',
    'SOURCE 原始输入 Q-bin、各端点自己的观测 Q-bin及支持外计数分别保存；按各自 Q-bin 的曲线不等于相同事件人口或因果干预。所有非有限和非正电荷计数保留，本次为零。', '',
    '## 已有噪声与学习增量', '',
    '| 输入路径 / 保存位置 | 基线 RMS 均值 [mV] | RMS W1 对 REAL DEV [mV] | ACF1 均值 | ACF1 W1 |',
    '|---|---:|---:|---:|---:|']
for domain in ['SOURCE', 'REAL_REFERENCE']:
    for name in ['RAW', 'INITIAL_POSTADC', 'SHORT', 'LONG', 'BASE']:
        r = m['baseline_paths'][domain][name]['fields']
        rms = r['baseline_RMS_V']
        acf = r['ACF1']
        am = acf['all_rows']['mean']
        aw = acf['W1_KS_vs_REAL_DEV_all_finite_rows']['W1']
        af = 'undefined' if am is None else f'{am:.6f}'
        wf = 'undefined' if aw is None else f'{aw:.6f}'
        lines.append(f"| {domain} / {name} | {1000*rms['all_rows']['mean']:.6f} | "
            f"{1000*rms['W1_KS_vs_REAL_DEV_all_finite_rows']['W1']:.6f} | {af} | {wf} |")
lines += ['',
    'REAL RAW → 初态 → LONG 的 RMS 失配为 0.003115 → 0.546959 → 0.890583 mV；LONG 的 ACF1 W1 为 0.407618。因此增量也参与失配，单纯减小初始 Gaussian 并不足以得到已验证的新机制。SOURCE RAW 的 baseline 方差为零，2000 条 ACF1 均 undefined，未填零或删行。', '',
    '| 两个新 baseline-local HGB | TRAIN BA | 留出 BA | 留出 BA 95% CI | 留出 AUC |',
    '|---|---:|---:|---|---:|']
for key in ['BASELINE_SOURCE_INITIAL', 'BASELINE_REAL_INITIAL']:
    r = m['classifiers'][key]
    ci = r['heldout_CI']['balanced_accuracy_95CI']
    lines.append(f"| {r['candidate']} | {r['training_balanced_accuracy']:.6f} | "
        f"{r['balanced_accuracy']:.3f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {r['AUC']:.6f} |")
p = m['REAL_initial_minus_RAW_RR_paired']
lines += ['',
    'baseline-local 是本区去均值、幅度归一化的时间点、ACF 和归一化谱视图，不含绝对 RMS；近 0.5 不能否定另列的 RMS 失配。训练与留出分数差明显，固定检验的功效和泛化限制保留。', '',
    f"同一留出行/组上，REAL 初态减 RAW REAL–REAL 的配对 ΔBA={p['BA_delta_first_minus_second']:+.6f}，95% CI [{p['BA_delta_95CI'][0]:+.6f}, {p['BA_delta_95CI'][1]:+.6f}]；"
    f"ΔAUC={p['AUC_delta_first_minus_second']:+.6f}，95% CI [{p['AUC_delta_95CI'][0]:+.6f}, {p['AUC_delta_95CI'][1]:+.6f}]。BA 区间跨零，不能将其写成已证实 BA 增大。旧 RAW RR 概率/分区直接复用，缺失的旧二维预测仍 unknown。", '',
    '四个功能分量 S/G/F/R 在每条 baseline100 内分别中心化，样本方差分母 99；4 个 Var 与所有 6 个有符号 2Cov 合成实际 preADC 方差。', '',
    '| 保存端点 | Var G | Var F | 2Cov(S,F) | 2Cov(G,F) | 实际总 Var [全部 mV²] |',
    '|---|---:|---:|---:|---:|---:|']
for arm, r in m['variance_decomposition'].items():
    t = r['term_summary']
    keys = ['Var_G', 'Var_F', '2Cov_S_F', '2Cov_G_F', 'actual_preADC_baseline_variance_V2']
    lines.append('| '+arm+' | '+' | '.join(f"{1e6*t[k]['mean']:.6f}" for k in keys)+' |')
lines += ['',
    'SOURCE 的 G/F 交叉项为负；REAL LONG 的 S/F 与 G/F 交叉项为正。四端点代数闭合的最大绝对误差不超过 1.02e-20 V²。它们是共享同一生成过程的功能分量，不是独立物理噪声；不能据交叉项直接删分支或声称因果。', '',
    '## 复现与边界', '',
    'phase0=训练 600 行，phase1=留出 250 行；原 fixed_R26_phase 字段实际为域列（0=SOURCE，2=TRAIN_REFERENCE），已保留原件并用 candidate_domain_column / training_phase / holdout_phase 显式澄清。修正无新计算。', '',
    '500 次区组 bootstrap 条件于这些固定分类器，不包括重新训练响应或分类器的不确定性。REAL 来自同一次采集，人工记录组不是独立采集批次。没有新增 REAL/EVAL/CAL 波形或 G4；原验收 A–F、未知来源语义及所有历史停止结论不变。', '',
    f"实际服务器统计 {remote[0]['elapsed_s']:.9f}s；原绘图 {local[0]['elapsed_s']:.9f}s；显示修复 {local[1]['elapsed_s']:.9f}s；累计科学 {science_s:.9f}s / 900s。实际 worker 峰 RSS {max(j['actual_worker_RSS_peak_bytes'] for j in jobs)} B，RSS 查询失败 0、GPU 0，全部三个作业 native0。5/5 HGB 完成，响应更新/ODE/EM/新生成均 0；6000 行 ADC 是既存输入的确定性特征处理。", '',
    '6 张最终图均已实际打开；charge-only 图例和 noise-by-charge 图外共享图例已修复，原两图及原脚本保留 display_initial。资源与原生链见 RESOURCE_RECEIPT.json / NATIVE_CHAIN.json；身份/预测/逐事件分量保持本地，发布使用 METRICS_COMPACT.json。', '',
    '## 下一步判断（只读推理，未运行）', '',
    '目前不足以把最后一个方法名额直接登记成已识别的新响应机制。最有根据的一步是先明确有噪/无噪输入的同处理契约和不变 A–F 下的总体人口要求，再据此固定一个同时接受 SOURCE 与已含电子噪声 REAL 的算子及对照。噪声状态依赖值得针对，但本轮不能识别如何分离观测信号与噪声；降低 Gaussian、换积分步数或重返已关闭的 R36 不构成新方法。charge-only uncertain 不是阻断证据，也不授权修改验收标准。', '']
text('REPORT.md', '\n'.join(lines))

repro = f'''# R47 复现入口

实际科学入口为 `numeric_worker.py`，由 `launch_server.py` 通过已保存的 `linux_supervise.py` / `linux_owned_worker.py` 导入前 PID/nonce/RSS 握手启动。实际 worker159116、supervisor159111；原生终态见 `numeric_worker.py.NATIVE_RECEIPT.json` 和 `numeric_worker.py.LOCAL_NATIVE_RECEIPT.json`。

服务器目录：`{cfg['remote_root']}`；解释器：`{cfg['server_python']}`。本轮实际环境 Python3.12.3 / NumPy1.26.4 / sklearn1.8.0 / SciPy1.11.4；CPU12、GPU0。

输入均只读引用 `config.json.inputs` 指定的 R46 服务器原件：实际 SOURCE/IDs、DEV_SOURCE_METADATA、REAL_DEV_AND_REFERENCE、原 C2ST phase0/phase1 选择、冻结 Q-bin 常数、SHORT/LONG 的保存输出及 S/G/F/R 分量、原 RAW RR 预测与分组。`SOURCE_REUSE_BINDINGS.json` 记录历史版本；正常上传/下载 native0，不宣称迁移位级验证。

在另一个新工作目录复制本轮必要脚本和 config，修改该新目录 remote_root，保留原 R46 输入路径，只在新目录部署并运行 `launch_server.py numeric_worker.py`。不能在本封存目录重跑或覆盖结果。该入口仅五个预定 HGB 与保存数组描述，不运行任何响应网络/ODE。

完整 `metrics.json` 和 `results/*PRIVATE.npz` 包含身份、索引与固定预测；发布汇总为 `METRICS_COMPACT.json`。`METRICS_COMPACT_EXECUTED.json` 保留原字段，元数据差异见 `METADATA_CORRECTION_RECEIPT.json`。

原图 `render_figures.py` 已执行；其一次性元数据澄清不可重复。`repair_display.py` 只从保存汇总修两幅显示，原图在 `results/figures/display_initial`，原脚本为 `render_figures_display_initial.py`。本机绘图解释器 `{cfg['CPU_python']}`，CPU4，缓存在本阶段 E 盘 cache。

原 RAW RR 分类器和 R46 全部旧结果直接复用。SOURCE 与 REAL 本来不是事件物理配对，人工组/区组区间也不能换成独立采集不确定性。本轮没有可保存/重放的新响应权重或候选。
'''
text('REPRODUCTION.md', repro)

tool_by_entry = {'numeric_worker.py': '67abfe', 'render_figures.py': '8819f9',
                 'repair_display.py': '0a38f7'}
chain = [{'tool_result_ID': tool_by_entry[j['entry']], 'entry': j['entry'],
          'actual_worker_PID': j['actual_worker_PID'],
          'supervisor_PID': j['supervisor_PID'], 'start_UTC': j['start_UTC'],
          'stop_UTC': j['stop_UTC'], 'native_exit': j['actual_worker_native_exit'],
          'elapsed_s': j['elapsed_s'], 'receipt': j['entry']+'.NATIVE_RECEIPT.json'}
         for j in jobs]
put('NATIVE_CHAIN.json', {'scientific_jobs': chain,
    'control_tools': {'first_control': 'a8dbb4', 'preparation_preflight': 'ff5053',
                      'deployment': '5aa3bc', 'normal_return': '7b1306'},
    'original_receipts_logs_retained': True,
    'current_finalization_PID': os.getpid(),
    'finalization_native_exit_is_authoritative_in_returned_tool_result': True})

figures = ['charge_population.png', 'charge_only_C2ST.png', 'baseline_paths.png',
           'baseline_variance_components.png', 'baseline_diagnostic_C2ST.png',
           'noise_by_charge.png']
put('FIGURE_QA.json', {'UTC': utc(), 'final_figures_count': len(figures),
    'executor_actually_opened_all_final_figures': True,
    'ROOT_actually_opened_all_six_and_reopened_two_repaired_figures': True,
    'ROOT_confirmation': 'collaboration message received before this seal',
    'figures': [{'path': 'results/figures/'+n,
                 'sha256': digest(ROOT/'results'/'figures'/n),
                 'actual_visual_QA': 'axes, units, support/undefined labels, data and legends legible'}
                for n in figures],
    'display_only_repairs': 'DISPLAY_REPAIR_RECEIPT.json',
    'original_two_PNG_and_script_preserved': True,
    'new_statistics_for_QA': False})

put('RESOURCE_RECEIPT.json', {'stage': m['stage'], 'UTC': utc(),
    'scientific_elapsed_s': science_s, 'science_limit_s': cfg['science_budget'],
    'all_attempts_included': True, 'scientific_jobs': chain,
    'actual_worker_RSS_peak_bytes': max(j['actual_worker_RSS_peak_bytes'] for j in jobs),
    'server_RSS_limit_bytes': cfg['RSS_limit_bytes'],
    'local_plot_RSS_limit_bytes': 4 << 30,
    'server_CPU_threads': 12, 'local_plot_CPU_threads': 4,
    'GPU_used': False, 'GPU_memory_bytes': 0,
    'memory_probe_failures': sum(j['memory_probe_failures'] for j in jobs),
    'actual_PID_handshake_before_imports_all_jobs': all(j['handshake_before_imports'] for j in jobs),
    'scientific_counts': counts, 'failures': failures,
    'metadata_repair_no_statistics': 'METADATA_CORRECTION_RECEIPT.json',
    'display_repair_elapsed_in_science_budget': True,
    'original_historical_outputs_referenced_not_copied': True,
    'archive_return_bytes': (ROOT/'RETURN_DIAGNOSTIC.tar').stat().st_size,
    'normal_transfer_native_exit': 0, 'migration_consistency_test_performed': False,
    'output_limit_bytes': cfg['output_limit_bytes'],
    'actual_stage_file_bytes_and_file_count_in_FINAL_RECEIPT': True,
    'control_finalization_is_not_a_new_scientific_job': True,
    'original_acceptance_A_F_unchanged': True,
    'new_response_candidate': False})

excluded = {'FINAL_MANIFEST.json', 'FINAL_RECEIPT.json'}
files = []
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and p.name not in excluded:
        check_time()
        if p.is_symlink():
            raise RuntimeError('unexpected stage symlink')
        files.append({'path': p.relative_to(ROOT).as_posix(),
                      'bytes': p.stat().st_size, 'sha256': digest(p)})
put('FINAL_MANIFEST.json', {'stage': m['stage'], 'UTC': utc(),
    'files': files, 'file_count': len(files),
    'manifested_bytes': sum(f['bytes'] for f in files),
    'self_reference_exclusions': sorted(excluded),
    'scientific_inputs_referenced_by_config_and_reuse_bindings': True,
    'sealed_historical_stages_not_modified': True})
pre_receipt_files = [p for p in ROOT.rglob('*') if p.is_file()]
base_total = sum(p.stat().st_size for p in pre_receipt_files)
receipt = {'stage': m['stage'], 'status': 'sealed', 'seal_PID': os.getpid(),
    'seal_UTC_before_final_receipt_write': utc(),
    'executor_deadline_UTC': cfg['deadline_UTC'],
    'scientific_jobs_all_actual_native_exit_0': True,
    'finalization_actual_native_exit_from_parent_tool_required': True,
    'scientific_elapsed_s': science_s, 'HGB_attempts': 5, 'HGB_completed': 5,
    'response_training_ODE_EM_generation': 0,
    'new_response_candidate': False, 'original_acceptance_unchanged': True,
    'stage_files_total': len(pre_receipt_files)+1,
    'stage_bytes_total': base_total,
    'output_limit_bytes': cfg['output_limit_bytes'],
    'FINAL_MANIFEST_sha256': digest(ROOT/'FINAL_MANIFEST.json'),
    'final_report': 'REPORT.md', 'resource_receipt': 'RESOURCE_RECEIPT.json',
    'final_figures': 6, 'failures_retained': failures,
    'self_reference_exclusions': sorted(excluded),
    'no_further_stage_writes': True}
# Account for this receipt's own byte length without a self-hash loop.
for _ in range(6):
    size = len((json.dumps(receipt, indent=2, ensure_ascii=False,
                           allow_nan=False)+'\n').encode('utf-8'))
    new_total = base_total + size
    if receipt['stage_bytes_total'] == new_total:
        break
    receipt['stage_bytes_total'] = new_total
assert receipt['stage_bytes_total'] <= cfg['output_limit_bytes']
put('FINAL_RECEIPT.json', receipt)
# The final receipt is the last write. Actual filesystem mtime and actual
# parent-command native exit are returned without touching the sealed stage.
last = ROOT/'FINAL_RECEIPT.json'
final_files = [p for p in ROOT.rglob('*') if p.is_file()]
assert sum(p.stat().st_size for p in final_files) == receipt['stage_bytes_total']
print(json.dumps({'seal_PID': os.getpid(), 'last_write_UTC':
    datetime.fromtimestamp(last.stat().st_mtime, timezone.utc).isoformat(),
    'control_elapsed_s': time.monotonic()-START,
    'files': len(final_files), 'bytes': receipt['stage_bytes_total'],
    'manifest_sha256': digest(ROOT/'FINAL_MANIFEST.json'),
    'receipt_sha256': digest(last), 'no_further_stage_writes': True}), flush=True)
