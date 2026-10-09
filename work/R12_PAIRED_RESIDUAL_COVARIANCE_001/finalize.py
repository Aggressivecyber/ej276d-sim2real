"""Control-only R12 report and one final manifest; no scientific array decoding."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
END = dt.datetime.fromisoformat('2026-10-09T04:44:33.504970+00:00')
T0 = time.monotonic()

def utc():
    return dt.datetime.now(dt.timezone.utc)

def write(name, value):
    if utc() >= END:
        raise TimeoutError('executor seal deadline reached; no further writes')
    data = value.encode('utf-8') if isinstance(value, str) else (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if len(data) > 2 * 1024**2:
        raise ValueError('control output exceeds 2 MiB')
    p = ROOT / name
    if p.exists():
        raise FileExistsError(p)
    if sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file()) + len(data) > 512 * 1024**2:
        raise ValueError('stage output cap exceeded')
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def read(name):
    p = ROOT / name
    n = p.stat().st_size
    if n > 2 * 1024**2:
        raise ValueError('control JSON exceeds 2 MiB')
    with p.open('rb') as f:
        b = f.read(n + 1)
    if len(b) != n:
        raise ValueError('control file size changed')
    return json.loads(b)

def thin_width(table):
    result = {k: table.get(k) for k in ['N', 'below_logQ_range', 'above_logQ_range', 'PSD_below_minus0_1', 'PSD_above0_6']}
    result['slices'] = []
    for row in table['slices']:
        fields = ['bin', 'logQ_edges', 'charge_edges_mVns', 'N', 'valid', 'converged', 'fit_attempts', 'failure_reasons', 'updates', 'variance_floor_ever', 'mean_gap_over_pooled_sigma', 'histogram_PSD_outside_grid_count']
        r = {k: row.get(k) for k in fields}
        r['bin_id_1based'] = int(row['bin']) + 1
        r['components'] = [{k: c.get(k) for k in ['order', 'mean', 'weight', 'effective_N_times_weight', 'responsibility_Kish_N', 'width_defined', 'sigma_PSD', 'FWHM_PSD', 'diagnostic_unqualified_sigma', 'local_PSD_per_logQ_slope', 'detrended_sigma_PSD', 'detrended_FWHM_PSD', 'diagnostic_unqualified_detrended_sigma']} for c in row.get('components', [])]
        result['slices'].append(r)
    return result

def number(v, digits=6):
    return 'undefined' if v is None else f'{v:.{digits}f}'

def main():
    print(json.dumps({'PID': os.getpid(), 'start_UTC': utc().isoformat(), 'control_only': True}), flush=True)
    m = read('metrics.json')
    native = read('REMOTE_NATIVE_RECEIPT.json')
    replay = read('REPLAY_RECEIPT.json')
    config = read('config.json')
    start = read('START_RECEIPT.json')
    transfer = read('RETURN_TRANSFER_RECEIPT.json')
    compact = {k: m[k] for k in ['status', 'models', 'baseline_and_curvature_variance_decomposition', 'domain_vs_REAL', 'C2ST', 'counters', 'versions', 'EVAL_read', 'same_acquisition_development', 'target_density_not_G4_mapping']}
    compact['domains'] = {}
    for name, domain in m['domains'].items():
        compact['domains'][name] = {'QC_valid': domain['QC_valid'], 'invalid': domain['invalid'], 'regions': {region: {k: row.get(k) for k in ['source_n', 'QC_n', 'HC_gamma_like_n', 'HC_neutron_like_n', 'unknown_n', 'reference_valid', 'frozen_reference_means', 'FoM']} for region, row in domain['regions'].items()}}
    compact['local'] = {name: {'source_n': row['source_n'], 'QC_n': row['QC_n'], 'regions': {region: {'QC_n': values['QC_n'], 'features': values['features']} for region, values in row['regions'].items()}} for name, row in m['local'].items()}
    ew = m['energy_conditional_width']
    compact['energy_conditional_width'] = {k: v for k, v in ew.items() if k != 'tables'}
    compact['energy_conditional_width']['tables'] = {name: thin_width(table) for name, table in ew['tables'].items()}
    compact['nearest_neighbours'] = m['nearest_neighbours']
    compact['publication_scope'] = 'Aggregated statistics only; no event IDs, predictions, nearest-neighbour indices, per-event waveforms or validation scores.'
    compact_sha = write('METRICS_COMPACT.json', compact)

    figures = ['charge_psd_full.png', 'charge_psd_main_body.png', 'c2st_paired.png', 'baseline_variance_decomposition.png', 'noise_and_curvature.png', 'conditional_waveforms.png', 'energy_peak_width.png', 'nearest_neighbours.png']
    qa = {'UTC': utc().isoformat(), 'method': 'Eight saved PNGs opened through view_image and visually inspected after the single numeric run.', 'images': [{'path': 'results/figures/' + name, 'axes_legends_counts_readable': True, 'no_material_clipping_or_overlap': True} for name in figures], 'specific_checks': ['All charge panels share ranges; main-body PSD view declares out-of-range counts and retains full-range figure.', 'Paired C2ST CI, chance line and REAL-REAL reference clearly separated.', 'Baseline variance units mV^2, divisor99, old scaled empirical / shrinkage terms and actual pre-ADC markers visible.', 'Prompt100:150 includes peak and early fall; curve labels distinguish output charge bins.', 'Unsupported width bins shown as crosses rather than interpolated values.', 'Nearest-neighbour plot explicitly states no-exact-copy is not proof of no memory.'], 'redraws': 0, 'scientific_recomputation_for_QA': 0}
    write('FIGURE_QA.json', qa)

    model_lines = []
    variance_lines = []
    c2st_lines = []
    for name, row in m['models'].items():
        old = row['source']
        recovery = row['covariance_recovery']
        model_lines.append(f'| {old} → {name} | {row["selected_K_frozen"]} | {recovery["alpha"]:.8f} | {recovery["dimension"]} | {recovery["recovered_relative_Frobenius_error"]:.3g} | {row["validation_NLL_new_minus_old"]:+.6f} |')
        b = m['baseline_and_curvature_variance_decomposition'][name]['baseline']
        variance_lines.append(f'| {old} | {b["old_expected_population_mean_V2"]*1e6:.6f} → {b["new_expected_population_mean_V2"]*1e6:.6f} | {b["actual_old_pre_ADC_mean_square_V2"]["mean"]*1e6:.6f} → {b["actual_new_pre_ADC_mean_square_V2"]["mean"]*1e6:.6f} | {b["old_shrinkage_term_V2"]*1e6:.6f} |')
        results = m['C2ST']['results']
        pair = m['C2ST']['paired_new_minus_old'][name]
        for view in ['full_historical_shape', 'full_voltage', 'baseline_local']:
            d = pair[view]
            ci = d['BA_delta_95CI']
            c2st_lines.append(f'| {old} | {view} | {results[old][view]["balanced_accuracy"]:.6f} → {results[name][view]["balanced_accuracy"]:.6f} | {d["BA_delta_first_minus_second"]:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}] |')

    width_support = {name: [int(row['bin']) + 1 for row in table['slices'] if row['valid']] for name, table in ew['tables'].items()}
    raw_rms = m['local']['RAW_EMPIRICAL_COV']['regions']['overall']['features']['baseline_RMS_V']['mean'] * 1000
    factor_rms = m['local']['FACTOR_EMPIRICAL_COV']['regions']['overall']['features']['baseline_RMS_V']['mean'] * 1000
    report = f'''R12 完成了冻结随机输入下的协方差配对实验。移除 Ledoit–Wolf 向各向同性目标收缩的部分，显著减小了 R11 偏大的基线涨落，但没有解决总体真实响应分布。RAW 完整形状可区分性基本不变；FACTOR 小幅改善后仍明显可区分，且两臂完整验证似然都变差。科学负结果保留。

本轮仅替换剩余 PCA 坐标的共享协方差；这里的 tail 是第32维之后的坐标，不是物理时间尾窗。冻结 RAW K4、FACTOR K8 的完整表示、head、线性条件均值、尺度、epsilon=1e-12 V²以及每条实际 component/head_white/tail_white。用 C0=C_old−epsilon I、mu=trace(C0)/d、S=(C0−alpha mu I)/(1−alpha) 恢复 TRAIN 经验协方差，只对称化，再用 S+epsilon I。两臂正定且无需改底、重拟合或扫描。共4000条新输出，旧4000条只读复用。

| 冻结来源 → 新臂 | K | alpha | tail维数 | 恢复相对Frobenius误差 | 599条完整NLL新−旧/nat |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(model_lines)}

NLL包含连续全维密度和各自Jacobian；越低越好。599条验证均合法。头与固定条件均值逐位保持一致，改变仅为 tail covariance/Cholesky。高维密度的条件数从约88/99增至1360/950，虽然仍正定，但经验协方差没有带来更好的验证泛化。

基线前100点的中心化样本方差使用分母99。解析条件期望为 (mu_bᵀHmu_b+tr(HC_b))/99；既存2000条固定head的预测与实际预ADC均方接近。单位均为mV²，不能与平均RMS混为同一统计量。

| 来源 | 解析期望旧→新/mV² | 实际2000均方旧→新/mV² | 旧各向同性收缩贡献/mV² |
| --- | ---: | ---: | ---: |
{chr(10).join(variance_lines)}

实际QC总体基线平均RMS：REAL 0.646423 mV，RAW 0.940192→{raw_rms:.6f} mV，FACTOR 1.012893→{factor_rms:.6f} mV。RMS的W1分别由0.298005→0.034538 mV及0.366500→0.082505 mV；但ACF1 W1由0.045100→0.061624、0.059633→0.089827，仍存在联合噪声结构差异。因此单个噪声幅度接近不是完整噪声成功。

30个固定HGB均使用相同600条分类训练和249条留出/每域，来自冻结R7位置及四臂共同QC。500次配对人工组bootstrap固定分类器，不含分类训练不确定性或独立采集变化；没有多重检验校正。REAL–REAL同采集TRAIN–DEV参照的完整shape BA为0.512048、voltage为0.546185，是采样参照而非等效验收线。

| 来源 | 视图 | BA旧→新 | 配对新−旧，95%CI |
| --- | --- | ---: | ---: |
{chr(10).join(c2st_lines)}

RAW/FACTOR的PSD W1几乎保持0.0196/0.00926，电荷W1几乎保持195.4/113.5 mV ns。RAW两版均有6条QC无效（1994/2000），FACTOR两版均2000/2000；无拒绝重采样、ADC裁剪或非有限波形。完整主总体保留所有有效事件，C2ST只按预定共同QC比较。平均曲线相近仍不能代表全波形联合分布接近。

按输出电荷区间观察，峰后曲率仍从低区过大转为高区不足，换共享协方差没有恢复真实曲率随电荷明显增加的关系。prompt=100:150包含峰及早期下降，不能将其全部解释为纯上升沿。各域仅16个新描述槽位/16次EM；有效宽度bin为 {json.dumps(width_support, ensure_ascii=False)}。RAW旧/新只能在共同bins7、8比较；低PSD原宽度MAE 0.010216→0.010401，高PSD 0.009117→0.008630。FACTOR两版无合格bin，四臂共同支持为空，无法宣称条件双峰宽度成功。失败分离度/支持原因逐箱保留，未修改阈值或重拟合。

冻结CAL的q4参考不收敛，q4仍全部unknown；q1 HC neutron-like仍0。整体表中的global后验只是独立诊断；粒子类别无独立真值，不以FoM或两成分拟合替代。新臂全2000条最近邻相对2592条TRAIN参考无精确拷贝，V平均距离约4.394/4.388 mV；这不证明没有记忆。

主worker153463/监督153462 native0，17.628519s；独立重放153727 native0，0.801853s，累计18.430372s。重放两臂4000条、1920000点，预ADC与后ADC均逐位相等。峰RSS1,058,848,768 B，4CPU线程；PCA/GMM/ridge重拟合0、响应更新0、新随机抽样0、EVAL读取0、新G4为0。本地取回正常SCP native0；没有迁移逐文件哈希或对等字节测试。八幅PNG均实际打开检查，完整范围及主体放大图并存。Matplotlib的Axes3D可选导入警告不影响本轮2D图。

后续应把“共享残差协方差修正幅度”与“残差条件结构不足”分开：本轮支持前者解释基线均方偏大，但剩余电荷条件曲率/双峰/全波形差异仍需单独建模。当前只是TRAIN拟合、同采集DEV诊断的目标密度；没有建立G4父事件到真实波形的逐事件物理对应，不从目标密度失败反推G4错误。下一轮由ROOT登记，本轮不追加实验。

完整结果见 metrics.json，新模型及生成输入见两个 *_EMPIRICAL_COV 子目录，科学重放见 REPLAY_RECEIPT.json，聚合发布版见 METRICS_COMPACT.json。Ledoit–Wolf的收缩定义可见[官方1.8文档](https://scikit-learn.org/1.8/modules/generated/sklearn.covariance.LedoitWolf.html)，实验实际恢复公式与冻结配方见 PROTOCOL.md。工程记录保留两次报告JSON摘要读取的schema错误（native exit1，工具cc7059、367caa）；发生在科学成功后，仅控制汇总，未重算或改变科学数值。
'''
    write('REPORT.md', report)
    repro = '''本轮服务器目录：/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R12_REMOTE_20261009_034933，解释器 /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python；实际入口 numeric_worker.py，由 remote_supervise.py 启动并随后执行 replay.py。CPU4，现有环境，不安装或升级。

只读依赖为兄弟 R11_REMOTE_20261009_020915 下 RAW_PCA_GMM/FACTOR_PCA_GMM 的 representation、冻结K4/K8、GENERATION_INPUTS、generated、metrics/results，及 R10_REMOTE_20261009_012338/inputs。精确路径映射在 config.json。新stage保存了两套恢复的representation、GMM和实际创新输入；这些科学文件可在新目录用 replay.py 重放，无需新抽样/拟合或调用旧R11入口。正式再次执行需作为后续任务登记，避免覆盖本轮一次性token与旧结果。

已实际执行的重放PID153727/exit0：4000条/1920000点，pre-ADC与post-ADC均bitwise，通过回执 REPLAY_RECEIPT.json。对照头、条件均值、创新保持的配对检查在 metrics.json。这里的位级检查是模型科学重放，不是迁移完整性测试。正常文件迁移只保留SCP/SSH退出与实际加载；未声称远端/本地逐文件内容相等。

完整 metrics.json、NPZ、NN索引、分类预测及验证逐值保留E/D；METRICS_COMPACT.json仅聚合结果用于版本管理。所有模型均同一次采集TRAIN/DEV诊断，不称外部盲测或G4条件物理映射。
'''
    write('REPRODUCTION.md', repro)
    failures = {'science_failures': [], 'control_summary_failures': [{'tool': 'cc7059', 'native_exit_code': 1, 'wall_s': 1.1522104, 'reason': 'report-only JSON summary treated width-table object as list; AttributeError', 'numeric_retry': False}, {'tool': '367caa', 'native_exit_code': 1, 'wall_s': 0.6867751, 'reason': 'report-only JSON summary requested bins rather than actual slices key; KeyError', 'numeric_retry': False}], 'original_tool_receipts_preserved_in_conversation': True, 'old_stages_modified': False}
    write('FAILURE_RECEIPT.json', failures)
    resource = {'stage': 'R12_PAIRED_RESIDUAL_COVARIANCE_001', 'registration_UTC': '2026-10-09T03:49:33.504970+00:00', 'first_native': {'tool': 'bdba53', 'PID': 25704, 'exit_code': 0, 'UTC': '2026-10-09T03:50:43.894096+00:00'}, 'executor_seal_stop_UTC': END.isoformat(), 'absolute_deadline_UTC': '2026-10-09T04:49:33.504970+00:00', 'science_native_tools': ['91da6a', 'c7e4e0'], 'local_bridge_PID': 60704, 'SSH_PID': 59616, 'remote': native, 'counters': m['counters'], 'replay_rows': sum(r['rows'] for r in replay['models']), 'replay_values': sum(r['values'] for r in replay['models']), 'replay_pass': replay['status'] == 'pass', 'numeric_limit_s': 900, 'RSS_limit_bytes': 4 * 1024**3, 'output_limit_bytes': 512 * 1024**2, 'CPU_threads': 4, 'numeric_cumulative_s': native['numeric_cumulative_s'], 'numeric_limit_pass': native['numeric_cumulative_s'] <= 900, 'RSS_peak_bytes': max(r['RSS_peak_bytes'] for r in native['history']), 'RSS_probe_failures': sum(r['RSS_probe_failures'] for r in native['history']), 'only_one_science_process_at_a_time': True, 'main_and_replay_sequential': True, 'transfer_tool': '5a8e62', 'transfer': transfer, 'upload_bytes': (ROOT/'UPLOAD.tar').stat().st_size, 'download_bytes': (ROOT/'REMOTE_RETURN.tar').stat().st_size, 'migration_equality_or_hash_test': False, 'data_IO_byte_total': None, 'data_IO_total_reason': 'Simplified task has no per-read byte ledger; explicit source payloads/normal transfer sizes and fixed decoded roles retained.', 'system_dependency_load_IO_not_measured': True, 'control_summary_failures': failures['control_summary_failures'], 'control_only_failures_are_not_numeric_retries': True, 'extra_numeric_after_replay': 0, 'GPU_allocated_bytes': 0, 'EVAL_read': 0, 'new_G4': 0, 'new_training': 0, 'output_bytes_before_final_manifest': sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()), 'source_binding_receipt': 'READINESS.json', 'versions': m['versions'], 'METRICS_COMPACT_sha256': compact_sha, 'seal_process_PID': os.getpid(), 'seal_native_exit_proof': 'Use actual tools.exec_command return; this in-process receipt does not certify its own native exit.'}
    write('RESOURCE_RECEIPT.json', resource)
    entries = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.name in {'FINAL_MANIFEST.json', 'FINAL_RECEIPT.json'}:
            continue
        if utc() >= END:
            raise TimeoutError('seal deadline during own-output byte hashing')
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024**2), b''):
                h.update(block)
        entries.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()})
    manifest = {'stage': 'R12_PAIRED_RESIDUAL_COVARIANCE_001', 'UTC': utc().isoformat(), 'members': entries, 'member_count': len(entries), 'member_bytes': sum(e['bytes'] for e in entries), 'scope': 'Single local new-stage artifact manifest; no old source rehash and no source/destination migration comparison.', 'self_reference_exclusions': ['FINAL_MANIFEST.json', 'FINAL_RECEIPT.json'], 'frozen_old_stages_modified': False}
    manifest_sha = write('FINAL_MANIFEST.json', manifest)
    before = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    receipt = {'stage': 'R12_PAIRED_RESIDUAL_COVARIANCE_001', 'PID': os.getpid(), 'UTC': utc().isoformat(), 'status': 'sealed_complete_scientific_negative_result', 'scientific_success_is_not_response_similarity_pass': True, 'science_native_exit_codes': [r['exit_code'] for r in native['history']], 'local_bridge_native_exit_code': 0, 'transfer_native_exit_code': transfer['exit_code'], 'numeric_cumulative_s': native['numeric_cumulative_s'], 'manifest_sha256': manifest_sha, 'manifest_member_count': len(entries), 'manifest_member_bytes': manifest['member_bytes'], 'METRICS_COMPACT_sha256': compact_sha, 'output_bytes_before_this_receipt': before, 'seal_within_executor_deadline': utc() < END, 'control_assembly_elapsed_s_before_receipt': time.monotonic()-T0, 'native_seal_exit_code': None, 'native_seal_exit_evidence': 'Actual tool result, supplied to ROOT after this command returns.', 'writes_stop_after_this_receipt': True, 'further_science_or_next_stage_started': False}
    receipt_sha = write('FINAL_RECEIPT.json', receipt)
    print(json.dumps({'PID': os.getpid(), 'final_receipt_UTC': receipt['UTC'], 'last_write_UTC_observed': utc().isoformat(), 'FINAL_MANIFEST_sha256': manifest_sha, 'FINAL_RECEIPT_sha256': receipt_sha, 'members': len(entries), 'member_bytes': manifest['member_bytes'], 'stage_file_count': len([p for p in ROOT.rglob('*') if p.is_file()]), 'stage_bytes': sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()), 'elapsed_s': time.monotonic()-T0, 'scientific_array_decodes': 0, 'fits': 0}), flush=True)

if __name__ == '__main__':
    main()
