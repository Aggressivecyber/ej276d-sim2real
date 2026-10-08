from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UTC = dt.timezone.utc


def now():
    return dt.datetime.now(UTC).isoformat()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    if dt.datetime.now(UTC) >= deadline:
        raise RuntimeError('R9 stage deadline')
    payload = value.encode('utf-8') if isinstance(value, str) else (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) + len(payload) > config['output_limit_bytes']:
        raise RuntimeError('R9 cumulative output cap')
    with (ROOT / name).open('xb') as f:
        f.write(payload)


config = read('config.json')
deadline = dt.datetime.fromisoformat(config['deadline_UTC'])
metrics = read('metrics.json')
native = read('numeric_1/NATIVE_RECEIPT.json')
worker = read('NUMERIC_RECEIPT.json')
inputs = read('INPUTS.json')
failures = read('CONTROL_FAILURES.json')
assert native['worker_native_exit'] == 0 and native['reason'] is None
assert metrics['counters'] == worker['counters']
assert metrics['counters']['HGB_attempts'] == 42
assert metrics['EVAL_original'] == 7600 and metrics['EVAL_QC'] == 7583 and metrics['REAL_main'] == 2000
assert metrics['response_updates'] == metrics['EVAL_feedback'] == metrics['SSH'] == metrics['new_G4'] == 0

clarification = {
    'kind': 'preserved_native_receipt_descriptive_field_correction',
    'UTC': now(), 'PID': os.getpid(),
    'originals_preserved': ['launch.py', 'numeric_1/NATIVE_RECEIPT.json', 'RUN_HISTORY.json'],
    'original_native_receipt_sha256': digest(ROOT / 'numeric_1/NATIVE_RECEIPT.json'),
    'original_launcher_sha256': digest(ROOT / 'launch.py'),
    'affected_fields': {
        'EVAL_read': {'original_literal': 0, 'actual_original_records_decoded': 7600, 'actual_QC_valid': 7583, 'actual_fixed_main_REAL': 2000},
        'response_updates': {'original_literal': native['response_updates'], 'actual_R9_updates': 0},
    },
    'cause': 'Reused launcher retained two stale descriptive literals; worker counters and data-role assertions record actual R9 use.',
    'authoritative_counter_files': ['NUMERIC_RECEIPT.json', 'metrics.json'],
    'authoritative_worker_receipt_sha256': digest(ROOT / 'NUMERIC_RECEIPT.json'),
    'metrics_sha256': digest(ROOT / 'metrics.json'),
    'not_a_scientific_rerun': True,
    'native_PID_exit_wall_RSS_unchanged': True,
}
write('RECEIPT_FIELD_CLARIFICATION.json', clarification)

qa_notes = {
    'c2st.png': 'Six views, all five candidates and both controls; intervals and axes readable.',
    'energy_peak_width.png': 'Ordered density-component curves and undefined gaps retained; all six panels readable.',
    'charge_psd.png': 'Complete eight-domain charge-PSD populations visible; units and QC counts readable.',
    'conditional_waveforms.png': 'All four output-charge regions, raw V and own-peak shape, medians and 10-90% bands readable.',
    'energy_psd_slices.png': 'All eight frozen native slices and full PSD density shown; empty OFF lowest slice retained.',
    'response_features.png': 'Complete QC-valid CDFs, including rise and baseline tails; labels readable.',
    'fluctuations.png': 'Baseline RMS, full-denominator ACF and normalized non-DC spectrum readable; physical-calibration limitation labeled.',
    'conditional_curvature.png': 'Prompt and postpeak absolute and local-relative curvature visible; prompt peak/early-fall caveat present.',
    'parent_migration.png': 'All 2000 parents retained; all five matrices and invalid columns visible; separate colorbar obscures no count.',
    'pure_front_examples.png': 'Fixed-hash pre-peak examples in V and own-peak shape readable; examples are illustrative, complete rise CDF separate.',
    'all_real_supplement_joint.png': 'Fixed main2000 and all7583 supplement distinguishable; R-R range explicitly descriptive rather than tolerance.',
}
assert set(qa_notes) == {p.name for p in (ROOT / 'results/figures').glob('*.png')}
qa = {'UTC': now(), 'method': 'All 11 existing PNGs actually viewed via view_image; no replot or scientific rerun needed.', 'passed': True,
      'figures': [{'path': 'results/figures/' + name, 'bytes': (ROOT / 'results/figures' / name).stat().st_size,
                   'sha256': digest(ROOT / 'results/figures' / name), 'visual_pass': True, 'observation': observation}
                  for name, observation in qa_notes.items()]}
write('FIGURE_QA.json', qa)

names = config['candidates']
domain_keys = ['PSD', 'Qtotal_mVns', 'peak_V', 'rise_10_90_ns', 'tail_charge_fraction', 'baseline_RMS_V', 'ACF1']


def distances(key):
    return {name: {region: {feature: record[feature] for feature in domain_keys} for region, record in domains.items()}
            for name, domains in metrics[key].items()}


def reference_summary(record):
    return {region: {key: values[key] for key in ['source_n', 'QC_n', 'HC_gamma_like_n', 'HC_neutron_like_n', 'unknown_n', 'reference_valid', 'FoM']}
            for region, values in record['regions'].items()}


widths = metrics['energy_conditional_width']
width_key_allow = ['N', 'distinct_records_with_positive_weight', 'valid', 'converged', 'fit_attempts', 'restarts', 'failure_reasons', 'components', 'updates', 'weights', 'means', 'variances', 'effective_counts', 'parameter_delta', 'variance_floor_ever', 'mean_gap_over_pooled_sigma', 'bin', 'logQ_edges', 'charge_edges_mVns', 'actual_charge_range_mVns']
width_summaries = {name: {'N': table['N'], 'slices': [{k: row[k] for k in width_key_allow if k in row} for row in table['slices']]}
                   for name, table in widths['tables'].items()}
compact = {
    'status': 'completed_fixed_retrospective_validation_all_five_candidates_fail_complete_response_similarity',
    'frozen_SHA256': metrics['frozen_SHA256'], 'frozen_UTC': metrics['frozen_UTC'],
    'candidates': names, 'candidate_selection_from_R9': False,
    'usage': {'EVAL_original': 7600, 'EVAL_QC': 7583, 'fixed_main_REAL': 2000, 'source_parents_each_candidate': 2000,
              'previously_viewed_same_acquisition_not_blind_or_external': True, 'EVAL_response_feedback': 0},
    'domain_counts': {n: {k: d[k] for k in ['QC_valid', 'invalid']} for n, d in metrics['domains'].items()},
    'frozen_reference_coverage_and_FoM': {n: reference_summary(d) for n, d in metrics['domains'].items()},
    'all_REAL_reference_coverage_and_FoM': reference_summary(metrics['REAL_all_supplement']),
    'CAL_reference_valid': metrics['CAL_reference_valid'], 'q4_no_fallback': True, 'unknown_all_preserved': True,
    'domain_vs_REAL': distances('domain_vs_REAL'), 'domain_vs_REAL_all': distances('domain_vs_REAL_all'),
    'joint_W1': {n: {k: row[k] for k in ['W1', 'counts', 'angles', 'scale']} for n, row in metrics['joint_sliced_W1'].items()},
    'joint_W1_REAL_all': {n: {k: row[k] for k in ['W1', 'counts', 'angles', 'scale']} for n, row in metrics['joint_sliced_W1_REAL_all'].items()},
    'REAL_REAL_summary': metrics['REAL_REAL_summary'], 'REAL_REAL_joint': metrics['REAL_REAL_joint'],
    'REAL_REAL_interpretation': '64 saved same-N artificial-group splits; 5-95% ranges are sampling references, not physical-equivalence thresholds.',
    'C2ST': metrics['C2ST'], 'paired_C2ST_vs_R3B': metrics['paired_C2ST_vs_R3B'],
    'C2ST_common_selection': {k: metrics['C2ST_common_selection'][k] for k in ['common_source_QC', 'source_records', 'train_each_domain', 'holdout_each_domain', 'all_candidates_and_views_same_source_real_indices', 'RR_definition', 'bootstrap_repeats']},
    'private_selection_summary': {'rule': 'Frozen R1 hashes with parent/artificial-group isolation, shared source and REAL rows across all candidates and views.',
                                  'split_file': 'results/C2ST_SPLITS.json', 'split_sha256': digest(ROOT / 'results/C2ST_SPLITS.json'),
                                  'data_roles_sha256': digest(ROOT / 'results/DATA_ROLES.json')},
    'energy_width': {'edges_logQ': widths['edges_logQ'], 'tables': width_summaries,
                     'common_support_paired_errors': widths['common_support_paired_errors'],
                     'all_coverage_and_failure_reasons': widths['all_coverage_and_failure_reasons'],
                     'REAL_MAIN_group_bootstrap_trends': widths['REAL_MAIN_group_bootstrap']['trend_slopes'],
                     'new_density_fits': 0, 'same_saved_data_and_density_definition': True, 'not_particle_truth': True},
    'source_nominal_condition': {n: {'definition': d['definition'],
                                   'migration_rows_nominal_cols_output_q1_q2_q3_q4_invalid': d['migration_rows_nominal_cols_output_q1_q2_q3_q4_invalid'],
                                   'regions': {r: {k: v[k] for k in ['entire_source_parent_count', 'QC_n', 'invalid_n', 'REAL_own_q_QC_n', 'vs_REAL_output_own_q']} for r, v in d['regions'].items()}}
                                 for n, d in metrics['source_nominal_condition'].items()},
    'source_association': metrics['source_association'],
    'three_view_within_parent_variation': metrics['three_view_within_parent_variation'],
    'controlled_pair_gap': 'R3B saved fixed-latent energy/composition pairs reused; newer arms have no separately stored such controlled pairs. Bundle replacement does not isolate these conditions.',
    'replay_reference': {'R8': 'Previously sealed JOINT and JOINT_SIGNAL both bitwise pass, 4000 rows total.',
                         'R8C': 'Previously sealed FULL_WAVE_STOCHASTIC and PULSE_STOCHASTIC_OFF both bitwise pass, 4000 rows total.',
                         'new_R9_replay_or_generation': 0},
    'branch_ablation_boundary': 'OFF removes an entire nonlinear stochastic pulse branch, including its possible conditional mean capacity; not a pure-variance ablation.',
    'counters': metrics['counters'], 'versions': metrics['versions'],
    'private_payloads_not_embedded': ['source/REAL event IDs and indices', 'individual C2ST predictions', '64 individual R-R event selections', 'individual waveform features'],
}
write('METRICS_COMPACT.json', compact)

def line(name):
    d = metrics['domain_vs_REAL'][name]['overall']
    return f"| {name} | {d['PSD']['W1']:.6f} | {d['Qtotal_mVns']['W1']:.1f} | {d['rise_10_90_ns']['W1']:.4f} | {metrics['joint_sliced_W1'][name]['W1']:.6f} |"

distance_rows = '\n'.join(line(name) for name in metrics['domain_vs_REAL'])
views = config['C2ST_views']
c2st_rows = '\n'.join('| ' + name + ' | ' + ' | '.join(f"{metrics['C2ST'][name+'|'+view]['balanced_accuracy']:.3f}" for view in views) + ' |' for name in names + ['REAL_vs_REAL', 'PERTURBED_REAL'])
width_rows = '\n'.join('| ' + ('低PSD成分' if item['component'] == 0 else '高PSD成分') + ' ' + item['field'] + ' | ' + ' | '.join(f"{item['mean_abs_error'][name]:.6f}" for name in names) + ' |' for item in widths['common_support_paired_errors']['primary_five_candidates'])
support_rows = '\n'.join('| ' + name + ' | ' + ', '.join(str(bin_id) for bin_id, d in record.items() if d['valid']) + ' | ' + '; '.join(f"bin{bin_id}: {','.join(d['failure_reasons'])}" for bin_id, d in record.items() if not d['valid']) + ' |' for name, record in widths['all_coverage_and_failure_reasons'].items())
q3_rows = '\n'.join(f"| {name} | {metrics['source_nominal_condition'][name]['regions']['q3']['vs_REAL_output_own_q']['PSD']['W1']:.6f} | {metrics['domain_vs_REAL'][name]['q3']['PSD']['W1']:.6f} |" for name in names)
rr = metrics['REAL_REAL_joint']
paired = metrics['paired_C2ST_vs_R3B']['JOINT_SIGNAL|postpeak_local_minus_R3B']
report = f'''# R9：五候选冻结后回顾性响应验证

五个候选均未达到完整真实响应相似。JOINT_SIGNAL的峰后局部差距减小，但完整波形仍高度可区分，电荷条件下的PSD峰位和宽度也未全面改善。FULL_WAVE_STOCHASTIC及PULSE_STOCHASTIC_OFF的边际PSD/电荷部分较好，仍同时出现过窄的上升时间分布、条件密度成分偏移及基线差异。该轮不采用任一候选为已经验证的整体响应替代，也不据这些结果追加EVAL驱动调参。

这是一轮**已查看同采集基准上的冻结后回顾性验证**。EVAL曾在R4–R6被查看，其汇总结论参与问题提出；它不构成新盲测、独立采集或外部工况验证。该轮响应更新为0，全部预定有限候选保留，EVAL未向训练、选点或随机种子回流。

## 固定输入与评价

冻结登记为`CONTINUATION_FROZEN_METHOD.json`，UTC {metrics['frozen_UTC']}，SHA256 `{metrics['frozen_SHA256']}`。READINESS在首次EVAL数值解码前记录70份唯一既存绑定和本轮执行源/config SHA。任务正文及候选补充均保留原字节。

原7600条EVAL记录解码一次，保留17条既存QC失败；7583条有效作为补充总体。主REAL2000精确沿用R4保存位置/原ID并核对原hash排序，不重新抽样。每个候选沿用同一2000个G4父事件和已保存生成；五候选都为2000/2000 QC有效。原SIM与历史GEN_W0作分布对照，不新增生成或额外C2ST。

signed PSD为`1-Qfast/Qtotal`，保存480点的前100点均值扣基线；fast100:150、total100:400、tail200:400、1ns。真实电压和自身峰归一化形状分别报告。CAL参考/边界、TRAIN拟合尺度和八窄logQ区间未修改；电荷只是当前观测代理，未升级为独立标定的粒子能量。密度成分、0.99参考后验和旧simulation source_label均不作为粒子真值。

## 完整分布与真实重采样参照

以下为同2000样本的W1；电荷单位mV·ns，上升时间单位ns。二维值用冻结TRAIN logQ/PSD尺度和128方向。

| 域 | PSD W1 | 电荷W1 | rise W1 | 二维sliced W1 |
|---|---:|---:|---:|---:|
{distance_rows}

64次已保存的同N真实整组半分，二维W1中位数为{rr['median']:.6f}，5–95%为[{rr['range5_95'][0]:.6f}, {rr['range5_95'][1]:.6f}]；PSD W1中位数为{metrics['REAL_REAL_summary']['overall']['PSD']['W1']['median']:.6f}。这些只是同采集、人工50记录组的**采样参照，不是物理等效阈值**。本轮没有预设可据此自动宣称等效的容差。

全7583有效REAL补充保持同一方向：R3B/JOINT/JOINT_SIGNAL的二维W1分别为0.194832/0.179237/0.181184，FULL/OFF为0.261309/0.200704。FULL/OFF的rise W1为1.842330/1.959759ns，明显偏离；主2000上对应1.787009/1.901557ns。靠PSD边际较近不能补偿联合响应与时间分布的失败。

## 固定评价器与局部响应

42个预定HGB全部执行，参数不搜索：5候选×6视图及阴阳对照各6。各域600训练/250留出，所有候选和视图共享固定父/REAL评价事件，人工组/父隔离。TRAIN-only填补/尺度，500次条件组bootstrap；区间条件于已拟合评价器，不含响应训练不确定性。

| 域 | 完整shape | 完整V | prompt局部 | postpeak局部 | tail局部 | baseline局部 |
|---|---:|---:|---:|---:|---:|---:|
{c2st_rows}

REAL–REAL完整shape BA为0.504（95%区间0.472502–0.540699），既定扰动REAL为0.996。候选完整shape为0.988–1.000，具有明显的完整响应残差。JOINT_SIGNAL相对R3B的postpeak BA差为{paired['BA_delta_first_minus_second']:.3f}（配对95%区间[{paired['BA_delta_95CI'][0]:.6f}, {paired['BA_delta_95CI'][1]:.6f}]），是局部改善；其完整shape BA仍0.994。各区域评价只取本区信息，不用全局峰归一化或PSD泄漏。prompt100:150包含峰和早期下降，不能把其曲率全部称为纯上升沿；真正峰前例图与完整rise CDF另报。

JOINT_SIGNAL峰后曲率的电荷依赖更接近REAL，但prompt曲率的电荷关系仍偏；FULL/OFF峰后曲率随电荷近乎平坦。所有模型基线RMS主体较接近而REAL尾部更宽；FULL/OFF ACF/谱与REAL仍不同。平均/中位波形及10–90%带用于描述，不能替代完整分布相似性证据。

## 条件峰宽与覆盖

精确复用相同保存波形的旧密度拟合，逐域核对八区计数和PSD直方图；本轮新增密度拟合0。主要五候选+REAL及加入SIM/W0的完整域族，共同合格区间均为**bins3–7**。以下为该同一支持集合的平均绝对误差，不用各候选不同覆盖的均值比较。

| 项目 | R3B | JOINT | JOINT_SIGNAL | FULL_WAVE_STOCHASTIC | PULSE_STOCHASTIC_OFF |
|---|---:|---:|---:|---:|---:|
{width_rows}

四个后续修正臂在共同bins3–7上，两成分的峰位、原宽度、去趋势宽度六项误差都比R3B大。不能据局部C2ST或总体PSD W1的改善宣称条件分布也改善。REAL两成分宽度随logQ下降的旧组bootstrap95%斜率区间仍为负；这是本采集、电荷代理和该密度描述下的证据，不强制1/sqrt(E)规律。

| 域 | 有效窄区间 | 保留的缺口 |
|---|---|---|
{support_rows}

CAL q4参考仍失败，全q4为unknown，FoM undefined，不回退全局参考。q1所有候选及REAL的HC neutron-like计数仍0。FULL/OFF的q1固定切点两侧不足20，因此FoM undefined；各域完整总体和unknown保留。全局FoM可计算也不能掩盖这些分区失败。主REAL的FoM为0.669813，全7583为0.663001，均为冻结切点带矩法而非重新两高斯拟合。

## 父条件、迁移与源关联

全部2000父事件参与源名义电荷四区统计，另报输出自身电荷四区，迁移矩阵保留invalid列。不能只比较输出落入有利电荷区的样本。以共同426个名义q3父为例，相同REAL q3的PSD W1如下：

| 候选 | 全名义q3父输出 | 输出自身q3筛选 |
|---|---:|---:|
{q3_rows}

源名义/输出筛选对应不同统计对象，不能互换排序或省略迁移。沉积组成/条件关联与每父3视图随机变化已保存，它们不是新独立实测。R3B旧固定latent能量/组成对直接引用；新臂没有单独保存这种条件对，bundle替换不能隔离能量或组成。R8/R8C的保存权重及实际随机输入独立重放均已逐位通过，按补充协议引用原封存回执，本轮未无故重放。

R8C的OFF删除整个非线性脉冲随机分支，也删除其可表达的条件均值容量，因此不是严格纯方差消融。固定噪声源替换/打乱的敏感性只说明输出使用了输入，不证明真实逐事件物理映射。FULL旧step1900内部验证异常保留，原因unknown；单次训练/一次重放不证明训练稳定性。

## 来源局限、回执和停止

既有G4沉积条件和clean模板没有提供真实逐步沉积时序，完整SIM输入不能补回不存在的时序。旧SIM噪声校准曾用79793实测事件的历史先验，不能说其成本只有本次TRAIN。历史SIM/W0物理电压来自SIM父峰代理；源能谱/混合组成、触发与探测器处理差异仍混杂。本轮域相似性检验不证明G4物理校准，也不能由相似性推断正确事件对应。需要独立采集/真实条件标定及可追踪标签，才能支持外部条件或粒子判别的进一步声明。

主作业监督PID60844、worker25004，启动工具0ca54f、完成工具e69698，原生exit0。数值累计45.137038s，峰RSS1,349,013,504B、内存探针失败0，4线程；42HGB、响应更新/新生成/新增密度拟合/SSH/新G4均0。控制准备曾因R4根目录STANDARDIZATION路径错误native1，失败原回执和unknown PID保留，未启动科学拟合。

原launcher/native/RUN_HISTORY中`EVAL_read:0`和训练receipt提示是复用代码的陈旧字面字段，原字节保留；`RECEIPT_FIELD_CLARIFICATION.json`绑定原SHA并给出实际7600读取、7583QC、固定2000主评价及更新0。有效PID/退出/墙钟/RSS不改，不为这个元数据错误重跑科学。

11张PNG均已实际目检，轴、图例、全部候选和失败缺口可见。`metrics.json`、分组/逐事件特征及预测完整保留本地；`METRICS_COMPACT.json`只含聚合统计、规则和摘要SHA，适合ROOT的代码/报告提交。当前停止于完整响应负结果，不追加拟合、评价或EVAL驱动配方。

主要文件：`config.json`、`PROTOCOL.md`、`READINESS.json`、`FROZEN_BINDING_CHECK.json`、`INPUTS.json`、`REPORT.md`、`METRICS_COMPACT.json`、`RESOURCE_RECEIPT.json`、`FIGURE_QA.json`、`FINAL_MANIFEST.json`、`FINAL_RECEIPT.json`。
复现入口及范围见`REPRODUCTION.md`；ROOT另负责总报告、Git及D盘复制后的核验。
'''
write('REPORT.md', report)

resource = {
    'stage': config['stage'], 'UTC': now(), 'closing_PID': os.getpid(),
    'first_actual_native': {'UTC': config['start_UTC'], 'PID': config['first_native_PID'], 'tool_chunk': config['first_native_chunk'], 'exit': config['first_native_exit']},
    'deadline_UTC': config['deadline_UTC'],
    'numeric_process': {'launch_tool_chunk': '0ca54f', 'completion_tool_chunk': 'e69698', 'supervisor_PID': native['supervisor_PID'],
                        'worker_PID': native['worker_PID'], 'native_exit': native['worker_native_exit'], 'start_UTC': native['start_UTC'], 'stop_UTC': native['stop_UTC'],
                        'wall_s': native['wall_s'], 'numeric_cumulative_wall_s': native['numeric_cumulative_wall_s'],
                        'peak_RSS_observed_bytes': native['peak_RSS_observed'], 'memory_probe_failures': native['memory_probe_failures'],
                        'threads': native['threads'], 'reason': native['reason']},
    'worker_counters': worker['counters'],
    'actual_EVAL_use': {'original_records_decoded': 7600, 'QC_valid': 7583, 'original_QC_failed': 17, 'fixed_main_REAL': 2000, 'previously_viewed_same_acquisition': True},
    'limits': {'numeric_cumulative_wall_s': 1800, 'threads': 4, 'RSS_bytes': config['RSS_limit_bytes'], 'stage_output_bytes': config['output_limit_bytes']},
    'bound_input_payload_bytes': sum(item['bytes'] for item in inputs),
    'bound_input_payload_note': 'Sum of 22 bound payload sizes from INPUTS; not a claim about all physical OS/system-library I/O or all repeated control reads.',
    'preserved_control_failures': failures,
    'control_failure_elapsed_s': 'unknown; original retained control receipt has exit1 and unknown PID, not elapsed wall.',
    'no_numeric_failures_or_retry': True,
    'native_descriptive_field_correction': 'RECEIPT_FIELD_CLARIFICATION.json; do not use original EVAL_read literal0 as evidence of no EVAL use.',
    'versions': worker['versions'],
    'old_stage_writes': 0, 'SSH': 0, 'install': 0, 'Git': 0, 'new_agents': 0,
    'figure_QA_actual_images_viewed': 11,
    'final_byte_counts': 'Exact final member bytes are recorded in FINAL_MANIFEST and FINAL_RECEIPT after this file is written.',
}
write('RESOURCE_RECEIPT.json', resource)

reproduction = '''# R9复现入口

本目录是已完成封存结果。直接运行会消费新的评价作业并改写run输出，不应在这个已封存目录再执行。必要复现应复制代码/config及已绑定输入引用到新的E/D工作目录，显式登记复现；不更改当前结果，也不据回顾性EVAL选新响应配方。

原入口为本目录cwd下`python -B launch.py`；launcher只启动`python -B run.py`一次。缓存TEMP/TMP为项目`.runtime-cache/tmp`，PYTHONDONTWRITEBYTECODE=1、PYTHONUTF8=1；数值环境OMP/OPENBLAS/MKL/NUMEXPR/BLIS/VECLIB线程为4。实际Python3.13.3、NumPy2.2.6、scikit-learn1.7.2。当前执行源SHA在READINESS，冻结响应模型/生成/实际随机输入SHA在CONTINUATION_FROZEN_METHOD和FROZEN_BINDING_CHECK，原22输入SHA在INPUTS。

已有R3B/R8/R8C生成和独立重放引用未修改；R9无响应推理或训练。run先核验READINESS/config/已知输入哈希，复用同R4主REAL位置、64真实组半分、CAL参考和TRAIN尺度，42固定分类器及500条件组bootstrap；密度结果从完全相同的已存波形拟合表复用并核对各区计数/直方图。本轮不复制/安装环境或读服务器。

完整本地载荷为metrics.json、results/DATA_ROLES.json、results/C2ST_SPLITS.json、results/C2ST_HOLDOUT_PREDICTIONS.npz、results/PER_EVENT_FEATURES.npz及results/REAL_REAL_REFERENCE.json。身份/逐次选择/预测不嵌入METRICS_COMPACT发布汇总。11图实际QA见FIGURE_QA。封存纯字节SHA清单排除其自身及FINAL_RECEIPT，由FINAL_RECEIPT绑定清单SHA；没有伪造自哈希。

原生回执的两个陈旧说明字段保留，机器可读更正见RECEIPT_FIELD_CLARIFICATION。本轮真实EVAL读取7600、有效7583、固定主2000；正确响应更新0。该更正不改变波形/指标或重跑科学。
'''
write('REPRODUCTION.md', reproduction)

members = []
for path in sorted(ROOT.rglob('*')):
    if path.is_file() and path.name not in {'FINAL_MANIFEST.json', 'FINAL_RECEIPT.json'}:
        members.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)})
manifest = {'stage': config['stage'], 'UTC': now(), 'seal_PID': os.getpid(), 'members': members,
            'member_count': len(members), 'member_bytes': sum(row['bytes'] for row in members),
            'self_reference_exclusions': ['FINAL_MANIFEST.json', 'FINAL_RECEIPT.json'],
            'science_decodes_during_seal': 0, 'no_followup_fit_or_inference': True}
write('FINAL_MANIFEST.json', manifest)
receipt = {'stage': config['stage'], 'status': 'sealed_complete_scientific_negative', 'seal_PID': os.getpid(),
           'UTC_before_last_write': now(), 'first_UTC': config['start_UTC'], 'deadline_UTC': config['deadline_UTC'],
           'manifest_sha256': digest(ROOT / 'FINAL_MANIFEST.json'), 'manifest_bytes': (ROOT / 'FINAL_MANIFEST.json').stat().st_size,
           'manifest_member_count': len(members), 'manifest_member_bytes': manifest['member_bytes'],
           'hash_member_total_bytes': manifest['member_bytes'],
           'numeric_launch_tool': '0ca54f', 'numeric_completion_tool': 'e69698', 'numeric_native_exit': 0,
           'supervisor_PID': 60844, 'worker_PID': 25004, 'worker_counters': worker['counters'],
           'figure_QA_pass': True, 'figures_viewed': 11, 'receipt_correction': 'RECEIPT_FIELD_CLARIFICATION.json',
           'seal_native_exit': 'observed by caller after this process terminates; not self-certified',
           'pure_control_seal_scientific_decodes': 0,
           'complete_response_similarity_pass': False, 'candidate_adoption_from_R9': False,
           'old_stages_modified': False, 'retrospective_same_acquisition_not_external': True,
           'key_sha256': {name: digest(ROOT / name) for name in ['REPORT.md', 'METRICS_COMPACT.json', 'metrics.json', 'RESOURCE_RECEIPT.json', 'FIGURE_QA.json', 'RECEIPT_FIELD_CLARIFICATION.json', 'REPRODUCTION.md']}}
write('FINAL_RECEIPT.json', receipt)
files = [path for path in ROOT.rglob('*') if path.is_file()]
last_write = max(path.stat().st_mtime_ns for path in files)
assert last_write <= int(deadline.timestamp() * 1e9)
assert sum(path.stat().st_size for path in files) <= config['output_limit_bytes']
print(json.dumps({'PID': os.getpid(), 'UTC_after_all_writes': now(), 'last_write_UTC': dt.datetime.fromtimestamp(last_write / 1e9, UTC).isoformat(),
                  'stage_file_count': len(files), 'stage_bytes': sum(path.stat().st_size for path in files),
                  'manifest_sha256': receipt['manifest_sha256'], 'final_receipt_sha256': digest(ROOT / 'FINAL_RECEIPT.json'),
                  'key_sha256': receipt['key_sha256'], 'science_decodes': 0, 'sealed': True}, ensure_ascii=False))
