"""Record the root's decision from completed local saved evidence; no scientific fits."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R49_LOCAL_HEUN128_RECOVERY_001'
PREFIX = 'work/' + STAGE + '/'
OUT = ROOT / 'reviews/R49_LOCAL_ROOT_DIRECTION_001.json'
assert not OUT.exists()


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def bind(name):
    raw = (ROOT / name).read_bytes()
    return {'path': name, 'bytes': len(raw), 'SHA256': hashlib.sha256(raw).hexdigest()}


M = read(PREFIX + 'metrics.json')
A = read('reviews/R49_LOCAL_ACCEPTANCE_OBSERVATION_001.json')
G = read(PREFIX + 'GATE_METRICS.json')
RP = read(PREFIX + 'REPLAY_RECEIPT.json')
N = read(PREFIX + 'TRAIN_SOLVER_CHECK.json')
P = 'GUIDED_NOISE_COMPLETION'
C = 'GUIDED_FIXED_NOISE_CONTROL'
R = 'GUIDED_COMPLETION_REAL_CONTROL'
VIEWS = ['full_historical_shape', 'normalized_u480', 'full_voltage', 'Q_PSD_2D', 'baseline_local']
assert N['both_pass'] and RP['rows'] == 8000
assert not A['A_pass'] and not A['B_pass'] and not G['primary_A_C_pass']
assert G['decision'] == 'close_fixed_R49_Heun128_recipe'
row = A['arms'][P]
views = row['A_five_views_each_domain']
pair_keys = ['paired_primary_minus_control_C2ST', 'paired_completion_minus_fixed_same_REAL_C2ST']
paired = {key: {v: {k: M[key][v][k] for k in ['BA_delta_first_minus_second', 'BA_delta_95CI']}
                for v in VIEWS} for key in pair_keys}
all_pair_CIs_include_zero = all(value['BA_delta_95CI'][0] <= 0 <= value['BA_delta_95CI'][1]
                               for group in paired.values() for value in group.values())
g = G['models'][P]
facts = {
    'SOURCE_primary_shape_BA': views[P]['full_historical_shape']['BA'],
    'SOURCE_primary_shape_BA_95CI': views[P]['full_historical_shape']['BA_95CI'],
    'BASE_SOURCE_shape_BA': M['C2ST']['results']['BASE']['full_historical_shape']['balanced_accuracy'],
    'SOURCE_primary_voltage_BA': views[P]['full_voltage']['BA'],
    'same_treated_REAL_primary_voltage_BA': views[R]['full_voltage']['BA'],
    'same_treated_REAL_primary_voltage_BA_95CI': views[R]['full_voltage']['BA_95CI'],
    'SOURCE_and_same_treated_REAL_original_five_views': views,
    'primary_overall_PSD_W1': row['B_PSD']['PSD_overall_W1']['value'],
    'control_overall_PSD_W1': A['arms'][C]['B_PSD']['PSD_overall_W1']['value'],
    'BASE_overall_PSD_W1': M['domain_vs_REAL']['BASE']['overall']['PSD']['W1'],
    'primary_fixed_Q1_7_PSD_W1': row['B_PSD']['PSD_common_Q_equal_mean_W1']['value'],
    'required_PSD_W1_max': .01,
    'required_density_failed_bins': {arm: A['arms'][arm]['B_required_density_failures'] for arm in [C, P]},
    'density_failure_reasons': {arm: M['energy_conditional_width']['tables'][arm]['slices'][2]['failure_reasons']
                                for arm in [C, P]},
    'failed_density_bins_not_removed_or_reinterpreted_as_physical_peak_absence': True,
    'primary_noise': row['B_noise'],
    'primary_actual_source_pure_had_valleys': g['gate3_source_preservation']['actual_source_pure_had_valleys'],
    'primary_whole_population_source_ranks': g['gate3_source_preservation']['whole_population_source_ranks'],
    'primary_failed_boundaries': [x for x in g['gate4_boundary_source']['boundaries'] if not x['pass']],
    'paired_primary_minus_control': paired,
    'all_five_original_SOURCE_and_REAL_BA_pair_CIs_include_zero': all_pair_CIs_include_zero,
    'independent_main_replay_rows': RP['rows'],
    'independent_main_replay_pass': RP['pass'],
    'six_original_stage_gates': {key: g[key]['pass'] for key in [
        'gate1_complete_response', 'gate2_PSD_width_centers_noise', 'gate3_source_preservation',
        'gate4_boundary_source', 'gate5_same_treated_REAL_control', 'gate6_numerical_replay']},
}
replay_text = '通过' if RP['pass'] else '未通过，失败细节保留在回执中'
record = {
    'stage': STAGE, 'recorded_UTC': datetime.now(timezone.utc).isoformat(),
    'actual_root_control_PID': os.getpid(),
    'decision_basis': 'Completed local generated-array evaluation and full local replay; rendered QA and native sealing are separately verified.',
    'primary_A_C_pass': False, 'primary_A_pass': False, 'primary_B_pass': False,
    'original_A_F_unchanged': True, 'compute_budget_enforced': False,
    'time_or_output_budget_stop': False,
    'current_family_status': 'fixed_R49_local_recipe_unaccepted_unbudgeted_project_policy_continues',
    'scientific_decision': G['decision'],
    'whole_noise_completion_family_declared_ineffective': False,
    'higher_solver_step_search_is_a_supported_remedy': False,
    'source_main_population_changed': False, 'fixed_control_used_as_fallback_winner': False,
    'new_scientific_fits_or_inference_for_this_review': 0,
    'remote_attempt_metrics_used_as_local_results': False,
    'remote_replay_terminal': 'unknown', 'facts': facts,
    'interpretation': (
        'Both local 128/256 numerical prerequisites pass. The primary candidate improves descriptive PSD distance '
        'relative to its fixed-noise control, but fails absolute PSD, full SOURCE response, same-treated REAL voltage, '
        'required Q3 density, source-conditional valleys and one boundary constraint. The five original-view paired '
        'BA intervals for both SOURCE and REAL include zero; they do not establish a uniform full-response advantage. '
        'The baseline_local view removes per-event offset and scale, so its REAL pass does not validate absolute DC '
        'or noise amplitude. ADC representation, baseline restoration and amplitude structure remain hypotheses for '
        'the REAL voltage discrepancy. No counterfactual causal test was performed. Explaining that discrepancy '
        'alone would not resolve the observed SOURCE normalized-shape or conditional-PSD failures.'),
    'next_direction': (
        'Retain R49 as a closed fixed recipe. A separately registered next protocol should first isolate the '
        'same-treated REAL voltage discrepancy using frozen outputs and controlled ADC/baseline representations, '
        'then define a response change that preserves source-conditional PSD while controlling noise. Maintain '
        'the original populations, all failed bins, fixed primary/control roles and original A-F thresholds. '
        'The earlier ADC source observation is a diagnostic hypothesis only; no ADC counterfactual study is claimed.'),
    'next_action': (
        'Complete native sealing, private Git publication and D normal-copy delivery for this fixed local stage. '
        'Retain the full negative evidence and the unknown old remote terminal state. Any subsequent scientific '
        'work requires a distinct recorded protocol under the existing no-compute-budget authorization. '
        'No A-C-qualified candidate exists for D/E validation.'),
    'Chinese_current_state': (
        '本机两臂128/256步数值检查通过，11968条输出及40个判别器/32个密度槽评价完成；'
        f'8000条独立重放{replay_text}。固定主候选A/B失败，原六项门结果保留。'
        '总体PSD W1点估计比固定噪声对照低，但完整SOURCE响应、同处理REAL电压、源条件PSD和必需Q3密度仍未达标。'),
    'Chinese_result_summary': (
        'R49本机恢复完成固定两臂的完整评价，主候选不满足原A–C联合验收。'
        'SOURCE形状BA从旧BASE的0.978降至0.780，仍未达到要求；主候选95%区间为0.745–0.817，'
        '同处理REAL电压BA为0.828（0.796–0.858）。总体/固定Q1–7 PSD W1为0.02890/0.03932，均超过0.01；'
        '旧BASE总体PSD W1为0.02307，固定噪声对照为0.04184。原五视图的SOURCE及REAL配对BA差区间均包含0，'
        '不能据此宣称主方案全面优于固定对照。两臂必需Q3密度均未在原固定定义内收敛，Q2–7峰位/宽度联合结果'
        '保持未定义；失败区保留，不推断物理粒子峰消失。主候选两处源pure-had谷比例和一处时间边界未满足约束。'
        f'本机8000条独立重放{replay_text}。固定配方关闭，整个噪声补全方法家族的可行性仍不能据此否定。'
        'REAL电压差异原因尚未确定，ADC表示仅为待检验假设；SOURCE形状与条件PSD仍须分别解决。'
        'D/E没有合格候选可进入，模型目标未完成。'),
    'bindings': [bind(name) for name in ['reviews/R49_LOCAL_ACCEPTANCE_OBSERVATION_001.json',
        PREFIX + 'metrics.json', PREFIX + 'GATE_METRICS.json', PREFIX + 'REPLAY_RECEIPT.json',
        PREFIX + 'TRAIN_SOLVER_CHECK.json', 'reviews/ROOT_R49_ADC_SOURCE_OBSERVATION_001.json']],
}
OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')
print(json.dumps({'receipt': OUT.relative_to(ROOT).as_posix(), 'PID': os.getpid(),
                  'primary_A_C_pass': False, 'replay_pass': RP['pass'], 'new_scientific_fits': 0}))
