from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
S = R / 'work/R54_CONTINUOUS_CHARGE_FLOW_001'
NOW = datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((S / name).read_text('utf8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


files = sorted((S / 'figures').glob('*.png'))
assert len(files) == 9
write(S / 'FIGURE_QA.json', {
    'UTC': NOW, 'owner': '/root',
    'method': 'Actually opened all nine R54 rendered PNGs with image viewer after terminal fetch.',
    'all_figures_opened': True, 'layout_pass': True,
    'scientific_acceptance_inferred_from_layout': False,
    'observations': [
        'Five-view error bars and acceptance band are visible; exact baseline upper CI is reported numerically.',
        'All eight PSD bins remain visible. Candidate Q2 and Q8 failed density fits are annotated.',
        'Noise, source association, full waveform moments, all nine boundaries, training and teacher examples are readable.',
    ],
    'figures': [{'path': p.relative_to(S).as_posix(), 'bytes': p.stat().st_size,
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'opened': True, 'layout_pass': True} for p in files],
})
M = read('METRICS_COMPACT.json')
G = read('GATE_METRICS.json')
ARM = 'CONTINUOUS_Q_RESPONSE'
CTRL = 'CONTINUOUS_Q_REAL_CONTROL'
A = M['acceptance_A_C'][ARM]
gates = G['models'][ARM]
assert not any(A[k]['pass'] for k in ['A', 'B', 'C'])
assert G['decision'] == 'close_fixed_R54_continuous_charge_recipe'
noise = gates['gate2_PSD_width_centers_noise']['measures']
direction = {
    'UTC': NOW, 'owner': '/root', 'scientific_decision': G['decision'],
    'model_goal_complete': False, 'application_goal_status': 'active',
    'A_B_C_pass': {k: A[k]['pass'] for k in ['A', 'B', 'C']},
    'source_five_views': A['A']['views'][ARM],
    'same_treated_REAL_five_views_pass': all(v['pass'] for v in A['A']['views'][CTRL].values()),
    'overall_PSD_W1': A['B']['overall_PSD_W1'],
    'common_Q1_7_PSD_W1': A['B']['common_Q1_7_PSD_W1'],
    'required_density_fit_failed_bins': gates['gate2_PSD_width_centers_noise']['new_required_density_failures'],
    'all_candidate_density_failures': M['density_support_gaps'][ARM],
    'source_purehad_valley_checks': gates['gate3_source_preservation']['actual_source_pure_had_valleys'],
    'source_ranks': gates['gate3_source_preservation']['whole_population_source_ranks'],
    'noise_relative_to_BASE': {k: noise[k] for k in ['baseline_RMS_V_overall_W1', 'ACF1_overall_W1']},
    'paired_R54_minus_R53': M['paired_R54_minus_frozen_R53'],
    'replay_rows': 4000, 'independent_portable_replay_pass': read('REPLAY_RECEIPT.json')['pass'],
    'scientific_limits': [
        'SOURCE shape and u480 BA point estimates decrease, but their paired 95 percent intervals include zero.',
        'Overall PSD and baseline noise improve; required PSD accuracy, density validity and source-valley checks remain failures.',
        'SOURCE baseline CI upper bound 0.6000918180468247 exceeds 0.60; display rounding cannot turn it into a pass.',
        'One common initialization and random stream do not quantify response-retraining uncertainty.',
        'Preserved SOURCE Q marginal differs from REAL; existing finite HGB evidence is not an impossibility proof.',
        'DEV is historically exposed. Final EVAL and independent acquisitions were not opened.',
    ],
    'next_action': '封存 R54 固定方案。下一轮先登记总体/条件 PSD 与原始源谷比例的必要条件兼容性诊断：保持固定 SOURCE 电荷、身份和权重，允许原数值容差及 ADC 引起的分箱迁移，求一个明确放宽的分布距离下界；不训练或选新模型，不改变原 A–F。只有经核验的下界越过原阈值才可报告目标冲突；否则该诊断仅说明尚不能排除可行性，随后据 TRAIN 端点与学习误差证据决定新机制。',
}
write(R / 'reviews/R54_ROOT_DIRECTION_001.json', direction)
print(json.dumps({'figures_opened': len(files), 'A_B_C_pass': direction['A_B_C_pass'],
                  'decision': G['decision'], 'goal_complete': False}, indent=2))
