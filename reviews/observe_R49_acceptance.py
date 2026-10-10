"""Apply original A/B thresholds to saved aggregates; no fits or event resampling."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R49_FROZEN_HEUN128_RESPONSE_001'
WORK = ROOT / 'work' / STAGE
OUT = ROOT / 'reviews/R49_ACCEPTANCE_OBSERVATION_001.json'
assert not OUT.exists()
metrics = json.loads((WORK / 'metrics.json').read_text('utf-8-sig'))
numeric = json.loads((WORK / 'TRAIN_SOLVER_CHECK.json').read_text('utf-8-sig'))
assert numeric['both_pass'] and numeric['DEV_read_or_generated'] == 0
PRIMARY = 'GUIDED_NOISE_COMPLETION'
CONTROL = 'GUIDED_FIXED_NOISE_CONTROL'
REAL_PAIRS = {CONTROL: 'GUIDED_FIXED_NOISE_REAL_CONTROL', PRIMARY: 'GUIDED_COMPLETION_REAL_CONTROL'}
views = ['full_historical_shape', 'normalized_u480', 'full_voltage', 'Q_PSD_2D', 'baseline_local']
width = metrics['energy_conditional_width']['tables']
required = list(range(1, 7))
arms = {}
for arm, real_control in REAL_PAIRS.items():
    classification = {}
    for name in [arm, real_control]:
        classification[name] = {}
        for view in views:
            value = metrics['C2ST']['results'][name][view]
            ci = value.get('BA_95CI')
            if ci is None:
                ci = value['heldout_CI']['balanced_accuracy_95CI']
            classification[name][view] = {'BA': value['balanced_accuracy'], 'BA_95CI': ci,
                'required_CI_containment': [.4, .6], 'pass': bool(.4 <= ci[0] <= ci[1] <= .6)}
    failed = {name: [i + 1 for i in required if not width[name]['slices'][i]['valid']]
              for name in ['REAL', arm]}
    component_tests = []
    for k in [0, 1]:
        rows = []
        for i in required:
            rc, ac = width['REAL']['slices'][i], width[arm]['slices'][i]
            valid = rc['valid'] and ac['valid'] and len(rc['components']) == len(ac['components']) == 2
            row = {'bin_1based': i + 1, 'valid': bool(valid), 'peak_abs_error': None,
                   'raw_SD_relative_abs_error': None}
            if valid:
                real, candidate = rc['components'][k], ac['components'][k]
                values = [real['mean'], candidate['mean'], real['sigma_PSD'], candidate['sigma_PSD']]
                if all(v is not None and math.isfinite(v) for v in values) and real['sigma_PSD'] > 0:
                    row['peak_abs_error'] = abs(candidate['mean'] - real['mean'])
                    row['raw_SD_relative_abs_error'] = abs(candidate['sigma_PSD'] - real['sigma_PSD']) / real['sigma_PSD']
                else:
                    row['valid'] = False
            rows.append(row)
        valid = all(row['valid'] for row in rows)
        peak = math.fsum(row['peak_abs_error'] for row in rows) / 6 if valid else None
        sd = math.fsum(row['raw_SD_relative_abs_error'] for row in rows) / 6 if valid else None
        component_tests.append({'component': k, 'rows': rows, 'all_required_bins_valid': valid,
            'peak_MAE': peak, 'peak_MAE_limit': .01, 'peak_pass': bool(valid and peak <= .01),
            'raw_SD_mean_relative_abs_error': sd, 'raw_SD_relative_limit': .1,
            'raw_SD_pass': bool(valid and sd <= .1), 'failed_bins_not_dropped': True})
    common_required = list(range(7))
    missing = {name: [i + 1 for i in common_required
        if metrics['PSD_main']['tables'][name]['bins'][i]['N'] < 40] for name in ['REAL', arm]}
    overall = metrics['domain_vs_REAL'][arm]['overall']['PSD']['W1']
    common = (math.fsum(metrics['PSD_main']['vs_REAL'][arm]['bins'][i]['W1'] for i in common_required) / 7
              if not any(missing.values()) else None)
    psd = {key: {'value': value, 'maximum': .01, 'pass': value is not None and value <= .01}
           for key, value in [('PSD_overall_W1', overall), ('PSD_common_Q_equal_mean_W1', common)]}
    noise = {}
    for key in ['baseline_RMS_V', 'ACF1']:
        base = metrics['domain_vs_REAL']['BASE']['overall'][key]['W1']
        value = metrics['domain_vs_REAL'][arm]['overall'][key]['W1']
        noise[key] = {'value': value, 'BASE': base, 'maximum': 1.1 * base, 'pass': value <= 1.1 * base}
    A = all(v['pass'] for domain in classification.values() for v in domain.values())
    B = (all(v['pass'] for v in psd.values()) and all(v['pass'] for v in noise.values())
         and all(v['peak_pass'] and v['raw_SD_pass'] for v in component_tests))
    arms[arm] = {'role': 'primary' if arm == PRIMARY else 'fixed_control_not_fallback',
        'A_pass': A, 'A_five_views_each_domain': classification, 'B_pass': B,
        'B_PSD': psd, 'B_commonQ_missing_required_support': missing, 'B_noise': noise,
        'B_required_density_failures': failed, 'B_peak_width_fixed_bins2_7': component_tests}

record = {'stage': STAGE, 'recorded_UTC': datetime.now(timezone.utc).isoformat(),
    'actual_root_control_PID': os.getpid(), 'status': 'original_A_B_applied_to_saved_aggregates',
    'primary_candidate': PRIMARY, 'scientific_threshold_version': 1,
    'acceptance_registry_sha256': hashlib.sha256((ROOT / 'PROJECT_ACCEPTANCE.json').read_bytes()).hexdigest(),
    'source': 'work/' + STAGE + '/metrics.json', 'arms': arms,
    'A_pass': arms[PRIMARY]['A_pass'], 'B_pass': arms[PRIMARY]['B_pass'],
    'RAW_REAL_REAL_Q_PSD_2D': metrics['C2ST']['results'].get('RAW_REAL_REAL', {}).get('Q_PSD_2D', 'unknown_not_measured'),
    'C_six_stage_gates_and_replay': 'recorded_in_final_root_review',
    'D_E': 'not_performed_in_R49', 'F_final_delivery': 'pending_seal_Git_D',
    'new_response_fits': 0, 'new_classifier_fits': 0, 'new_density_fits': 0,
    'no_population_or_threshold_changes': True,
    'B_is_absolute_B_definition_C_separately_requires_all_original_stage_gates': True}
OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')
print(json.dumps({'status': record['status'], 'A_pass': record['A_pass'], 'B_pass': record['B_pass'],
    'primary_required_density_failures': arms[PRIMARY]['B_required_density_failures'], 'new_fits': 0,
    'receipt': OUT.relative_to(ROOT).as_posix()}))
