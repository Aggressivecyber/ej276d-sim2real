"""Pure v5 decision rules. No file reads, learned models, or science at import.

The original five statistical/source stage rules retain their numerical lines.
Stage six binds the actual noniterative Q+shape method and independent replay.
Undefined metrics fail their rule; a missing schema raises rather than passes.
"""
from __future__ import annotations

import math
from statistics import mean

VIEWS = ('full_historical_shape', 'normalized_u480', 'full_voltage',
         'Q_PSD_2D', 'baseline_local')
BOUNDARIES = ('first_99_100', 'first_149_150', 'first_399_400',
              'second_center_99', 'second_center_100', 'second_center_149',
              'second_center_150', 'second_center_399', 'second_center_400')
LSB = 1 / 16383.


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def le(x, limit):
    return finite(x) and finite(limit) and x <= limit


def interval(ci):
    return isinstance(ci, (list, tuple)) and len(ci) == 2 and all(map(finite, ci)) and ci[0] <= ci[1]


def score_aliases(row):
    out = dict(row)
    ci = row.get('heldout_CI', {})
    if isinstance(ci, dict):
        for target, original in [('BA_95CI', 'balanced_accuracy_95CI'), ('AUC_95CI', 'AUC_95CI')]:
            if original in ci:
                out[target] = ci[original]
    return out


def measure(candidate, baseline):
    change = candidate / baseline - 1 if finite(candidate) and finite(baseline) and baseline > 0 else (
        0. if finite(candidate) and finite(baseline) and candidate == baseline == 0 else None)
    return {'candidate': candidate, 'BASE': baseline, 'relative_change': change,
            'pass': le(change, .10)}


def density_row_ok(row, component=None, field=None):
    if row.get('valid') is not True or len(row.get('components', [])) != 2:
        return False
    return component is None or finite(row['components'][component].get(field))


def numerical_gate(c, evidence, arm, ctrl):
    training = evidence['training_admission']
    replay = evidence['replay']
    charge = evidence['charge']
    population = evidence['population']
    endpoints = replay.get('endpoints', {})
    names = [arm, ctrl]
    needed = ['pre_ADC_V', 'waveforms_V', 'predicted_target_Q_Vns', 'ADC_integer_codes',
              'evaluation_canonical_V', 'original_weight', 'response_applied_all_rows']
    replay_fields = all(
        endpoints.get(n, {}).get('rows') == 2000 and
        endpoints.get(n, {}).get('bitwise_pass') is True and
        set(needed) <= set(endpoints.get(n, {}).get('checks', {})) and
        all(v is True for v in endpoints[n]['checks'].values()) for n in names)
    counters = replay.get('actual_response_counters', {})
    checks = {
        'fixed_completed_training_and_native_exit': training.get('completed_updates') == c['updates'] and
            training.get('native_exit') == 0 and training.get('numeric_gate_recomputed') is True,
        'independent_noniterative_NumPy_reference': evidence.get('independent_numpy_reference_pass') is True,
        'training_full_population_and_endpoint_binding': evidence.get('training_records_revalidated') is True,
        'all_preADC_f32_Qstar_error_le1e_5': all(charge[n]['rows'] == 2000 and
            charge[n]['predicted_Qstar_positive_finite'] is True and
            le(charge[n]['actual_f32_preADC_max_relative_error'], 1e-5) for n in names),
        'all_main_rows_processed': all(population[n]['rows'] == 2000 and
            population[n]['all_rows_full_response'] is True for n in names),
        'row_metadata_and_charge_trace_verified': all(charge[n]['row_metadata_consistent'] is True and
            population[n]['identity_order_weight_match'] is True for n in names),
        'main4000_independent_replay': replay.get('rows') == 4000 and replay_fields and
            replay.get('pass') is True and replay.get('independent_process') is True and
            replay.get('complete_Q_head_shape_lifting_noise_Box_ADC_replayed') is True and
            replay.get('both_domains_computed_before_any_candidate_output_read') is True and
            replay.get('saved_Qstar_or_prepared_used_as_input') is False and
            replay.get('strict_integer_codes_and_actual_canonical_voltage_checked_separately') is True and
            counters.get('Q_head_forward_rows') == 4000 and counters.get('shape_forward_rows') == 4000 and
            counters.get('Box_capture_rows') == 4000 and counters.get('Box_capture_calls') == 16,
        'generation_replay_evaluation_native_exit0': evidence.get('prior_native_entries_pass') is True,
        'same_bound_public_SOURCE_REAL_operator': evidence.get('same_public_operator_and_bindings') is True,
        'charge_migration_full_mass_accounted': evidence.get('v5_charge_descriptions_complete') is True,
    }
    return {'checks': checks, 'pass': all(checks.values()),
            'fixed_updates': c['updates'], 'training_admission': training,
            'independent_reference': evidence['numerical_reference'],
            'charge_error_summaries': {n: charge[n] for n in names},
            'replay_rows': replay.get('rows'), 'precision_reference_type': 'independent_NumPy_float64_full_Q_shape',
            'input_SOURCE_Q_hard_conservation_required': False,
            'physical_equivalence_or_independent_acquisition_claim': False}


def evaluate(c, m, evidence):
    """Return six original stage gates and final A/B/C for the one fixed model."""
    arm, ctrl = c['SOURCE_arm'], c['REAL_arm']
    if c['acceptance_version'] != 5 or c['updates'] <= 0 or arm == ctrl:
        raise ValueError('incorrect prospective candidate/acceptance binding')
    cs = {n: {v: score_aliases(x) for v, x in rows.items()} for n, rows in m['C2ST']['results'].items()}
    psd, width = m['PSD_main'], m['energy_conditional_width']
    required = list(range(1, 7))
    for n in ['REAL', 'BASE', arm, ctrl]:
        if len(width['tables'][n]['slices']) != 8 or [s['bin'] for s in width['tables'][n]['slices']] != list(range(8)):
            raise ValueError('eight original density slots must remain, including failures: ' + n)
    gaps = {n: [{'bin_id_1based': s['bin'] + 1, 'N': s['N'], 'failure_reasons': s['failure_reasons']}
               for s in t['slices'] if not s['valid']] for n, t in width['tables'].items()}

    paired = m['C2ST']['paired'][arm]
    limits = {'full_historical_shape': .934, 'normalized_u480': .928}
    g1 = {'paired_minus_BASE': {v: paired[v] for v in limits}, 'fixed_point_limits': limits,
          'candidate_BA': {v: cs[arm][v]['balanced_accuracy'] for v in limits},
          'both_BASE_CI_upper_negative': all(interval(paired[v]['BA_delta_95CI']) and
              paired[v]['BA_delta_95CI'][1] < 0 for v in limits),
          'both_fixed_R37_point_limits': all(le(cs[arm][v]['balanced_accuracy'], limit + 1e-14)
              for v, limit in limits.items())}
    g1['pass'] = g1['both_BASE_CI_upper_negative'] and g1['both_fixed_R37_point_limits']

    common = list(range(7))
    missing_q = {n: [i+1 for i in common if psd['tables'][n]['bins'][i]['N'] < 40]
                 for n in ['REAL', 'BASE', arm]}
    pw = {}
    for n in ['BASE', arm]:
        values = [psd['vs_REAL'][n]['bins'][i]['W1'] for i in common]
        pw[n] = mean(values) if not missing_q['REAL'] and not missing_q[n] and all(map(finite, values)) else None
    meas = {k + '_overall_W1': measure(m['domain_vs_REAL'][arm]['overall'][k]['W1'],
            m['domain_vs_REAL']['BASE']['overall'][k]['W1']) for k in ['PSD', 'baseline_RMS_V', 'ACF1']}
    meas['PSD_common_Q_equal_mean_W1'] = {**measure(pw[arm], pw['BASE']),
        'common_bins_1based': [i+1 for i in common], 'N_each_domain_ge': 40,
        'fixed_required_bins': True, 'unsupported_required_bins_by_domain': missing_q}
    # Compute center/raw-SD errors directly from the fixed six per-bin fits.
    # The old aggregated table is not an authoritative pass flag.
    errors, centers = {}, {}
    for component in [0, 1]:
        for field in ['mean', 'sigma_PSD']:
            missing = {n: [i+1 for i in required if not density_row_ok(
                width['tables'][n]['slices'][i], component, field)] for n in ['REAL', 'BASE', arm]}
            values = {n: mean([abs(width['tables'][n]['slices'][i]['components'][component][field] -
                width['tables']['REAL']['slices'][i]['components'][component][field]) for i in required])
                if not missing['REAL'] and not missing[n] else None for n in ['BASE', arm]}
            key = 'component' + str(component) + ('_peak_position_MAE' if field == 'mean' else '_raw_width_MAE')
            meas[key] = {**measure(values[arm], values['BASE']), 'common_bins_1based': [2,3,4,5,6,7],
                         'failed_required_bins_by_domain': missing, 'density_failure_not_hidden_by_intersection': True}
            errors[key] = values
            if field == 'mean':
                centers[str(component)] = values[arm]
    newfail = [i+1 for i in required if not density_row_ok(width['tables'][arm]['slices'][i])]
    basefail = [i+1 for i in required if not density_row_ok(width['tables']['BASE']['slices'][i])]
    zeros = evidence['zero_baseline_counts']
    g2 = {'measures': meas, 'BASE_valid_required_bins_1based': [i+1 for i in required if i+1 not in basefail],
          'BASE_required_density_failures': basefail, 'new_required_density_failures': newfail,
          'all_density_support_gaps': gaps, 'zero_baseline_counts': zeros,
          'zero_baseline_not_increased': zeros[arm] <= zeros['BASE']}
    g2['pass'] = len(meas) == 8 and all(v['pass'] for v in meas.values()) and not newfail and not basefail and g2['zero_baseline_not_increased']

    primary = m['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4']
    valleys, ranks = [], []
    for b, count in [('bin3', 213), ('bin4', 181)]:
        x, base = primary[arm][b]['valley_fraction'], primary['BASE'][b]['valley_fraction']
        delta = x-base if finite(x) and finite(base) else None
        n = primary[arm][b]['source_count']
        valleys.append({'actual_source_bin': b, 'N': n, 'candidate': x, 'BASE': base, 'delta': delta,
                        'pass': n == count and primary['BASE'][b]['source_count'] == count and le(delta, .02)})
    for field in ['source_Q_to_output_Q', 'source_PSD_to_output_PSD']:
        x = m['source_association_actual_preADC_input'][arm]['composition_and_PDG']['all'][field]['Spearman']
        base = m['source_association_actual_preADC_input']['BASE']['composition_and_PDG']['all'][field]['Spearman']
        delta = base-x if finite(x) and finite(base) else None
        ranks.append({'measure': field, 'candidate': x, 'BASE': base, 'BASE_minus_candidate': delta, 'pass': le(delta, .05)})
    gen = m['generation'][arm]
    pop = evidence['population'][arm]
    sensitivity = m['source_probes'][arm]['rows']['source_roll_no_fixed_points']['waveform_RMS_vs_correct_reference_V']['mean']
    g3 = {'actual_source_pure_had_valleys': valleys, 'whole_population_source_ranks': ranks,
          'source_ID_order_composition_weights_preserved': gen['all_source_ID_order_weights_and_composition_preserved'] is True and
              pop['identity_order_weight_match'] is True and m['data_roles']['DEV_composition'] == {'mixed':65,'pure_EE':417,'pure_had':1518} and
              m['primary_DEV_population_reweighted'] is False,
          'internal_constraint_projection': m['explicit_constraint_projection'][arm],
          'no_new_invalid_nonfinite_ADC_clip': m['domains'][arm]['invalid'] == 0 and pop['pre_ADC_finite'] is True and
              pop['pre_ADC_outside_range_sample_count'] == 0,
          'outside_TRAIN_support_rows': pop['outside_original_support_rows'],
          'all_outside_rows_use_full_response': pop['outside_original_support_rows'] == 417 and pop['all_rows_full_response'] is True and
              gen['BASE_fallback_N'] == 0 and gen['full_response_applied_N'] == 2000,
          'source_roll_RMS_V': sensitivity, 'sensitivity_not_correct_physical_mapping': True}
    g3['pass'] = all(x['pass'] for x in valleys+ranks) and all(g3[k] for k in [
        'source_ID_order_composition_weights_preserved', 'no_new_invalid_nonfinite_ADC_clip', 'all_outside_rows_use_full_response']) and finite(sensitivity) and sensitivity > 0

    bd = m['boundary_differences']['results']
    bounds = []
    for key in BOUNDARIES:
        x, base, other = (bd[n][key]['W1_V'] for n in [arm, 'BASE', 'INDEPENDENT_CFM'])
        limit = max(base, other) + LSB if finite(base) and finite(other) else None
        bounds.append({'location': key, 'candidate_W1_V': x, 'BASE_W1_V': base,
                       'INDEPENDENT_W1_V': other, 'limit_V': limit, 'pass': le(x, limit)})
    gp = m['guide_probe'][arm]
    guide = dict(gp)
    guide['pass'] = (gp['rows'] == 128 and le(.95, gp['output_nonidentical_fraction']) and
        finite(gp['preADC_output_RMS_V']['median']) and gp['preADC_output_RMS_V']['median'] > LSB)
    g4 = {'boundaries': bounds, 'full_SOURCE_perturbation': guide,
          'pass': all(x['pass'] for x in bounds) and guide['pass']}

    cp = m['C2ST']['same_treated_REAL_control_paired'][ctrl]
    control_rows = {}
    for view in ['full_historical_shape', 'normalized_u480', 'baseline_local']:
        row = cp[view]
        control_rows[view] = {'BASE_G_REAL_BA': cs['BASE_REAL_CONTROL'][view]['balanced_accuracy'],
            'new_G_REAL_BA': cs[ctrl][view]['balanced_accuracy'], **row,
            'point_delta_le0_01': le(row['BA_delta_first_minus_second'], .01 + 1e-14),
            'CI_upper_le0_10': interval(row['BA_delta_95CI']) and row['BA_delta_95CI'][1] <= .10}
        control_rows[view]['pass'] = control_rows[view]['point_delta_le0_01'] and control_rows[view]['CI_upper_le0_10']
    g5 = {'pre_registered_same_treated_scope': True, 'comparison': 'same public response on REAL minus frozen BASE on REAL',
          'three_views': control_rows, 'RAW_RR_absolute_effects_descriptive': {
              n: m['C2ST']['paired'][n] for n in ['BASE_REAL_CONTROL', ctrl]},
          'RAW_identity_not_required': True, 'pass': all(v['pass'] for v in control_rows.values())}
    g6 = numerical_gate(c, evidence, arm, ctrl)
    all_gates = {'gate1_complete_response': g1, 'gate2_PSD_width_centers_noise': g2,
                 'gate3_source_preservation': g3, 'gate4_boundary_source': g4,
                 'gate5_same_treated_REAL_control': g5, 'gate6_numerical_replay': g6}
    retained = all(v['pass'] for v in all_gates.values())
    all_gates.update(retain=retained, role='primary')
    absolute = {}
    for name in [arm, ctrl]:
        absolute[name] = {}
        for view in VIEWS:
            row = cs[name][view]
            ci = row.get('BA_95CI')
            absolute[name][view] = {'BA': row['balanced_accuracy'], 'BA_95CI': ci,
                'pass': interval(ci) and ci[0] >= .4 and ci[1] <= .6}
    relative = {}
    for component in [0, 1]:
        failed, vals = [], []
        for i in required:
            r, s = width['tables']['REAL']['slices'][i], width['tables'][arm]['slices'][i]
            if not density_row_ok(r, component, 'sigma_PSD') or not density_row_ok(s, component, 'sigma_PSD'):
                failed.append(i+1)
                continue
            x, b = s['components'][component]['sigma_PSD'], r['components'][component]['sigma_PSD']
            if b <= 0:
                failed.append(i+1)
            else:
                vals.append(abs(x-b)/b)
        value = mean(vals) if not failed and len(vals) == 6 else None
        relative[str(component)] = {'required_bins_1based': [2,3,4,5,6,7], 'failed_required_bins': failed,
            'mean_absolute_relative_rawSD_error': value, 'pass': le(value, .10)}
    global_psd = m['domain_vs_REAL'][arm]['overall']['PSD']['W1']
    bpass = le(global_psd, .01) and le(pw[arm], .01) and all(le(x, .01) for x in centers.values()) and all(
        x['pass'] for x in relative.values()) and g2['pass']
    acceptance = {arm: {
        'A': {'views': absolute, 'pass': all(x['pass'] for rows in absolute.values() for x in rows.values())},
        'B': {'overall_PSD_W1': global_psd, 'common_Q1_7_PSD_W1': pw[arm],
              'peak_MAE_required_Q2_7': centers, 'raw_width_relative_errors': relative,
              'original_noise_side_effects_required': True, 'pass': bpass},
        'C': {'all_original_six_stage_gates': retained, 'pass': retained},
        'D_E': 'not_evaluated; requires separate registration after primary A-C pass', 'role': 'primary'}}
    passed = all(acceptance[arm][key]['pass'] for key in ['A','B','C'])
    gate = {'acceptance_version': 5, 'recipe': c['recipe_description'], 'models': {arm: all_gates},
            'primary': arm, 'primary_retain': retained, 'primary_A_C_pass': passed,
            'decision': 'primary_pending_separately_registered_D_E' if passed else 'close_fixed_R77_full_response',
            'prespecified_rules_changed_after_data': False, 'control_not_fallback': True,
            'new_Q_diagnostic_threshold_added': False, 'true_missing_region_event_rate': 'unknown'}
    return gate, acceptance
