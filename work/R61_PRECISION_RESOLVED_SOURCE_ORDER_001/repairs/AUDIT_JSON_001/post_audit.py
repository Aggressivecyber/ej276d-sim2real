"""Read-only checks of the frozen training record and explicit projection effects."""
from pathlib import Path
import hashlib
import json
import numpy as np
import evaluation as ev
from precision_audit import precision_single_factor_audit


def constraint_description(d):
    n = len(d['constraint_projection_range_active'])
    fields = ['pre_constraint_out_of_range_sample_count','constraint_projection_RMS_V',
        'constraint_projection_maxabs_V','constraint_projection_rail_sample_count',
        'pre_constraint_charge_relative_error']
    def describe(use):
        return {'N':int(use.sum()),
            'range_active_events':int(d['constraint_projection_range_active'][use].sum()),
            'rail_sample_total':int(d['constraint_projection_rail_sample_count'][use].sum()),
            'fields':{k:ev.summary(d[k][use]) for k in fields}}
    result = describe(np.ones(n, bool))
    result.update(internal_model_operation='Euclidean box-charge projection after constant-field Heun',
        zero_ADC_clip_count_does_not_mean_zero_model_projection=True,
        equal_original_parent_weights=True, pass_claim=False,
        inside_TRAIN_support=describe(d['inside_TRAIN_support']),
        outside_TRAIN_support=describe(~d['inside_TRAIN_support']),
        by_original_charge_region=[dict(region=int(b),**describe(d['source_Qbin_actual_preADC']==b)) for b in range(-1,9)])
    return result


def training_audit(root, cfg, receipt):
    root=Path(root);p=root/cfg['arm']
    def read(n):return json.loads((p/n).read_text('utf8'))
    init=read('INITIALIZATION.json');rng0=read('INITIAL_RANDOM_STATES.json');rng1=read('FINAL_RANDOM_STATES.json')
    with np.load(p/'PARENT_EXPOSURES_PRIVATE.npz',allow_pickle=False) as z:
        exposures={k:z[k].copy() for k in ['source_counts','REAL_reference_counts','REAL_identity_counts','source_bin0','REAL_bin0']}
    with np.load(p/'LOSS_BY_UPDATE_PRIVATE.npz',allow_pickle=False) as z:
        losses=z['values'].copy();columns=z['columns'].tolist()
    checks={
        'sole_50000':receipt['updates']==cfg['updates']==50000 and receipt['selection']=='sole final50000',
        'one_fresh_fit_no_teacher':receipt['new_response_training_runs']==1 and receipt['teacher_waveforms_constructed']==0 and not receipt['R58_law_or_atoms_training_target'] and not init['loaded_previous_trained_response'],
        'all_eligible_parents_seen':receipt['actual_unique_SOURCE_REAL_reference_REAL_identity']==[6593,21047,21047],
        'new_seed_matches':receipt['model_seed']==cfg['model_seed']==init['seed'] and receipt['PCG64_seeds']==cfg['PCG64_seeds'],
        'no_validation_decode':all(receipt[k]==0 for k in ['DEV_read','EVAL_read','CAL_decode']) and not receipt['source_val_target_val_decoded'],
        'loss_records_complete_finite':losses.shape==(50000,20) and np.isfinite(losses).all(),
        'counter_totals':receipt['counter']['new_response_updates']==50000 and receipt['counter']['SOURCE_training_rows']==16000000 and receipt['counter']['REAL_reference_rows']==16000000 and receipt['counter']['REAL_identity_rows']==3200000,
        'positive_exposures_and_totals':all(np.all(exposures[k]>0) and int(exposures[k].sum())==total for k,total in [('source_counts',16000000),('REAL_reference_counts',16000000),('REAL_identity_counts',3200000)]),
        'exact_stratified_draw_totals':all(int(exposures[count][exposures[bins]==b].sum())==1600000 for count,bins in [('source_counts','source_bin0'),('REAL_reference_counts','REAL_bin0')] for b in range(-1,9)),
        'initial_state_hash_matches':receipt['initial_state_SHA']==init['state_tensor_SHA'],
        'six_consumed_stream_hashes':set(receipt['stream_hashes'])==set(cfg['PCG64_seeds']) and all(len(bytes.fromhex(h))==32 for h in receipt['stream_hashes'].values()),
        'all_RNG_states_saved':set(rng0)==set(rng1)==set(cfg['PCG64_seeds']),
        'initial_PCG64_states_match_seeds':all(rng0[k]==np.random.PCG64(seed).state for k,seed in cfg['PCG64_seeds'].items()),
        'q_error_within_original_tolerance':receipt['max_output_relative_q_error_f32']<=1e-5 and receipt['max_output_relative_q_error_f64']<=1e-5,
        'GPU_within_cap':max(receipt['GPU_allocated_peak_bytes'],receipt['GPU_reserved_peak_bytes'])<=cfg['GPU_limit_bytes'],
        'no_network_h_or_PDG':not receipt['h_PDG_network_inputs'],
    }
    col={k:losses[:,i] for i,k in enumerate(columns)}
    checks['seven_losses_sum']=bool(np.allclose(col['total_loss'],sum(col[k] for k in cfg['loss_weights']),rtol=1e-12,atol=1e-12))
    precision=precision_single_factor_audit(root,cfg,receipt,losses,columns)
    checks.update(precision['checks'])
    fields=['SOURCE_global_range_projection_fraction','REAL_range_projection_fraction',
        'SOURCE_global_projection_RMS_V','REAL_projection_RMS_V']
    return {'pass':all(checks.values()),'checks':checks,
        'weight_file_SHA256':hashlib.sha256((p/'FINAL_FLOW.pt').read_bytes()).hexdigest(),
        'loss_columns':columns,'all_updates_recorded':len(losses),
        'constraint_loss_trace_summary':{k:ev.summary(col[k]) for k in fields},
        'TRAIN_summary_measure':'per-update minibatch estimates, SOURCE weighted by original region mass; not exhaustive frozen TRAIN inference',
        'stream_reconstruction_executed':False,'realized_exposures_uniform':False,
        'equal_parent_global_measure_in_expectation':True,
        'comparison_with_R59_is_single_factor_ablation':precision['pass'],
        'single_factor_precision_order':precision}
