"""Read saved TRAIN records to check the preregistered single-factor comparison."""
from pathlib import Path
import hashlib
import json
import numpy as np
import evaluation as ev


def precision_single_factor_audit(root,cfg,receipt,losses,columns):
    root=Path(root);p=root/cfg['arm'];h=Path(cfg['historical_R59_root'])/'DIRECT_DISTRIBUTION_RESPONSE'
    read=lambda path:json.loads(path.read_text('utf8'))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    historical=read(h/'TRAINING_RECEIPT.json')
    with np.load(p/'PRECISION_ORDER_BY_UPDATE_PRIVATE.npz',allow_pickle=False)as z:
        values=z['values'].copy();names=z['columns'].tolist()
    expected=['original_order_loss','omitted_order_loss','original_nonzero_pair_mass','resolved_pair_fraction']
    assert names==expected and values.shape==(50000,4),(names,values.shape)
    old=values[:,0];omitted=values[:,1];resolved=losses[:,columns.index('source_order')]
    with np.load(p/'PARENT_EXPOSURES_PRIVATE.npz',allow_pickle=False)as z,np.load(h/'PARENT_EXPOSURES_PRIVATE.npz',allow_pickle=False)as oldz:
        fields=['source_ids','source_counts','REAL_reference_counts','REAL_identity_counts','source_bin0','REAL_bin0','source_h','source_original_psd','source_original_q']
        equal={k:bool(np.array_equal(z[k],oldz[k]))for k in fields}
        ids=z['source_ids'].copy();original_psd=z['source_original_psd'].copy()
    with np.load(root/'SOURCE_PSD_PRECISION_PRIVATE.npz',allow_pickle=False)as z:
        lo=z['PSD_interval_low'];hi=z['PSD_interval_high']
        precision_contains=bool(np.array_equal(ids,z['source_ids'])and np.array_equal(original_psd,z['source_original_psd'])and np.isfinite(lo).all()and np.isfinite(hi).all()and np.all(lo<=original_psd)and np.all(original_psd<=hi))
    metadata=read(root/'SOURCE_ORDER_PRECISION_DEFINITION.json')
    integrity=receipt['single_factor_integrity']
    keys=['same_R59_initial_state','all_six_consumed_stream_hashes_equal_R59','all_parent_exposures_equal_R59','final_random_states_equal_R59','frozen_R59_distribution_statistics_reused']
    checks={
        'R59_initial_state_equal':receipt['initial_state_SHA']==historical['initial_state_SHA']==cfg['expected_R59_initial_state_SHA'],
        'R59_all_six_consumed_streams_equal':receipt['stream_hashes']==historical['stream_hashes']==cfg['expected_R59_stream_hashes'],
        'R59_parent_arrays_and_exposures_equal':all(equal.values()),
        'R59_final_PCG64_states_equal':read(p/'FINAL_RANDOM_STATES.json')==read(h/'FINAL_RANDOM_STATES.json'),
        'R59_frozen_distribution_statistics_byte_identical':sha(root/'TRAIN_DISTRIBUTION_CONSTANTS.npz')==sha(Path(cfg['input_paths']['distribution_statistics'])),
        'single_factor_receipt_assertions_match':all(integrity[k]is True for k in keys)and integrity['sole_change']==cfg['sole_scientific_change'],
        'precision_trace_complete_finite_nonnegative':bool(np.isfinite(values).all()and np.all(values>=0)and receipt['precision_order_trace_rows']==50000),
        'original_order_equals_resolved_plus_omitted':bool(np.allclose(old,resolved+omitted,rtol=1e-10,atol=1e-10)),
        'original_pair_mass_and_resolved_fraction_valid':bool(np.all(values[:,2]>0)and np.all(values[:,2]<=1+1e-12)and np.all(values[:,3]<=1+1e-12)),
        'precision_intervals_cover_original_source':precision_contains,
        'precision_rule_frozen_original_denominator':metadata['interval_rule']==receipt['source_order_rule']==cfg['source_order_rule']and metadata['all_original_denominator_pairs_retained']is True and metadata['new_threshold_fit']is False,
    }
    checks={k:bool(v)for k,v in checks.items()}
    return {'pass':all(checks.values()),'checks':checks,'historical_R59_parent_array_checks':equal,
        'precision_trace_columns':names,'rows':len(values),
        'original_minus_resolved_minus_omitted_maxabs':float(np.max(np.abs(old-resolved-omitted))),
        'trace_summary':{n:ev.summary(values[:,i])for i,n in enumerate(names)},
        'source_interval_width_summary':ev.summary(hi-lo),
        'same_actual_consumption_not_only_same_seeds':True,'single_training_seed':True,
        'independent_training_seed_variability_estimated':False,
        'TRAIN_only_saved_records_no_new_forward_or_fit':True}
