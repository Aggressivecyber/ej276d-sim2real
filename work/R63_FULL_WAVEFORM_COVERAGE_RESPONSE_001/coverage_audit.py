"""Saved-record audit; reconstruct only the new independent direction stream."""
from pathlib import Path
import hashlib,json
import numpy as np
import evaluation as ev

def reconstruct_temporal_stream(seed,updates):
    rng=np.random.Generator(np.random.PCG64(seed));digest=bytes(32)
    for _ in range(updates):
        p=rng.standard_normal((31,64));p/=np.linalg.norm(p,axis=0,keepdims=True);p=p.astype('<f4')
        digest=hashlib.sha256(digest+p.tobytes()).digest()
    return digest.hex(),rng.bit_generator.state

def coverage_extension_audit(root,cfg,receipt,losses,columns):
    root=Path(root);p=root/cfg['arm'];h=Path(cfg['historical_R61_root'])/'PRECISION_RESOLVED_RESPONSE';updates=cfg['updates']
    read=lambda path:json.loads(path.read_text('utf8'));sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    historical=read(h/'TRAINING_RECEIPT.json');expected=cfg['expected_R61_stream_hashes'];stream='baseline_projection_directions'
    rng1=read(p/'FINAL_RANDOM_STATES.json');old_rng1=read(h/'FINAL_RANDOM_STATES.json')
    with np.load(p/'PRECISION_ORDER_BY_UPDATE_PRIVATE.npz',allow_pickle=False)as z:values=z['values'].copy();names=z['columns'].tolist()
    assert names==['original_order_loss','omitted_order_loss','original_nonzero_pair_mass','resolved_pair_fraction']and values.shape==(updates,4)
    original=values[:,0];omitted=values[:,1];resolved=losses[:,columns.index('source_order')]
    with np.load(p/'PARENT_EXPOSURES_PRIVATE.npz',allow_pickle=False)as z,np.load(h/'PARENT_EXPOSURES_PRIVATE.npz',allow_pickle=False)as oldz:
        fields=['source_ids','source_counts','REAL_reference_counts','REAL_identity_counts','source_bin0','REAL_bin0','source_h','source_original_psd','source_original_q']
        equal={k:bool(np.array_equal(z[k],oldz[k]))for k in fields};ids=z['source_ids'].copy();original_psd=z['source_original_psd'].copy()
    with np.load(root/'SOURCE_PSD_PRECISION_PRIVATE.npz',allow_pickle=False)as z:
        lo=z['PSD_interval_low'];hi=z['PSD_interval_high']
        contains=bool(np.array_equal(ids,z['source_ids'])and np.array_equal(original_psd,z['source_original_psd'])and np.isfinite(lo).all()and np.isfinite(hi).all()and np.all(lo<=original_psd)and np.all(original_psd<=hi))
    with np.load(root/'SOURCE_PSD_PRECISION_PRIVATE.npz',allow_pickle=False)as z,np.load(h.parent/'SOURCE_PSD_PRECISION_PRIVATE.npz',allow_pickle=False)as oldz:
        interval_equal=all(np.array_equal(z[k],oldz[k])for k in z.files)
    meta=read(root/'SOURCE_ORDER_PRECISION_DEFINITION.json');integrity=receipt['objective_extension_integrity']
    keys=['same_R61_initial_state','all_six_consumed_stream_hashes_equal_R61','all_parent_exposures_equal_R61','original_six_final_random_states_equal_R61','frozen_R61_distribution_statistics_reused','seventh_direction_stream_recorded_independently']
    with np.load(p/'COVERAGE_BY_UPDATE_PRIVATE.npz',allow_pickle=False)as z:coverage=z['values'].copy();cn=z['columns'].tolist()
    expected_columns=['original_seven_total','coordinate_RMS','temporal_joint','temporal_marginal','temporal_sliced',*[n+'_coordinate_RMS'for n in cfg['feature_names']]]
    assert cn==expected_columns and coverage.shape==(updates,9)
    col={n:losses[:,i]for i,n in enumerate(columns)};cover={n:coverage[:,i]for i,n in enumerate(cn)}
    old_losses=[k for k in cfg['loss_weights']if k not in ['coordinate_RMS','temporal_joint']]
    with np.load(root/'TRAIN_TEMPORAL_CONSTANTS.npz',allow_pickle=False)as z:temporal={k:z[k].copy()for k in z.files}
    with np.load(root/'TRAIN_DISTRIBUTION_CONSTANTS.npz',allow_pickle=False)as z:real_counts=z['REAL_counts'].copy()
    tm=read(root/'TRAIN_TEMPORAL_CONSTANTS.json');center=temporal['temporal_center'];scale=temporal['temporal_scale']
    stream_hash,stream_state=reconstruct_temporal_stream(cfg['PCG64_seeds'][stream],updates)
    checks={
        'R61_initial_state_equal':receipt['initial_state_SHA']==historical['initial_state_SHA']==cfg['expected_R61_initial_state_SHA'],
        'R61_original_six_consumed_streams_equal':{k:receipt['stream_hashes'][k]for k in expected}==historical['stream_hashes']==expected,
        'R61_parent_arrays_and_exposures_equal':all(equal.values()),
        'R61_original_six_final_PCG64_states_equal':{k:rng1[k]for k in expected}==old_rng1,
        'R61_frozen_distribution_statistics_byte_identical':sha(root/'TRAIN_DISTRIBUTION_CONSTANTS.npz')==sha(Path(cfg['input_paths']['distribution_statistics'])),
        'extension_receipt_assertions_match':all(integrity[k]is True for k in keys)and integrity['sole_change']==cfg['sole_scientific_change'],
        'precision_trace_complete_finite_nonnegative':np.isfinite(values).all()and np.all(values>=0)and receipt['precision_order_trace_rows']==updates,
        'original_order_equals_resolved_plus_omitted':np.allclose(original,resolved+omitted,rtol=1e-10,atol=1e-10),
        'original_pair_mass_and_resolved_fraction_valid':np.all(values[:,2]>0)and np.all(values[:,2]<=1+1e-12)and np.all(values[:,3]<=1+1e-12),
        'precision_intervals_cover_original_source_and_match_R61':contains and interval_equal,
        'precision_rule_frozen_original_denominator':meta['interval_rule']==receipt['source_order_rule']==cfg['source_order_rule']and meta['all_original_denominator_pairs_retained']is True and meta['new_threshold_fit']is False,
        'coverage_trace_complete_finite_nonnegative':np.isfinite(coverage).all()and np.all(coverage>=0)and receipt['coverage_trace_rows']==updates,
        'original_seven_trace_decomposition':len(old_losses)==7 and np.allclose(cover['original_seven_total'],sum(col[k]for k in old_losses),rtol=1e-12,atol=1e-12),
        'new_terms_trace_match':all(np.array_equal(cover[k],col[k])for k in ['coordinate_RMS','temporal_joint']),
        'temporal_half_sum':np.allclose(cover['temporal_joint'],.5*(cover['temporal_marginal']+cover['temporal_sliced']),rtol=1e-12,atol=1e-12),
        'coordinate_four_view_mean':np.allclose(cover['coordinate_RMS'],coverage[:,5:].mean(1),rtol=1e-12,atol=1e-12),
        'nine_terms_sum':np.allclose(col['total_loss'],coverage[:,:3].sum(1),rtol=1e-12,atol=1e-12),
        'temporal_constants_shape_finite_floors':center.shape==scale.shape==(10,31)and np.isfinite(center).all()and np.isfinite(scale).all()and np.all(scale>=np.asarray(cfg['temporal_scale_floor'])),
        'temporal_constants_saved_metadata_equal':np.array_equal(center,tm['center'])and np.array_equal(scale,tm['scale'])and np.array_equal(temporal['temporal_REAL_counts'],real_counts)and int(real_counts.sum())==tm['fitted_REAL_TRAIN_rows']==cfg['REAL_N'],
        'temporal_constants_hash_and_preupdate_record':sha(root/'TRAIN_TEMPORAL_CONSTANTS.npz')==tm['new_statistics_SHA256']and sha(root/'TRAIN_DISTRIBUTION_CONSTANTS.npz')==tm['old_statistics_SHA256']and tm['fitted_before_first_update']is True and tm['scientific_response_fit']is False and tm['teacher_fit']is False,
        'independent_seventh_stream_consumption_reconstructed':stream_hash==receipt['stream_hashes'][stream]and stream_state==rng1[stream]and receipt['temporal_direction_calls']==updates,
        'all_nine_outer_weights_one':len(cfg['loss_weights'])==9 and all(v==1. for v in cfg['loss_weights'].values()),
    }
    checks={k:bool(v)for k,v in checks.items()}
    return {'pass':all(checks.values()),'checks':checks,'historical_R61_parent_array_checks':equal,'precision_trace_columns':names,'coverage_trace_columns':cn,'rows':updates,
        'original_minus_resolved_minus_omitted_maxabs':float(np.max(np.abs(original-resolved-omitted))),
        'precision_trace_summary':{n:ev.summary(values[:,i])for i,n in enumerate(names)},'coverage_trace_summary':{n:ev.summary(coverage[:,i])for i,n in enumerate(cn)},
        'source_interval_width_summary':ev.summary(hi-lo),'same_original_six_actual_consumption_not_only_same_seeds':True,
        'new_seventh_stream_independently_reconstructed':True,'new_seventh_stream_SHA256':stream_hash,'single_training_seed':True,'independent_training_seed_variability_estimated':False,
        'saved_records_only_no_new_scientific_forward_or_fit':True}
