from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast
R=Path(__file__).resolve().parents[1];S=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001';C=json.loads((S/'config.json').read_text('utf8'))
check=json.loads((S/'ARTIFICIAL_CHECK_LOCAL.json').read_text('utf8'));binding=json.loads((S/'INPUT_BINDING.json').read_text('utf8'))
assert check['scientific_data_read'] is False and check['response_training_updates']==0
assert check['integrated_loss_and_network']['ten_region_all_seven_losses_finite'] and check['integrated_loss_and_network']['complete_network_box_charge_gradient_finite_nonzero']
assert check['integrated_loss_and_network']['optimizer_updates']==0 and check['worker_peak_RSS_bytes']<4*1024**3
for f in binding['original_acceptance']:assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
registered=json.loads((R/'reviews/R59_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
assert hashlib.sha256((S/'config.json').read_bytes()).hexdigest()==registered['config']['sha256']
assert hashlib.sha256((S/'PROTOCOL.md').read_bytes()).hexdigest()==registered['protocol']['sha256']
for f in check['code_binding']:
    if f['path'].startswith('work/'):
        assert hashlib.sha256((R/f['path']).read_bytes()).hexdigest()==f['sha256'],f['path']
files=list(S.glob('*.py'))+list((R/'reviews').glob('*R59*root*.py'))
for f in files:ast.parse(f.read_text('utf8'),filename=str(f))
review={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','stage':C['stage'],'ready_for_single_registered_training':True,
    'scientific_execution_started':False,'new_response_training_runs_started':0,'model_goal_complete':False,
    'original_A_F_and_registration_hashes_unchanged':True,'only_legal_TRAIN_fields_decoded_by_training_code':['source','target','source_ids','h','source_bin','target_bin'],
    'sampling_review':{'all_SOURCE_REAL_parents_eligible':True,'global_original_equal_parent_expectation':True,'exact_uniform_realized_draw_counts_claimed':False,
        'all_ten_regions_used':True,'source_h_used_only_for_registered_valley_loss':True,'source_h_or_PDG_fed_to_network':False},
    'method_review':{'no_fixed_cross_domain_waveform_targets':True,'no_R58_law_atoms_as_supervision':True,'time_and_state_constant_source_conditioned_field':True,
        'training_analytic_endpoint_and_128_256_Heun_inference_explained':True,'network_call_counts_separate_from_mathematical_field_evaluations':True,
        'box_charge_projection_is_explicit_model_component':True,'internal_constraint_corrections_must_be_reported':True,'same_exact_operator_for_REAL':True,
        'known_Q_population_mismatch_retained_not_claimed_solved':True},
    'implementation_details':{'identity_uses_same_ADC_and_canonical_representation':True,'fixed_training_updates':50000,'checkpoint_selection':'only final50000',
        'rolling_same_run_recovery_every_updates':5000,'rolling_recovery_path':'cache/RECOVERY_SAME_RUN.pt','recovery_is_not_a_candidate_endpoint':True,
        'six_PCG64_streams_with_state_and_chained_actual_bytes_hashes':True,'initial_chain_digest':'32 zero bytes','full_sample_count_and_dense_loss_records':True,
        'CPU_and_CUDA_artificial_checks_before_scientific_training':True,'server_CUDA_artificial_pass':'pending actual process'},
    'artificial_check_receipt':'ARTIFICIAL_CHECK_LOCAL.json','artificial_implementation_fixes_before_science':[
        'The local monitoring helper initially imported unavailable psutil; replaced it with native Windows affinity/peak-memory calls. No scientific data were read.',
        'A toy parent-weight float comparison exceeded its arbitrary 1e-17 threshold by one rounding unit. Replaced the proof with exact Fraction equality and separately reported 1.38778e-17 float roundoff. Original scientific thresholds unchanged.'
    ],
    'local_check':{'torch':check['torch'],'peak_RSS_bytes':check['worker_peak_RSS_bytes'],'CPU_affinity':check['CPU_affinity'],
        'formula':check['result'],'integrated_loss_and_network':check['integrated_loss_and_network']},
    'AST_files':len(files),'code_files':[{'path':f.relative_to(R).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]}
(S/'ROOT_IMPLEMENTATION_REVIEW.json').write_bytes((json.dumps(review,ensure_ascii=False,indent=2)+'\n').encode('utf8'))
print(json.dumps({'ready_for_single_registered_training':True,'AST_files':len(files),'parameters':check['integrated_loss_and_network']['network_parameter_count'],
    'response_training_runs_started':0,'server_CUDA_check_pending':True,'model_goal_complete':False},indent=2))
