from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,difflib
R=Path(__file__).resolve().parents[1];S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001';H=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
read=lambda p:json.loads(p.read_text('utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
C=read(S/'config.json');check=read(S/'ARTIFICIAL_CHECK_LOCAL.json');binding=read(S/'INPUT_BINDING.json');old=read(H/'config.json')
assert check['scientific_data_read']is False and check['response_training_updates']==0
assert check['integrated_loss_and_network']['all_seven_loss_values_and_gradients_exactly_match_R61']
assert check['integrated_loss_and_network']['ten_region_all_nine_losses_finite']and check['integrated_loss_and_network']['complete_network_box_charge_gradient_finite_nonzero']
assert check['integrated_loss_and_network']['optimizer_updates']==0 and check['worker_peak_RSS_bytes']<4*1024**3
for record in binding['original_acceptance']:assert sha(Path(record['path']))==record['sha256']
registered=read(R/'reviews/R63_REGISTRATION_001/RECEIPT.json')
assert sha(S/'config.json')==registered['config']['sha256']
assert sha(S/'PROTOCOL.md')==sha(R/C['protocol'])==registered['protocol']['sha256']
for record in check['code_binding']:
 if record['path'].startswith('work/'):assert sha(R/record['path'])==record['sha256'],record['path']
for name in ['network.py','noise_core.py','source_flow_core.py','precision_order.py','linux_owned_worker.py','NOISE_CALIBRATION.npz','UNIT_CHARGE_CONSTANTS.npz']:assert sha(S/name)==sha(H/name),name
assert sha(S/'R61_distribution_loss.py')==sha(H/'distribution_loss.py') and sha(S/'R61_artificial_check.py')==sha(H/'artificial_check.py')
assert {k:C['PCG64_seeds'][k]for k in old['PCG64_seeds']}==old['PCG64_seeds']
assert {k:C['loss_weights'][k]for k in old['loss_weights']}==old['loss_weights'] and len(C['loss_weights'])==9
assert C['PCG64_seeds']['baseline_projection_directions']==20261011631 and C['updates']==50000
files=list(S.glob('*.py'))+list((R/'reviews').glob('*R63*root*.py'))
for p in files:ast.parse(p.read_text('utf8'),filename=str(p))
train=(S/'train_final.py').read_text('utf8');loss=(S/'distribution_loss.py').read_text('utf8')
assert 'fit_statistics('not in train and 'torch.load('not in train and train.count('fit_temporal_statistics(real,rraw,C)')==1
assert all(x in train for x in ["initial_hash==C['expected_R61_initial_state_SHA']","==C['expected_R61_stream_hashes']","R61_parent_exposures","R61_final_random_states","objective.temporal_direction_calls==50000"])
assert "row[10]"not in train and "row[11]"not in train and "columns.index('max_relative_q_f64')"in train
assert "old_total,pieces,diagnostics=super().__call__"in loss
names=['config.json','INPUT_BINDING.json','PROTOCOL.md','ARTIFICIAL_CHECK_LOCAL.json','NOISE_CALIBRATION.npz','NOISE_CALIBRATION.json','UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json']+[p.name for p in S.glob('*.py')]
review={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','stage':C['stage'],'ready_for_single_registered_training':True,'scientific_execution_started':False,
 'new_response_training_runs_started':0,'model_goal_complete':False,'original_A_F_and_registration_hashes_unchanged':True,
 'frozen_R61_network_noise_endpoint_precision_order_exact':True,'manual_source_and_driver_diff_review_completed':True,
 'scientific_objective_extension':C['sole_scientific_change'],'all_original_source_parents_and_regions_preserved':True,'no_new_teacher_or_domain_flag':True,'same_operator_for_REAL':True,
 'same_R61_initial_state_and_original_six_actual_stream_hashes_required':True,'new_temporal_statistics_only_from_REAL_TRAIN_before_updates':True,
 'server_CPU_and_CUDA_artificial_checks_required_before_science':True,'server_artificial_checks_status':'pending actual execution',
 'fixed_training_endpoint':50000,'local_check':check,'AST_files':len(files),
 'reviewed_driver_diff':''.join(difflib.unified_diff((H/'train_final.py').read_text('utf8').splitlines(True),train.splitlines(True),fromfile='R61/train_final.py',tofile='R63/train_final.py')),
 'files':[{'name':n,'bytes':(S/n).stat().st_size,'sha256':sha(S/n)}for n in names]}
(S/'ROOT_IMPLEMENTATION_REVIEW.json').write_bytes((json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
print(json.dumps({'ready_for_single_registered_training':True,'bound_files':len(names),'parameters':check['integrated_loss_and_network']['network_parameter_count'],'new_training_started':False,'scientific_scope':'preregistered_two_term_coverage_extension'},indent=2))
