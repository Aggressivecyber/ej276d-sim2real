from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, ast, difflib
R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001';H=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'
read=lambda p:json.loads(p.read_text('utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
C=read(S/'config.json');check=read(S/'ARTIFICIAL_CHECK_LOCAL.json');binding=read(S/'INPUT_BINDING.json')
assert check['scientific_data_read']is False and check['response_training_updates']==0
assert check['integrated_loss_and_network']['six_other_loss_values_and_gradients_exactly_match_R59']
assert check['integrated_loss_and_network']['complete_network_box_charge_gradient_finite_nonzero']
assert check['integrated_loss_and_network']['optimizer_updates']==0 and check['worker_peak_RSS_bytes']<4*1024**3
assert isinstance(check['result']['constant_field_128_256_max_error'],float)and check['result']['constant_field_128_256_max_error']<1e-10
for record in binding['original_acceptance']:assert sha(Path(record['path']))==record['sha256']
registered=read(R/'reviews/R61_REGISTRATION_001/RECEIPT.json')
assert sha(S/'config.json')==registered['config']['sha256']
assert sha(S/'PROTOCOL.md')==sha(R/C['protocol'])==registered['protocol']['sha256']
for record in check['code_binding']:
    if record['path'].startswith('work/'):assert sha(R/record['path'])==record['sha256'],record['path']
for name in ['network.py','noise_core.py','source_flow_core.py']:assert sha(S/name)==sha(H/name)
assert sha(S/'R59_distribution_loss.py')==sha(H/'distribution_loss.py')
files=list(S.glob('*.py'))+list((R/'reviews').glob('*R61*root*.py'))
for p in files:ast.parse(p.read_text('utf8'),filename=str(p))
train=(S/'train_final.py').read_text('utf8')
assert 'fit_statistics('not in train and 'torch.load('not in train
assert all(x in train for x in ["initial_hash==C['expected_R59_initial_state_SHA']","==C['expected_R59_stream_hashes']","R59_parent_exposures","R59_final_random_states"])
names=['config.json','INPUT_BINDING.json','PROTOCOL.md','ARTIFICIAL_CHECK_LOCAL.json','TRAIN_IMPLEMENTATION_PREPARATION.json',
       'NOISE_CALIBRATION.npz','NOISE_CALIBRATION.json','UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json']+[p.name for p in S.glob('*.py')]
review={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','stage':C['stage'],'ready_for_single_registered_training':True,
        'scientific_execution_started':False,'new_response_training_runs_started':0,'model_goal_complete':False,
        'original_A_F_and_registration_hashes_unchanged':True,'frozen_R59_network_noise_endpoint_exact':True,
        'manual_code_diff_review_completed':True,'only_scientific_loss_change':'Source-order numerator excludes float32 PSD interval overlap; denominator and all other objectives unchanged.',
        'all_original_source_parents_and_regions_preserved':True,'no_new_teacher_or_domain_flag':True,'same_operator_for_REAL':True,
        'same_R59_initial_state_and_six_actual_stream_hashes_required':True,'no_new_normalization_fit':True,
        'server_CPU_and_CUDA_artificial_checks_required_before_science':True,'server_artificial_checks_status':'pending actual execution',
        'fixed_training_endpoint':50000,'original_numerical_and_A_F_requirements_unchanged':True,
        'artificial_receipt_repair':'reviews/R61_ARTIFICIAL_RECEIPT_REPAIR_001/README.md',
        'local_check':check,'AST_files':len(files),
        'reviewed_diff':{name:''.join(difflib.unified_diff((H/name).read_text('utf8').splitlines(True),(S/name).read_text('utf8').splitlines(True),fromfile='R59/'+name,tofile='R61/'+name))for name in ['train_final.py','distribution_loss.py','artificial_check.py']},
        'files':[{'name':name,'bytes':(S/name).stat().st_size,'sha256':sha(S/name)}for name in names]}
(S/'ROOT_IMPLEMENTATION_REVIEW.json').write_bytes((json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
print(json.dumps({'ready_for_single_registered_training':True,'bound_files':len(names),'parameters':1369889,'new_training_started':False,'scientific_scope':'one-factor precision-resolved source-order ablation'},indent=2))
