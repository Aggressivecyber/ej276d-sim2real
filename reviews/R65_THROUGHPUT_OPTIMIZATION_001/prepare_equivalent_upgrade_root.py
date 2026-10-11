"""Prepare, never install, one explicitly recorded implementation upgrade."""
from pathlib import Path
import ast
import copy
import ctypes
import hashlib
import json

R = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
S = R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
D = P/'prepared_upgrade'
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(), 8)
assert not D.exists()

def sha(body):
    return hashlib.sha256(body).hexdigest()

review = json.loads((S/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))
for row in review['files']:
    assert sha((S/row['name']).read_bytes()) == row['sha256'], row['name']
gpu = json.loads((P/'attempt_001/CUDA_ARTIFICIAL_RECEIPT.json').read_text('utf8'))
assert gpu['status'] == 'passed' and all(row['byte_equal'] for row in gpu['checks'])
assert gpu['actual64_forward_plus_VJP_benchmark']['candidate_fractional_time_reduction'] > .15
guard = json.loads((P/'attempt_001/GUARD_RECEIPT.json').read_text('utf8'))
assert guard['benchmark_native_exit'] == 0 and guard['training_resumed']

helper_path = S/'network_update_check/network_update_check.py'
old_helper = helper_path.read_text('utf8')
candidate = (P/'agent_candidate/gradient_pack_candidate.py').read_text('utf8')
candidate_ast = ast.parse(candidate)
candidate_functions = [ast.get_source_segment(candidate,node) for node in candidate_ast.body
                       if isinstance(node,ast.FunctionDef) and node.name in ['_flatten_legacy','flatten_gradients_cpu_f64']]
assert len(candidate_functions) == 2
new_function = '\n\n'.join(candidate_functions).replace('def flatten_gradients_cpu_f64(', 'def _flatten_gradients_cpu_f64(')
start = old_helper.index('def _flatten_gradients_cpu_f64(')
stop = old_helper.index('\n\ndef parameter_vjp_cpu_f64(', start)
new_helper = old_helper[:start]+new_function+old_helper[stop:]
ast.parse(new_helper)

old_train = (S/'train_final.py').read_text('utf8')
new_train = old_train
old_anchor = "    request_path = R/'RECOVERY_REQUEST.json'\n    recovery_requested = request_path.exists()\n    recovery = None\n"
new_anchor = '''    request_path = R/'RECOVERY_REQUEST.json'
    recovery_requested = request_path.exists()
    assert recovery_requested, 'This equivalent implementation revision only continues the existing run'
    checkpoint_frozen_binding = frozen_binding
    implementation_upgrade_applied = None
    requested_binding = read('RECOVERY_REQUEST.json')['frozen_binding']
    if requested_binding != frozen_binding:
        upgrade = read('IMPLEMENTATION_UPGRADE.json')
        assert upgrade['schema_version'] == 1 and upgrade['kind'] == 'equivalent_gradient_pack_only'
        assert upgrade['prior_frozen_binding'] == requested_binding
        assert requested_binding['implementation_review_SHA'] == 'e0a4b101886f5cb10b205364fe8414ca2db1fa211cbb058de7d4113f85ba24f5'
        assert all(requested_binding[key] == frozen_binding[key] for key in
                   ('protocol_SHA','config_SHA','input_binding_SHA','environment'))
        assert upgrade['logical_run_id'] == logical_run_id and upgrade['scientific_parameters_changed'] is False
        assert upgrade['checkpoint_SHA256'] == read('RECOVERY_REQUEST.json')['checkpoint_SHA256']
        assert len(upgrade['changed_source_files']) == 3 and {record['name'] for record in upgrade['changed_source_files']} == {
            'train_final.py','network_update_check/network_update_check.py','network_update_check/SOURCE_BINDING.json'}
        for record in upgrade['changed_source_files']:
            assert sha(R/record['name']) == record['after_SHA256'], record['name']
            archived = (R/record['before_archive_relative_path']).resolve(strict=True)
            assert (R/'native_attempts').resolve() in archived.parents
            assert sha(archived) == record['before_SHA256'], record['name']
        for kind in ('CPU','CUDA'):
            evidence = upgrade['equivalence_evidence'][kind]
            assert sha(R/evidence['path']) == evidence['SHA256']
            assert read(evidence['path'])['status'] == 'passed'
        checkpoint_frozen_binding = requested_binding
        implementation_upgrade_applied = {
            'declaration_path':'IMPLEMENTATION_UPGRADE.json', 'declaration_SHA256':sha(R/'IMPLEMENTATION_UPGRADE.json'),
            'from_implementation_review_SHA':requested_binding['implementation_review_SHA'],
            'to_implementation_review_SHA':frozen_binding['implementation_review_SHA'],
            'same_protocol_config_inputs_environment':True, 'checkpoint_bytes_modified':False,
            'gradient_pack_only':True, 'scientific_parameters_changed':False,
        }
        # Validate the new implementation before deserializing the original checkpoint.
        from artificial_suite import run_all
        suite = run_all(R, C, frozen_binding['implementation_review_SHA'])
        assert suite['pass']
        write('ARTIFICIAL_SUITE_RECEIPT.json', suite)
    recovery = None
'''
assert new_train.count(old_anchor) == 1
new_train = new_train.replace(old_anchor, new_anchor)
for old, new in [
    ("assert request['frozen_binding'] == frozen_binding", "assert request['frozen_binding'] == checkpoint_frozen_binding"),
    ("assert publication['logical_run_id'] == logical_run_id and publication['frozen_binding'] == frozen_binding", "assert publication['logical_run_id'] == logical_run_id and publication['frozen_binding'] == checkpoint_frozen_binding"),
    ("assert recovery['logical_run_id'] == logical_run_id and recovery['frozen_binding'] == frozen_binding", "assert recovery['logical_run_id'] == logical_run_id and recovery['frozen_binding'] == checkpoint_frozen_binding"),
    ("'bindings_verified':True,'previous_PID_confirmed_dead':True,", "'bindings_verified':True,'previous_PID_confirmed_dead':True,\n            'implementation_upgrade':implementation_upgrade_applied,"),
    ("        assert np.all(countsDS == updates) and np.all(countsDR == updates)\n    else:", "        assert np.all(countsDS == updates) and np.all(countsDR == updates)\n        write('SAME_RUN_RESTORE_RECEIPT.json', {\n            'UTC':utc(),'PID':os.getpid(),'native_attempt_id':attempt_id,'logical_run_id':logical_run_id,\n            'committed_update_restored':updates,'checkpoint_SHA256':request['checkpoint_SHA256'],\n            'model_state_SHA256_after_restore':state_hash(model.state_dict()),\n            'model_state_SHA256_in_checkpoint':state_hash(recovery['model']),\n            'implementation_upgrade':implementation_upgrade_applied,\n            'all_eight_random_states_and_original_optimizer_restored':True,\n            'current_frozen_binding':frozen_binding,\n        })\n    else:"),
    ("'native_attempt_id':attempt_id,'frozen_binding':frozen_binding,\n        'source_parents'", "'native_attempt_id':attempt_id,'frozen_binding':frozen_binding,\n        'implementation_upgrade':implementation_upgrade_applied,\n        'source_parents'"),
]:
    assert new_train.count(old) == 1, old
    new_train = new_train.replace(old,new)
tree = ast.parse(new_train)
def main_update_loop(source):
    found = [n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
             and n.target.id == 'step' and isinstance(n.iter,ast.Call) and getattr(n.iter.func,'id',None) == 'range']
    assert len(found) == 1
    return ast.dump(found[0], include_attributes=False)
assert main_update_loop(old_train) == main_update_loop(new_train)

D.mkdir()
(D/'network_update_check').mkdir()
(D/'train_final.py').write_text(new_train, encoding='utf8', newline='\n')
(D/'network_update_check/network_update_check.py').write_text(new_helper, encoding='utf8', newline='\n')
binding = json.loads((S/'network_update_check/SOURCE_BINDING.json').read_text('utf8'))
binding['helper_sha256'] = sha((D/'network_update_check/network_update_check.py').read_bytes())
(D/'network_update_check/SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n',encoding='utf8',newline='\n')
record = {'status':'prepared_not_installed','changed_source_files':[], 'scientific_training_loop_AST_unchanged':True,
          'same_parameter_VJP_and_C_reverse_and_clip_Adam_callers':True,
          'only_parameter_pack_implementation_and_explicit_binding_migration_changed':True,
          'live_training_files_changed':False, 'checkpoint_bytes_modified':False}
for name in ['train_final.py','network_update_check/network_update_check.py','network_update_check/SOURCE_BINDING.json']:
    record['changed_source_files'].append({'name':name,'before_SHA256':sha((S/name).read_bytes()),'after_SHA256':sha((D/name).read_bytes())})
(D/'PREPARATION_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps(record))
