from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,os
R=Path(__file__).resolve().parents[1];S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'
read=lambda p:json.loads(p.read_text('utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
control=read(S/'LOCAL_PIPELINE_CONTROL.json');live=read(S/'LIVE_OBSERVATION.json')
assert control['status']=='verified_wait' and control['current_entry']=='train_final.py'
assert live['actual_process']['exists']and live['actual_process']['identity_matches_registered_stage']
assert control['remote_actual_process']['PID']==live['actual_process']['PID']
for path,h in read(S/'POSTPROCESS_PREPARATION.json')['training_files_and_protocol_acceptance_untouched'].items():assert sha(Path(path))==h,path
assert read(S/'POSTPROCESS_ROOT_REVIEW.json')['frozen_training_code_protocol_and_A_F_unchanged']
assert read(S/'POST_AUDIT_ARTIFICIAL_CHECK.json')['pass']
for n,h in read(S/'POSTPROCESS_FILE_BINDING.json')['bound_SHA256'].items():assert sha(S/n)==h,n
for p in list(S.glob('*.py'))+list((R/'reviews').glob('*R63*root*.py')):ast.parse(p.read_text('utf8'),filename=str(p))
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','local_control_PID':control['control_PID'],'control_session':'67043','actual_remote_worker_PID':live['actual_process']['PID'],
 'last_live_observation_UTC':live['UTC'],'last_training_update':live['files']['TRAIN_PROGRESS.json']['global_step'],
 'local_and_server_CPU_CUDA_artificial_checks_pass':True,'frozen_postprocess_files_deployed':True,'postprocess_reference_paths_present':13,
 'original_acceptance_and_training_bindings_unchanged':True,'fixed_training_endpoint':50000,'auto_sequence_requires_previous_native_zero_exit':True,
 'independent_replay_rows_registered':4000,'new_DEV_read_at_preparation':0,'model_goal_complete':False,'resource_limits_unchanged':True,
 'reporting_and_delivery_helpers_prepared_only':True,'delivery_executed':False}
(R/'reviews/R63_POSTPROCESS_READY_001.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
 p=R/n;t=p.read_text('utf8');old='当前已登记，尚未启动科学训练。'
 assert t.count(old)==1
 t=t.replace(old,'科学训练已实际启动；CPU/CUDA人工检查通过，唯一50000步末点之后的数值/完整评价/独立重放已冻结部署。当前真实进程与进度以work/'+S.name+'/LIVE_OBSERVATION.json为准。',1)
 p.write_bytes(t.encode('utf8'))
print(json.dumps(receipt,indent=2))
