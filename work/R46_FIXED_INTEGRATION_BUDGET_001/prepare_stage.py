from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, os, shutil, tarfile

R=Path(__file__).resolve().parent;P=R.parent.parent;OLD=R.parent/'R45_EXTENDED_BUDGET_NEURAL_FLOW_001'
REG=json.loads((P/'reviews/R46_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
REMOTE='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R46_REMOTE_20261010_014939_UTC'
cfg=json.loads((OLD/'config_remote.json').read_text('utf8'))
cfg.update(stage=R.name,registered_UTC=REG['registered_UTC'],deadline_UTC=REG['executor_seal_deadline_UTC'],
 absolute_deadline_UTC=REG['stage_deadline_UTC'],remote_root=REMOTE,CPU_threads=12,threads=12,
 science_budget=14400,numeric_wall_limit_s=14400,training_total_limit_s=0,training_wall_s=0,
 RSS_limit_bytes=48*1024**3,GPU_limit_bytes=0,output_limit_bytes=4*1024**3,disk_free_reserve_bytes=8*1024**3,
 updates=0,HGB_max_fits=40,HGB_cap=40,HGB_plan=40,density_EM_max=32,width_slots=32,
 solver_steps=64,solver_check_steps=128,generation_chunk=64,main_generation_rows=8000,replay_rows=8000,
 new_response_training_runs=0,parameter_initialization='frozen R43/R45 model; no optimizer load or updates',
 execution={'training':'none','numeric_generation_evaluation_replay':'server CPU12/chunk64',
   'local_auxiliary':'CPU4 E/D','actual_PID_nonce_RSS_before_imports':True})
short='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R43_REMOTE_20261009_233131_UTC/SCALAR_TIME_FLOW_FINAL/FINAL_FLOW.pt'
long='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R45_REMOTE_20261010_010004_UTC/EXTENDED_FLOW_FINAL/FINAL_FLOW.pt'
cfg['arms']=['SHORT_2950_HEUN64','LONG_50000_HEUN64']
cfg['endpoints']=['SHORT_2950_HEUN64','SHORT_2950_REAL_CONTROL','LONG_50000_HEUN64','LONG_50000_REAL_CONTROL']
cfg['endpoint_weight_paths']=dict(zip(cfg['endpoints'],[short,short,long,long]))
cfg['numerical_order']=['SHORT_2950_HEUN64','LONG_50000_HEUN64']
cfg['numerical_weights_registered']={k:REG['input_bindings'][j] for k,j in [('SHORT_2950_HEUN64',0),('LONG_50000_HEUN64',1)]}
cfg['require_both_pass_before_DEV']=True
(R/'config.json').write_bytes((json.dumps(cfg,indent=2)+'\n').encode())
cfg['constants_path']=REMOTE+'/UNIT_CHARGE_CONSTANTS.npz'
(R/'config_remote.json').write_bytes((json.dumps(cfg,indent=2)+'\n').encode())
records=[]
for n in ['network.py','source_flow_core.py','linux_supervise.py','linux_owned_worker.py','launch_server.py','supervise.py','owned_worker.py']:
 raw=(OLD/n).read_bytes();text=raw.decode('utf8').replace('\r\r\n','\n').replace('\r\n','\n').replace('\r','\n')
 before=ast.dump(ast.parse(text),include_attributes=False)
 if n in ['linux_supervise.py','launch_server.py']:
  text=text.replace("['train_extended.py','check_train_solver.py','generate_all.py','evaluate_saved.py','replay_all.py']",
    "['check_models.py','generate_all.py','evaluate_saved.py','replay_all.py']")
 if n=='linux_supervise.py':
  text=text.replace("if meminfo['MemAvailable']<60129542144:raise RuntimeError('resource admission RAM below56GiB; no science submitted')",
   "if meminfo['MemAvailable']<60129542144:\n    cfg['RSS_limit_bytes']=min(cfg['RSS_limit_bytes'],max(0,meminfo['MemAvailable']-8*1024**3))\n    startResource['RSS_quota_bytes']=cfg['RSS_limit_bytes'];startResource['lower_quota_reason']='MemAvailable below56GiB'\n    write('SERVER_START_RESOURCE.json',startResource)\n    if cfg['RSS_limit_bytes']<4*1024**3:raise RuntimeError('insufficient lower RAM quota')")
  # Refresh only our project's actual argv/cwd before starting each scientific child.
  marker="(ROOT/(entry+'.used')).write_text(utc(),'utf8')"
  inserted="""own=[]
for d in Path('/proc').iterdir():
    if not d.name.isdigit() or int(d.name)==os.getpid():continue
    try:
        a=(d/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace');c=str((d/'cwd').resolve())
        if 'python' in a and ('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305' in a or c.startswith('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305')):own.append({'PID':int(d.name),'argv':a,'cwd':c})
    except(OSError,PermissionError):pass
startResource['other_own_project_jobs']=own;write('SERVER_START_RESOURCE.json',startResource)
if own:raise RuntimeError(('other project science status must be resolved',own))
"""
  text=text.replace(marker,inserted+marker)
 ast.parse(text);(R/n).write_bytes(text.encode('utf8'))
 if n in ['network.py','source_flow_core.py','linux_owned_worker.py','supervise.py','owned_worker.py']:
  assert before==ast.dump(ast.parse(text),include_attributes=False)
 records.append({'name':n,'origin':str(OLD/n),'origin_sha256':hashlib.sha256(raw).hexdigest(),
  'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'LF_only':True,
  'scientific_core_AST_unchanged':n in ['network.py','source_flow_core.py']})
for n in ['R42_TRAIN_SOLVER_CHECK_PRIVATE.npz','UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json']:
 shutil.copy2(OLD/n,R/n)
protocol='''# R46 fixed integration budget

Authority: tasks/R46_FIXED_INTEGRATION_BUDGET_001.md, registration, numerical protocol v2 / acceptance file version3. SHORT2950 control then LONG50000 primary, frozen weights/constants. No optimizer, parameter update, new Monte Carlo TRAIN state or model selection. CPU12/chunk64, float64 coordinates and float32 network, candidate64/reference128. Reuse exact128 TRAIN rows and white. Each must satisfy median<=.25LSB,p95<=1LSB and all f64/f32 q errors<=1e-5. Both checks completed even if either numerical result fails; any failure closes this recipe before DEV. No256/solver search.

If both pass: two sets of SOURCE2000+REAL2000+joint1024+probe832+source perturb128, total11968. HGB40/EM32 maximum, fixed600/250 and500 group bootstrap; old results/reference fits reused. SHORT is not selected as fallback. LONG vs SHORT is equal inference and unequal historical training budget. Same CPU12/chunk64 independent8000 main replay. Old six scientific gates plus source population and same-treated REAL semantics retained. Prepared code is not execution evidence.

New independent server leaf R46_REMOTE_20261010_014939_UTC. One scientific child, actual PID+nonce acknowledged before imports; tree RSS48GiB or recorded lower quota if memory admission falls below56GiB, CPU12, GPU jobs0, disk8GiB reserve, output4GiB, science14400s, executor06:49:39.896743UTC. Local auxiliariesCPU4/RSS4GiB and all new files/cache E/D. Normal transfer/native exits, no migration content consistency tests, installation, agents or Git. R43–R45 sealed originals immutable.
'''
(R/'PROTOCOL.md').write_bytes(protocol.encode())
(R/'INPUT_BINDINGS.json').write_bytes((json.dumps({'registration_bindings':REG['input_bindings'],
 'scientific_source_copies':records,'no_source_or_destination_migration_consistency_test':True},indent=2)+'\n').encode())
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'remote_root':REMOTE,
 'scientific_core_AST_unchanged':True,'CPU_numeric_source_ready':(R/'check_models.py').is_file(),
 'scientific_array_decode_or_forward_fit':0,'DEV_body_copies':0,'source_records':records}
(R/'PREPARATION_RECEIPT.json').write_bytes((json.dumps(rec,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in rec.items() if k!='source_records'}))
