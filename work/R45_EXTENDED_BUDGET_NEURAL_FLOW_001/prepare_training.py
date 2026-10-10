from pathlib import Path
from datetime import datetime, timezone
import ast, json, hashlib, tarfile, os, shutil

R=Path(__file__).resolve().parent; P=R.parent.parent
old=P/'work/R43_SCALAR_TIME_NEURAL_FLOW_001'
pre=json.loads((R/'SERVER_PREFLIGHT.json').read_text())
reg=json.loads((P/'reviews/R45_REGISTRATION_001/RECEIPT.json').read_text())
remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R45_REMOTE_20261010_010004_UTC'
old_remote='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R43_REMOTE_20261009_233131_UTC'
for name in ['network.py','source_flow_core.py']:
    shutil.copy2(old/name,R/name)
for name in ['linux_supervise.py','linux_owned_worker.py']:
    shutil.copy2(P/'work/R44_PSD_VELOCITY_FIT_DIAGNOSTIC_001'/name,R/name)
cfg=json.loads((old/'config.json').read_text())
cfg.update(stage=reg['stage'],registered_UTC=reg['registered_UTC'],deadline_UTC=reg['executor_seal_deadline_UTC'],
    absolute_deadline_UTC=reg['stage_deadline_UTC'],remote_root=remote_root,CPU_threads=12,CPU_local_threads=4,
    RSS_limit_bytes=48*1024**3,GPU_limit_bytes=10*1024**3,GPU_allocated_limit_bytes=10*1024**3,
    science_budget=14400,training_wall_s=10800,training_total_limit_s=10800,numeric_wall_limit_s=14400,
    output_limit_bytes=4*1024**3,disk_free_reserve_bytes=8*1024**3,
    inherited_updates=2950,new_updates=47050,total_updates=50000,updates=50000,
    arms=['EXTENDED_FLOW_FINAL'],endpoints=['EXTENDED_FLOW_FINAL','EXTENDED_FLOW_REAL_CONTROL'],
    selection='sole total50000; original fixed128 TRAIN solver prerequisite; no intermediate DEV',
    objective='only original full480 standardized tangent velocity MSE',
    parameter_initialization='inherit R43 final model and Adam2950; not new weights',
    training_t_rule='PCG64 restore R43 final_t_RNG; uniform32.astype(float32) each new update',
    training_white_seed=20261010452,probe_selection_seed=20261010453,probe_t_seed=20261010454,probe_white_seed=20261010455,
    probe_selection_order='one PCG64 flow; bin0..7 choose512 occurrence positions without replacement',
    execution={'training':'server CUDA CPU12 RSS48GiB/GPU10GiB','generation_evaluation_replay':'same server CPU12 chunk64',
               'local_auxiliary':'CPU4 E/D','actual_PID_nonce_RSS_before_imports':True},
    old_G_reader_D_used=False,old_response_weights_loaded=True,resource_policy_version=2,
    first_actual_command=json.loads((R/'FIRST_CONTROL_RECEIPT.json').read_text()))
cfg['R43_root']=str(old)
cfg['TRAIN_path']=cfg['training_input_paths']['TRAIN_PRIVATE.npz']
cfg['inherited_weight']=str(old/'SCALAR_TIME_FLOW_FINAL/FINAL_FLOW.pt')
cfg['inherited_optimizer']=str(old/'SCALAR_TIME_FLOW_FINAL/FINAL_OPTIMIZER.pt')
cfg['inherited_coupling']=str(old/'SCALAR_TIME_FLOW_FINAL/ACTUAL_CFM_STREAM_PRIVATE.npz')
cfg['inherited_train_receipt']=str(old/'SCALAR_TIME_FLOW_FINAL/TRAINING_RECEIPT.json')
cfg['constants_from']=str(old/'UNIT_CHARGE_CONSTANTS.npz');cfg['constants_json_from']=str(old/'UNIT_CHARGE_CONSTANTS.json')
cfg['constants_path']=str(R/'UNIT_CHARGE_CONSTANTS.npz')
cfg['endpoint_weight_paths']={n:str(R/'EXTENDED_FLOW_FINAL/FINAL_FLOW.pt')for n in cfg['endpoints']}
remote=cfg.copy()
remote['R43_root']=old_remote
remote['TRAIN_path']=pre['R43_config']['training_input_paths']['TRAIN_PRIVATE.npz']
for key,name in [('inherited_weight','SCALAR_TIME_FLOW_FINAL/FINAL_FLOW.pt'),('inherited_optimizer','SCALAR_TIME_FLOW_FINAL/FINAL_OPTIMIZER.pt'),
                 ('inherited_coupling','SCALAR_TIME_FLOW_FINAL/ACTUAL_CFM_STREAM_PRIVATE.npz'),('inherited_train_receipt','SCALAR_TIME_FLOW_FINAL/TRAINING_RECEIPT.json'),
                 ('constants_from','UNIT_CHARGE_CONSTANTS.npz'),('constants_json_from','UNIT_CHARGE_CONSTANTS.json')]:
    remote[key]=old_remote+'/'+name
remote['constants_path']=remote_root+'/UNIT_CHARGE_CONSTANTS.npz'
remote['endpoint_weight_paths']={n:remote_root+'/EXTENDED_FLOW_FINAL/FINAL_FLOW.pt'for n in cfg['endpoints']}
(R/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(R/'config_remote.json').write_text(json.dumps(remote,indent=2)+'\n')
(R/'INPUT_BINDINGS.json').write_text(json.dumps({'from_registration':reg['input_bindings'],'task_SHA':reg['task_sha256'],
    'source_versions_frozen_from_R43':True,'no_new_whole_source_or_migration_hash_audit':True,
    'server_training_inputs':{k:remote[k]for k in ['TRAIN_path','inherited_weight','inherited_optimizer','inherited_coupling','inherited_train_receipt','constants_from','constants_json_from']}},indent=2)+'\n')
g=(R/'linux_supervise.py').read_text()
g=g.replace("['diagnose_frozen.py']","['train_extended.py','check_train_solver.py','generate_all.py','evaluate_saved.py','replay_all.py']")
g=g.replace("limit=min(remaining,datetime.fromisoformat(cfg['deadline_UTC']).timestamp()-time.time()-120)",
    "training_used=sum(r['elapsed_s']for r in history if r['entry'].startswith('train_'))\nentry_limit=cfg['training_total_limit_s']-training_used if entry.startswith('train_')else remaining\nlimit=min(remaining,entry_limit,datetime.fromisoformat(cfg['deadline_UTC']).timestamp()-time.time()-600)")
g=g.replace("if meminfo['MemAvailable']<60129542144 or int(gpuinfo)*1048576<11811160064 or otherGPU:raise RuntimeError('resource admission not satisfied; no forward submitted')",
    "if meminfo['MemAvailable']<60129542144:raise RuntimeError('resource admission RAM below56GiB; no science submitted')\nif entry.startswith('train_')and(int(gpuinfo)*1048576<11811160064 or otherGPU):raise RuntimeError('GPU admission not satisfied; no science submitted')\nimport shutil\nif shutil.disk_usage(ROOT).free<cfg['disk_free_reserve_bytes']:raise RuntimeError('disk reserve admission failed')")
g=g.replace("last_GPU_query=0.","last_GPU_query=0.;last_progress_query=0.;last_progress_step=None;last_disk_query=0.")
g=g.replace("while p.poll()is None:\n            if row", "while p.poll()is None:\n            if time.monotonic()-last_progress_query>=30:\n                last_progress_query=time.monotonic()\n                progress=ROOT/'TRAIN_PROGRESS.json'\n                if progress.exists():\n                    try:\n                        item=json.loads(progress.read_text())\n                        if item.get('global_step')!=last_progress_step:\n                            last_progress_step=item.get('global_step');print(json.dumps({'actual_training_progress':item}),flush=True)\n                    except(json.JSONDecodeError,OSError):pass\n            if time.monotonic()-last_disk_query>=20:\n                last_disk_query=time.monotonic()\n                if shutil.disk_usage(ROOT).free<cfg['disk_free_reserve_bytes']:reason='disk_free_reserve_limit'\n                if sum(q.stat().st_size for q in ROOT.rglob('*')if q.is_file())>cfg['output_limit_bytes']:reason='stage_output_limit'\n            if row")
ast.parse(g);(R/'linux_supervise.py').write_text(g)
protocol='''# R45 fixed training-budget experiment\n\nRead the original task and registration as authority. Extend the actual R43 model+Adam2950 to50000, exactly47050 new updates. Original94400 source/target/matched-target/bin/ordinal fixed; global_step k uses batch(k-1)%2950. Original constants, projection, float32 network/float64 coordinates, MSE, lr.0002, Adam(.9,.999),eps1e-8,clip5 unchanged. Additional t resumes final_t_RNG; white independent PCG64(20261010452). Save per-batch hashes, fixed replay draw algorithm, start/final RNG, every5000 engineering checkpoint, exposure counts. Intermediates are not candidates.\n\nProbe4096 states: one independent selection PCG64(20261010453), bins0..7 in order,512 without replacement from each occurrence pool; independent t/white20261010454/455. Evaluate2950 then5000..50000 with chunk32. No optimizer or selection uses the probe. State/MSE changes are a budget diagnostic, not complete-response acceptance.\n\nOnly50000 final is eligible for original128 TRAIN inputs/white CPU12 chunk64 Heun32 vs64 prerequisite (median<=.25LSB,p95<=1LSB,all preADC relative q<=1e-5). Fail stops all DEV generation/new classifiers/EM/main replay. Pass continues original SOURCE2000+REAL2000 controls,1024repeat/832source/128source perturbation,20HGB/16density slots,4000 independent CPU12 replay. The original six gates retain thresholds; no support fallback, clipping or model rescue.\n\nUnique server R45_REMOTE_20261010_010004_UTC, Python existing3.12.3/Torch2.12 CUDA,CPU12,RSS48GiB,GPU allocated/reserved/process10GiB; startRAM>=56GiB,GPUfree>=11GiB/no other compute; disk reserve8GiB/output4GiB. Train including probes/setup/failures<=10800s, all science<=14400s, executor seal06:00:04.557894UTC. Local auxiliaryCPU4/RSS4GiB/E-D only. Beforeimports realPID+nonce RSS guard, process-tree sampling, gaps timestamps. No source/destination migration content tests, installs, new G4/EVAL/CAL, new agents or Git.\n'''
(R/'PROTOCOL.md').write_text(protocol)
names=['train_extended.py','probe_core.py','network.py','source_flow_core.py','linux_supervise.py','linux_owned_worker.py']
for n in names:ast.parse((R/n).read_text())
with tarfile.open(R/'UPLOAD_TRAIN.tar','w',format=tarfile.USTAR_FORMAT)as t:
    for n in names:t.add(R/n,arcname=n)
    t.add(R/'config_remote.json',arcname='config.json')
bind={n:{'bytes':(R/n).stat().st_size,'sha256':hashlib.sha256((R/n).read_bytes()).hexdigest()}for n in names+['config_remote.json','config.json','PROTOCOL.md']}
(R/'PRETRAIN_FREEZE.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source_bindings':bind,
    'updates_before_launch':0,'server_preflight':pre,'normal_upload_no_content_tests':True},indent=2)+'\n')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'AST_pass':True,'upload_bytes':(R/'UPLOAD_TRAIN.tar').stat().st_size,'source_paths':names,'remote_root':remote_root}))
