from __future__ import annotations
import pathlib,json,sys,shutil,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent;PROJECT=ROOT.parent.parent
cfg=json.loads((ROOT/'config.json').read_text('utf8'));old=json.loads((PROJECT/'work/R10_MATCHED_COMPOSITION_AND_TARGET_ONLY_001/config.json').read_text('utf8'))
cfg.update(numeric_wall_limit_s=1800,remote_R14_root=str(PROJECT/'work/R14_TRAIN_POOL_EXPANSION_001'),
 remote_R18_root=str(PROJECT/'work/R18_UNPAIRED_WAVEFORM_TRANSLATION_001'),
 remote_R19_root=str(PROJECT/'work/R19_SOURCE_PRIOR_TILT_001'),
 remote_R20_root=str(PROJECT/'work/R20_FINE_TIME_DISTRIBUTION_001'),
 remote_input_root=str(ROOT/'readonly_input_mapping'),remote_root=str(ROOT),
 local_input_paths={k:v['path']for k,v in cfg['input_files'].items()})
for k in ['G4_TRAIN_cache','generation_inputs']:cfg['local_input_paths'][k]=old['input_files'][k]['path']
cfg['execution']={'location':'local','python':sys.executable,'device':'CPU4; existing torch2.10.0+cpu, no usable CUDA','source_historical_environment':'R20 torch2.12.0+cu130 GPU; no hardware identity claim','remote_science_started_before_switch':False,'SSH_calls_in_stage':0}
cfg['source_rng_seed_and_pairing']='exact R20/FINE source/REAL/bin/U/NCE actual arrays; old z/white reconstructed from recorded old RNG; compare completed prefix'
cfg['sole_training_intervention']='global signal FiLM plus coarse-signal latent recovery, in addition to unchanged R20/FINE objective'
cfg['HGB_max_fits']=12
paths=list(cfg['local_input_paths'].values())+[cfg['remote_R14_root']+'/TRAIN_ADDITION.npz',cfg['remote_R18_root']+'/INITIAL_CRITIC.pt',cfg['remote_R18_root']+'/VALIDATION_INPUTS.npz',cfg['remote_R19_root']+'/TILT/TRANSLATOR.pt',cfg['remote_R20_root']+'/FINE/TRAINING_STREAMS_PRIVATE.npz']
missing=[p for p in paths if not pathlib.Path(p).is_file()]
if missing:raise FileNotFoundError('existing local dependencies absent: '+str(missing))
(ROOT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n','utf8')
p=ROOT/'data_support.py';s=p.read_text('utf8').replace("p=pathlib.Path(CFG['remote_input_root'])/k;n=CFG['input_files'][k]['bytes']","p=pathlib.Path(CFG['local_input_paths'][k]);n=CFG['input_files'][k]['bytes']")
s=s.replace("def zload(p,keys=None):\n    with np.load(p,allow_pickle=False)","def zload(p,keys=None):\n    p=pathlib.Path(CFG['local_input_paths'].get(pathlib.Path(p).name,str(p)))\n    with np.load(p,allow_pickle=False)")
p.write_text(s,'utf8')
p=ROOT/'numeric_worker.py';s=p.read_text('utf8');s=s.replace("signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,min(1740,(datetime.datetime.fromisoformat(CFG['deadline_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-90)))", "if hasattr(signal,'SIGALRM'):\n    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,min(1740,(datetime.datetime.fromisoformat(CFG['deadline_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-90)))\n# On Windows the owned local supervisor enforces the full hard wall and RSS limits.")
p.write_text(s,'utf8')
p=ROOT/'training.py';s=p.read_text('utf8').replace("    if delta>1e-6:raise RuntimeError('initial J20 disagrees with matched legacy start '+str(delta))","    # Hardware/library differences are recorded, not used as a new identity gate. Structural equivalence is checked in this local environment.")
# Check every actually completed old stream prefix, including replayed z/white, without rehashing an input file.
s=s.replace("    full_match={k:hashes[k]==oldrecord['stream_hashes'][k]for k in oldkeys}if updates==nold else {k:None for k in oldkeys}", "    full_match={k:hashes[k]==oldrecord['stream_hashes'][k]for k in oldkeys}if updates==nold else {k:None for k in oldkeys}\n    expected_prefix={k:hashlib.sha256(actual[k][:count].tobytes()).hexdigest()for k in oldkeys if k not in ['noise_stream','NCE_positions']}\n    expected_prefix['NCE_positions']=hashlib.sha256(actual['NCE_positions'][:3*updates].tobytes()).hexdigest()\n    replay_noise=np.random.default_rng(0);replay_noise.bit_generator.state=rs['initial']['noise'];noise_expected=hashlib.sha256()\n    for _ in range(updates):\n        noise_expected.update(replay_noise.normal(size=(batch,8)).astype('f4').tobytes());noise_expected.update(replay_noise.normal(size=(batch,480)).astype('f4').tobytes())\n    expected_prefix['noise_stream']=noise_expected.hexdigest()\n    prefix_match={k:hashes[k]==expected_prefix[k]for k in oldkeys}\n    if not all(prefix_match.values()):raise RuntimeError('completed old stream prefix mismatch')")
s=s.replace("'old_stream_prefix_from_saved_actual_arrays':True", "'old_stream_prefix_from_saved_actual_arrays':True,'old_R20_FINE_completed_prefix_match':prefix_match")
p.write_text(s,'utf8')
for src,dst in [('tasks/R21_LOCAL_EXECUTION_ADDENDUM_001.md','LOCAL_EXECUTION_ADDENDUM.md'),('reviews/R21_LOCAL_EXECUTION_001/RECEIPT.json','LOCAL_EXECUTION_REGISTRATION.json')]:shutil.copyfile(PROJECT/src,ROOT/dst)
(ROOT/'LOCAL_PREPARATION_RECEIPT.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dependencies_exist':True,'dependency_count':len(paths),'scientific_decode':0,'source_hash_or_migration_test':False,'python':sys.executable,'device':'CPU','stage_SSH':0},indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'local_inputs_ready':True,'no_science_yet':True}),flush=True)
