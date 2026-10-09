from pathlib import Path
import datetime, os, json, shutil, hashlib, ast
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent.parent
old=PROJECT/'work/R21_GLOBAL_STOCHASTIC_SIGNAL_001'
def write(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
for n in ['model.py','legacy_model.py']:
    shutil.copyfile(old/n,ROOT/n)
cfg={'stage':'R22_FROZEN_GENERATOR_CRITIC_DIAGNOSTIC_001','start_UTC':'2026-10-09T10:24:03.7367011+00:00',
 'seal_deadline_UTC':'2026-10-09T11:12:41.881019+00:00','science_s':1200,'train_s_per_arm':300,'train_s_total':600,
 'updates':4000,'batch':32,'lr':.0002,'betas':[.5,.999],'clip':5.,'threads':4,'RSS_limit':4*1024**3,'GPU_limit':4*1024**3,'output_limit':192*1024**2,
 'CPU_python':'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe',
 'GPU_python':'E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe',
 'G_weights':str(old/'GLOBAL_STYLE/TRANSLATOR.pt'),'D_initial':str(PROJECT/'work/R18_UNPAIRED_WAVEFORM_TRANSLATION_001/INITIAL_CRITIC.pt'),
 'old_config':str(old/'config.json'),'tilt_table':str(old/'FROZEN_TILT_TABLE.json'),
 'G4_cache':str(PROJECT/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/data/G4_TRAIN_CACHE.npz'),
 'REAL_addition':str(PROJECT/'work/R14_TRAIN_POOL_EXPANSION_001/TRAIN_ADDITION.npz'),
 'edges':[7.071309847742635,7.354357548030852,7.6374052483190695,7.920452948607288,8.203500648895506,8.486548349183723,8.769596049471941,9.052643749760158,9.335691450048376],
 'seeds':{'train_style':20261009221,'train_old_noise':20261009222,'val_style':20261009223,'val_old_noise':20261009224,'sampling':20261009225,'bootstrap':20261009226},
 'HGB':{'max_iter':100,'learning_rate':.06,'max_leaf_nodes':15,'min_samples_leaf':20,'l2_regularization':1.,'early_stopping':False,'random_state':20261008},
 'selection':'equal8bin TILT-weighted generated / uniform REAL balanced validation MSE, include0/every250/final; earliest strict min',
 'scaler':'original Torch float32 input_channels, then float64 mean/population SD on all21047 REALfit; perchannel floor max(.05 medianSTD,1e-6)',
 'gradient':'mean((D(GEN)-1)^2) input voltage derivative over same64 hashfirst validation sourceparents; report per-wave RMS; unit V^-1; no update',
 'bootstrap':'500 paired fixed-classifier cluster draws within each bin/domain; source parent independently, REAL50-record groups resampled; equal-bin/TILT weights renormalized each draw; artificial groups same acquisition',
 'EVAL_numeric':0,'DEV_numeric':0,'CAL_numeric':0,'SSH':0,'new_G4':0,'new_EM':0,'new_G_updates':0,'new_reader_updates':0}
write('config.json',cfg)
write('START_RECEIPT.json',{'first_actual_tool':'509bf7','first_native_PID':60920,'first_native_exit':0,'first_UTC':cfg['start_UTC'],'bootstrap_PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'science_started':False})
for src,n in [(PROJECT/'tasks/R22_FROZEN_GENERATOR_CRITIC_DIAGNOSTIC_001.md','TASK.md'),(PROJECT/'reviews/R22_REGISTRATION_001/RECEIPT.json','REGISTRATION.json')]:shutil.copyfile(src,ROOT/n)
write('SOURCE_BINDINGS.json',[{'name':n,'bytes':(ROOT/n).stat().st_size,'sha256':hashlib.sha256((ROOT/n).read_bytes()).hexdigest()}for n in ['model.py','legacy_model.py','TASK.md','REGISTRATION.json']])
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':str(ROOT),'status':'prepared_controls'}),flush=True)
