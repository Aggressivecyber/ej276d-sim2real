from pathlib import Path
import os,json,shutil,datetime
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent.parent;old=PROJECT/'work/R21_GLOBAL_STOCHASTIC_SIGNAL_001'
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
for n in ['model.py','legacy_model.py','fine_core.py','prior.py','evaluation.py','assessment.py','common_math.py','diagnostic_core.py','energy_width.py','data_support.py','random_diagnostics.py','FINE_CONSTANTS.npz','FINE_CONSTANTS_RECEIPT.json','FROZEN_TILT_TABLE.json','VALLEY_DEFINITION.json','REFERENCE_METRICS.json','TRAIN_BAND_REFERENCE.json']:
    shutil.copyfile(old/n,ROOT/n)
c=json.loads((old/'config.json').read_text('utf8'))
c.update(stage='R23_ONLINE_STANDARDIZED_CRITIC_001',deadline_UTC='2026-10-09T12:11:36.829269+00:00',stage_deadline_UTC='2026-10-09T12:21:36.829269+00:00',remote_root=str(ROOT),
    updates=4000,training_wall_s=450,numeric_wall_limit_s=1800,HGB_max_fits=14,HGB_cap=14,width_slots=16,arms=['RAW','STANDARDIZED'],
    G_initial=str(old/'GLOBAL_STYLE/TRANSLATOR.pt'),reader_initial=str(old/'GLOBAL_STYLE/STYLE_READER.pt'),
    R21_root=str(old),R22_root=str(PROJECT/'work/R22_FROZEN_GENERATOR_CRITIC_DIAGNOSTIC_001'),
    seed=20261009230,source_REAL_bin_seed=20261009231,old_noise_seed=20261009232,style_train_seed=20261009233,NCE_seed=20261009234,
    science_budget=1800,CPU_python='C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe',GPU_python='E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe',
    source_population_preserved=True,G_D_update_ratio='1:1',critic_intervention='fixed R22 percoordinate centering AND scaling, not separated',initial_D_pretrained_R22=False)
c['execution']={'location':'local','GPU_actual_scientific_PID_required':True,'CPU_python':c['CPU_python'],'GPU_python':c['GPU_python']}
write('config.json',c);write('START_RECEIPT.json',{'first_native_tool':'8c0487','PID':57564,'UTC':'2026-10-09T10:52:22.758204+00:00','exit':0,'stage_science_started':False,'bootstrap_PID':os.getpid()})
shutil.copyfile(PROJECT/'tasks/R23_ONLINE_STANDARDIZED_CRITIC_001.md',ROOT/'TASK.md');shutil.copyfile(PROJECT/'reviews/R23_REGISTRATION_001/RECEIPT.json',ROOT/'REGISTRATION.json')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bootstrap':'complete'}),flush=True)
