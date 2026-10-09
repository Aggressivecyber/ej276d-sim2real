from __future__ import annotations
import pathlib,json,shutil,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent
OLD=ROOT.parent/'R20_FINE_TIME_DISTRIBUTION_001'
shared=['assessment.py','common_math.py','diagnostic_core.py','evaluation.py','energy_width.py','data_support.py','fine_core.py','prior.py','launch.py','retrieve.py','remote_supervise.py']
for n in shared:shutil.copyfile(OLD/n,ROOT/n)
shutil.copyfile(OLD/'model.py',ROOT/'legacy_model.py')
for n in ['INPUT_BINDINGS.json','VALLEY_DEFINITION.json','TRAIN_BAND_REFERENCE.json','PROBE_INPUTS_CORRECTED.npz','REPEAT_INPUTS.npz','FROZEN_TILT_TABLE.json','FINE_CONSTANTS.npz','FINE_CONSTANTS_RECEIPT.json']:
    shutil.copyfile(OLD/n,ROOT/n)
shutil.copyfile(OLD/'metrics.json',ROOT/'REFERENCE_METRICS.json')
cfg=json.loads((OLD/'config.json').read_text('utf8'))
cfg.update(stage='R21_GLOBAL_STOCHASTIC_SIGNAL_001',arms=['GLOBAL_STYLE'],
 remote_R20_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R20_REMOTE_20261009_084753',
 remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R21_REMOTE_20261009_092148',
 deadline_UTC='2026-10-09T10:41:48.905481+00:00',stage_deadline_UTC='2026-10-09T10:51:48.905481+00:00',
 style_dim=8,style_reconstruction_weight=1.,new_module_seed=20261009210,
 style_train_seed=20261009211,style_val_seed=20261009212,style_main_seed=20261009213,style_repeat_seed=20261009214,
 old_reference_training_allowed=False,selection='same J20 without style reconstruction term',
 scientific_package='three-layer global FiLM plus coarse-signal latent reconstruction; not individual mechanism isolation',
 style_zero_identity=True,repeated_rows_planned=3072,repeated_rows_cap=4096,
 HGB_plan=7,HGB_cap=12,width_slots=8,structural_check_rows=32)
cfg['execution']['remote_root']=cfg['remote_root']
(ROOT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n','utf8')
for n in ['launch.py','retrieve.py','remote_supervise.py']:
    p=ROOT/n;s=p.read_text('utf8').replace('R20','R21')
    if n=='remote_supervise.py':
        s=s.replace("'FROZEN_TILT_TABLE.json']","'FROZEN_TILT_TABLE.json','legacy_model.py']")
    p.write_text(s,'utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_only':True,'copied_old_sources_readonly':shared+['legacy_model.py'],'arrays_decoded':0}),flush=True)
