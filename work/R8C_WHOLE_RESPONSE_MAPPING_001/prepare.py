from __future__ import annotations
import pathlib,json,hashlib,ast,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
R8=PROJECT/'work/R8_LOCAL_JOINT_FLUCTUATION_001'
old=json.loads((R8/'config.json').read_text(encoding='utf-8'))
start=json.loads((ROOT/'START_RECEIPT.json').read_text())
sources=[]
def source(p):
    b=p.read_bytes();sources.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});return b.decode('utf-8')
for name in ['evaluation.py','common_math.py','diagnostic_core.py','energy_width.py']:
    (ROOT/name).write_text(source(R8/name),encoding='utf-8')
a=source(R8/'assessment.py')
a=a.replace("names=['R3B','JOINT','JOINT_SIGNAL']","names=[n for n in domains if n!='REAL']")
a=a.replace("for name in ['R3B','JOINT','JOINT_SIGNAL']:","for name in [n for n in domains if n!='REAL']:")
# Width fitting is implemented in the new driver; do not reuse its hardcoded arms.
(ROOT/'assessment.py').write_text(a,encoding='utf-8')
s=source(PROJECT/'work/R3B_ENERGY_CONDITIONAL_WIDTH_001/model.py')
node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='observables')
(ROOT/'observable_source.py').write_text('import torch\n'+ast.get_source_segment(s,node)+'\n',encoding='utf-8')
s=source(PROJECT/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/run.py')
node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='clean_expectation')
(ROOT/'clean_source.py').write_text('import numpy as np\n'+ast.get_source_segment(s,node)+'\n',encoding='utf-8')
s=source(R8/'launch.py').replace('R8_response_started','R8C_response_started').replace('paired_joint_signal_response','whole_response_mapping').replace('paired_joint_signal','whole_response_mapping')
(ROOT/'launch.py').write_text(s,encoding='utf-8')
inp={k:v for k,v in old['input_files'].items() if k not in ['model','generation_rng']}
for k,p,n,h in [
 ('R8_metrics',R8/'metrics.json',1946665,'12683a60571d7b615dac71ea14142f29be896a6aee2767a0d4f4d9e9217c6d99'),
 ('R8_JOINT',R8/'arms/JOINT/generated.npz',7735246,'918698b6f35c5ff43ca21fdbb30c1bc1f4d79777279fa240a012c3a37a16bea4'),
 ('R8_JOINT_SIGNAL',R8/'arms/JOINT_SIGNAL/generated.npz',10983126,'8617792d5b7533581f36d2be89f90b6becfb417fc04db0937d36b3766ee028ee'),
]:
    assert p.stat().st_size==n
    inp[k]={'path':str(p),'bytes':n,'sha256':h}
cfg={**start,'numeric_wall_limit_s':2400,'seed':2026100891,'projection_seed':2026100892,
 'source_permutation_seed':2026100893,'generation_seeds':[2026100894,2026100895,2026100896],
 'validation_seed':2026100897,'validation_n':384,'arms':['FULL_WAVE_STOCHASTIC','PULSE_STOCHASTIC_OFF'],
 'max_updates_per_arm':2000,'training_soft_limit_s':1600,'training_finalization_reserve_s':30,
 'batch_size':64,'learning_rate':.0002,'Adam_betas':[.5,.9],'weight_decay':.0001,
 'gradient_norm_max':5.,'generation_chunk':64,'validation_interval':100,
 'encoder':{'channels':24,'kernel':5,'dilations':[1,2,4,8,16,32,64],'convolutions_per_residual_block':2,'activation':'SiLU','padding':'zero same','stride':1},
 'pulse_branch':{'z_pulse':16,'z_base':4,'white_points':512,'dilations':[1,4,16],'output_scale_V':.1,'output_last_layer_initialization':'zero'},
 'mean_branch':{'gain':'exp(linear_c4_MLP16), initial intercept TRAIN median Qtotal/source clean Qtotal','signed_residual_scale_V':.1,'output_last_layer_initialization':'zero','reads_noise':False},
 'baseline_branch':{'z_base':4,'white_points':512,'conditional_FIR_taps':31,'FIR_normalization':'L2, no zero-sum constraint','RMS':'exp(MLP(c4,zbase4) plus initial logRMS_sd*zbase0)','initial_FIR':'centered 31 taps of IFFT sqrt(TRAIN-fit baseline mean spectrum), L2 normalized','driving_white_mean':'population zero, no per-event centering'},
 'critic':{'channels':24,'scales':[1,2,4],'dilations':[1,2,4],'kernel':5,'spectral_norm':True,'stride':1,'inputs':'V/.1 and own baseline-subtracted peak-normalized shape'},
 'loss_coefficients':{'hinge_adv':1.,'full_joint_SW':1.,'conditional_CDF':2.,'occupancy':.5,'baseline_joint':.5},
 'joint_features':{'scalar_fields':['PSD','log_Qtotal_mVns','log_peak_V','rise10_90_ns','tail_charge_fraction'],'scalar_IQR_floors':[.05,.5,.5,1.,.05],'blocks':['scalar5','V480/.1','ownshape480','d1_V479/.1','d2_V478/.1'],'coordinate_IQR_floors':[None,.005,.02,.005,.005],'balance':'each standardized block divided by sqrt(its dimension)','projections':128,'smoothing':False},
 'baseline_joint':{'fields':['log_RMS_ddof1','ACF_lag1_to8_full100_den','nonDC_normalized_power50'],'IQR_floors':[.2,.02,.005],'projections':128,'den_floor':1e-12,'log_RMS_floor_V':1e-7},
 'CDF':{'PSD_grid_points':128,'PSD_grid_limits':[-.1,.6],'temperature':.0025,'logQ_centers':'fixed8 bin midpoints','logQ_soft_width':'one fixed narrow-bin width'},
 'validation':'fixed384 G4 internal-val and REAL internal-val selections/noise; SW batch projection quantiles, full599 validation CDF/occupancy and joint baseline target; no adversarial; strict lowest score incl0',
 'source_controls':{'fixed_positions':'floor(j*1999/63), j=0..63','shuffle':'same permutation of x,c,Q on these64','replacement':'full ID SHA order first allowed G4 TRAIN-fit parent'},
 'HGB_max_fits':12,'C2ST':old['C2ST'],'train_each_domain':600,'holdout_each_domain':250,
 'bootstrap_repeats':500,'bootstrap_seed':old['bootstrap_seed'],'C2ST_views':old['C2ST_views'],
 'density_EM_limit':16,'input_files':inp,'evaluation':old['evaluation'],
 'EVAL_read':0,'SSH':0,'new_G4':0,'new_agents':0,'Git':0,
 'full_clean_is_fixed_total_deposition_source_description_not_new_timing':True,
 'retrospective_same_acquisition_DEV':True,'CAL_q4_fallback':False,
 'numeric_ledger':'all worker/replay/plot child wall times incl failures in RUN_HISTORY; no reset after repair'}
(ROOT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'PREPARATION_SOURCE_READS.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_reads':len(sources),'numeric_array_decode':0,'config':str(ROOT/'config.json')}))
