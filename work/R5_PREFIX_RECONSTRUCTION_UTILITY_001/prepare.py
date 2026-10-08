from __future__ import annotations
import pathlib,json,hashlib,ast,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parent.parent
def bind(p):
    p=pathlib.Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
start=json.loads((ROOT/'START_RECEIPT.json').read_text());frozen=json.loads((BASE/'FROZEN_METHOD.json').read_text());bindings=[]
for s in frozen['model_and_generation_bindings']+frozen['definitions_and_tasks']:
    p=BASE/s['path'];actual=bind(p)
    if actual['sha256']!=s['sha256'] or actual['bytes']!=s['bytes']:raise RuntimeError('freeze drift '+str(p))
    bindings.append(actual)
for n in ['common_math.py','evaluation.py']:(ROOT/n).write_bytes((BASE/'work/R4_FROZEN_HELDOUT_RESPONSE_001'/n).read_bytes())
for stage,dest in [('R2_SMOOTH_STOCHASTIC_RESPONSE_001','model_R2.py'),('R3B_ENERGY_CONDITIONAL_WIDTH_001','model_flexible.py')]:
    (ROOT/dest).write_bytes((BASE/'work'/stage/'model.py').read_bytes())
source=pathlib.Path('E:/GAN/simulation_provenance_audit_20261004/sources/active_prepare_sim_data.py');text=source.read_text();tree=ast.parse(text)
names={'TAU_EE','FRAC_EE','TAU_HAD','FRAC_HAD','RISE_NS','YIELD','QUENCH','CAL_A','CAL_B'}
parts=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets):parts.append(ast.get_source_segment(text,node))
    if isinstance(node,ast.FunctionDef) and node.name=='pulse_rate':parts.append(ast.get_source_segment(text,node))
digitizer=pathlib.Path('E:/GAN/fullwave_execution_20261004_0930/executed_source/gan/scripts/digitizer.py')
(ROOT/'digitizer.py').write_bytes(digitizer.read_bytes())
noise=pathlib.Path('E:/GAN/fullwave_execution_20261004_0930/executed_source/gan/scintcpunet/noise_spectrum.py')
(ROOT/'noise_spectrum.py').write_bytes(noise.read_bytes())
interface=BASE/'work/P1K_REAL_REFERENCE_AND_REPEATABILITY_001/interface_core.py';st=interface.read_text();tr=ast.parse(st)
functions=[ast.get_source_segment(st,n) for n in tr.body if isinstance(n,ast.FunctionDef) and n.name in ['_normalize_waveform','_window_waveform','constant_fraction_tp0']]
(ROOT/'sim_generator.py').write_text('from __future__ import annotations\nimport numpy as np\nfrom digitizer import digitize\nfrom noise_spectrum import spectrum_noise\n'+'\n\n'.join(parts+functions)+'\n',encoding='utf-8')
launcher=(BASE/'work/R4_FROZEN_HELDOUT_RESPONSE_001/launch.py').read_text().replace('R4_evaluation_started','R5_utility_started').replace('frozen_heldout_evaluation','prefix_reconstruction_utility')
launcher=launcher.replace("str(ROOT/'run.py')","str(ROOT/('replay.py' if os.environ.get('R5_MODE')=='replay' else 'run.py'))")
(ROOT/'launch.py').write_text(launcher,encoding='utf-8')
cfg={**start,'stage':'R5','numeric_wall_limit_s':1800,'threads':4,'RSS_limit_bytes':4294967296,'output_limit_bytes':536870912,
     'budgets':[256,1024],'full_REAL':2592,'arms':['REAL','REAL+SIM','REAL+R2','REAL+R3','REAL+R3B'],
     'models':['Ridge','HGB'],'targets':['PSD','logQ'],'planned_fits':44,'synth_N':2000,
     'synth_parent_order':'full SHA256 R5-synth-v1|parent_id,first2000 of bound8192 G4 TRAIN cache',
     'real_order':'full SHA256 source_id UTF8,ascending,nested256/1024 inside REAL TRAIN internal fit2592',
     'seed':2026100852,'bootstrap_seed':2026100853,'bootstrap_repeats':1000,'generation_chunk':256,
     'prefix_range':[0,150],'baseline_range':[0,100],'pulse_prefix':[100,150],'features':'50 baseline-subtracted pulse samples / prefix peak, log prefix peak, log prefix baseline sample RMS',
     'prefix_peak_floor_V':1e-8,'prefix_RMS_floor_V':1e-12,'feature_N':52,
     'scaling':'only corresponding REAL subset,mean/population SD;zeroSD->1;shared across all arms at a budget',
     'sample_weights':{'REAL_only':'each1/n,total1','mixed_REAL':'each0.5/n,total0.5','mixed_synth':'each0.5/2000,total0.5'},
     'Ridge':{'alpha':.001},'HGB':{'loss':'squared_error','max_iter':150,'learning_rate':.05,'max_leaf_nodes':15,'min_samples_leaf':20,'l2_regularization':1,'early_stopping':False,'random_state':202610085},
     'model_selection':'none;599 internal validation check only;fit all44 and freeze before EVAL input/prediction',
     'margin_edges_PSD':[0,.005,.01,.02,.05,.1],'response_fits':0,'SSH':0,
     'replay':'fresh process all44 saved estimators,fixed first64 EVAL QC prefix inputs,rtol=atol1e-12;no refit',
     'task_SHA256':start['task_sha256'],'FROZEN_METHOD_SHA256':hashlib.sha256((BASE/'FROZEN_METHOD.json').read_bytes()).hexdigest(),
     'original_SIM':'source-bound active analytic kernel,0.6258014992938228mV full-window RMS,fixedWelch spectrum,14bit,CFD;seed2026100852 for this new fixed TRAIN-parent realization',
     'SIM_historical_noise_calibration_REAL_count':79793,'not_online_trigger_or_FPGA_validation':True}
write('config.json',cfg)
write('PROTOCOL.json',cfg)
external=[bind(p) for p in [source,digitizer,noise,interface,BASE/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/data/G4_TRAIN_CACHE.npz',BASE/'work/R4_FROZEN_HELDOUT_RESPONSE_001/EVAL_EXTRACT.npz',ROOT/'SIM_RECIPE.json']]
sources=[]
for p in sorted(ROOT.glob('*.py')):ast.parse(p.read_bytes());sources.append(bind(p))
write('READINESS.json',{'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),'frozen_bindings':bindings,'local_input_bindings':external,'execution_sources':sources,'new_response_fit':0,'SSH':0,'planned44fits':True})
print(json.dumps({'kind':'R5_ready','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':len(sources),'local_dependencies':len(external),'SSH':0}),flush=True)
