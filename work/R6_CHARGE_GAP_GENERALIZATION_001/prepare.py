from __future__ import annotations
import pathlib,json,hashlib,ast,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parent.parent
def bind(p):
    p=pathlib.Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
start=json.loads((ROOT/'START_RECEIPT.json').read_text());freeze=json.loads((BASE/'FROZEN_METHOD.json').read_text());bindings=[]
for s in freeze['model_and_generation_bindings']+freeze['definitions_and_tasks']:
    a=bind(BASE/s['path']);assert a['bytes']==s['bytes'] and a['sha256']==s['sha256'];bindings.append(a)
for n in ['common_math.py','evaluation.py']:(ROOT/n).write_bytes((BASE/'work/R4_FROZEN_HELDOUT_RESPONSE_001'/n).read_bytes())
for stage,dest in [('R2_SMOOTH_STOCHASTIC_RESPONSE_001','model_R2.py'),('R3B_ENERGY_CONDITIONAL_WIDTH_001','model.py')]:
    (ROOT/dest).write_bytes((BASE/'work'/stage/'model.py').read_bytes())
(ROOT/'energy_width.py').write_bytes((BASE/'work/R3B_ENERGY_CONDITIONAL_WIDTH_001/energy_width.py').read_bytes())
source=(BASE/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/run.py').read_text();tree=ast.parse(source)
parts=[ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['clean_expectation','condition']]
(ROOT/'source_condition.py').write_text("import numpy as np\nimport torch\nfrom model_R2 import observables\nCFG={'energy_log_center':-.5}\n\n"+'\n\n'.join(parts)+'\n',encoding='utf-8')
stageconfigs={}
for tag,stage in [('R2','R2_SMOOTH_STOCHASTIC_RESPONSE_001'),('R3','R3_FLEXIBLE_TEMPORAL_RESPONSE_001'),('R3B','R3B_ENERGY_CONDITIONAL_WIDTH_001')]:
    c=json.loads((BASE/'work'/stage/'config.json').read_text());stageconfigs[tag]={k:c[k] for k in ['max_updates','batch_size','learning_rate','training_wall_s','validation_seed']}
cfg={**start,'stage':'R6','numeric_wall_limit_s':2400,'generation_seed':2026100862,'generation_chunk':256,'gap_charge_mVns':[3703.7782948464155,6711.804904974998],
     'curriculum':stageconfigs,'arms':['FULL','GAP'],'initialization':'master202610086 before any constructor; shared trainable parameters initialized only from allowed GAP-fit statistics; each arm owns its TRAIN-only fixed noise spectrum buffer; all trainable parameters byte equal, no FULL statistics copied to GAP',
     'TRAIN_STATISTICS':'noise spectrum, feature scales and conditionalCDF targets computed independently using each permitted arm fit; shared init statistics derived from GAP-fit subset only',
     'G4_exclusion':'nominal clean saved480 CFD-aligned 100:400 signed charge; no learned gain and no full512 integral substitute',
     'curriculum_seeds':'same master202610086 training generator and constructor reset at each arm/stage; original fixedvalidation2026100821; no EVAL selection',
     'R3_transition':'own selected R2 checkpoint, strictFalse only temporal_/time_ missing; new zero-output time heads',
     'R3B_transition':'own selected R3 strictTrue; centers and CDF targets from own allowedREAL fit',
     'C2ST':json.loads((BASE/'work/R4_FROZEN_HELDOUT_RESPONSE_001/config.json').read_text())['C2ST'],
     'q3_C2ST_support':'all five candidates and REAL must support fixed200train/100test each under inherited group hash; otherwise whole common comparison undefined, no replacement',
     'RR':'64 deterministic group-half splits; same-N per candidate output q3 count if both halves can supply at every repeat, otherwise preserve unsupported reference; common feasible minimum-N additionally listed; no physical equivalence',
     'RR_seed_prefix':'R6-RR-group-v1|repeat|group / R6-RR-event-v1|repeat|source_id',
     'narrow_width_edges':'exact existing R3B REAL TRAIN development edges; diagnostic EM fixed500/1e-8/no restart; not training arm target centers',
     'prior_exposure':'CAL thresholds/source analytic constants/history prior and architecture fullDEV development are allowed and remain disclosed; current GAP fit excludes interval',
     'EVAL_loading':'only after both fitted model/RNG/main generation saved','SSH':0,'new_G4':0,'EVAL_fit':0,'FROZEN_METHOD_SHA256':hashlib.sha256((BASE/'FROZEN_METHOD.json').read_bytes()).hexdigest()}
write('config.json',cfg);write('PROTOCOL.json',cfg)
launcher=(BASE/'work/R5_PREFIX_RECONSTRUCTION_UTILITY_001/launch.py').read_text().replace('R5_MODE','R6_MODE').replace('R5_utility_started','R6_curriculum_started').replace('prefix_reconstruction_utility','paired_gap_response').replace("'response_training':0","'response_training':'two fixed1600/2000/2000 curricula'")
(ROOT/'launch.py').write_text(launcher,encoding='utf-8')
sources=[]
for p in sorted(ROOT.glob('*.py')):ast.parse(p.read_bytes());sources.append(bind(p))
write('READINESS.json',{'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),'frozen_bindings':bindings,'execution_sources':sources,'no_scientific_values_accessed_during_preparation':True,'SSH':0,'planned_optimizer_updates':11200})
print(json.dumps({'status':'ready','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scripts':len(sources),'task_SHA256':start['task_SHA256'],'deadline':start['deadline_UTC']}),flush=True)
