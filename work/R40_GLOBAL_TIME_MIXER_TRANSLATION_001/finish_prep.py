from pathlib import Path
from datetime import datetime,timezone
import json, ast, hashlib, os
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R39_FULL_JOINT_KERNEL_TRANSLATION_001';BASE=ROOT.parent/'R32_CONDITIONAL_MEAN_TRANSLATION_001'
cfg=json.loads((ROOT/'config.json').read_text('utf8'));rc=json.loads((ROOT/'remote_config.json').read_text('utf8'))
record=json.loads((BASE/'SERVER_FRESH_CONTROL/TRAINING_RECEIPT.json').read_text('utf8'))
for c in [cfg,rc]:
    c['BASE_initial_J20']=record['initial_validation']['J'];c['initial_J20_tolerance']=1e-6
    c['remote_fresh_reference']='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R32_REMOTE_20261009_170201_UTC/SERVER_FRESH_CONTROL/FRESH_PROJECTION_STREAM.json'
for n,c in [('config.json',cfg),('remote_config.json',rc)]: (ROOT/n).write_text(json.dumps(c,indent=2,allow_nan=False)+'\n','utf8')
t=(OLD/'deploy_training.py').read_text('utf8').replace("'kernel_core.py'","'initialization_check.py'")
(ROOT/'deploy_training.py').write_text(t,'utf8')
t=(OLD/'retrieve.py').read_text('utf8').replace('JOINT_KERNEL_FINAL','GLOBAL_TIME_MIXER_FINAL')
start=t.index("    for q in sorted((p/'failure001_remote')");end=t.index('    for q in sorted(p.iterdir())',start)
t=t[:start]+t[end:]
t=t.replace("'KERNEL_FORMULA_CHECK.json'","'INITIAL_EQUIVALENCE.json'")
(ROOT/'retrieve.py').write_text(t,'utf8')
p=ROOT/'linux_supervise.py';s=p.read_text('utf8').replace(',896)',',900)');p.write_text(s,'utf8')
protocol='''# R40: frozen whole-time neural mixing

Authoritative R40_GLOBAL_TIME_MIXER_TRANSLATION_001 only. Original R32 full neural configuration and R28 fresh-fine implementation; STANDARDIZED critic, unchanged basic/style losses, no mean/MMD/calibration. Sole final2950 endpoint; J20 best internal checkpoint is record only. Source full480 enters network; no post-G transport or BASE fallback. All2000 parents preserved including417 outside old TRAIN support.

After final16-channel style FiLM before both mean and electronic heads, add one shared-channel Linear480->128/GELU(exact)/Linear128->480 residual. W1 default init seed20261010040 in CPU/CUDA restoring fork_rng; W2 weight/bias zero.123488 new params, four new state keys only. Original G/reader/D identical saved parameters. First64 SOURCE+first64 REAL actual update rows/noise compared old/new with preADC/signal/electronic bitwise equality and zero delta (256 forwards). Initial J20 saved same-server comparison abs tolerance1e-6. Original Torch RNG preserved by old/new module checks. First/last mixer gradients and final parameter norms persist.

Original R26 94400 rows/2950 batches32, 2946 homogeneous and4 mixed, original source/REAL/bin/ordinal/NCE/z-white/style order, fixed initialization and newAdam .0002(.5,.999),clip5,D1. Fresh independent PCG64(20261009281),shape480/diff479/baseline100 order,128unitdirections each/8850 groups; original fixed J20 and validation schedule unchanged. Data source6593/target21047 loaded; actual exposure3752/18739. NCE negative positions are within a waveform. Unique training cap900s includes initialization/validation/sealing; actual PID/RSS guard before scientific imports.

Main2000 and newREALcontrol2000, joint1024/probe832/SOURCE input perturbation128, and finalmodulezero128 descriptive only. Reuse R39 BASE_REAL outputs,10predictions/density/replay; old RAW9refs no refit,2D unknown. New two endpoints only20HGB/16density slots; reference fits0. Fixed source/control phase0/2,realphase1,600/250 artificialgroups,500bootstrap. Actual unquantized source-Q bins213/181 vs historyADC207/182 always separate. Independent CPU replay4000 from saved inputs+newG,chunk128, all functional components/prepost identical. Disabled-module outputs are not candidates or a training counterfactual.

All six exact task gates conjunctive. Undefined not pass; failure closes this single architecture recipe, no width/layer/seed/location/weight/endpoint search. REAL control is paired against same-treated reused BASE_REAL,not stochastic identity. Same acquisition/no independent particle truth retained. Time MLP reference: https://proceedings.neurips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html; shared time mixing only,not full MLP-Mixer reproduction. Extra parameters/forwardFLOPs confounded,not proof original convolution lacked global information.

Science1800s,CPU4,actualRSS/GPUallocated4GiB,output512MiB. Local/cache/temp E/D only. Server independent own directory, no install/newdata/EVAL/CAL/G4/Git/agents. Normal transfers without migration equality testing. Executor deadline23:39:22.826524UTC; root23:49:22.826524UTC.
'''
(ROOT/'PROTOCOL.md').write_text(protocol,'utf8')
bindings={}
for p in ROOT.glob('*.py'):
    b=p.read_bytes();ast.parse(b.decode('utf8'));bindings[p.name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
r={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source_bindings':bindings,'science_started':False,'full_neural_configuration_origin':str(BASE/'config.json'),'initial_J20_reference':cfg['BASE_initial_J20']}
(ROOT/'SOURCE_BINDINGS.json').write_text(json.dumps(r,indent=2)+'\n','utf8');print(json.dumps({'PID':os.getpid(),'UTC':r['UTC'],'AST_count':len(bindings),'training_entry_ready':True}),flush=True)
