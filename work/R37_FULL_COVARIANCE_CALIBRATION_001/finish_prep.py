from pathlib import Path
import json,ast,hashlib,os
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R36_CONDITIONAL_DIFFUSION_EDIT_001'
deploy=(OLD/'deploy_training.py').read_text('utf8')
deploy=deploy.replace("files=['train_server.py','diffusion_core.py','flow_core.py','linux_supervise.py','linux_owned_worker.py','remote_config.json','PROTOCOL.md']",
    "files=['train_server.py','covariance_core.py','model.py','legacy_model.py','reorder_core.py','flow_core.py','linux_supervise.py','linux_owned_worker.py','remote_config.json','PROTOCOL.md']")
(ROOT/'deploy_training.py').write_text(deploy,'utf8')
retrieve=(OLD/'retrieve.py').read_text('utf8').replace('CONDITIONAL_DIFFUSION_EDIT','FULL_COVARIANCE_CALIBRATION')
retrieve=retrieve.replace("['DIFFUSION_CONSTANTS.npz','CONDITION_CONSTANTS.json','TRAINING_ROW_BINDING.json','DIFFUSION_FORMULA_CHECK.json','MODEL_STRUCTURE.json']",
    "['STATISTICS_INPUT_BINDING.json','BASE_FORWARD_HASHES.json','COVARIANCE_FORMULA_CHECK.json','STATISTICS_FAILURE.json','REORDER_CONSTRUCTION_RECEIPT.json']")
retrieve=retrieve.replace('128*1024**2','384*1024**2')
(ROOT/'retrieve.py').write_text(retrieve,'utf8')
protocol='''# R37 frozen full480 covariance calibration

One registered statistical question, no additional neural updates. Task and registration control the exact counters and gates. Server fit uses the original R33 source/target/bin/ordinal/z_white/style sequence:2950 batches32,94400 rows/domain; original frozen R32 G output hash must match. Two independent marginal means and unbiased covariance(ddof1) per original source-Q bin use float64 centered batch combination; repeated draws retained, not called independent acquisition counts. Original REAL TRAIN480 affine constants reused in float64.

Single prescribed eigenvalue floor for each Cs/Ct:1e-6*trace(C)/480. N>480 and trace>0 required; original/intermediate min eigenvalue relative negative limit1e-8. No lower rank, spectral scan, change of bins, arbitrary rotation, posterior protection, smoothing or charge scaling. Eight A use exactly Cs_reg^-1/2 [Cs_reg^1/2 Ct_reg Cs_reg^1/2]^1/2 Cs_reg^-1/2. The middle principal root is numerically evaluated by SVD of H=Cs_reg^1/2 Ct_reg^1/2 (H H.T is the same middle matrix), avoiding squaring small singular values; its residual and actual intermediate eigenvalues are saved. This changes neither the Gaussian map nor its floor. Actual A must be positive definite and regularized covariance relative Frobenius residual<=1e-6. Unregularized training covariance residual reported separately.

Deployment: complete480 source+original z/white/style -> frozen G -> original source signed-Q bin -> float64 ys -> mt+A(ys-ms) -> original inverse TRAIN affine -> float32 -> original14bitADC once. G generation/replay CPU chunk128, matrix float64. All2000 original parent IDs/composition/weights/order remain; outside source-Q support retain G. No REAL-library lookup, target self-generation or new random source. New calibration is a TRAIN-fitted linear layer after a frozen neural response, not a newly trained NN.

One source2000, joint128x8=1024, corrected original832 source probes and guide128. Guide probe is first128 supported main parents, same original bin and same G randomness; perturb only150:400 by fixed demeaned sine at segmentRMS1mV and save Qfast/Qtotal/PSD numerical residuals. Save exact auxiliary/main preADC. Algebraic baseS/baseE/rounding/calibration delta includes cross terms; final physical branch split unknown. Independent CPU full source->G->map->singleADC replay2000 must match all saved fields bytewise without loading saved outputs as inference input.

Fixed original10HGB views600/250 and500 group bootstrap, old BASE/R33/R35/R36/RAW RR controls reused. New8 density slots<=8EM, deterministic old settings and per-slot immediate persistence. Old RAW RR Q_PSD_2D prediction absent ->unknown, no refit. TRAIN-only map; DEV descriptions do not fit transformation. Whole population, output-Q and actual source-Q conditions separate; actual pure-had bins3/4=213/181, postADC historical reference207/182. All fixed boundary/peak-center/width/noise/valley/rank gates preserved; peak center MAE <=BASE+10% newly registered before this execution. Undefined fails; no scans after negative results.

Unique server900s fit with actual PID/nonce/RSS guard then localCPU4 generation/evaluation/replay. Whole science1800s incl failed attempts and figures; actual RSS/GPU allocated each4GiB, exclusive new outputs512MiB. Executor seal deadline22:01:09.839435UTC. Normal SSH/scp exit and loading, no migration-content tests or installs. All local storage/temp/cache E/D. Source is G4 total-deposit analytic waveform, not newly recovered microscopic timing; same-acquisition artificial50record groups, historicalDEV exposure, no blind-test claim. EVAL/CAL/newREAL/longraw/newG4/extra neural fits0. Failure closes this fixed Gaussian recipe only; project active.
'''
(ROOT/'PROTOCOL.md').write_text(protocol,'utf8')
bindings={}
for p in ROOT.glob('*.py'):
    ast.parse(p.read_text('utf8'));bindings[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(ROOT/'PREPARATION_SOURCE_RECEIPT.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
    'source_bindings':bindings,'source_only':True,'no_scientific_arrays_or_formulas_executed':True},indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'AST_files':len(bindings),'ready_for_unique_statistics_fit':True}),flush=True)
