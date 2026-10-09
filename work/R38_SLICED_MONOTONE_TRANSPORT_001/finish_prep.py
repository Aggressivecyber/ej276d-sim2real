from pathlib import Path
import ast,json,hashlib,os
from datetime import datetime,timezone
R=Path(__file__).resolve().parent;old=R.parent/'R37_FULL_COVARIANCE_CALIBRATION_001'
arm='SLICED_MONOTONE_TRANSPORT'
def replace(s,before,after):
    assert before in s,before
    return s.replace(before,after)
s=(old/'deploy_training.py').read_text('utf8').replace('covariance_core.py','sliced_core.py')
(R/'deploy_training.py').write_text(s,'utf8')
s=(old/'retrieve.py').read_text('utf8').replace('FULL_COVARIANCE_CALIBRATION',arm).replace('COVARIANCE_FORMULA_CHECK.json','SLICED_FORMULA_CHECK.json')
s=s.replace("'STATISTICS_FAILURE.json','REORDER_CONSTRUCTION_RECEIPT.json'", "'STATISTICS_FAILURE.json','REORDER_CONSTRUCTION_RECEIPT.json','DIRECTION_INITIALIZATION.json'")
(R/'retrieve.py').write_text(s,'utf8')
s=(old/'generate_all.py').read_text('utf8').replace('FULL_COVARIANCE_CALIBRATION',arm)
s=s.replace('one TRAIN-fitted full480 Gaussian Monge map per original sourceQ bin; support outside base unchanged',
    'TRAIN-fitted32-layer nonlinear sliced monotone maps per original sourceQ bin; original frozen G and outside-support fallback')
s=s.replace('flow_named_fields_are_compatibility_aliases_for_calibration_delta','flow_named_fields_are_compatibility_aliases_for_sliced_transform_delta')
(R/'generate_all.py').write_text(s,'utf8')
s=(old/'evaluate_saved.py').read_text('utf8').replace('FULL_COVARIANCE_CALIBRATION',arm)
s=replace(s,"'CONDITIONAL_DIFFUSION_EDIT']","'CONDITIONAL_DIFFUSION_EDIT','FULL_COVARIANCE_CALIBRATION']")
s=replace(s,"old=json.loads((pathlib.Path(CFG['R36_root'])/'metrics.json')", "old=json.loads((pathlib.Path(CFG['R37_root'])/'metrics.json')")
s=replace(s,"('CONDITIONAL_DIFFUSION_EDIT',pathlib.Path(CFG['R36_root'])/'CONDITIONAL_DIFFUSION_EDIT/generated.npz'),(SOURCE_ARM",
    "('CONDITIONAL_DIFFUSION_EDIT',pathlib.Path(CFG['R36_root'])/'CONDITIONAL_DIFFUSION_EDIT/generated.npz'),\n        ('FULL_COVARIANCE_CALIBRATION',pathlib.Path(CFG['R37_root'])/'FULL_COVARIANCE_CALIBRATION/generated.npz'),(SOURCE_ARM")
s=s.replace('CALIBRATION_vs_BASE','SLICED_vs_BASE').replace('CALIBRATION_vs_ORDERED','SLICED_vs_ORDERED').replace('R37 saved evaluation','R38 saved evaluation')
(R/'evaluate_saved.py').write_text(s,'utf8')
s=(old/'replay_all.py').read_text('utf8').replace('FULL_COVARIANCE_CALIBRATION',arm).replace('TRAIN_calibration','TRAIN_sliced_transform')
(R/'replay_all.py').write_text(s,'utf8')
protocol='''# R38 frozen source-driven sliced monotone transport

Exact task and registration govern one bounded recipe. Original R32 CONTROL_FINAL G and all seven original R33 streams,2950 batch32=94400 rows per domain; source/REAL independent, no PSD coupling/reweight/target retrieval. Neural updates0; statistical directions/maps are fitted and counted. Source/REAL float32 lifted into original TRAIN480 mu/scale in float64. Original source-Q bin decided once; unsupported parents retain G. Original inputs and old results remain immutable.

PCG64(20261010381) generates480x16 initial normals in layer-major0..31/bin-minor0..7 order, reduced QR positive diagonal, saved seed/version/initial hashes. Eight bins each32 layers,16 orthogonal columns,257 knots(.005...995),256 groups/4096 maps. Direction subset floor((j+.5)*N/min(1024,N)), each marginal in its original bin row order. Exactly20 float64 CPU gradient-ascent steps per group, empirical sorted W2 squared mean, tangent G-A sym(A.T G); Fro norm<=1e-12 is counted no-op, otherwise step0.1 normalized tangent and QR. Fixed step20, no best-J/line search/restart. Record before/after objective, Euclidean/tangent norm,orthogonality and actual nonascending steps. 5120 directional steps max; subset/full projection calls counted; not a same-FLOP comparison to R37.

All full-bin projected knots use numpy.quantile method linear. Strict finite/increasing x/y knots required; any zero/negative difference stops entire recipe, no jitter/slopes floor/knot reduction. Piecewise linear maps extend tails by slope1 translations, without clamping. X <- X+(T(X@A)-X@A)@A.T, Y stays fixed. Actual group parameters and trace saved immediately; only all256 groups complete is a valid endpoint. Artificial identity/monotonicity/continuity/inverse checks and actual TRAIN-subset layer inverse errors recorded. Endpoint cannot be selected from DEV or intermediate layers. Continuous piecewise-linear maps are not claimed smooth; ADC is not invertible.

Fullsource+originalz/white/style->G128 CPU->per-bin32 float64 layers->inverse TRAIN affine->float32->original14bitADC once; inference and replay use identical per-bin whole-group matrix order, not differently chunked transform calls. No new random source/REAL-library/PSD baseline charge protection/smoothing. Main2000 preserved, joint1024,corrected source832,extra guide128 firstsupported sourceorder with fixedbin and same G randomness. Guide150:400 fixeddemeaned sine segmentRMS1mV, report f32 Qfast/Qtotal/PSD drift. Exact main/aux basepreADC/out64/cast-rounding/delta retained. Legacy calibration/flow field names mean this nonlinear transform delta; no CFM or R37 covariance layer. Additive cross terms remain; physical final branch splitunknown. Independent fullsource->G->32layers->singleADC replay2000, every field and ID compared bytewise.

New10fixedHGB/8densityslots<=8EM; original600/250 rows/groups and500 fixed-classifier blockbootstrap. BASE/R33/R35/R36/R37 and RAW RR reused without refitting; old missing RR QPSD prediction unknown. Complete shape493/u480 CI upper againstBASE negative and each BA<=R37 .924/.918+.01, all two-center/two-raw-width/global/commonQPSD/noise RMS/ACF differences<=BASE+10%, BASEvalidbins2-7preserved. Actual preADC NNsourceQ purehad3/4N213/181 valleydelta<=.02, not historicalADC207/182; rankdrops<=.05,no invalid/clips,9fixedboundaries and guidegate unchanged. All numericgroups/replay/QA/resources required, undefinedfails. These are developer retention rules, not physical equivalence. Anyfail closes this recipe; no layer/K/knot/stepsize/seed/endpoint/protection rescues.

Source waveform is existing G4 total-deposit analytic representation, not microscopic timing. Sourcecomp/PDG inference separately reported; density components not particletruth, same-acquisition50-record artificialgroups/historicalDEV exposure retained. TRAIN-only transformations, DEV descriptive covariance/PSD never used to fit maps, no EVAL/CAL/newREAL/G4/installation. Server sameenvironment CUDA only frozenG then CPU4 float64 stats, unique fit900s/science1800s inclfailures/figures, actualPID/nonce/RSS4GiB andGPUallocated4GiB, outputs512MiB. Normal transfer exits and actual loading, no migration-content tests. Executor seal22:37:06.331729UTC; project stays active after single fixedrecipe closure.

Construction reference: [Dai and Seljak, Sliced Iterative Normalizing Flows, PMLR139 (2021)](https://proceedings.mlr.press/v139/dai21a/dai21a.pdf), iterated orthogonal one-dimensional transport. This registered implementation uses finite QR steps and linear quantile knots rather than the paper's optimization/interpolation choices. It is not a full reproduction, an optimum guarantee, a new algorithm or proof of full joint equality.
'''
(R/'PROTOCOL.md').write_text(protocol,'utf8')
bindings={}
for p in R.glob('*.py'):
    ast.parse(p.read_text('utf8'));bindings[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(R/'LOCAL_PREGEN_FREEZE.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
    'entries':bindings,'ast_pass':True,'source_only':True},indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'AST_files':len(bindings),'source_only':True,'ready_for_unique_fit':True}),flush=True)
