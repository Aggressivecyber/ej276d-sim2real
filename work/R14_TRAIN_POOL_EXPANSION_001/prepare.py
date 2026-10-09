from __future__ import annotations
import os, json, struct, shutil, ast, hashlib, datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
OLD=BASE/'work/R13_FACTORIZED_CONDITIONAL_DENSITY_001'
def write(n,v): (ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
roles=(BASE/'work/P1J_CURRENT_ROLE_INDEX_001/run_001/role.bin').read_bytes()
groups=struct.unpack('<100000H',(BASE/'work/P1J_CURRENT_ROLE_INDEX_001/run_001/group.bin').read_bytes())
idx=struct.unpack('<6800I',(BASE/'work/P1K_REAL_REFERENCE_AND_REPEATABILITY_001/selection_original_index.bin').read_bytes())
vg=json.loads((BASE/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/results/DATA_ROLES.json').read_text())['REAL_internal_validation_artificial_groups']
assert len(roles)==100000 and all(g==i//50 for i,g in enumerate(groups))
old_fit=[i for i in idx[2000:5200] if groups[i] not in vg]
candidates=[i for i,r in enumerate(roles) if r==1 and groups[i] not in vg]
oldset=set(old_fit);new=[i for i in candidates if i not in oldset]
assert len(old_fit)==2600 and len(candidates)==21100 and len(new)==18500 and len(set(groups[i] for i in candidates))==422
assert all(roles[i]==1 for i in candidates) and set(new).isdisjoint(idx) and new==sorted(new)
(ROOT/'new_original_index.bin').write_bytes(struct.pack('<18500I',*new))
(ROOT/'new_groups.bin').write_bytes(struct.pack('<18500H',*(groups[i] for i in new)))
(ROOT/'candidate_original_index.bin').write_bytes(struct.pack('<21100I',*candidates))
write('SELECTION.json',{'original_TRAIN_pool_records':21700,'TRAIN_pool_groups':434,'internal_val_groups':vg,'internal_val_candidate_records':600,'unchanged_internal_val_QC_records':599,'all_fit_candidate_records':21100,'fit_candidate_groups':422,'old_fit_candidate_records':2600,'old_fit_QC_records':2592,'new_raw_records':18500,'new_raw_groups':370,'new_order':'original_index ascending; valid rows appended after original2592','REAL_DEV_CAL_EVAL_candidates_added':0,'source_identity_string_decodes':0,'actual_QC_valid_new':'unknown_until_extraction'})
for n in ['density.py','evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','common_math.py','conditional_density.py','deploy.py','launch.py','retrieve.py']:
    s=(OLD/n).read_text(encoding='utf-8').replace('R13','R14').replace('1900','2500') if n in ['deploy.py','launch.py','retrieve.py'] else (OLD/n).read_text(encoding='utf-8')
    (ROOT/n).write_text(s,encoding='utf-8')
(ROOT/'interface_core.py').write_text((BASE/'work/R4_FROZEN_HELDOUT_RESPONSE_001/interface_core.py').read_text(encoding='utf-8'),encoding='utf-8')
cfg=json.loads((OLD/'config.json').read_text())
remote='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R14_REMOTE_20261009_044129'
cfg.update(stage='R14',registered_UTC='2026-10-09T04:41:29.425185+00:00',deadline_UTC='2026-10-09T06:01:29.425185+00:00',absolute_deadline_UTC='2026-10-09T06:11:29.425185+00:00',remote_root=remote,first_native_PID=48404,first_native_tool='2111ce',first_native_UTC='2026-10-09T04:46:07.253679+00:00',numeric_wall_limit_s=2400,HGB_max_fits=42,density_description_slots_max=32,extraction_limit_s=600,maximum_updates=16300,original_fit_N=2592,equal_updates=2000)
cfg['execution']['remote_root']=remote
cfg['remote_R13_root']='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R13_REMOTE_20261009_042246'
cfg['models']=['UPDATES_GLOBAL','UPDATES_CONDITIONAL','EXPOSURES_GLOBAL','EXPOSURES_CONDITIONAL']
cfg['all_comparison_names']=['SMALL_GLOBAL','SMALL_CONDITIONAL']+cfg['models']
cfg['C2ST_selection']='six candidate common QC, original R7 600/250 pools, sameN including REAL and RR; no expansion'
cfg['NN_selection']='all8000 new draws; original2592 and actual expanded-fit references separately, no inference from reference-cardinality differences'
cfg['remote_source']=json.loads((BASE/'work/R4_FROZEN_HELDOUT_RESPONSE_001/config.json').read_text())['remote_source']
cfg['remote_source']['memory_limit_bytes']=cfg['RSS_limit_bytes']
write('config.json',cfg)
write('START_RECEIPT.json',{'PID':48404,'first_native_tool':'2111ce','UTC':'2026-10-09T04:46:07.253679+00:00','registered_UTC':cfg['registered_UTC'],'deadline_UTC':cfg['deadline_UTC'],'scientific_cumulative_s':0,'R13_terminal':True})
(ROOT/'PROTOCOL.md').write_text('''# R14 fixed TRAIN expansion

Before any raw science: 21700 TRAIN_POOL rows minus exactly12 original internal-val groups equals21100 candidates (422 groups). Reuse old2600 candidates/2592 valid and unchanged599 validation; stream only remaining18500 rows, save processed V/QC/indices/groups, discard raw10000-point rows. No CAL/DEV/EVAL addition, source-name decoding, whole-container SHA pass or migration consistency test. Source header/CRC/EOF stream code and 20-point mean/14-bit ADC/f32 baseline and peak/CFD480 window are reused unchanged from R4. Only time axis and selected new raw row values are decoded; other decompressed bytes may pass through CRC.

Fixed R13 FACTOR basis/head scale/ridge mean and R12 empirical L were estimated on original2592 and remain fixed. Start each network from actual R13 saved original initial state, same architecture/permutations/seed/Adam lr0.0003 weightdecay1e-5 batch128 clip5. Only two continuous network fits. Each100 and step0 evaluate all original599 in float32; maintain earliest exact tie independently for LARGE_EQUAL_UPDATES (<=2000) and LARGE_EQUAL_EXPOSURES (<=U=100*ceil(2000*Nlarge/2592/100), at most16300). No optimizer restart/schedule/data-driven recipe change. Per-network600s/combined1200s; expose budget-incomplete comparisons instead of substituting one. Report original and new average draws per record separately. Same seed and initial parameters do not imply same minibatch trajectory after N changes.

Final scores CPUfloat64 with full Jacobian and frozen scales/L on original599 and entire new fit; original R13 selected networks score once on original2592 as report-only comparison. Four new2000 arms: each update budget FLOW_GLOBAL and FLOW_CONDITIONAL. Same actual head_white/tail_white/draw positions; within each budget same physical head and residual innovations for tail contrast. No redraw/rejection/qPSD clamp/newCFD. Head-change comparisons are draw-position pairing, not G4 event matching.

Compare four new arms plus R13 SMALL_GLOBAL/SMALL_CONDITIONAL and sameN RR. Original R7 fixed600/250 event/group pools only; all six common QC; no fallback fill, support150/75. Six fixed views/HGB at most42,500 paired artificial-group bootstraps conditional on fixed classifiers. Contrasts: updates minus small and exposures minus updates at each tail; conditional minus global within each new budget. No adjusted multiple-testing significance/equivalence claims. Whole QC population retained; inherited q4 failed CAL reference and q1 HC-neutron absence remain.

Eight frozen log-charge bins and original width definition; four new arms32 maximum EM description slots. Reuse REAL/R13 old tables; per-arm, pairwise and all-six common support reported explicitly, empty=undefined. Components are densities, not particle truth. All8000 new nearest queries against original2592 and actual expanded fit in bounded chunks; no-copy is not no-memory, smaller distance to a larger reference is not evidence of memorization. Model replay once in new process, all8000 actual inputs, premax1e-10V/post array_equal; no fit or new random draws. No repeated Jacobian audit.

Registered04:41:29.425185UTC, executor seal06:01:29.425185, overall06:11:29.425185. Numeric incl extraction/train/eval/plots/replay2400s, extract600s, pernet600s and totaltrain1200s. Single scientific child at a time, CPU4/RSS4GiB/GPUallocated4GiB/output512MiB. All local writes E/D; isolated remote directory. Historical stages read-only; no Git/install/newG4/EVAL. Negative scientific results kept; ordinary engineering failures cost budget. Larger sample remains one acquisition, and added REAL events count toward paper A costs. This target-density comparison does not yet establish G4 response or utility.
''',encoding='utf-8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_raw':len(new),'all_candidates':len(candidates),'scientific_started':False}))
