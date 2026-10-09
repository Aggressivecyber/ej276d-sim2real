from pathlib import Path
from datetime import datetime,timezone
import json,shutil,ast,hashlib,os

ROOT=Path(__file__).resolve().parent; P=ROOT.parent.parent
OLD=ROOT.parent/'R32_CONDITIONAL_MEAN_TRANSLATION_001'
REG=json.loads((P/'reviews/R33_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
if (ROOT/'PREPARATION_RECEIPT.json').exists():raise RuntimeError('preparation already done')
cfg=json.loads((OLD/'config.json').read_text('utf8'))
cfg.update(stage=ROOT.name,registered_UTC=REG['registered_UTC'],deadline_UTC=REG['executor_seal_deadline_UTC'],
    absolute_deadline_UTC=REG['stage_deadline_UTC'],science_budget=2400,training_wall_s=900,training_total_limit_s=1800,
    arms=REG['arms'],endpoints=REG['arms'],updates=2950,HGB_max_fits=20,HGB_cap=20,HGB_plan=20,
    density_EM_max=16,width_slots=16,main_generation_rows=4000,repeated_rows_cap=2048,source_probe_rows=1664,replay_rows=4000,
    first_actual_command={'tool':'63fd7e','PID':51776,'native_exit':0,'UTC':'2026-10-09T17:53:14.399903+00:00'},
    remote_root=REG['execution']['remote_root'],server_python=REG['execution']['python'],
    R32_root=str(OLD),base_G=str(OLD/'SERVER_FRESH_CONTROL/FINAL_TRANSLATOR.pt'),
    model_seed=20261010330,t_seed=20261010331,coupling_seed=20261010332,
    flow_lr=.0002,flow_betas=[.9,.999],flow_eps=1e-8,flow_weight_decay=0,
    solver_steps=32,solver_check_steps=64,solver_check_rows_each=32,solver_error_limit_V=1/16383,
    sole_training_intervention='within original Q-bin coupling: random permutation versus minimum squared480 affine-voltage assignment',
    endpoint_selection_uses_DEV=False,selection='actual final only',sigma=0.,
    architecture='497->512 SiLU->512 SiLU->512 SiLU->480; zero last layer',
    execution={'training':'registered Linux same-platform two sequential flow fits','evaluation_replay':'local CPU4',
               'CPU_python':cfg['CPU_python'],'actual_PID_nonce_RSS_before_imports':True},
    endpoint_weight_paths={a:str(ROOT/a/'FINAL_FLOW.pt')for a in REG['arms']})
for name in ['model.py','legacy_model.py','reorder_core.py','evaluation.py','diagnostic_core.py','assessment.py',
             'energy_width.py','data_support.py','endpoint_common.py','temporal_common.py','common_math.py',
             'mean_description.py','supervise.py','owned_worker.py','linux_owned_worker.py','launch_remote.py']:
    shutil.copy2(OLD/name,ROOT/name)
for name in ['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz',
             'DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json',
             'DATA_ROLES.json','VALLEY_DEFINITION.json','FROZEN_TILT_TABLE.json']:
    shutil.copy2(OLD/name,ROOT/name)
for name in ['supervise.py','owned_worker.py']:
    text=(ROOT/name).read_text('utf8')
    # Keep the proven actual-PID/RSS chain, timestamp any newly observed query gaps.
    if name=='supervise.py':
        text=text.replace("'memory_probe_failures':0,","'memory_probe_failures':0,'memory_probe_failure_events':[],")
        text=text.replace("row['memory_probe_failures']+=1", "row['memory_probe_failures']+=1;row['memory_probe_failure_events'].append({'UTC':utc(),'PID':row['actual_worker_PID'],'launcher_poll':p.poll(),'context':'active_actual_PID_query'})")
    (ROOT/name).write_text(text,'utf8')
text=(OLD/'linux_supervise.py').read_text('utf8')
text=text.replace("['pid_probe.py','train_server.py']","['train_server.py']")
text=text.replace("'memory_probe_failures':0,","'memory_probe_failures':0,'memory_probe_failure_events':[],")
text=text.replace("row['memory_probe_failures']+=1", "row['memory_probe_failures']+=1;row['memory_probe_failure_events'].append({'UTC':utc(),'PID':p.pid,'poll':p.poll(),'context':'actual_PID_query'})")
(ROOT/'linux_supervise.py').write_text(text,'utf8')
payloads={'TRAIN_PRIVATE.npz':cfg['training_input_paths']['TRAIN_PRIVATE.npz'],
    'R26_TRAINING_STREAMS_PRIVATE.npz':cfg['consumed_R26_stream'],
    'R23_TRAINING_RANDOM_STATES.json':cfg['actual_training_random_states'],
    'R23_TRAINING_RECEIPT.json':cfg['actual_training_receipt'],
    'R26_TRAINING_RECEIPT.json':cfg['consumed_R26_receipt'],'BASE_TRANSLATOR.pt':cfg['base_G']}
cfg['remote_input_sources']={n:{'path':str(p),'bytes':Path(p).stat().st_size}for n,p in payloads.items()}
remote=json.loads(json.dumps(cfg)); REMOTE=cfg['remote_root']
remote['training_input_paths']={'TRAIN_PRIVATE.npz':REMOTE+'/inputs/TRAIN_PRIVATE.npz'}
for key,name in [('consumed_R26_stream','R26_TRAINING_STREAMS_PRIVATE.npz'),('actual_training_random_states','R23_TRAINING_RANDOM_STATES.json'),
                 ('actual_training_receipt','R23_TRAINING_RECEIPT.json'),('consumed_R26_receipt','R26_TRAINING_RECEIPT.json'),('base_G','BASE_TRANSLATOR.pt')]:
    remote[key]=REMOTE+'/inputs/'+name
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
write('config.json',cfg);write('remote_config.json',remote)
write('REMOTE_RESOURCE_PREFLIGHT.json',{'tool':'b380be','native_exit':0,'remote_PID':30001,'UTC':'2026-10-09T17:55:18.844901+00:00',
    'owned_project_jobs_argv_cwd':[],'MemAvailable_kB':61076292,'GPU':'RTX5070,12227MiB total,242MiB used,0percent',
    'scipy_metadata_version':'1.11.4','torch_metadata_version':'2.12.0','new_leaf_existed':False,'others_modified':False})
manifest=json.loads((OLD/'FINAL_MANIFEST.json').read_text('utf8'))
entry=next(x for x in manifest['members']if x['path']=='SERVER_FRESH_CONTROL/FINAL_TRANSLATOR.pt')
write('FROZEN_BASE_IDENTITY.json',{'path':cfg['base_G'],'binding':entry,'binding_source':str(OLD/'FINAL_MANIFEST.json'),
    'base_step':2950,'current_source_rehashed_for_migration':False,'base_G_updates_this_stage':0})
protocol='''# R33 frozen residual flow protocol

Complete source480 -> frozen R32 SERVER_FRESH_CONTROL FINAL step2950 G(z8,white480,style8) preADC -> full480 conditional flow -> one14bit ADC. No unconditional Gaussian start, REAL retrieval or target event at inference. Two arms differ only in statistical within-original-Q coupling: PCG64(20261010332) uniform permutation versus exact SciPy linear_sum_assignment on all480 common-affine squared voltage coordinates, equal weights and preserved marginals. Singleton/no-cross-Q, mixed batches split by original bins. Statistical coupling is not a physical pairing.

Common REAL TRAIN-fit21047 raw voltage means and populationSD per frozen8 Q bins, floor1e-6V, no DEV or599validation. Stored frozen actual source and REAL row-bin bindings are checked against actual signed-charge bins. Identical common affine T_b applies to both domains. Network497->512SiLU->512SiLU->512SiLU->480, zero final layer, no norm/dropout; seed20261010330, same initialstate/newAdam(.0002,.9,.999,eps1e-8,clip5), same actual94400 source/target/bin/originalordinal/oldz/white/style rows perarm. No G/D/reader updates. Same t PCG64(20261010331), f32 U[0,1); loss mean480/32 squared velocity error at linear path, sigma0. Each<=2950updates/900s, final only, no val/DEV model selection. Log all updates and actual initial G output per-batch hashes; no all-wave training cache.

CPU inference fixed32-step Heun/64NFE, common TRAIN affine inverse, exactly once final14bit[-.5,.5]ADC. Unsupported actual preADC source-Q: base fallback and saved flow_applied false, no boundary clamping. First32 DEV parents perarm64-step check once, strict maxpreADC difference<1/16383V, no step tuning. Final flow increment is not a physical signal/electronics split; keep base_signal/base_electronic/increment and total covariance/cross terms, physical final attribution unknown.

Two actual final endpoints, original2000 parents417pureEE/1518purehad/65mixed/IDs/order/weight. Use original10C2ST views600train250holdout/500clusterbootstrap, max20HGB; max16 density slots, reference fits not repeated, all undefined gaps. Baseline R32 CONTROL_FINAL and REAL-RR saved predictions/statistics reused. Full/output-Q metrics plus actual NN sourcepreADC Q213/181 purehad bins3/4 and old SOURCE_CLEAN postADC reference Q207/182 separately. Actual source-Q is primary source association/flow condition; ADC reference retained for historical comparisons only.

Gate exactly task/registration: OT vsbase full493/u480 paired upperCI<0; OT not worse points vsindependent plus>=1upperCI<0; no>10percent PSDW1/widthMAE/baselineRMS/ACF1 worsening; no newinvalid/clip, sourceQ/PSDcorr loss>.05, purehad actual-sourcevalley increase>2pp; solver/replay mandatory. Unknown support cannot pass. Mean closeness/trainingloss alone not response success, finite same-acquisition historicalDEV not freshblindtest.

2400s science incl preparation/checks/failures/train/generation/20HGB/16EM/bootstrap/plots/replay,4threads,actualPID-RSS4GiB,GPUallocated4GiB,512MiB scientific output. Server two sequential fits; localCPU generation/evaluation/replay; one science worker. Normal transfer only no migration consistency tests. E/D local storage, EVAL/CAL/newREAL/G4/install/Git/agents0. End executor19:42:39.460916UTC. All previous rounds immutable.

Method reference: Tong et al. section3.2.2/3.2.3 https://arxiv.org/html/2302.00482v3 and author conditional_flow_matching.py. General-source linear-path velocity regression used as a registered fixed recipe; no novelty or ideal-theorem success claim.
'''
(ROOT/'PROTOCOL.md').write_text(protocol,'utf8')
write('PREPARATION_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
    'first_actual_command':cfg['first_actual_command'],'actual_science_started':False,
    'reused_input_bytes':sum(v['bytes']for v in cfg['remote_input_sources'].values()),'migration_content_tests':0,'old_stage_writes':0})
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'prepared':True,
    'reused_input_bytes':sum(v['bytes']for v in cfg['remote_input_sources'].values())}))
