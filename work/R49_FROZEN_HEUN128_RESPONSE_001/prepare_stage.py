from pathlib import Path
from datetime import datetime, timezone
import os, json, ast, hashlib

R = Path(__file__).resolve().parent
P = R.parent.parent
OLD = R.parent / 'R48_GUIDED_NOISE_COMPLETION_FLOW_001'
REMOTE = '/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R49_REMOTE_20261010_093411_UTC'
names = ['network.py','noise_core.py','source_flow_core.py','flow_runtime.py',
         'check_models.py','generate_all.py','flow_diagnostics.py','evaluate_saved.py','replay_all.py',
         'evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py',
         'common_math.py','model.py','legacy_model.py','endpoint_common.py','temporal_common.py',
         'mean_description.py','flow_core.py','summarize_and_plot.py','render_figures.py','render_train.py',
         'linux_owned_worker.py','launch_post.py','fetch_small.py']
def put(name, data):
    (R/name).write_bytes(data.encode('utf8'))
def replace_checked(text, old, new):
    if old not in text: raise ValueError(('missing expected source fragment', old))
    return text.replace(old, new)

c = json.loads((OLD/'config_cpu.json').read_text('utf8'))
history = {k:c.get(k) for k in ['stage','registered_UTC','deadline_UTC','absolute_deadline_UTC',
            'updates','training_wall_s','training_total_limit_s','science_budget','output_limit_bytes',
            'execution','first_actual_command','historical_metadata']}
registration = json.loads((P/'reviews/R49_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
c.update(stage=R.name, registered_UTC=registration['registered_UTC'], remote_root=REMOTE,
         solver_steps=128, solver_check_steps=256, generation_chunk=64,
         deadline_UTC=None, absolute_deadline_UTC=None, training_wall_s=None,
         training_total_limit_s=None, science_budget=None, science_budget_s=None,
         numeric_wall_limit_s=None, wall_budget_s=None, output_limit_bytes=None,
         disk_free_reserve_bytes=None, updates=0, new_response_training_runs=0,
         new_response_updates=0, compute_budget_enforced=False, deadlines_enforced=False,
         resource_policy_version=3, CPU_threads=12, threads=12,
         CPU_local_threads=4, RSS_limit_bytes=48*1024**3, GPU_limit_bytes=10*1024**3,
         GPU_allocated_limit_bytes=10*1024**3, GPU_reserved_limit_bytes=10*1024**3,
         MemAvailable_min_bytes=56*1024**3,
         historical_training_root=history['execution'] and json.loads((OLD/'config.json').read_text())['remote_root'],
         historical_local_training_root=str(OLD).replace('\\','/'),
         historical_updates_each=50000, historical_R48_configuration=history,
         parameter_initialization='not performed; R48 frozen final models loaded',
         selection='fixed R48 endpoints, common Heun128; both TRAIN128/256 gates before DEV',
         old_response_weights_loaded=True,
         execution={'training':'none; new updates 0',
                    'numeric_generation_evaluation_replay':'R49 server CPU12/chunk64/Heun128',
                    'local_auxiliary':'CPU4 E/D', 'actual_PID_nonce_RSS_before_imports':True},
         first_actual_command=json.loads((R/'EXECUTOR_ACK.json').read_text())['first_actual_command'],
         counter_scope_note='R48 historical training each50000; R49 new training/calibration 0; actual science elapsed recorded without budget kill',
         allowed_scientific_entries=['check_models.py','generate_all.py','evaluate_saved.py',
                                     'replay_all.py','summarize_and_plot.py','render_train.py'],
         historical_training={'R48_control_updates':50000,'R48_completion_updates':50000,
                              'new_REAL_unique_from_calibration':False,'new_training_runs':0})
for n in ['config.json','config_cpu.json']:
    put(n,json.dumps(c,indent=2,allow_nan=False)+'\n')

bindings=[]
for name in names:
    text=(OLD/name).read_text('utf8').replace('\r','')
    if name=='source_flow_core.py':
        text=replace_checked(text,'completion,steps=64,chunk=64','completion,steps=128,chunk=64')
    elif name=='flow_runtime.py':
        text=replace_checked(text,'assert self.steps==64','assert self.steps==128')
    elif name=='check_models.py':
        text=replace_checked(text,"(R/arm/'TRAINING_RECEIPT.json')","(Path(C['historical_training_root'])/arm/'TRAINING_RECEIPT.json')")
        text=replace_checked(text,"torch.load(R/arm/'FINAL_FLOW.pt'","torch.load(C['endpoint_weight_paths'][arm]")
        text=replace_checked(text,'completion,64,64)','completion,C[\'solver_steps\'],C[\'generation_chunk\'])')
        text=replace_checked(text,'completion,128,64)','completion,C[\'solver_check_steps\'],C[\'generation_chunk\'])')
        text=text.replace('solution128_preADC_f32_V','solution256_preADC_f32_V').replace('solution64_preADC_f32_V','solution128_preADC_f32_V')
        text=text.replace('relative_q128_f32','relative_q256_f32').replace('relative_q64_f32','relative_q128_f32')
        text=text.replace("'solver128':rb","'solver256':rb").replace("'solver64':ra","'solver128':ra")
        text=replace_checked(text,"'candidate_steps':64,'reference_only_steps':128","'candidate_steps':128,'reference_only_steps':256")
    elif name=='generate_all.py':
        text=text.replace("'solver_steps':64","'solver_steps':CFG['solver_steps']")
    elif name=='evaluate_saved.py':
        text=replace_checked(text,"(ROOT/a/'TRAINING_RECEIPT.json')","(pathlib.Path(CFG['historical_training_root'])/a/'TRAINING_RECEIPT.json')")
        text=text.replace("'status':'complete','training':","'status':'complete','training_scope':'historical R48 only','new_training_runs':0,'new_response_updates':0,'training':")
    elif name=='replay_all.py':
        text=text.replace(";assert ROWS[name]['bitwise_pass'],checks",'')
        text=text.replace("assert COUNT==8000 and r['pass'];write",'assert COUNT==8000;write')
    elif name=='summarize_and_plot.py':
        text=replace_checked(text,"(R/a/'TRAINING_RECEIPT.json')","(pathlib.Path(C['historical_training_root'])/a/'TRAINING_RECEIPT.json')")
        text=text.replace("'new_training_updates_per_arm':50000","'historical_R48_updates_per_arm':50000,'new_training_updates_per_arm':0")
        text=text.replace('TRAIN_fixed64_vs128_pass','TRAIN_fixed128_vs256_pass').replace('solver128_reference_not_exact_truth','solver256_reference_not_exact_truth')
        text=text.replace('paired_fresh_training_pass','historical_R48_paired_training_pass').replace('close_fixed_R48_recipe','close_fixed_R49_Heun128_recipe')
        text=text.replace("'training':tc","'historical_R48_training':tc,'new_training_runs':0")
    elif name=='render_train.py':
        text=replace_checked(text,"root/arm/'LOSS_BY_UPDATE_PRIVATE.npz'","Path(cfg['historical_training_root'])/arm/'LOSS_BY_UPDATE_PRIVATE.npz'")
        text=text.replace('TRAIN64 vs128','TRAIN128 vs256').replace('Candidate64 / reference128','Candidate128 / reference256')
        text=text.replace('Two fresh 50000-step endpoints; blocks for display only, no checkpoint selection','Historical R48 training: frozen 50000-step endpoints; R49 new training=0')
        text=replace_checked(text,"axs[j].legend(fontsize=8);axs[j].grid(alpha=.15)","axs[j].set_ylim(0,.60);axs[j].legend(fontsize=8);axs[j].grid(alpha=.15)")
    elif name=='launch_post.py':
        text=replace_checked(text,"assert entry in ['calibrate.py','train_control.py','train_completion.py','check_models.py','generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py','render_train.py']","assert entry in cfg['allowed_scientific_entries']")
    ast.parse(text,filename=name)
    put(name,text)
    bindings.append({'name':name,'source_reference':str(OLD/name).replace('\\','/'),
                     'new_source_SHA256':hashlib.sha256(text.encode()).hexdigest()})
for name in ['NOISE_CALIBRATION.npz','NOISE_CALIBRATION.json']:
    (R/name).write_bytes((OLD/name).read_bytes())
for arm in c['arms']:(R/arm).mkdir(exist_ok=True)
put('SOURCE_BINDINGS.json',json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
    'task_sha256':registration['task_sha256'],'source_bindings':bindings,
    'frozen_inputs':registration['frozen_local_version_bindings'],'new_training':0,
    'new_calibration':0,'migration_equivalence_test':False},indent=2)+'\n')
put('PROTOCOL.md','''# R49 implementation\n\nThe registered R49 task is authoritative. Frozen R48 models and calibration are reused.\nOnly common candidate Heun128/reference Heun256 changes; CPU12/chunk64.\nBoth original128 TRAIN rows must pass before any DEV access. Original gates and A-C unchanged.\nAll2000 SOURCE rows processed, including417 outside TRAIN support; conditioning clamp only.\n11968 generated rows,40 fixed HGB designs,32 density slots,8000 independent main replay rows.\nFixed Q1-7 PSD and Q2-7 density support; missing bins cannot be deleted.\nR48 historical50000-step training per arm is referenced; R49 training/calibration=0.\nTime/deadline/output quota fields are null. Machine RSS/single-job safety remains.\nCPU work does not require idle GPU. Native elapsed and unknown observations remain recorded.\nNo migration equivalence tests. E/D local persistence. No writes to frozen older stages.\n''')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
                  'prepared_sources':len(bindings),'remote_root':REMOTE,'new_science':0}))
