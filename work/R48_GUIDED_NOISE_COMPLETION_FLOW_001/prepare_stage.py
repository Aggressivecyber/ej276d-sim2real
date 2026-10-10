from pathlib import Path
from datetime import datetime, timezone
import json, subprocess, sys, os, hashlib, ast

R=Path(__file__).resolve().parent
P=R.parent.parent
BASE='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305'
REMOTE=BASE+'/R48_REMOTE_20261010_073406_UTC'
PY='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'

def write(name, data):
    (R/name).write_bytes((json.dumps(data,indent=2,allow_nan=False)+'\n').encode('utf8'))

cfg=json.loads((P/'work/R46_FIXED_INTEGRATION_BUDGET_001/config_remote.json').read_text('utf8'))
cfg.update(stage=R.name,registered_UTC='2026-10-10T07:34:06.929385+00:00',
 deadline_UTC='2026-10-10T13:04:06.929385+00:00',absolute_deadline_UTC='2026-10-10T13:34:06.929385+00:00',
 remote_root=REMOTE,server_python=PY,CPU_threads=12,CPU_local_threads=4,RSS_limit_bytes=48*1024**3,
 GPU_allocated_limit_bytes=10*1024**3,GPU_limit_bytes=10*1024**3,output_limit_bytes=160*1024**2,
 science_budget=14400,training_wall_s=5400,training_total_limit_s=10800,updates=50000,
 arms=['GUIDED_FIXED_NOISE_CONTROL','GUIDED_NOISE_COMPLETION'],
 endpoints=['GUIDED_FIXED_NOISE_CONTROL','GUIDED_FIXED_NOISE_REAL_CONTROL','GUIDED_NOISE_COMPLETION','GUIDED_COMPLETION_REAL_CONTROL'],
 numerical_order=['GUIDED_FIXED_NOISE_CONTROL','GUIDED_NOISE_COMPLETION'],
 model_seed=20261010480,PCG64_seeds={'source_t':20261010481,'source_epsilon':20261010482,'identity_t':20261010483,'identity_epsilon':20261010484},
 loss_weights=[.5,.5],training_cycle_batches=2950,architecture='R43 scalar time U-Net with full fixed guide, input channels3',
 TRAIN_path=BASE+'/R33_REMOTE_20261009_175239_UTC/inputs/TRAIN_PRIVATE.npz',
 original_stream=BASE+'/R43_REMOTE_20261009_233131_UTC/SCALAR_TIME_FLOW_FINAL/ACTUAL_CFM_STREAM_PRIVATE.npz',
 constants_path=BASE+'/R43_REMOTE_20261009_233131_UTC/UNIT_CHARGE_CONSTANTS.npz',
 train_solver_input=BASE+'/R46_REMOTE_20261010_014939_UTC/R42_TRAIN_SOLVER_CHECK_PRIVATE.npz',
 source_stream_local=str(P/'work/R43_SCALAR_TIME_NEURAL_FLOW_001/SCALAR_TIME_FLOW_FINAL/ACTUAL_CFM_STREAM_PRIVATE.npz'),
 objective='0.5 SOURCE tangent velocity MSE + 0.5 same REAL identity tangent velocity MSE',
 selection='fixed50000 each only; both64/128 TRAIN gates before DEV',
 HGB_max_fits=40,HGB_cap=40,HGB_plan=40,density_EM_max=32,width_slots=32,main_generation_rows=8000,replay_rows=8000,
 old_G_reader_D_used=False,old_response_weights_loaded=False,parameter_initialization='fresh Torch seed20261010480 same each arm')
cfg['endpoint_weight_paths']={a:REMOTE+'/'+a+'/FINAL_FLOW.pt' for a in cfg['arms']}
cfg['endpoint_weight_paths']['GUIDED_FIXED_NOISE_REAL_CONTROL']=cfg['endpoint_weight_paths']['GUIDED_FIXED_NOISE_CONTROL']
cfg['endpoint_weight_paths']['GUIDED_COMPLETION_REAL_CONTROL']=cfg['endpoint_weight_paths']['GUIDED_NOISE_COMPLETION']
oldkeys=['t_seed','initial_noise_V','first_actual_command','inherited_updates','new_updates','total_updates',
 'training_t_rule','training_white_seed','probe_selection_seed','probe_t_seed','probe_white_seed','probe_selection_order',
 'numerical_weights_registered','parameter_initialization_unused','inherited_weight','inherited_optimizer','inherited_coupling',
 'inherited_train_receipt','frozen_R42_coupling','frozen_R42_train_receipt','historical_R42_initial_state','R43_root']
cfg['historical_metadata']={k:cfg.pop(k)for k in oldkeys if k in cfg}
cfg['execution']={'training':'R48 server CUDA fresh guide+REAL-identity 2 arms sequential',
 'numeric_generation_evaluation_replay':'R48 server CPU12/chunk64','local_auxiliary':'CPU4 E/D',
 'actual_PID_nonce_RSS_before_imports':True}
cfg['first_actual_command']=json.loads((R/'EXECUTOR_ACK.json').read_text('utf8'))
write('config.json',cfg)
for name in ['linux_owned_worker.py','linux_supervise.py','launch_server.py']:
    s=(P/'work/R46_FIXED_INTEGRATION_BUDGET_001'/name).read_text('utf8')
    if name=='linux_supervise.py':
        s=s.replace("['check_models.py','generate_all.py','evaluate_saved.py','replay_all.py']", "['calibrate.py','train_control.py','train_completion.py','check_models.py','generate_all.py','evaluate_saved.py','replay_all.py']")
        s=s.replace("cfg['training_total_limit_s']-training_used if entry.startswith('train_')else remaining", "min(cfg['training_wall_s'],cfg['training_total_limit_s']-training_used) if entry.startswith('train_')else remaining")
        s=s.replace("item.get('global_step')", "(item.get('arm'),item.get('global_step'))")
        a=s.index("if meminfo['MemAvailable']<60129542144:")
        b=s.index("if entry.startswith('train_')and",a)
        s=s[:a]+"if meminfo['MemAvailable']<60129542144:raise RuntimeError('R48 startup MemAvailable below registered56GiB; no implicit lower quota')\n"+s[b:]
        s=s.replace("while p.poll()is None:\n", "while p.poll()is None:\n",1)
    if name=='launch_server.py':
        s=s.replace("['check_models.py','generate_all.py','evaluate_saved.py','replay_all.py']", "['calibrate.py','train_control.py','train_completion.py','check_models.py','generate_all.py','evaluate_saved.py','replay_all.py']")
    (R/name).write_bytes(s.replace('\r\n','\n').replace('\r','').encode('utf8'))
s=(P/'work/R43_SCALAR_TIME_NEURAL_FLOW_001/network.py').read_text('utf8')
s=s.replace('nn.Conv1d(2,32,3,padding=1)','nn.Conv1d(3,32,3,padding=1)').replace('def forward(self,y,t,b):','def forward(self,y,t,b,g):')
s=s.replace('torch.cat((y[:,None,:],self.position.to(y)', 'torch.cat((y[:,None,:],g.to(y)[:,None,:],self.position.to(y)')
(R/'network.py').write_bytes(s.replace('\r\n','\n').replace('\r','').encode('utf8'))
for name in ['AGENTS.md','tasks/R48_GUIDED_NOISE_COMPLETION_FLOW_001.md','reviews/R48_REGISTRATION_001/RECEIPT.json','PROJECT_ACCEPTANCE.md','PROJECT_ACCEPTANCE.json']:
    d=(P/name).read_bytes()
    if 'tasks/' in name and hashlib.sha256(d).hexdigest()!='901517e92b400f65a7206ad01e2fff8e7fbb6625ae2b0f995c1af91bd874bb8f':raise ValueError('task bytes changed')
write('PROTOCOL.json',{'task':str(P/'tasks/R48_GUIDED_NOISE_COMPLETION_FLOW_001.md'),'task_SHA':'901517e92b400f65a7206ad01e2fff8e7fbb6625ae2b0f995c1af91bd874bb8f','arms':cfg['arms'],
 'loss_weights':cfg['loss_weights'],'seeds':cfg['PCG64_seeds'],'tau_quantile':.05,'single_periodogram_kernel':True,'actual_input_b':True,'identity_b':'actual b(REAL)',
 'original_stream_resort':False,'conditioning_source_target_projection':'SOURCE b(x) for both source unit-charge and matched REAL unit-charge; ID b(r) separately',
 'solver':[64,128],'both_pass_before_DEV':True,'original_A_F':str(P/'PROJECT_ACCEPTANCE.md'),'source_population_unchanged':True,
 '160MiB_each_location':True,'migration_consistency_tests':False,'old_stages_read_only':True,'training_flow_only_cycle_original94400_rows':True})
script=(R/'server_preflight.py').read_text('utf8')
cmd=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102',PY+' -B -']
q=subprocess.run(cmd,input=script,text=True,capture_output=True,timeout=45)
(R/'SERVER_PREFLIGHT.stdout.log').write_text(q.stdout,encoding='utf8',newline='\n')
(R/'SERVER_PREFLIGHT.stderr.log').write_text(q.stderr,encoding='utf8',newline='\n')
write('RESOURCE_CHECK_NATIVE.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'native_exit':q.returncode,'argv':cmd,'science_updates':0})
if q.returncode:raise RuntimeError(('server preflight failed',q.returncode,q.stderr))
obj=json.loads(q.stdout);write('SERVER_PREFLIGHT.json',obj)
assert not obj['own_project_jobs'] and not obj['GPU_compute_apps'] and obj['GPU_free_bytes']>=11*1024**3 and obj['disk_free_bytes']>=8*1024**3
assert not obj['remote_leaf_exists']
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'preflight':obj,'source_ready':True,'science_started':False}),flush=True)
