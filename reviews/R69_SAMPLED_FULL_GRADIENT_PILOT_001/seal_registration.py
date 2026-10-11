"""Seal one prospective root-reviewed attempt before any scientific execution."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,shutil,ast
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R69_SAMPLED_FULL_GRADIENT_PILOT_001'
A=R/'reviews/R69_DIRECTION_DECISION_001/optimization_review'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'registration_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def row(p):return dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
assert not(P/'REGISTRATION.json').exists()and not(P/'LAUNCH_RECEIPT.json').exists()
manifest=read(A/'SHA256_MANIFEST.json')
for v in manifest['reviewed_inputs']+manifest['artifacts']:assert sha(Path(v['absolute_path']))==v['sha256'],v['path']
assert read(A/'ARTIFICIAL_CHECK_RECEIPT.json')['checks_passed']==35
assert manifest['native_exit_code']==0
assert sha(R/'PROJECT_ACCEPTANCE.json')=='b0e4c13302dc67b0eec0799e77fdbab603e29a293cb32bbc26fa0734f025ac9e'
assert sha(R/'PROJECT_ACCEPTANCE.md')=='d6f68a1e902d901bcceafaa1d23f97b1a8ff5ad9f52acf46b00b82d7a6bdb04d'
for n in ['sampled_blocks.py','check_sampled_blocks.py']:
 if not(S/n).exists():shutil.copy2(A/n,S/n)
 assert sha(S/n)==sha(A/n)
assert sha(S/'PROTOCOL.md')==sha(R/'tasks'/f'{S.name}.md')
assert sha(S/'reference_full_step.py')==sha(R/'work/R67_EQUAL_COST_REPAIR_001/full_step.py')
assert sha(S/'mini_step.py')==sha(R/'work/R67_EQUAL_COST_REPAIR_001/mini_step.py')
reference=(S/'reference_full_step.py').read_text('utf8')
prefix=reference.split("inflight = 'single_final_cast_clip_Adam'",1)[0]
prefix=prefix.replace("for group in optimizer.param_groups:\n    group['lr'] = rate\n",'')
assert (S/'reference_gradient.py').read_text('utf8').strip()==prefix.strip()
ref_ast=ast.parse((S/'reference_gradient.py').read_text('utf8'))
for n in ast.walk(ref_ast):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):assert n.func.attr not in {'step','backward','clip_grad_norm_'}
now=datetime.now(timezone.utc).isoformat()
C=read(S/'config.json');C['registered_UTC']=now;write(S/'config.json',C)
for f in [*S.glob('*.py'),P/'control.py',P/'seal_registration.py']:compile(f.read_text('utf8'),str(f),'exec')
sources=[row(f)for f in sorted(S.iterdir())if f.is_file()]
review=dict(UTC=now,reviewer='/root',status='PASS_STATIC_ROOT_REVIEW',files=sources,
 scope='Prospective single no-update complete-gradient admission, then only the conditional capped pilot',
 independent_review='reviews/R69_DIRECTION_DECISION_001/optimization_review/OPTIMIZATION_REVIEW.md',
 unchanged_reference_full_and_mini=True,reference_is_exact_prefix_without_LR_mutation=True,
 full_population_current_D_before_sampled_VJP=True,HT_scale_once_no_block_length_reweight=True,
 independent_PCG64_block_stream=True,final_single_parent_block_retained=True,
 preclip_gradient_only_unbiased=True,clip_Adam_or_trajectory_unbiased_claim=False,
 original_C_once_clip_once_Adam_once=True,zero_model_updates_at_admission=True,
 both_arm_caps_200s_including_complete_step_overshoot=True,hard_total_cap_s=600,
 synthetic_helper_checks=35,artificial_scope='CPU exact Fraction and pure NumPy helper checks; no quality proof',
 original_A_F_unchanged=True,R65_remains_retired=True,no_scan_no_retry_no_automatic_long_resume=True,
 original_acceptance_files=[row(R/'PROJECT_ACCEPTANCE.json'),row(R/'PROJECT_ACCEPTANCE.md')],
 controller= row(P/'control.py'))
write(S/'ROOT_IMPLEMENTATION_REVIEW.json',review)
bound=[dict(name=f.name,path=C['remote_root']+'/'+f.name,bytes=f.stat().st_size,sha256=sha(f))for f in sorted(S.iterdir())if f.is_file()]
old=read(R/'work/R67_EQUAL_COST_REPAIR_001/FILE_BINDING.json')['files']
bound.extend(v for v in old if v['name'].startswith(('R65/','R63/')))
write(S/'FILE_BINDING.json',dict(UTC=now,files=bound))
local=[row(f)for f in sorted(S.iterdir())if f.is_file()]+[row(P/n)for n in ['control.py','seal_registration.py']]+[row(R/'tasks'/f'{S.name}.md'),row(A/'SHA256_MANIFEST.json')]
reg=dict(UTC=now,status='REGISTERED_BEFORE_SCIENCE',stage=S.name,owner='/root',local_files=local,
 protocol='tasks/'+S.name+'.md',remote_root=C['remote_root'],total_worker_cap_s=600,
 prospective_first_endpoint='one unchanged full-gradient admission before response updates or probes',
 source_parents=6593,REAL_parents=21047,k=4,M=104,sample_seed=C['block_sampler_seed'],
 no_final_evaluation_or_new_HGB=True,old_R65_checkpoint_SHA256=C['checkpoint_SHA256'],
 next_action_on_admission_failure='Close fixed K4 with zero response training/probes; no k or seed scan',
 original_A_F_unchanged=True,model_goal_complete=False,resource=dict(guard,**memory()))
snap=P/'before_registration_metadata';snap.mkdir()
meta=['AGENTS.md','STATUS.md','PLAN.md','PROJECT_OBJECTIVE.md','PROJECT_PROGRESS.json','PROGRAM.json','jobs.json','experiments.json']
for n in meta:shutil.copy2(R/n,snap/n)
write(P/'REGISTRATION.json',reg)
current=dict(stage=S.name,status='registered_not_started',remote_root=C['remote_root'],wall_cap_s=600)
for n in ['PROJECT_PROGRESS.json','PROGRAM.json']:
 d=read(R/n);d.update(updated_UTC=now,status='R69_registered_gradient_admission',project_status='R69_registered_gradient_admission',
  current_stage=S.name,current_task='tasks/'+S.name+'.md',current_stage_registration='reviews/'+S.name+'/REGISTRATION.json',
  current_stage_scope='One no-update full-gradient admission; conditional current-law sampled-VJP equal-cost TRAIN pilot',
  current_activity='R69_registered_before_science',current_stage_progress='registered_not_started',current_job=current,
  active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,paused_remote_scientific_jobs=0,
  remote_job_state='no_active_worker_at_preflight_R65_retired',global_single_physical_job_confirmed=True,
  application_goal_status_note='Active full original A-F goal; R65 retired on user cost hold',
  application_goal_status_observed_UTC=now,model_goal_complete=False,full_A_F_proven=False,
  latest_scientific_decision='Register exactly one K4 cost admission; conditional pilot only if its frozen error and time gates pass',
  next_action='Launch the hash-bound owned600-second worker once; no expensive R65 restart',
  execution_coordination='Root controls science and delivery; two scoped auxiliary static/synthetic reviews completed',
  current_stage_response_training_runs_started=0)
 write(R/n,d)
jobs=read(R/'jobs.json');assert not any(j['stage']==S.name for j in jobs['jobs'])
jobs['jobs'].append(dict(stage=S.name,status='registered_not_started',owner='/root',registered_UTC=now,
 protocol='tasks/'+S.name+'.md',output_root='work/'+S.name,remote_root=C['remote_root'],active_scientific_jobs=0,model_goal_complete=False))
jobs.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/'jobs.json',jobs)
ex=read(R/'experiments.json')
for j in jobs['jobs']:
 if j['stage'].startswith(('R65_','R66_','R67_','R68_')):
  matches=[x for x in ex['experiments']if x['id']==j['stage']]
  e=matches[0]if matches else dict(id=j['stage'])
  if not matches:ex['experiments'].append(e)
  e.update(status=j['status'],active_scientific_jobs=0,current_job=j.get('current_job',j.get('native')),reconciliation='jobs.json and native closure receipts',updated_UTC=now)
assert not any(e['id']==S.name for e in ex['experiments'])
ex['experiments'].append(dict(id=S.name,**{k:v for k,v in jobs['jobs'][-1].items()if k!='stage'}))
ex.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/'experiments.json',ex)
header='# Current: R69 registered one bounded gradient admission\n\nThe fixed K4 candidate retains all current SOURCE/REAL distribution laws and samples four of the original104 parameter-VJP blocks. First measure its exact conditional error from one complete unchanged gradient; if either predeclared relative-MSE gate fails, stop with zero training/probes. Only a pass may enter the same-start equal-time comparison, within a total600-second worker cap. No k/seed scan or R65 restart. Original population, charge and A-F remain unchanged and incomplete. See tasks/'+S.name+'.md.\n\n## Earlier records\n\n'
for n in ['AGENTS.md','STATUS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/n).write_text(header+(R/n).read_text('utf8'),encoding='utf8')
print(json.dumps(dict(status=reg['status'],registered_UTC=now,bound_remote_files=len(bound),bound_local_files=len(local),new_scientific_execution=0)),flush=True)
