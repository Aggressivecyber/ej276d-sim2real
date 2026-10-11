"""Freeze R66 implementation and protocol before any new scientific computation."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,sys,shutil,math
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R66_EQUAL_COST_PILOT_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert not(P/'REGISTRATION.json').exists()
retire=json.loads((P/'R65_RETIREMENT_RECEIPT.json').read_text())
assert retire['native_exit']==-15 and retire['durable_checkpoint_update']==500
old=R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001';mini=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'
c=json.loads((old/'config.json').read_text());now=datetime.now(timezone.utc)
records=[]
for name,directory in [('full',old),('mini',mini)]:
    text=(directory/'train_final.py').read_text('utf8');tree=ast.parse(text)
    loops=[n for n in ast.walk(tree)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='step']
    assert len(loops)==1
    loop=loops[0];body=[]
    for n in loop.body:
        if isinstance(n,ast.If)and(('progress_interval_updates'in ast.unparse(n.test))if name=='full'else('step % 1000'in ast.unparse(n.test))):break
        body.append(n)
    assert len(body)>20 and len(body)<len(loop.body)
    module=ast.Module(body=body,type_ignores=[]);source=ast.unparse(module)+'\n'
    assert ast.dump(ast.parse(source),include_attributes=False)==ast.dump(module,include_attributes=False)
    target=S/(name+'_step.py');target.write_text(source,encoding='utf8');compile(source,str(target),'exec')
    records.append(dict(arm=name,origin=str(directory/'train_final.py'),origin_SHA256=sha(directory/'train_final.py'),
        origin_first_line=body[0].lineno,origin_last_line=body[-1].end_lineno,output=target.name,output_SHA256=sha(target),
        normalized_AST_SHA256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest(),AST_identical=True))
fixed=c['lr_final']+.5*(c['lr']-c['lr_final'])*(1+math.cos(math.pi*(501-c['warmup_updates'])/(c['updates']-c['warmup_updates'])))
remote='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R66_ROOT_'+now.strftime('%Y%m%d_%H%M%S_UTC')
pc=dict(stage='R66_EQUAL_COST_PILOT_001',registered_UTC=now.isoformat(),remote_root=remote,R65_root=c['remote_root'],
    server_python=c['server_python'],CPU_threads=12,RSS_limit_bytes=48*1024**3,GPU_limit_bytes=10*1024**3,
    MemAvailable_min_bytes=c['MemAvailable_min_bytes'],science_budget=600,deadline_UTC=None,output_limit_bytes=None,
    allowed_scientific_entries=['pilot.py'],GPU_scientific_entries=['pilot.py'],
    checkpoint_SHA256=retire['checkpoint_SHA256'],checkpoint_model_SHA256='f5b02963603abb2566d8e3d757e1efd6476fcb2cf1ffb69ad366e85f6867f9d0',
    fixed_learning_rate=fixed,full_updates=30,full_arm_cap_s=180,worker_cap_s=600,
    probe_seeds=list(range(202610116601,202610116609)),REAL_control_seed=202610116699,bootstrap_seed=202610116680,
    source_parents=6593,REAL_parents=21047,original_acceptance_unchanged=True,automatic_long_resume=False,
    diagnostic_only=True,original_R65_terminal_updates=50000)
write(S/'config.json',pc)
shutil.copy2(old/'linux_owned_worker.py',S/'linux_owned_worker.py')
sup=(old/'linux_supervise.py').read_text('utf8')
assert sup.count("if any(cfg[k] is not None for k in ['science_budget','deadline_UTC','output_limit_bytes']):")==1
sup=sup.replace('import os, sys, json, time, uuid, subprocess, signal, shutil','import os, sys, json, time, uuid, subprocess, signal, shutil, threading')
sup=sup.replace("if any(cfg[k] is not None for k in ['science_budget','deadline_UTC','output_limit_bytes']):\n    raise ValueError('R61 requires explicit null budget fields')", "if cfg['science_budget'] != 600 or cfg['CPU_threads'] != 12:\n    raise ValueError('R66 requires preregistered600-second budget')")
sup=sup.replace("'compute_budget_enforced':False", "'compute_budget_enforced':True").replace("'limit_s':None", "'limit_s':600")
sup=sup.replace("start=time.monotonic()", "start=time.monotonic()\nos.environ['PILOT_START_MONOTONIC']=str(start)",1)
needle="    row['launcher_PID']=p.pid;write('REMOTE_ACTIVE_JOB.json',row);print(json.dumps({'started':row}),flush=True)"
assert sup.count(needle)==1
sup=sup.replace(needle,needle+"\n    deadline_fired=threading.Event()\n    def hard_deadline():\n        if p.poll() is None:\n            deadline_fired.set()\n            try:os.killpg(p.pid,signal.SIGKILL)\n            except ProcessLookupError:pass\n    watchdog=threading.Timer(max(0.,start+600-time.monotonic()),hard_deadline)\n    watchdog.daemon=True;watchdog.start()")
needle="row.update(actual_worker_native_exit=p.returncode,launcher_native_exit=p.returncode,"
assert sup.count(needle)==1
sup=sup.replace(needle,"watchdog.cancel()\nif deadline_fired.is_set():reason='hard600_second_worker_budget'\n"+needle)
(S/'linux_supervise.py').write_text(sup,encoding='utf8')
for f in S.glob('*.py'):compile(f.read_text('utf8'),str(f),'exec')
review=dict(UTC=now.isoformat(),root_owner='/root',agent_review='reviews/R66_EQUAL_COST_PILOT_001/design_review/DESIGN_REVIEW.md',
    agent_review_SHA256=sha(P/'design_review/DESIGN_REVIEW.md'),root_learning_rate_decision='Both arms use common fixed original update501 LR',
    exact_AST_update_bodies=records,old_numerical_suite_reused='work/R65_FULL_EMPIRICAL_D_RESPONSE_001/ARTIFICIAL_SUITE_RECEIPT.json',
    no_new_loss_or_derivative_formulas=True,numerical_path_modules_bound_by_prior115_file_review=True,
    static_compilation_pass=True,new_scientific_computation_started=False,local_guard=guard,local_memory=memory())
write(S/'ROOT_IMPLEMENTATION_REVIEW.json',review)
shutil.copy2(R/'tasks/R66_EQUAL_COST_PILOT_001.md',S/'PROTOCOL.md')
files=[]
for f in sorted(S.iterdir()):
    if f.is_file():files.append(dict(name=f.name,path=remote+'/'+f.name,bytes=f.stat().st_size,sha256=sha(f)))
# The original files stay on R65 and are checked independently against its115-file review.
for f in [old/'ROOT_IMPLEMENTATION_REVIEW.json',old/'ARTIFICIAL_SUITE_RECEIPT.json',old/'config.json']:
    files.append(dict(name='R65/'+f.name,path=c['remote_root']+'/'+f.name,bytes=f.stat().st_size,sha256=sha(f)))
write(S/'FILE_BINDING.json',dict(files=files,scope='Frozen pilot plus immutable R65 definitions/receipt; all data separately checked by R65 binding'))
backup=P/'before_registration_metadata';backup.mkdir()
for name in ['PROJECT_PROGRESS.json','PROGRAM.json','AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:shutil.copy2(R/name,backup/name)
stage='R66_EQUAL_COST_PILOT_001';task='tasks/'+stage+'.md'
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
    d=json.loads((R/name).read_text('utf8'))
    d.update(updated_UTC=now.isoformat(),updated_at_utc=now.isoformat(),status='R66_registered_capped_benefit_pilot',
        project_status='R66_registered_capped_benefit_pilot',application_goal_status='active',current_stage=stage,current_task=task,
        current_activity='equal_cost_benefit_pilot_preparation',current_stage_progress='registered_not_started',
        active_scientific_jobs=0,current_job=None,current_stage_registration='reviews/R66_EQUAL_COST_PILOT_001/REGISTRATION.json',
        current_stage_scope='shared_checkpoint_fixed_LR_equal_cost_TRAIN_pilot',
        next_action='Run one preregistered600-second pilot after frozen installation/admission; no automatic long-run restart.',
        goal_status_note='Goal resumed by subsequent goal continuation; original full A-F remains incomplete. Latest user cost objection controls R65 continuation.',
        execution_coordination='Root leads and reviews. One scoped gpt-6.1-sol/max static review completed; no active auxiliary agents.',
        latest_user_agent_policy='Scoped auxiliary analysis uses gpt-6.1-sol/max with root leading and reviewing.',
        compute_budget_enforced=True,deadlines_enforced=True,execution_policy='User-requested benefit-first pilot capped600 seconds',
        latest_permission_status='User asks proof before high cost; one capped comparison, expensive R65 remains on cost hold.')
    d['R65_cost_hold']=dict(status='retired_incomplete_cost_hold',native_exit=-15,retirement_receipt='reviews/R66_EQUAL_COST_PILOT_001/R65_RETIREMENT_RECEIPT.json',
        completed_updates_before_stop=613,inflight_update=614,durable_checkpoint_update=500,automatic_resume=False,original_terminal_updates=50000)
    d['latest_user_execution_policy'].update(R65_continuation='retired_incomplete_cost_hold',new_scientific_pilot_registered=True,
        pilot_stage=stage,pilot_wall_cap_s=600,old_50000_endpoint_role='Incomplete original R65 retained; diagnostic pilot cannot replace its scientific endpoint')
    write(R/name,d)
header='# Current: R66 capped equal-cost benefit pilot registered\n\nR65 processes have exited (-15) and176 files were archived; original update500 checkpoint hash is unchanged. The50000-update endpoint remains incomplete and on cost hold. R66 is a separate, single-worker600-second TRAIN-only local comparison from the same model/Adam/RNG state, with common fixed update501 learning rate, full30 updates versus equal-time original mini updates, eight common full-parent probes and one complete same-treated-REAL control per model. The baseline cost gate must fit all work; otherwise stop inconclusive without extending or dropping metrics. No DEV/EVAL, population reduction, A-F changes or automatic long-run restart. Positive local evidence permits only a separately capped confirmation. Root leads/reviews; the scoped6.1-sol/max static review is finished. All new local files remain on E/D. See tasks/R66_EQUAL_COST_PILOT_001.md.\n\n## Earlier records\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/name).write_text(header+(R/name).read_text('utf8'),encoding='utf8')
receipt=dict(UTC=now.isoformat(),stage=stage,status='REGISTERED_BEFORE_SCIENTIFIC_EXECUTION',protocol=task,protocol_SHA256=sha(R/task),
    implementation_review_SHA256=sha(S/'ROOT_IMPLEMENTATION_REVIEW.json'),file_binding_SHA256=sha(S/'FILE_BINDING.json'),
    remote_root=remote,resource_limits=dict(CPU_threads=12,RSS_bytes=48*1024**3,GPU_bytes=10*1024**3,wall_seconds=600),
    source_checkpoint_SHA256=retire['checkpoint_SHA256'],new_scientific_computation_started=False,original_A_F_unchanged=True)
write(P/'REGISTRATION.json',receipt)
print(json.dumps(receipt,indent=2))
