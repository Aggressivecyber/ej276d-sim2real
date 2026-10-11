"""Close the invalid R66 attempt; register one repair inside the ORIGINAL cap."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,shutil,ast
P=Path(__file__).resolve().parent;R=P.parents[1];OLD=R/'work/R66_EQUAL_COST_PILOT_001'
N=R/'reviews/R67_EQUAL_COST_REPAIR_001';S=R/'work/R67_EQUAL_COST_REPAIR_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply
guard=apply(P)
def read(f):return json.loads(f.read_text('utf8'))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def write(f,v):f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert not N.exists()and not S.exists();N.mkdir();S.mkdir()
collection=read(P/'COLLECTION_RECEIPT.json');result=read(OLD/'RESULT.json')
assert result['status']=='INCONCLUSIVE_EXECUTION'and collection['native']['actual_worker_native_exit']==1
assert result['error']=="TimeoutError('full30 arm exceeded180 seconds')"
assert read(OLD/'ENVIRONMENT.json')['CPU_affinity']==[11]
spent=collection['native']['elapsed_s'];assert spent+395<600
now=datetime.now(timezone.utc)
decision=dict(UTC=now.isoformat(),R66_status='CLOSED_INVALID_EXECUTION_NO_COMPARISON',
    cause='Read-only installer pinned itself to CPU11 and launched the science supervisor without restoring its12-CPU allocation.',
    R66_completed_full_updates='unknown_between0_and29_no_endpoint_saved',R66_mini_updates=0,
    R66_actual_native_wall_s=spent,R66_no_comparative_quality_result=True,
    root_repair_reason='Correct this demonstrated launcher error, once, within unspent original total600-second allocation; do not extend that allocation.',
    departure_from_original_no_retry_plan_explicit=True,
    original_overall_cap_s=600,repair_hard_cap_s=395,combined_max_native_wall_s=spent+395,
    repair_full_endpoint_updates=20,repair_full_arm_cap_s=120,
    endpoint_selection_basis='Remaining compute capacity and historical per-update cost only. No trained pilot endpoint exists or was examined.',
    original_R65_50000_unchanged_incomplete=True,original_AF_unchanged=True,automatic_long_resume=False,
    no_further_repair_after_this=True)
write(P/'ROOT_INVALID_CLOSURE_AND_BUDGET_REPAIR.json',decision)
(R/'R66_REPORT.md').write_text('# R66 capped pilot: invalid execution\n\nThe worker ended natively with exit1 after '+f'{spent:.6f}'+' seconds. Full30 hit its180-second arm limit. The launcher unintentionally passed CPU affinity[11] to both supervisor and worker, although torch threads were12. No full30 endpoint or mini comparison was obtained; this is not evidence for or against full-D quality. The number of completed full updates was not durably published and remains unknown (0..29). The original update500 checkpoint is unchanged; partial RAM was discarded. Eight baseline probes and one complete REAL control were preserved, but baseline values cannot demonstrate improvement.\n\nRoot registers exactly one new R67 comparison, with actual12-CPU launch verification and20 full updates, solely to fit the unspent part of the original600-second total allocation. R66 is closed and never reinterpreted. R67 cap395 seconds plus the observed R66 native time is below600. This is an explicit operational revision of the initial no-retry plan; no endpoint quality was available to select the revision. No further retry or long-run restart is authorized by this repair.\n',encoding='utf8')
c=read(OLD/'config.json');c.update(stage='R67_EQUAL_COST_REPAIR_001',registered_UTC=now.isoformat(),
    remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R67_ROOT_'+now.strftime('%Y%m%d_%H%M%S_UTC'),
    science_budget=395,worker_cap_s=395,full_updates=20,full_arm_cap_s=120,
    exact_CPU_affinity=list(range(12)),aggregate_budget_s=600,prior_attempt_native_wall_s=spent,
    prior_invalid_attempt='R66_EQUAL_COST_PILOT_001',remaining_reserve_after_repair_max_s=600-spent-395)
write(S/'config.json',c)
for name in ['full_step.py','mini_step.py','linux_owned_worker.py']:shutil.copy2(OLD/name,S/name)
pilot=(OLD/'pilot.py').read_text('utf8')
pilot=pilot.replace('DEADLINE=START+600.0\nPC=json.loads((P/\'config.json\').read_text(\'utf8\'))',
    "PC=json.loads((P/'config.json').read_text('utf8'))\nDEADLINE=START+PC['worker_cap_s']")
pilot=pilot.replace("os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:12])", "assert sorted(os.sched_getaffinity(0))==PC['exact_CPU_affinity'],'R67 requires exact12-CPU allocation before scientific imports'")
pilot=pilot.replace('full_updates=30','full_updates=PC[\'full_updates\']')
pilot=pilot.replace('arm_deadline=time.monotonic()+180',"arm_deadline=time.monotonic()+PC['full_arm_cap_s']")
pilot=pilot.replace("raise TimeoutError('full30 arm exceeded180 seconds')","raise TimeoutError('full20 arm exceeded120 seconds')")
pilot=pilot.replace('if elapsed>180:',"if elapsed>PC['full_arm_cap_s']:")
pilot=pilot.replace('if n==30:break',"if n==PC['full_updates']:break")
pilot=pilot.replace('budget_estimate=time.monotonic()-START+360+',"budget_estimate=time.monotonic()-START+2*PC['full_arm_cap_s']+")
pilot=pilot.replace('required_training_reserve_s=360',"required_training_reserve_s=2*PC['full_arm_cap_s']")
pilot=pilot.replace('limit_s=600,pass_gate=budget_estimate<=600',"limit_s=PC['worker_cap_s'],pass_gate=budget_estimate<=PC['worker_cap_s']")
pilot=pilot.replace('if budget_estimate>600:',"if budget_estimate>PC['worker_cap_s']:")
pilot=pilot.replace('the fixed600-second budget','the fixed395-second remaining budget')
pilot=pilot.replace("assert RESULT['elapsed_s']<600", "assert RESULT['elapsed_s']<PC['worker_cap_s']")
assert 'START+600'not in pilot and 'n==30'not in pilot and 'elapsed>180'not in pilot
(S/'pilot.py').write_text(pilot,encoding='utf8')
sup=(OLD/'linux_supervise.py').read_text('utf8')
sup=sup.replace("os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:cfg['CPU_threads']])", "assert sorted(os.sched_getaffinity(0))==cfg['exact_CPU_affinity'],'Actual12-CPU supervisor allocation required'")
sup=sup.replace("cfg['science_budget'] != 600", "cfg['science_budget'] != 395")
sup=sup.replace('R66 requires preregistered600-second budget','R67 requires preregistered395-second budget')
sup=sup.replace("'limit_s':600", "'limit_s':395").replace('start+600-time.monotonic()','start+395-time.monotonic()').replace('hard600_second_worker_budget','hard395_second_worker_budget')
(S/'linux_supervise.py').write_text(sup,encoding='utf8')
for f in S.glob('*.py'):compile(f.read_text('utf8'),str(f),'exec')
protocol=(R/'tasks/R66_EQUAL_COST_PILOT_001.md').read_text('utf8')
protocol=protocol.replace('# R66: capped local equal-cost benefit test','# R67: one corrected launch within the original cumulative budget')
protocol=protocol.replace('600 seconds of worker wall time','395 seconds of worker wall time').replace('Full arm: exactly 30 completed updates','Full arm: exactly 20 completed updates').replace('180 seconds maximum for this arm','120 seconds maximum for this arm')
protocol=protocol.replace('elapsed + 360 + 2*1.10*E0 + 20 <= 600 seconds','elapsed + 240 + 2*1.10*E0 + 20 <= 395 seconds')
prefix='R66 has been closed as an invalid execution: native time '+f'{spent:.9f}'+' seconds, full30 timeout caused by unintended single-CPU affinity, no comparative endpoint. Root explicitly revises the original no-retry plan for ONE corrected launch within the SAME cumulative600-second allocation. R67 cap395 plus R66 actual time is '+f'{spent+395:.9f}'+' seconds. No additional budget is allocated. Full20 is fixed before any endpoint quality observation, solely to fit remaining time. All scientific loss formulas, starting state, LR, eight probe seeds, all parent populations, harm thresholds, primary criteria and control remain those of R66. Actual CPU affinity must be exactly0..11 before scientific imports, and torch threads12. No further retry follows this attempt.\n\n'
protocol=protocol.replace('\n\nThe user asked','\n\n'+prefix+'The user asked',1)
(R/'tasks/R67_EQUAL_COST_REPAIR_001.md').write_text(protocol,encoding='utf8');(S/'PROTOCOL.md').write_text(protocol,encoding='utf8')
review=read(OLD/'ROOT_IMPLEMENTATION_REVIEW.json')
review.update(UTC=now.isoformat(),repair_decision=decision,pilot_source_SHA256=sha(S/'pilot.py'),
    frozen_update_body_hashes_unchanged=all(sha(S/n)==sha(OLD/n)for n in ['full_step.py','mini_step.py']),
    new_loss_derivative_or_metric_changes=False,execution_only_policy_changes=['full20 endpoint','120-second arm cap','395-second worker cap','240-second training reserve','mandatory12-CPU allocation'],
    static_compilation_pass=True,new_scientific_computation_started=False)
write(S/'ROOT_IMPLEMENTATION_REVIEW.json',review)
rows=[]
for f in sorted(S.iterdir()):rows.append(dict(name=f.name,path=c['remote_root']+'/'+f.name,bytes=f.stat().st_size,sha256=sha(f)))
rows.extend(x for x in read(OLD/'FILE_BINDING.json')['files']if x['name'].startswith(('R65/','R63/')))
write(S/'FILE_BINDING.json',dict(files=rows))
before=N/'before_registration_metadata';before.mkdir()
for name in ['PROJECT_PROGRESS.json','PROGRAM.json','AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:shutil.copy2(R/name,before/name)
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
    d=read(R/name);d.update(status='R67_registered_single_budget_repair',project_status='R67_registered_single_budget_repair',
        current_stage=c['stage'],current_task='tasks/'+c['stage']+'.md',active_scientific_jobs=0,current_job=None,
        current_activity='corrected_launch_within_original_budget',current_stage_progress='registered_not_started',
        current_stage_registration='reviews/R67_EQUAL_COST_REPAIR_001/REGISTRATION.json',
        next_action='One exact12-CPU R67 launch,20 full updates,395-second hard cap; cumulativeR66+R67 below600. No further retry or long-run resume.',
        latest_diagnostic_stage='R66_EQUAL_COST_PILOT_001',latest_completed_stage='R66_EQUAL_COST_PILOT_001',
        latest_completed_report='R66_REPORT.md',latest_scientific_decision='R66 invalid execution yields no benefit evidence; exactly one corrected capped comparison preregistered before any trained endpoint observation.')
    d['latest_user_execution_policy'].update(pilot_stage=c['stage'],pilot_wall_cap_s=395,cumulative_benefit_test_cap_s=600,prior_invalid_attempt_actual_wall_s=spent)
    write(R/name,d)
header='# Current: R67 single corrected comparison inside the original600-second total budget\n\nR66 is closed INVALID: full30 timeout after unintended single-CPU launch, no comparative endpoint. Root corrects the launcher once under the unspent original budget: R67 full20 versus equal-time mini, worker cap395 seconds, exact CPU0..11 checked before imports. Together with observed R66 native time201.553185 seconds, total cap is596.553185 seconds. Eight probes, full populations, starting model/Adam/RNG, fixedLR and metrics remain unchanged. No further retry or expensive R65 restart. Original A-F and50000 remain incomplete. See tasks/R67_EQUAL_COST_REPAIR_001.md and reviews/R66_EQUAL_COST_PILOT_001/ROOT_INVALID_CLOSURE_AND_BUDGET_REPAIR.json.\n\n## Earlier records\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/name).write_text(header+(R/name).read_text('utf8'),encoding='utf8')
receipt=dict(UTC=now.isoformat(),stage=c['stage'],status='REGISTERED_BEFORE_NEW_SCIENTIFIC_EXECUTION',
    protocol_SHA256=sha(S/'PROTOCOL.md'),FILE_BINDING_SHA256=sha(S/'FILE_BINDING.json'),remote_root=c['remote_root'],
    root_decision=decision,root_implementation_review_SHA256=sha(S/'ROOT_IMPLEMENTATION_REVIEW.json'),local_guard=guard)
write(N/'REGISTRATION.json',receipt)
print(json.dumps(dict(stage=c['stage'],remote_root=c['remote_root'],new_cap_s=395,cumulative_max_s=spent+395,protocol_SHA256=sha(S/'PROTOCOL.md')),indent=2))
