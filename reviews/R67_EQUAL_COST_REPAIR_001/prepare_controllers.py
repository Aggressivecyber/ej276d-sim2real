"""Create controllers with explicit child CPU allocation; no science or upload."""
from pathlib import Path
import ast,json,hashlib,sys
P=Path(__file__).resolve().parent;R=P.parents[1];OLD=R/'reviews/R66_EQUAL_COST_PILOT_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply
apply(P)
s=(OLD/'install_launch_observe.py').read_text('utf8')
start=s.index('    # Bind the original mini runner')
end=s.index('    package=P/',start)
s=s[:start]+"    binding=json.loads((S/'FILE_BINDING.json').read_text())\n    assert any(x['name']=='R63/train_final.py'for x in binding['files'])\n"+s[end:]
s=s.replace('R66_','R67_').replace('R66 native','R67 native').replace('R67_EQUAL_COST_PILOT_001','R67_EQUAL_COST_REPAIR_001')
needle="stdout=out,stderr=err,start_new_session=True)"
assert s.count(needle)==1
s=s.replace(needle,"stdout=out,stderr=err,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(cfg['exact_CPU_affinity'])))")
s=s.replace("wall_cap_s=600", "wall_cap_s=C['worker_cap_s']")
s=s.replace("cwd=str((f/'cwd').resolve()))", "cwd=str((f/'cwd').resolve()),CPU_affinity=sorted(os.sched_getaffinity(pid)))")
needle="old=Path(cfg['R65_root']);retired=json.loads((old/'R65_COST_HOLD_TERMINATION.json').read_text())"
assert s.count(needle)==1
s=s.replace(needle,needle+"\nprior=Path('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R66_ROOT_20261011_070346_UTC')\nprior_native=json.loads((prior/'pilot.py.NATIVE_RECEIPT.json').read_text())\nassert prior_native['actual_worker_native_exit']==1\nassert prior_native['elapsed_s']+cfg['worker_cap_s']<cfg['aggregate_budget_s']\nfor key in ['actual_worker_PID','supervisor_PID']:\n q=Path('/proc')/str(prior_native[key])\n if q.exists():assert(q/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'")
compile(s,'R67_install_launch_observe','exec');(P/'install_launch_observe.py').write_text(s,encoding='utf8')
c=(OLD/'collect_completed.py').read_text('utf8').replace('R66_','R67_').replace('R67_EQUAL_COST_PILOT_001','R67_EQUAL_COST_REPAIR_001')
a=c.index('deviation=dict(');b=c.index("for name in ['PROJECT_PROGRESS.json'",a)
c=c[:a]+"assert environment['CPU_affinity']==list(range(12))\nresource_proof=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_CPU_affinity=environment['CPU_affinity'],torch_threads=12,\n    prior_attempt_native_wall_s=C['prior_attempt_native_wall_s'],repair_native_wall_s=receipt['native']['elapsed_s'],\n    cumulative_native_wall_s=C['prior_attempt_native_wall_s']+receipt['native']['elapsed_s'],original_cumulative_cap_s=600)\nassert resource_proof['cumulative_native_wall_s']<=600\nwrite(P/'CUMULATIVE_BUDGET_AND_CPU_RECEIPT.json',resource_proof)\n"+c[b:]
c=c.replace('wall_cap_s=600',"wall_cap_s=C['worker_cap_s']")
compile(c,'R67_collect_completed','exec');(P/'collect_completed.py').write_text(c,encoding='utf8')
(P/'CONTROLLER_ROOT_REVIEW.json').write_text(json.dumps(dict(explicit_preexec_child_CPU0_to11=True,supervisor_and_worker_assert_exact12_before_imports=True,
    frozen_science_files_changed=False,original_R66_proven_terminal_before_launch=True,
    scripts=[dict(name=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in [P/'install_launch_observe.py',P/'collect_completed.py']]),indent=2)+'\n',encoding='utf8')
print('R67 controllers compiled; explicit child CPU allocation and cumulative budget assertions installed.')
