"""Read and seal actual R77 training admission while its owned job runs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'admission_review_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
C=read(S/'config.json');remote=C['remote_root'];out=P/'ROOT_TRAIN_ADMISSION_REVIEW.json';assert not out.exists()
code='from pathlib import Path\nimport json,os\nos.sched_setaffinity(0,{11})\np=Path('+repr(remote)+')\n'
code+="names=['COST_ADMISSION.json','FULL_PARENT_EQUIVALENCE.json','DATA_RECEIPT.json','PROGRESS.json','REMOTE_ACTIVE_JOB.json']\nprint(json.dumps({n:json.loads((p/n).read_text())for n in names}))\n"
run=subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,encoding='utf8',timeout=30)
(P/'ADMISSION_READ.stderr.log').write_text(run.stderr,'utf8');assert run.returncode==0,run.stderr
evidence=json.loads(run.stdout);write(P/'ADMISSION_NATIVE_TEXT_EVIDENCE.json',evidence)
cost=evidence['COST_ADMISSION.json'];eq=evidence['FULL_PARENT_EQUIVALENCE.json'];data=evidence['DATA_RECEIPT.json'];progress=evidence['PROGRESS.json']
assert cost['pass_gate']and cost['estimate_full_run_seconds']<=14400 and cost['engineering_updates']==50
assert eq['pass_all']and [v['update']for v in eq['checks']]==[1,25,50]
assert all(v['pass_gate']for v in eq['checks'])and all(v['quality_selection']is False for v in eq['checks'])
assert cost['optimizer_empty']and cost['all_RNG_restored']and cost['engineering_parameters_discarded']
assert cost['initial_state_SHA256']==cost['restored_state_SHA256']==data['initial_state_SHA256']
assert cost['per_update_complete_SOURCE_rows']==data['per_update_SOURCE_rows']==6593
assert cost['per_update_complete_REAL_rows']==data['per_update_REAL_rows']==21047 and cost['all_parameter_VJP_rows']==27640
assert not data['heldout_arrays_read']and data['all_loss_terms_full_parents']and data['Q_head_frozen']
assert progress['phase']=='full_training'and progress['main_updates']>=1 and progress['parameter_VJP_rows']==27640
assert progress['firstpass_rows']==27640 and progress['neural_forward_calls']==1 and progress['replay_rows']==0
assert C['updates']==25000
now=datetime.now(timezone.utc).isoformat();review=dict(UTC=now,root_admission_review_pass=True,actual_full_training_started=True,
    actual_observed_step=progress['main_updates'],observed_worker_PID=evidence['REMOTE_ACTIVE_JOB.json']['actual_worker_PID'],
    unique_fixed_endpoint=25000,SOURCE_per_update=6593,REAL_per_update=21047,full_parameter_VJP_rows_per_update=27640,
    cost_estimated_total_hours=cost['estimate_full_run_seconds']/3600,measured_p90_seconds=cost['p90_update_seconds'],
    initial_state_restored_SHA256=cost['restored_state_SHA256'],all_RNG_and_empty_Adam_restored=True,
    full_parent_same_start_value_gradient_Adam_pass=True,heldout_read=False,quality_based_decisions=False,
    main_training_complete=False,model_goal_complete=False,application_goal_status='active',guard={**guard,**memory()})
write(out,review)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);assert d['current_stage']==P.name
    d.update(updated_UTC=now,status='R77_full_data_fixed25000_training_active',current_activity='complete27640_parent_updates_to_unique25000',
        current_model_fit_role='R71 Q head frozen; R77 complete waveform training active; held-out A-F not yet evaluated',
        latest_admission_review=out.relative_to(R).as_posix(),next_action='Continue fixed full training; prepare frozen held-out evaluation and independent4000 replay while waiting')
    write(R/name,d)
prefix=f'''# 当前执行：R77全数据25000主训练已通过准入并实际启动

本轮CPU/CUDA人工与原R72完整人口第1/25/50步值/梯度/Adam对照通过。50工程步后恢复原初始权重、空Adam及全部RNG；p90 {cost['p90_update_seconds']:.6f}秒，保守25000+尾检{cost['estimate_full_run_seconds']/3600:.6f}小时<=4。实际科学worker {review['observed_worker_PID']} 已开始正式更新；每步完整6593 SOURCE+21047 REAL与27640参数VJP。唯一终点25000、全部验收不变，不按中途表现选检查点。

训练及尾检未完成，held-out A–F尚未执行。训练期间推进冻结评价/独立4000重放静态实现与REAL-only GAP可用性审计；新科学作业仍等待本作业原生终止后单独登记。旧R65停止，长期目标active，根/既有辅助/资源/单作业/E-D保持。

根实际准入核验：reviews/{P.name}/ROOT_TRAIN_ADMISSION_REVIEW.json。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:(R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
print(json.dumps(review,ensure_ascii=False),flush=True)
