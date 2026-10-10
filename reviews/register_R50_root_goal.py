from pathlib import Path
from datetime import datetime, timezone
import os, json, shutil, hashlib
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STAGE='R50_FROZEN_VOLTAGE_ATTRIBUTION_001'
OUT=ROOT/'work'/STAGE
REG=ROOT/'reviews/R50_ROOT_REGISTRATION_001'
UTC=datetime.now(timezone.utc).isoformat()
def read(p): return json.loads((ROOT/p).read_text('utf-8-sig'))
def write(p,x):
    p=ROOT/p; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def bind(p):
    p=ROOT/p
    return {'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

assert not REG.exists() and not OUT.exists()
goal=read('reviews/ROOT_LONG_TERM_GOAL_20261010_001.json')
assert goal['goal']['status']=='active'
program=read('PROGRAM.json'); progress=read('PROJECT_PROGRESS.json'); budget=read('RESOURCE_BUDGET.json')
assert program['status']=='R49_delivered_model_goal_unmet'
assert budget['compute_budget_enforced'] is False
REG.mkdir(); OUT.mkdir(parents=True)
snapshots=['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','PROJECT_ACCEPTANCE.md','PROJECT_ACCEPTANCE.json',
           'SERVER_RESOURCE_POLICY.md','PROGRAM.json','PROJECT_PROGRESS.json','RESOURCE_BUDGET.json','jobs.json','experiments.json']
for name in snapshots: shutil.copy2(ROOT/name,REG/name)
task='tasks/'+STAGE+'.md'
remote='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R50_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
previous=read('work/R49_LOCAL_HEUN128_RECOVERY_001/config_cpu.json')
cfg={'stage':STAGE,'registered_UTC':UTC,'owner':'/root','subagents_allowed':False,'protocol':task,
     'remote_root':remote,'server_python':previous['server_python'],'CPU_threads':12,'RSS_limit_bytes':48*1024**3,
     'GPU_limit_bytes':10*1024**3,'MemAvailable_min_bytes':56*1024**3,
     'science_budget':None,'deadline_UTC':None,'output_limit_bytes':None,'compute_budget_enforced':False,
     'allowed_scientific_entries':['numeric_worker.py'],'planned_HGB_fits':24,'response_training_runs':0,
     'new_generation_rows':0,'density_fits':0,'EVAL_read':0,'CAL_decode':0,'C2ST':previous['C2ST'],
     'bootstrap_seed':previous['bootstrap_seed'],'bootstrap_repeats':previous['bootstrap_repeats'],
     'cases':['RAW_REFERENCE','RAW_LEGACY_ADC','FIXED_PRE','FIXED_POST','COMPLETION_PRE','COMPLETION_POST'],
     'core_views':['native_voltage','canonical_ADC_voltage'],
     'decomposition_cases':['RAW_REFERENCE','FIXED_POST','COMPLETION_POST'],
     'decomposition_views':['baseline_subtracted_voltage','baseline_mean','baseline_RMS','pulse_Q_PSD_peak'],
     'diagnostic_only':True,'original_acceptance_unchanged':True,'EVAL_remains_closed':True}
write('work/'+STAGE+'/config.json',cfg)
banner='# 当前执行：主助手独立推进长期目标与 R50\n\n'+(
 '用户要求设置明确结束条件的长期目标，并要求后续全部由主助手独立完成。应用目标已实际创建为 active，'
 '唯一完成条件是同一冻结方法满足原 A–F 并完成实际交付，详见 LONG_TERM_GOAL_20261010.md。'
 '不使用子智能体；本条覆盖下文历史角色分工。计算预算仍不设置，机器安全、单科学作业、源人口和E/D约束保留。'
 '服务器已恢复并经只读核实；旧远端PID不存在，原生退出未知，不补记成功。'
 '当前按 tasks/R50_FROZEN_VOLTAGE_ATTRIBUTION_001.md 登记冻结输出的24个诊断比较，0响应训练。'
 '本轮科学终点和后续判断写在协议内；模型目标尚未完成。\n\n## 以下为历史记录，执行角色和当前状态以上文为准\n\n')
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','SERVER_RESOURCE_POLICY.md']:
    (ROOT/name).write_text(banner+(ROOT/name).read_text('utf8'),'utf8')
ap='PROJECT_ACCEPTANCE.md'
(ROOT/ap).write_text('# 当前应用目标\n\n2026-10-10用户明确要求后，应用长期目标已实际创建为active；见LONG_TERM_GOAL_20261010.md。原A–F数值、数据范围和历史结果保持。下文旧无goal观测仅作历史。\n\n'+(ROOT/ap).read_text('utf8'),'utf8')
acceptance=read('PROJECT_ACCEPTANCE.json')
acceptance['previous_application_goal_observation']=acceptance['application_goal_observation']
acceptance['application_goal_observation']={'value':'active','observed_UTC':UTC,'receipt':'reviews/ROOT_LONG_TERM_GOAL_20261010_001.json','explicit_user_request':True}
acceptance['document_sha256']=bind(ap)['sha256'];write('PROJECT_ACCEPTANCE.json',acceptance)
location={'mode':'server_CPU_frozen_saved_array_diagnostics','host':'rylan@100.82.231.102','remote_root':remote,
          'CPU_threads':12,'job_tree_RSS_bytes':48*1024**3,'current_stage_GPU_used':False,'preflight_required':True,
          'server_user_reported_state':'online','remote_previous_attempt_terminal':'unknown_process_absent_after_restart',
          'local_files':'E/D only','project_concurrency_max':1}
current_job={'stage':STAGE,'status':'registered_not_launched','active_scientific_jobs':0}
next_action='Execute the fixed 24-fit voltage/ADC/baseline attribution protocol, record results and source-PSD follow-up; original A-F unchanged.'
program.update(executor_agent='/root',executor_model=None,executor_reasoning_effort=None,
    root_role='direction_implementation_execution_review_delivery',subagents_allowed=False,
    status='long_term_goal_active_R50_registered',current_stage=STAGE,current_task=task,next_prepared_task=None,
    current_stage_scope='frozen_saved_REAL_voltage_attribution_no_response_fit',current_stage_progress='registered_not_launched',
    execution_location=location,current_job=current_job,updated_at_utc=UTC,next_action=next_action,
    latest_permission_status='Explicit root-only long-term goal; no compute quota; server online and stage registered.',
    goal_status_note='create_goal returned active on explicit user request; model goal remains unmet.',
    goal_registration_receipt='reviews/ROOT_LONG_TERM_GOAL_20261010_001.json',application_goal_status='active')
write('PROGRAM.json',program)
progress.update(updated_UTC=UTC,project_status='long_term_goal_active_R50_registered',application_goal_status='active',
    current_stage=STAGE,current_task=task,current_stage_progress='registered_not_launched',current_activity='root_preparing_R50_server_diagnostics',
    current_stage_scope=program['current_stage_scope'],execution_location=location,current_job=current_job,next_action=next_action,
    current_direction=next_action,current_stage_registration='reviews/R50_ROOT_REGISTRATION_001/RECEIPT.json',
    current_executor_ack=None,executor_start_observation=None,active_scientific_jobs=0,active_local_scientific_jobs=0,
    active_remote_scientific_jobs=0,execution_owner='/root',subagents_allowed=False,model_goal_complete=False)
write('PROJECT_PROGRESS.json',progress)
jobs=read('jobs.json');jobs['jobs'].append({'stage':STAGE,'kind':'frozen_saved_waveform_diagnostic','status':'registered_not_launched',
    'registered_UTC':UTC,'owner':'/root','remote_root':remote,'protocol':task,'science_s':None,'planned_HGB_fits':24,'response_training_runs':0})
write('jobs.json',jobs)
exp=read('experiments.json');exp['experiments'].append({'id':STAGE,'kind':'controlled_ADC_baseline_attribution',
    'status':'registered','protocol':task,'output_root':'work/'+STAGE,'owner':'/root','registered_UTC':UTC,'model_candidate':False})
write('experiments.json',exp)
budget['improvement_block']['registered_stages'].append({'stage':STAGE,'kind':'diagnostic_not_response_method',
    'registered_UTC':UTC,'science_budget':None,'planned_HGB_fits':24,'new_training_runs':0,'status':'registered'})
budget['policy_sha256']=bind('SERVER_RESOURCE_POLICY.md')['sha256']
budget['current_execution_owner']='/root'; budget['current_registered_stage']=STAGE
write('RESOURCE_BUDGET.json',budget)

# Copy only frozen payload columns; no derived scientific feature or fit here.
realfile=Path(previous['frozen_eval_input_root'])/'REAL_DEV_AND_REFERENCE_PRIVATE.npz'
splitfile=Path(previous['R26_root'])/'results/C2ST_PRIVATE.npz'
paths=[realfile,splitfile]; packed={}
with np.load(realfile,allow_pickle=False) as z:
    for key in ['real_V','real_ids','real_groups','reference_V','reference_ids','reference_groups']:packed[key]=z[key].copy()
with np.load(splitfile,allow_pickle=False) as z:
    for phase in [0,1]:
        for j in [1,2]:packed[f'phase{phase}_{j}']=z[f'phase{phase}_{j}'].copy()
for tag,name in [('FIXED','GUIDED_FIXED_NOISE_REAL_CONTROL'),('COMPLETION','GUIDED_COMPLETION_REAL_CONTROL')]:
    file=ROOT/'work/R49_LOCAL_HEUN128_RECOVERY_001'/name/'generated.npz';paths.append(file)
    with np.load(file,allow_pickle=False) as z:
        assert np.array_equal(packed['reference_ids'],z['reference_ids'])
        assert np.array_equal(packed['reference_groups'],z['reference_groups'])
        packed[tag+'_PRE']=z['pre_ADC_V'].copy();packed[tag+'_POST']=z['waveforms_V'].copy()
np.savez_compressed(OUT/'INPUT_PRIVATE.npz',**packed)
input_receipt={'source':'sealed local R49 arrays, not historical remote arrays','copied_column_count':len(packed),
               'source_bindings':[bind(str(p.relative_to(ROOT))) for p in paths],
               'payload':bind('work/'+STAGE+'/INPUT_PRIVATE.npz'),'EVAL_read':0,'CAL_decode':0,'new_generation':0}
write('work/'+STAGE+'/INPUT_BINDING.json',input_receipt)
for name in ['linux_supervise_post.py','linux_owned_worker.py']:
    shutil.copy2(ROOT/'work/R49_FROZEN_HEUN128_RESPONSE_001'/name,OUT/name)
supervisor=OUT/'linux_supervise_post.py'
supervisor.write_text(supervisor.read_text('utf8').replace('R49 requires explicit null budget fields','R50 requires explicit null budget fields'),'utf8')
receipt={'UTC':UTC,'root_control_PID':os.getpid(),'stage':STAGE,'owner':'/root','goal_status':'active',
         'goal_receipt':'reviews/ROOT_LONG_TERM_GOAL_20261010_001.json','protocol':bind(task),
         'config':bind('work/'+STAGE+'/config.json'),'remote_root':remote,'previous_control_snapshot':str(REG.relative_to(ROOT)),
         'original_A_F_thresholds_unchanged':True,'compute_budget_enforced':False,'response_training_runs':0,
         'scientific_execution_started':False,'previous_server_state_receipts':['reviews/SERVER_RETURN_READONLY_20261010_001.json',
         'reviews/SERVER_RETURN_PROCESS_RECONCILIATION_20261010_001.json'],'input_binding':'work/'+STAGE+'/INPUT_BINDING.json'}
write('reviews/R50_ROOT_REGISTRATION_001/RECEIPT.json',receipt)
print(json.dumps({'stage':STAGE,'registered_UTC':UTC,'control_PID':os.getpid(),'remote_root':remote,
                  'input_bytes':(OUT/'INPUT_PRIVATE.npz').stat().st_size,'HGB_plan':24,'goal_status':'active'},indent=2))
