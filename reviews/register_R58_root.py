from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,subprocess,os
R=Path(__file__).resolve().parents[1];ID='R58_JOINT_DISTRIBUTION_FEASIBILITY_001';S=R/'work'/ID;REG=R/'reviews/R58_REGISTRATION_001';NOW=datetime.now(timezone.utc).isoformat()
def read(n):return json.loads((R/n).read_text('utf8'))
def write(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert not S.exists() and not REG.exists()
assert read('PROGRAM.json')['status']=='R57_delivered_long_term_goal_active' and read('PROGRAM.json')['active_scientific_jobs']==0
assert read('R57_DELIVERY_COMPLETION_RECEIPT.json')['status']=='delivered'
old=read('work/R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001/config.json')
cfg={k:old[k] for k in ['server_python','CPU_threads','RSS_limit_bytes','GPU_limit_bytes','MemAvailable_min_bytes','science_budget','deadline_UTC','output_limit_bytes','compute_budget_enforced']}
inputs={k:old['input_paths'][k] for k in ['TRAIN_path','constants_path','valley_definition']};inputs['R57_features']=old['remote_root']+'/FEATURES_PRIVATE.npz'
code='from pathlib import Path\nimport json,hashlib\ninputs='+repr(inputs)+'\nprint(json.dumps({k:{"path":p,"bytes":Path(p).stat().st_size,"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest()}for k,p in inputs.items()}))\n'
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=60)
assert p.returncode==0,p.stderr;binding=json.loads(p.stdout)
assert binding['R57_features']['sha256']==bind(R/'work/R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001/FEATURES_PRIVATE.npz')['sha256']
S.mkdir();REG.mkdir();meta=['AGENTS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','PROJECT_OBJECTIVE.md','RESOURCE_BUDGET.json','jobs.json','experiments.json']
for n in meta:shutil.copy2(R/n,REG/n)
cfg.update(stage=ID,owner='/root',subagents_allowed=False,registered_UTC=NOW,remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R58_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'),protocol='tasks/'+ID+'.md',input_paths=inputs,
    allowed_scientific_entries=['numeric_worker.py'],GPU_scientific_entries=[],SOURCE_N=6593,REAL_N=21047,target_quantile_atoms=512,source_rank_groups=64,
    PSD_overall_max=.01,PSD_conditional_mean_max=.01,source_valley_delta_max=.02,LP_attempts=1,LP_method='highs-ds',LP_options={'presolve':True,'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,'threads':12,'parallel':False},verification_tolerance=1e-7,
    original_DEV_BASE_PSD_rho_for_context=.4189898393117022,original_DEV_rho_drop_limit=.05,
    response_training_runs=0,HGB_fits=0,density_fits=0,ODE_generation=0,teacher_waveforms_constructed=0,DEV_read=0,CAL_decode=0,EVAL_read=0)
write('work/'+ID+'/config.json',cfg)
for n in ['noise_core.py','canonical.py']:(S/n).write_text((R/'work/R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001'/n).read_text('utf8').rstrip()+'\n','utf8')
for n in ['linux_supervise_post.py','linux_owned_worker.py']:
    text=(R/'work/R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001'/n).read_text('utf8').replace('R57 requires','R58 requires').rstrip()+'\n';(S/n).write_text(text,'utf8')
shutil.copy2(R/cfg['protocol'],S/'PROTOCOL.md')
write('work/'+ID+'/INPUT_BINDING.json',{'UTC':NOW,'inputs':binding,'binding_only_no_new_statistics':True,'SSH_native_exit':0,'acceptance':bind(R/'PROJECT_ACCEPTANCE.json'),'R57_report':bind(R/'R57_REPORT.md'),'frozen_C_definition_source':bind(R/'work/R54_CONTINUOUS_CHARGE_FLOW_001/summarize_and_plot.py'),'baseline_context_source':bind(R/'work/R54_CONTINUOUS_CHARGE_FLOW_001/GATE_METRICS.json'),'baseline_value_is_DEV_context_not_TRAIN_acceptance':True})
banner='# 当前执行：R58 联合分布约束检查\n\nR57已关闭并交付。用户询问教师机制后，本轮去掉固定目标波形假设，仅检查全TRAIN源等权下PSD总体/条件分布、谷约束及源秩关联的联合可行性，并单列ADC积分硬界。一次固定LP，0教师波形、0神经训练；原A–F和原验收源人口不变。主助手独立、无计算配额、单科学作业及本机E/D约束保持。\n\n## 以下为历史记录\n\n'
for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/n).write_text(banner+(R/n).read_text('utf8'),'utf8')
action='Solve the single registered joint TRAIN distribution LP and verify its original-parent marginals, exact PSD distances, valleys, attainable rank and separate ADC integral bounds; no teacher waveforms or neural training.'
for n in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    x=read(n);x.update(status='R58_registered_long_term_goal_active',current_stage=ID,current_task=cfg['protocol'],current_stage_scope='joint_TRAIN_distribution_feasibility',current_stage_progress='registered_not_launched',current_job={'stage':ID,'status':'registered_not_launched','active_scientific_jobs':0},next_action=action,current_stage_registration='reviews/R58_REGISTRATION_001/RECEIPT.json',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,subagents_allowed=False,application_goal_status='active',model_goal_complete=False)
    x['execution_location'].update(mode='server_CPU_joint_distribution_LP',remote_root=cfg['remote_root'],current_stage_GPU_used=False,preflight_required=True)
    if n=='PROGRAM.json':x['updated_at_utc']=NOW
    else:x.update(project_status=x['status'],updated_UTC=NOW,current_activity='root_preparing_R58',current_direction=action)
    write(n,x)
b=read('RESOURCE_BUDGET.json');b['current_registered_stage']=ID;b['improvement_block']['registered_stages'].append({'stage':ID,'kind':'joint_distribution_diagnostic_not_response_method','registered_UTC':NOW,'science_budget':None,'new_training_runs':0,'new_HGB_fits':0,'status':'registered'});write('RESOURCE_BUDGET.json',b)
for n,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    x=read(n);assert not any(v[ident]==ID for v in x[key]);x[key].append({ident:ID,'status':'registered','owner':'/root','registered_UTC':NOW,'protocol':cfg['protocol'],'output_root':'work/'+ID,'remote_root':cfg['remote_root'],'model_candidate':False});write(n,x)
write('reviews/R58_REGISTRATION_001/RECEIPT.json',{'UTC':NOW,'stage':ID,'root_control_PID':os.getpid(),'protocol':bind(R/cfg['protocol']),'config':bind(S/'config.json'),'input_binding':bind(S/'INPUT_BINDING.json'),'owner':'/root','application_goal_status':'active','original_A_F_unchanged':True,'scientific_execution_started':False,'response_training_runs':0,'remote_root':cfg['remote_root']})
print(json.dumps({'stage':ID,'UTC':NOW,'remote_root':cfg['remote_root'],'scientific_execution_started':False},indent=2))
