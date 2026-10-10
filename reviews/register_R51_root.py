from pathlib import Path
from datetime import datetime,timezone
import json,shutil,hashlib,os,ast
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
R=Path(__file__).resolve().parents[1];STAGE='R51_CANONICAL_SOURCE_TEACHER_DIAGNOSTIC_001';S=R/'work'/STAGE;REG=R/'reviews/R51_REGISTRATION_001';UTC=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,x):(R/p).write_bytes((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
def bind(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert not S.exists() and not REG.exists()
p=read('PROGRAM.json');assert p['status']=='R50_delivered_long_term_goal_active' and p['executor_agent']=='/root'
S.mkdir();REG.mkdir()
for name in ['AGENTS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','PROJECT_OBJECTIVE.md','RESOURCE_BUDGET.json','SERVER_RESOURCE_POLICY.md','jobs.json','experiments.json']:
 shutil.copy2(R/name,REG/name)
old=read('work/R49_LOCAL_HEUN128_RECOVERY_001/config_cpu.json');server=read('work/R48_GUIDED_NOISE_COMPLETION_FLOW_001/config.json');r50=read('work/R50_FROZEN_VOLTAGE_ATTRIBUTION_001/config.json')
remote='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R51_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
cfg={k:r50[k] for k in ['server_python','CPU_threads','RSS_limit_bytes','GPU_limit_bytes','MemAvailable_min_bytes','science_budget','deadline_UTC','output_limit_bytes','compute_budget_enforced','C2ST','bootstrap_seed','bootstrap_repeats']}
cfg.update(stage=STAGE,registered_UTC=UTC,owner='/root',subagents_allowed=False,remote_root=remote,
 protocol='tasks/'+STAGE+'.md',allowed_scientific_entries=['numeric_worker.py'],new_HGB_fits=32,reused_HGB_results=3,
 canonical_views=['full_historical_shape','normalized_u480','full_voltage','Q_PSD_2D','baseline_local'],
 domains=['BASE_SOURCE','R49_FIXED_SOURCE','R49_COMPLETION_SOURCE','BASE_REAL','R49_FIXED_REAL','R49_COMPLETION_REAL','RAW_REFERENCE'],
 server_R50_root=r50['remote_root'],TRAIN_path=server['TRAIN_path'],original_stream=server['original_stream'],
 source_local_TRAIN=old['training_input_paths']['TRAIN_PRIVATE.npz'],source_local_stream=server['source_stream_local'],
 response_training_runs=0,new_generation_rows=0,density_fits=0,EVAL_read=0,CAL_decode=0)
write('work/'+STAGE+'/config.json',cfg)
fresh={};source_bindings=[]
def load_selected(path,keys):
 path=Path(path);source_bindings.append(bind(path))
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
actual=load_selected(Path(old['frozen_eval_input_root'])/'ACTUAL_GENERATION_INPUTS.npz',['source_waveforms_V','source_ids'])
meta=load_selected(Path(old['frozen_eval_input_root'])/'DEV_SOURCE_METADATA.npz',['source_ids','composition'])
assert np.array_equal(actual['source_ids'],meta['source_ids'])
fresh.update(SOURCE_INPUT=actual['source_waveforms_V'],source_ids=actual['source_ids'],composition=meta['composition'])
for name,path in [('BASE_SOURCE',Path(old['R32_root'])/'CONTROL_FINAL/generated.npz'),
 ('R49_FIXED_SOURCE',R/'work/R49_LOCAL_HEUN128_RECOVERY_001/GUIDED_FIXED_NOISE_CONTROL/generated.npz'),
 ('R49_COMPLETION_SOURCE',R/'work/R49_LOCAL_HEUN128_RECOVERY_001/GUIDED_NOISE_COMPLETION/generated.npz')]:
 a=load_selected(path,['waveforms_V','source_ids']);assert np.array_equal(a['source_ids'],actual['source_ids']);fresh[name]=a['waveforms_V']
base_real=load_selected(old['old_BASE_REAL_generated'],['waveforms_V','reference_ids','reference_groups'])
with np.load(R/'work/R50_FROZEN_VOLTAGE_ATTRIBUTION_001/INPUT_PRIVATE.npz',allow_pickle=False) as z:
 assert np.array_equal(base_real['reference_ids'],z['reference_ids']) and np.array_equal(base_real['reference_groups'],z['reference_groups'])
fresh['BASE_REAL']=base_real['waveforms_V']
fresh.update(load_selected(Path(old['R26_root'])/'results/C2ST_PRIVATE.npz',['phase0_0','phase1_0']))
np.savez_compressed(S/'SOURCE_INPUT_PRIVATE.npz',**fresh)
shutil.copy2(Path(old['frozen_eval_input_root'])/'VALLEY_DEFINITION.json',S/'VALLEY_DEFINITION.json')
width=read(str(Path(old['local_input_paths']['width_definition']).relative_to(R)))
write('work/'+STAGE+'/Q_DEFINITION.json',{'edges_logQ':width['edges_logQ'],'source':old['local_input_paths']['width_definition']})
blocks=['import numpy as np\nimport sys\nev=sys.modules[__name__]\nREGIONS={"baseline":np.arange(100)}\n']
for file,names in [('evaluation.py',['crossing','features','c2st_matrix']),('diagnostic_core.py',['matrix'])]:
 path=R/'work/R49_LOCAL_HEUN128_RECOVERY_001'/file;text=path.read_text('utf8');nodes={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
 for name in names:blocks.append(ast.get_source_segment(text,nodes[name]))
 source_bindings.append(bind(path))
(S/'feature_definitions.py').write_bytes(('\n\n'.join(blocks)+'\n').encode())
write('work/'+STAGE+'/INPUT_BINDING.json',{'UTC':UTC,'local_source_bindings':source_bindings,'new_payload':bind(S/'SOURCE_INPUT_PRIVATE.npz'),
 'R50_input_and_predictions_reused_from_remote':r50['remote_root'],'TRAIN_and_stream_reused_from_remote':[server['TRAIN_path'],server['original_stream']],
 'original_frozen_feature_functions_verbatim_AST_extraction':True,'EVAL_read':0,'CAL_decode':0,'new_numeric_fit_before_registration':0})
for name in ['linux_supervise_post.py','linux_owned_worker.py']:shutil.copy2(R/'work/R50_FROZEN_VOLTAGE_ATTRIBUTION_001'/name,S/name)
f=S/'linux_supervise_post.py';f.write_bytes(f.read_bytes().replace(b'R50 requires',b'R51 requires'))
shutil.copy2(R/cfg['protocol'],S/'PROTOCOL.md')
banner='# 当前执行：主助手独立推进 R51\n\n长期目标active，原A–F和无计算预算政策保持。R50已完成ADC表示归因并交付。现按 tasks/'+STAGE+'.md 执行共同ADC表示下的SOURCE与训练目标诊断：32个新HGB、3个复用结果、原94400行关联的一次固定干预，0响应训练。主助手独立完成，不使用子智能体；服务器CPU12/RSS48GiB、单科学作业，本机只E/D。当前科学终点和下一机制判定以该协议为准。\n\n## 以下为历史记录\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
 (R/name).write_bytes((banner+(R/name).read_text('utf8')).encode())
job={'stage':STAGE,'status':'registered_not_launched','active_scientific_jobs':0}
next_action='Run the fixed canonical SOURCE comparisons and teacher/global-rank coupling diagnostic; decide the next response mechanism from the preregistered evidence.'
for fn in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 x=read(fn);x.update(current_stage=STAGE,current_task=cfg['protocol'],current_stage_scope='canonical_SOURCE_and_teacher_coupling_diagnostic',
  current_stage_progress='registered_not_launched',current_job=job,next_action=next_action,current_stage_registration='reviews/R51_REGISTRATION_001/RECEIPT.json',
  active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,subagents_allowed=False,application_goal_status='active')
 x['execution_location']={**x['execution_location'],'remote_root':remote,'mode':'server_CPU_frozen_source_and_teacher_diagnostics'}
 if fn=='PROGRAM.json':x.update(status='R51_registered_long_term_goal_active',updated_at_utc=UTC)
 else:x.update(project_status='R51_registered_long_term_goal_active',updated_UTC=UTC,current_activity='root_preparing_R51',current_direction=next_action)
 write(fn,x)
b=read('RESOURCE_BUDGET.json');b['current_registered_stage']=STAGE;b['improvement_block']['registered_stages'].append({'stage':STAGE,'kind':'diagnostic_not_response_method','registered_UTC':UTC,'science_budget':None,'new_training_runs':0,'new_HGB_fits':32,'reused_HGB_results':3,'status':'registered'});write('RESOURCE_BUDGET.json',b)
for fn,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 x=read(fn);x[key].append({ident:STAGE,'status':'registered','owner':'/root','registered_UTC':UTC,'protocol':cfg['protocol'],'output_root':'work/'+STAGE,'remote_root':remote,'model_candidate':False});write(fn,x)
write('reviews/R51_REGISTRATION_001/RECEIPT.json',{'UTC':UTC,'stage':STAGE,'root_control_PID':os.getpid(),'protocol':bind(R/cfg['protocol']),'config':bind(S/'config.json'),
 'owner':'/root','application_goal_status':'active','original_A_F_unchanged':True,'scientific_execution_started':False,'response_training_runs':0,'remote_root':remote})
print(json.dumps({'stage':STAGE,'UTC':UTC,'remote_root':remote,'input_bytes':(S/'SOURCE_INPUT_PRIVATE.npz').stat().st_size,'new_HGB':32,'reused_HGB':3},indent=2))
