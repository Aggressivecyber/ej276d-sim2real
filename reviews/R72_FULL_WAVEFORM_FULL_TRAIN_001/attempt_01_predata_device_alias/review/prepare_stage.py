"""Prepare the single R72 candidate; no scientific arrays are decoded here."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, shutil, sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'preparation_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (P/'PREPARATION_RECEIPT.json').exists()
assert read(R/'jobs.json')['active_scientific_jobs']==0
assert read(R/'R71_DELIVERY_RECEIPT.json')['all_copied_file_SHA256_verified']
now=datetime.now(timezone.utc);S.mkdir(parents=True,exist_ok=True)
old=R/'work/R71_REAL_CHARGE_FULL_TRAIN_001';oldconfig=read(old/'config.json')
C={k:v for k,v in oldconfig.items() if k not in {'input_paths','hidden','dtype','loss_weights','loss_scope'}}
C.update(stage=P.name,remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R72_ROOT_'+now.strftime('%Y%m%d_%H%M%S_UTC'),
    model_seed=20261011720,source_epsilon_seed=20261011721,real_epsilon_seed=20261011722,artificial_seed=20261011723,
    training_SOURCE_epsilon_seed=20261011724,training_REAL_epsilon_seed=20261011725,
    directions_seed=20261011726,temporal_directions_seed=20261011727,
    hidden=128,shape_dtype='float32',physical_dtype='float64',shape_input_features=482,
    shape_network=[482,128,128,480],Q_head_frozen=True,Q_head_component='R71_unique50000',
    neural_replay_chunk=27640,generation_chunk=256,loss_feature_chunk=64,
    loss_scope='complete SOURCE and same-treated REAL against original REAL canonical reference; current output charge bins',
    postprocess_reserve_seconds=1800,progress_interval_updates=100,recovery_interval_updates=5000,
    training_noise='one independent CUDA-generator float32 normal vector per original parent per update; reused within complete VJP replay',
    loss_parameter_gradient='sum nine exact full-population endpoint term gradients in float64; replay all original rows in order, cast at original output and parameter dtype',
    endpoint='sole completed shape update50000 with frozen R71 charge head; no checkpoint selection',
    input_paths={**oldconfig['input_paths'],'Q_head_weights':oldconfig['remote_root']+'/FINAL_HEAD.pt',
                 'Q_head_statistics':oldconfig['remote_root']+'/HEAD_STATISTICS.npz'})
legacy=read(R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/config.json')
for key in ['PSD_loss_scale','valley_delta','valley_smoothing','feature_names','feature_scale_floor','noise_scale_floor',
            'projection_directions','temporal_feature_count','temporal_lags','temporal_band_bin_count','temporal_band_count',
            'temporal_projection_directions','temporal_scale_floor','temporal_power_floor','loss_weights']:
    C[key]=legacy[key]
C['new_statistics_fit']=True
C['new_statistics_scope']='Only legal original canonical REAL TRAIN, actual canonical charge raw bins; source labels only for source constraints'
write(S/'config.json',C)
for name in ['public_charge_head.py','physical_operators.py','common_response.py','numpy_reference.py','linux_owned_worker.py','linux_supervise.py']:
    if name=='linux_supervise.py':
        text=(old/name).read_text('utf8').replace('R71 has no wall-clock','R72 has no wall-clock')
        (S/name).write_text(text,'utf8')
    else:shutil.copy2(old/name,S/name)
# The original interval implementation is independently retained for artificial comparisons.
interval=(R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/precision_order.py').read_text('utf8').replace('from noise_core import q_np','from numpy_reference import q as q_np')
(S/'precision_order_reference.py').write_text(interval,'utf8')
bindings=read(old/'INPUT_BINDING.json')['inputs']
for key,filename in [('Q_head_weights','FINAL_HEAD.pt'),('Q_head_statistics','HEAD_STATISTICS.npz')]:
    bindings[key]=dict(path=C['input_paths'][key],sha256=sha(old/filename),bytes=(old/filename).stat().st_size)
write(S/'INPUT_BINDING.json',dict(UTC=now.isoformat(),inputs=bindings,scope='TRAIN-only inputs, frozen R71 Q weights and constants; no DEV/EVAL/CAL arrays'))
control=(R/'reviews/R71_REAL_CHARGE_FULL_TRAIN_001/control.py').read_text('utf8').replace('R71_FROZEN_INSTALL','R72_FROZEN_INSTALL')
(P/'control.py').write_text(control,'utf8')
snapshot=P/'before_preparation';snapshot.mkdir()
for name in ['AGENTS.md','STATUS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json']:
    shutil.copy2(R/name,snapshot/name)
prefix='''# 当前执行：R72 完整波形全量方法准备，长期目标 active

R71 电荷组件50000步、根复算、私有Git/D盘交付均完成；完整响应A–F未完成。R72拟冻结同一公共Q头，训练482→128→128→480的全480点残差，按输出canonical Q重建完整SOURCE及同处理REAL分布损失，保留原源精度排序、谷、父事件、身份配对和边界/数值验收。128宽度是未看本轮科学数组之前的单一架构，32宽草案未执行；无模型扫描。

先完成人工值/梯度/重放核验并登记，再实测完整人口和全部反向的步时；保守预计50000步加尾检≤4小时则直接全量唯一终点。工程步后恢复初始权重、空Adam及全部随机流。当前尚未科学注册或启动；旧R65不恢复，v5数字/人口不改。主助手统筹，既有gpt-6.1-sol/max辅助；服务器CPU12/RSS48GiB/GPU10GiB、本机共享CPU4/RSS4GiB、单科学作业、全部新增本地E/D保持。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:
    prior=(R/name).read_bytes();nl=b'\r\n' if prior.count(b'\r\n')>prior.count(b'\n')//2 else b'\n'
    (R/name).write_bytes(prefix.encode().replace(b'\n',nl)+prior)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);d.update(updated_UTC=now.isoformat(),status='R72_full_waveform_preparation',current_stage=P.name,
      current_stage_scope='full480 response after frozen learned REAL-charge head',current_activity='implementation_and_artificial_validation',
      current_task=None,current_stage_registration=None,numeric_execution_released=False,active_scientific_jobs=0,current_job=None,
      application_goal_status='active',model_goal_complete=False,
      next_action='Bind one complete-parent objective and measure full update; automatically train sole50000 endpoint if conservative cost<=4h',
      latest_delivered_stage=old.name,latest_stage_publication='R71_DELIVERY_RECEIPT.json',latest_completed_backup='R71_DELIVERY_RECEIPT.json')
    write(R/name,d)
for f in list(S.glob('*.py'))+[P/'control.py']:
    ast.parse(f.read_text('utf8'))
write(P/'PREPARATION_RECEIPT.json',dict(UTC=now.isoformat(),stage=P.name,remote_root=C['remote_root'],science_arrays_decoded=False,
  training_jobs_launched=0,previous_delivery=read(R/'R71_DELIVERY_RECEIPT.json'),guard={**guard,**memory()}))
print(json.dumps(dict(stage=P.name,remote_root=C['remote_root'],science_arrays_decoded=False,guard={**guard,**memory()})))
