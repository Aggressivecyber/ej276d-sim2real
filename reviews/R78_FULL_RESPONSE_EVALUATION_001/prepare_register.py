"""Register the sole completed R77 endpoint for fixed held-out response evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import shutil
import sys

P = Path(__file__).resolve().parent
R = P.parents[1]
S = R/'work'/P.name
T = R/'work/R77_FIXED25000_FULL_DATA_TRAIN_001'
TP = R/'reviews/R77_FIXED25000_FULL_DATA_TRAIN_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard = apply(P/'registration_runtime')


def read(path):
    return json.loads(path.read_text('utf8'))


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


assert not S.exists() and not (P/'REGISTRATION.json').exists()
assert read(R/'jobs.json')['active_scientific_jobs'] == 0
assert read(T/'RESULT.json')['status'] == 'FULL_WAVEFORM_TRAIN_ENDPOINT_AND_POSTCHECKS_COMPLETE'
assert read(TP/'COLLECTION_RECEIPT.json')['native_exit'] == 0
# The user cancelled additional duplicate recomputation.  Existing native TRAIN
# receipts are sufficient to bind evaluation; Git/backup is delivery, not an
# additional prerequisite before the already-authorized held-out calculation.
steering = read(P/'USER_EVALUATION_INSTRUCTION.json')
assert steering['formal_evaluation_authorized'] and steering['extra_numeric_recomputation_cancelled']
native = read(T/'train_full.py.NATIVE_RECEIPT.json')
assert type(native['actual_worker_native_exit'])is int and native['actual_worker_native_exit'] == 0
assert native['termination_reason'] is None
train = read(T/'config.json')
receipt = read(T/'TRAINING_RECEIPT.json')
assert train['updates'] == receipt['completed_updates'] == receipt['unique_selected_endpoint'] == 25000
assert read(T/'POSTCHECK_RECEIPT.json')['numerical_and_TRAIN_replay_pass']
current = read(R/'CURRENT_ACCEPTANCE.json')
assert current['version'] == 5
for field,digest in [('document','document_sha256'),('registry','registry_sha256')]:
    assert sha(R/current[field]) == current[digest]

# All core production code must have its preparation completion before copying.
adapter_root = R/'reviews/R74_EXACT_LOSS_FULL_TRAIN_001/evaluation_candidate'
for folder in [adapter_root,P/'evaluator_candidate',P/'root_summary_candidate',
               P/'static_adapter_review',P/'static_summary_review',P/'summary_receipt_only']:
    assert (folder/'COMPLETION_RECEIPT.json').is_file(),folder
summary = read(P/'root_summary_candidate/COMPLETION_RECEIPT.json')
for name,digest in summary['production_sources'].items():
    assert sha(P/'root_summary_candidate'/name) == digest,name
assembly = read(P/'STATIC_ASSEMBLY_RECEIPT.json')
assert assembly['root_static_integration_review_pass']
for row in assembly['production_sources']:
    assert sha(R/row['source']) == row['sha256'],row['source']
S.mkdir()
provenance = []
for row in assembly['production_sources']:
    source,target = R/row['source'],S/row['name']
    assert not target.exists(),row['name']
    shutil.copy2(source,target)
    if target.suffix == '.py':
        ast.parse(target.read_text('utf8'))
    provenance.append({**row,'copied_SHA256':sha(target)})

now = datetime.now(timezone.utc).isoformat()
remote = '/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R78_EVAL_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
c = read(P/'config.disabled.json')
c.update(execution_enabled=True,remote_root=remote,output_root=remote,registered_UTC=now,
    binding_status='Actual completed R77 weights, native/numeric receipts and fixed references prospectively bound',
    R65_root=train['R65_root'],compute_budget_enforced=False)
for key,row in c['training'].items():
    path = T/Path(row['path']).name
    assert row['path'] == train['remote_root']+'/'+path.name
    row['sha256'] = sha(path)
for key in ['shape_weights','frozen_Q_weights','shape_statistics']:
    row = c['model'][key]
    path = T/Path(row['path']).name
    assert row['path'] == train['remote_root']+'/'+path.name
    row['sha256'] = sha(path)
assert c['model']['shape_weights']['sha256'] == receipt['final_weight_file_SHA256']
assert c['model']['frozen_Q_weights']['sha256'] == receipt['Q_head_weight_SHA256']
for name,row in c['runtime_sources'].items():
    assert row['path'] == train['remote_root']+'/'+name and sha(T/name) == row['sha256']
assert c['runtime_source_directory'] == train['remote_root']
assert all(row['sha256'] != '0'*64 for kind in ['training','model','inputs','runtime_sources']for row in c[kind].values())
write(S/'config.json',c)
post = read(P/'config_post.disabled.json')
post.update(evaluation_adapter_config=remote+'/config.json',execution_enabled=True,
    registered_UTC=now,remote_root=remote,output_root=remote,
    frozen_model_file_SHA256=receipt['final_weight_file_SHA256'])
write(S/'config_post.json',post)

protocol = '''# R78：唯一 R77 完整响应的冻结评价

本任务只评价已完成 R77 唯一25000 shape更新与冻结R71唯一50000 Q-head。全部6593 SOURCE和21047 REAL的TRAIN方法、九项与原始父权重已固定；不得按本次评价选择检查点、重训、修改随机流或选择对照替代。原生R77退出0、原定TRAIN独立NumPy64+64参考与TRAIN4000序列化重放已完成，直接引用原生回执。用户已明确要求进行评价并取消额外重复复算；不再执行根侧重复TRAIN数值计算，也不将Git/备份设成评价启动前置。当前登记不构成A–F成功。

生成使用完全相同的公共SOURCE/REAL算子：完整480输入→独立预测Q*→电荷lifting与有色噪声→482→128→128→480残差→原Box约束→原float32 ADC及严格共同canonical码表示。无域标记、REAL身份旁路或BASE回退。raw明确float64、原保存white明确float32，chunk256含末批、straight_through=False；保留原z/style但其不进入算子。绑定唯一权重、统计、四份原R77运行模块及全部输入哈希。

按原顺序一次生成SOURCE2000、同处理TRAIN_REFERENCE REAL2000、8噪声重复1024、源扰动832和guide128，共5984。REAL比较目标仍为原real_* DEV2000，不与输入reference_*对调；原SOURCE组成1518 pure_had、417 pure_EE、65 mixed，原支持外417和原谷母群213/181完整保留。历史ADC谷母群207/182仅另报。原父身份、组成、权重、原始条件/随机流、独立Q*、preADC和canonical Q、全部条件迁移与支持外质量保存。

生成后新进程重新载入同一权重与原输入/随机流，先完整重算两域4000，再读取主保存输出比较全部payload字段、preADC、canonical电压和整数码。此独立held-out4000与TRAIN序列化检查分开计数。任何数值或进程失败保留原始结果并停止本执行，不能删除行后通过。

评价只读取保存响应，不进行神经前向/更新。固定原R26三个角色选择槽：SOURCE0、REAL DEV target1、REAL TRAIN_REFERENCE control2，每域600分类训练+250分类留出，不重分。原五视图、HGB超参数、500次分组bootstrap与种子2026100871保持；新增HGB正好10次，密度描述槽正好16（两域各8），EM尝试至多16，失败/空槽完整保留。BASE、BASE_REAL、RAW_RR的原R51分类预测与原R39参考宽度复用，没有重拟合历史对照。11份历史参考首次读取前逐项验证绑定。最终EVAL与CAL波形解码均0；CAL既有固定描述和已暴露DEV的使用不代表新独立采集证据。

所有现行v5 A/B数字及原六项门保留：两域各五视图BA95%CI全部包含于[0.40,0.60]；总体及固定共同Q1–7 PSD W1≤0.01；必需Q2–7两峰位MAE≤0.01、原REAL raw SD平均相对差≤10%，支持/拟合失败不删除；原噪声、源谷/秩、九边界、guide与同处理REAL门保持。第6门使用事前R77非迭代独立NumPy参考（median≤0.25LSB、p95≤1LSB）与独立预测Q*的float32 preADC积分相对差≤1e-5及新4000重放。不能换成原SOURCE Q硬守恒或虚构迭代N/2N结果。

Q描述保留原2000均匀质量，正式评价一次计算输入Q0、预测Q*、preADC和共同canonical四轴CDF/KS/W1/计数/权重/支持外，以及三个相邻11×11迁移表（含低/高支持外与非法格）。聚合入口直接引用正式评价导出的数值证据与原生回执，不重复读取原波形或重算Q/ADC/身份/CDF/迁移，不将引用称为独立复算；不另加结果驱动电荷阈值。所有真实非法值作为失败保留；不能归零、删除或重新归一化冒充通过。

执行顺序固定为generate_full_response.py、replay_full_response.py、evaluate_saved.py、summarize_and_plot.py。每个入口由原受控Linux监督器核验真实PID、原生退出、RSS及适用GPU峰值，顺序单科学作业；GPU入口仅前两个。CPU0..11/RSS48GiB/GPU10GiB，无科学墙钟/阶段体积预算；本机辅助共享CPU4/RSS4GiB，每个辅助1CPU/1GiB。新增本地文件仅E/D。bash协调器不是科学回执，真实科学终态以四份native worker回执和进程消失为准。

主A–C任一失败则封存这个固定R77方案的负结果、实际图像、成本、原生记录与交付；不能降低门槛或把工程/局部通过说成模型完成。A–C全部通过才另行绑定D任务效用与E的REAL-only FULL/GAP设计；本任务不执行D/E、不验证真实无观测事件率或物理能量标定，它们仍unknown。当前应用长期目标读取为paused，本轮直接用户指令授权本次评价，不据此篡改应用目标状态或自动启动下一模型。完成后根实际打开8类科学图复核，私有Git仅保存源码/配置/紧凑指标/图，私有逐事件数组和权重保留E及D备份。
'''
(S/'PROTOCOL.md').write_text(protocol,'utf8')
task = R/'tasks'/(P.name+'.md')
assert not task.exists()
task.write_text(protocol,'utf8')
write(S/'SOURCE_PROVENANCE.json',dict(UTC=now,files=provenance,
    root_static_assembly_receipt_SHA256=sha(P/'STATIC_ASSEMBLY_RECEIPT.json')))
files = [dict(name=f.name,path=remote+'/'+f.name,sha256=sha(f),bytes=f.stat().st_size)
    for f in sorted(S.iterdir())if f.is_file()]
write(S/'FILE_BINDING.json',dict(UTC=now,files=files,stage=P.name,
    scope='Fixed sole R77 endpoint inference/replay/evaluation/aggregate and figures'))
review = dict(UTC=now,root_static_review_pass=True,
    production_sources_match_static_assembly=True,current_v5_binding_verified=True,
    sole_R77_terminal_model_native_numeric_receipts_pass=True,
    R77_Git_and_D_delivery_deferred_to_stage_delivery=True,source_private_arrays_decoded_by_registration=False,
    new_scientific_job_started=False,root_static_assembly_SHA256=sha(P/'STATIC_ASSEMBLY_RECEIPT.json'),
    user_instruction_SHA256=sha(P/'USER_EVALUATION_INSTRUCTION.json'),config_SHA256=sha(S/'config.json'),
    config_post_SHA256=sha(S/'config_post.json'),resource={**guard,**memory()})
write(P/'ROOT_REGISTRATION_REVIEW.json',review)
snapshot = P/'before_registration'
snapshot.mkdir()
for name in ['AGENTS.md','STATUS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','jobs.json','experiments.json']:
    shutil.copy2(R/name,snapshot/name)
entry = dict(id=P.name,stage=P.name,status='registered_not_started',defined_UTC=now,
    kind='sole_completed_R77_frozen_full_response_evaluation',task='tasks/'+P.name+'.md',
    registration='reviews/'+P.name+'/REGISTRATION.json',remote_root=remote,
    response_updates=0,new_HGB_fits=10,density_description_slots=16,density_EM_attempts_max=16,
    generated_rows=5984,independent_replay_rows=4000,acceptance_version=5,D_E_executed=False,
    model_goal_complete=False)
for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
    d = read(R/name)
    assert not any(row.get('stage',row.get('id')) == P.name for row in d[key])
    d[key].append(entry.copy())
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0)
    write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d = read(R/name)
    d.update(updated_UTC=now,status='R78_registered_frozen_R77_evaluation_not_started',
        current_stage=P.name,current_task=entry['task'],current_stage_registration=entry['registration'],
        current_job=None,current_activity='fixed5984_generation_independent4000_replay_original_A_C_evaluation',
        active_scientific_jobs=0,application_goal_status='paused',model_goal_complete=False,
        current_evaluation_directly_authorized=True,extra_numeric_recomputation=False,
        next_action='Run four sequential native entries on sole frozen R77 endpoint; no new response training',
        current_stage_delivery_status='pending_evaluation_and_delivery')
    write(R/name,d)
prefix = '''# 当前执行：R78唯一R77完整响应评价已登记

R77全部25000更新及原定TRAIN数值/序列化检查已正常完成exit0并收集到E。用户明确要求进行评价并取消额外重复复算。现绑定唯一完整响应，原SOURCE/同处理REAL及5984生成、原定独立4000重放、原10HGB/16密度槽和v5 A–C。汇总直接引用正式评价与已有原生证据，不再重复波形/Q/ADC/CDF/迁移数值计算。评价不重训、不选检查点或对照、不改数字；失败区/支持外原父全部保留。D/E另行登记，当前没有A–F成功结论；应用长期目标paused，本次评价由直接用户指令授权，实际Git/D交付待本轮完成。

任务：tasks/R78_FULL_RESPONSE_EVALUATION_001.md。按generate、replay、evaluate、summary顺序单作业，资源/E-D/根与既有辅助政策保持。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:
    (R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
local = [f for f in S.iterdir()if f.is_file()]+[task,P/'prepare_register.py',P/'control.py',P/'collect.py',P/'ROOT_REGISTRATION_REVIEW.json']
write(P/'REGISTRATION.json',dict(UTC=now,status='REGISTERED_BEFORE_SCIENCE',stage=P.name,
    local_files=[dict(path=f.relative_to(R).as_posix(),sha256=sha(f),bytes=f.stat().st_size)for f in local],
    training_endpoint=25000,generated_rows=5984,independent_replay_rows=4000,new_HGB=10,
    density_slots=16,acceptance_version=5,remote_root=remote))
print(json.dumps(dict(stage=P.name,registered=True,remote_root=remote,files=len(files),
    main_response_updates=0,source_arrays_decoded=False)),flush=True)
