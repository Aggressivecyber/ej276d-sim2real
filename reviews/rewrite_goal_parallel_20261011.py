from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, os

R=Path(__file__).resolve().parents[1]
D=R/'reviews/PROJECT_GOAL_PARALLEL_20261011_001'
APP=json.loads((D/'APP_GOAL_RECEIPT.json').read_text('utf8'))
goal=APP['application_result']['goal']
assert goal['status']=='active' and '按需使用多智能体' in goal['objective']
assert not (D/'RECEIPT.json').exists(), 'Revision already applied'
now=datetime.now(timezone.utc).isoformat()
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,text):
    tmp=path.with_name(path.name+'.parallel-rewrite-tmp')
    tmp.write_bytes(text.encode('utf8'))
    os.replace(tmp,path)
def wjson(path,value):
    write(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'
protected=[R/'PROJECT_ACCEPTANCE.json',R/'PROJECT_ACCEPTANCE.md',R/'tasks/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001.md',
           S/'config.json',S/'config_post.json',*sorted(S.glob('*.py'))]
protected=[p for p in protected if p.is_file()]
before_protected={str(p.relative_to(R)):digest(p) for p in protected}
names=['LONG_TERM_GOAL_20261010.md','AGENTS.md','PROJECT_OBJECTIVE.md','PLAN.md','README.md',
       'PROGRAM.json','PROJECT_PROGRESS.json','reviews/finalize_R63_root.py']
(D/'before').mkdir(parents=True,exist_ok=False)
for name in names:
    source=R/name
    target=D/'before'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
policy = """# 当前执行要求：等待期间主动研究，按需多智能体并行

2026-10-11 用户明确允许按需使用多智能体，并要求在作业等待期间主动调研、分析既有数据、思考原因、准备后续方法及评估并行效率。主助手统筹科学方向、证据综合、关键判断、作业协调、Git和交付；可把独立调研、合法既有结果分析、代码准备及检查交给多智能体并行处理，交付可复核的来源、分析、假设或实施方案。

所有智能体共同遵守原数据用途与资源限制：服务器CPU12/RSS48GiB/GPU10GiB、单科学作业，本机辅助工作合计CPU4/RSS4GiB，新增数据/源码/日志/缓存/临时文件只存E/D。新科学计算先登记再顺序执行；当前R63仍按原冻结50000步末点及后处理协议运行。不得在运行中更改冻结方案、提前打开最终评价选参，或以辅助工作替代原A–F验收。

本节覆盖下文及旧阶段文件中的“主助手独立、不使用子智能体”执行分工；旧科学方案和历史结果保留。完整长期目标见 [LONG_TERM_GOAL_20261010.md](LONG_TERM_GOAL_20261010.md)，实际应用目标回执见 [目标修订回执](reviews/PROJECT_GOAL_PARALLEL_20261011_001/APP_GOAL_RECEIPT.json)。

"""
old='原A–F保持；主助手独立、无子智能体/计算配额、CPU12/RSS48GiB/GPU10GiB、单科学作业、本机E/D。'
new='原A–F保持；主助手统筹并按需使用多智能体开展独立辅助工作，无计算配额，CPU12/RSS48GiB/GPU10GiB、单科学作业、本机E/D。'
for name in ['AGENTS.md','PROJECT_OBJECTIVE.md','PLAN.md','README.md']:
    p=R/name
    text=p.read_text('utf8')
    if name!='README.md':
        assert old in text, name
        text=text.replace(old,new,1)
    write(p,policy+text)
long_text='# 长期目标与结束条件（2026-10-11修订）\n\n'+goal['objective']+'\n\n'
long_text+='本次修订来自用户关于等待期间主动研究、提高并行效率及允许多智能体的明确要求。主助手保留科学判断与交付责任；所有代理共享原安全资源限额。原A–F文件及正在运行的R63科学协议保持原字节，旧目标文件备份于本次修订回执目录。\n\n'
long_text+='应用目标已实际建立为 active，未设置 token 预算。回执：reviews/PROJECT_GOAL_PARALLEL_20261011_001/APP_GOAL_RECEIPT.json。R49归因诊断为已完成的历史首项，当前从已登记R63继续，不重新执行历史首项。\n'
write(R/'LONG_TERM_GOAL_20261010.md',long_text)
coordination={
    'root_responsibility':['scientific_direction','evidence_synthesis','key_decisions','job_coordination','git','delivery'],
    'subagents_allowed':True,
    'waiting_work_required':['literature_research','analysis_of_permitted_saved_data','mechanism_and_failure_reasoning','next_method_and_validation_preparation','dependency_and_resource_based_parallelism'],
    'parallelism':'independent_auxiliary_tasks_within_shared_resource_and_data_role_limits',
    'max_active_scientific_jobs':1,
    'no_changes_to_live_frozen_method':True,
    'no_final_evaluation_based_selection':True,
    'local_auxiliary_CPU_limit_total':4,
    'local_auxiliary_RSS_GiB_limit_total':4,
    'output_drives':['E','D']
}
receipt_ref='reviews/PROJECT_GOAL_PARALLEL_20261011_001/APP_GOAL_RECEIPT.json'
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    p=R/name
    v=json.loads(p.read_text('utf8'))
    v.update(subagents_allowed=True,executor_agent='/root',
             root_role='scientific_direction_evidence_synthesis_job_coordination_review_delivery',
             execution_coordination=coordination,
             latest_user_steering='作业等待期间主动调研、既有数据分析、原因推理及后续方案准备；按需使用多智能体并行，保留原A-F和资源限制',
             latest_user_revision=receipt_ref,goal_registration_receipt=receipt_ref,goal_status_receipt=receipt_ref,
             goal_status_note='Application goal recreated active on explicit user rewrite; original A-F retained; root-led optional multi-agent auxiliary work required while waiting; no compute quotas.',
             application_goal_status='active',
             latest_permission_status='User permits on-demand multi-agent auxiliary research and analysis; one scientific job and existing resource/data-role limits remain.',
             next_action='Continue the same frozen R63 update50000 endpoint and queued original numerical/A-C/replay pipeline; while waiting, conduct evidence-backed parallel literature research, permitted saved-data analysis, and conditional next-method preparation; D/E only after A-C passes.')
    v['updated_at_utc' if name=='PROGRAM.json' else 'updated_UTC']=now
    wjson(p,v)
p=R/'reviews/finalize_R63_root.py'
text=p.read_text('utf8')
old_final='主助手独立执行，不使用子智能体；无计算配额，单科学作业和本机 E/D 存储限制保留。'
new_final='主助手统筹并按需使用多智能体，等待期间主动调研、分析既有证据和准备后续方法；无计算配额，单科学作业、共享资源和本机 E/D 存储限制保留。'
assert old_final in text
write(p,text.replace(old_final,new_final))
after_protected={str(p.relative_to(R)):digest(p) for p in protected}
assert before_protected==after_protected, 'Scientific freeze changed'
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    v=json.loads((R/name).read_text('utf8'))
    assert v['subagents_allowed'] is True and v['execution_coordination']['max_active_scientific_jobs']==1
    assert v['goal_status_receipt']==receipt_ref
for name in ['AGENTS.md','PROJECT_OBJECTIVE.md','PLAN.md','README.md']:
    assert (R/name).read_text('utf8').startswith(policy)
compile((R/'reviews/finalize_R63_root.py').read_text('utf8'),str(R/'reviews/finalize_R63_root.py'),'exec')
result={'UTC':now,'status':'project_goal_and_execution_policy_updated','application_goal_status':'active',
        'subagents_allowed':True,'waiting_research_and_analysis_required':True,
        'scientific_freeze_unchanged':before_protected==after_protected,
        'protected_scientific_files':before_protected,
        'updated_files':{name:{'before_SHA256':digest(D/'before'/name),'after_SHA256':digest(R/name)} for name in names},
        'same_live_job_retained':json.loads((S/'ROOT_CONTINUATION_OBSERVATION.json').read_text('utf8')),
        'scientific_jobs_started_by_this_revision':0}
wjson(D/'RECEIPT.json',result)
print(json.dumps({'status':result['status'],'application_goal_status':'active','subagents_allowed':True,'protected_scientific_files_unchanged':len(protected),'updated_files':names,'new_scientific_jobs':0},ensure_ascii=False))
