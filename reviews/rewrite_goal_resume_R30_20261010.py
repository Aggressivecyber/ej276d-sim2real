"""Apply explicit user goal rewrite and resume the unstarted R30 scope."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
STAGE = "R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001"
REG_REL = "reviews/R30_RESUME_20261010_001"
REG = ROOT / REG_REL
NOW = datetime.now(timezone.utc)
def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))
def save(name, obj):
    (ROOT / name).write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
def write(name, s):
    (ROOT / name).write_bytes(s.encode("utf-8"))

assert not subprocess.check_output(["git","diff","--cached","--name-only"],cwd=ROOT).strip()
p = read("PROGRAM.json")
old = read("reviews/R30_REGISTRATION_001/RECEIPT.json")
goal = read("reviews/PROJECT_GOAL_REWRITE_20261010_001.json")
assert goal["goal"]["status"] == "active"
assert p["current_stage"] == STAGE and p["active_scientific_jobs"] == 0
assert not (ROOT / "work" / STAGE).exists(), "Unstarted-scope assumption must be checked"
assert hashlib.sha256((ROOT / old["task"]).read_bytes()).hexdigest() == old["task_sha256"]
assert not REG.exists()
REG.mkdir()
snapshot_names = ["PROJECT_OBJECTIVE.md","PROJECT_PROGRESS.json","PROGRAM.json","PLAN.md","experiments.json","AGENTS.md"]
for name in snapshot_names:
    shutil.copy2(ROOT/name, REG/name)
seal = NOW + timedelta(seconds=3000)
deadline = NOW + timedelta(seconds=3600)
receipt = {
    "stage":STAGE,"registered_UTC":NOW.isoformat(),"user_request":"重写目标 推进",
    "reason":"Resume R30 after user status, waveform preview and goal rewrite; no R30 work directory or science existed.",
    "original_registration":"reviews/R30_REGISTRATION_001/RECEIPT.json",
    "original_registered_UTC":old["registered_UTC"],"original_stage_deadline_UTC":old["stage_deadline_UTC"],
    "task":old["task"],"task_sha256":old["task_sha256"],
    "task_addendum":"tasks/R30_RESUME_ADDENDUM_20261010_001.md",
    "science_spent_before_resume_s":0,"R30_directory_absent_before_resume":True,
    "stage_wall_budget_s":3600,"executor_seal_budget_s":3000,"root_delivery_reserve_s":600,
    "stage_deadline_UTC":deadline.isoformat(),"executor_seal_deadline_UTC":seal.isoformat(),
    "scientific_cumulative_budget_s":600,"HGB_fits_max":15,"response_updates":0,
    "response_inference_or_generation":0,"density_EM_max":0,
    "CPU_threads":4,"RSS_max_bytes":4*1024**3,"GPU_allocated_max_bytes":4*1024**3,
    "output_max_bytes":128*1024**2,"max_active_scientific_jobs":1,
    "execution":"local_guarded_CPU4_saved_output_diagnostic","SSH_allowed":False,
    "new_EVAL":0,"new_CAL_waveform_decode":0,"new_G4":0,"new_installation":0,
    "science_scope_and_sample_split_changed":False,
    "goal_receipt":"reviews/PROJECT_GOAL_REWRITE_20261010_001.json",
    "project_goal_status":"active"
}
save(REG_REL+"/RECEIPT.json",receipt)
addendum = f"""# R30 恢复执行补充（2026-10-10）

用户明确要求“重写目标 推进”。当前应用目标已重新创建为 active。R30 原方案已登记，但在状态沟通、波形预览和目标改写期间未启动；恢复前 work/{STAGE} 不存在，科学成本为0。原登记和所有历史结果保留。

R30 科学方案仍完全采用 tasks/R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001.md，任务SHA {old['task_sha256']}。本补充只替换执行墙钟起点：{NOW.isoformat()}；执行器封存截止 {seal.isoformat()}，根交付截止 {deadline.isoformat()}。最多600秒累计科学计算、15次HGB、CPU4、实际worker RSS4GiB、GPU分配4GiB、新文件128MiB；响应训练/推理/生成均为0。沿用导入前实际worker PID及RSS守护，失败尝试计入预算。

本轮在本机用保存输出完成，不SSH、不安装、不读取新REAL/EVAL/CAL/G4。既有执行器连续完成实现、运行、必要自检、图表检查和交付，不等待子步骤许可，不派生代理。保留六域统一训练计数至少20的条件区、各域全局回退、固定留出区每域至少10的描述门槛及原任务全部解释边界。保存变换参数、索引、预测和紧凑统计即可，无须复制全部变换后大数组。

根负责方向、关键结果判断、私有Git和D盘正常副本。R30之后的新训练任务另按诊断结果登记单一改动与比较预算；服务器已经恢复，届时先只读确认资源/现有作业和本项目独立目录，再决定该训练任务执行位置。R30本轮地点不变。
"""
write(receipt["task_addendum"],addendum)
objective = """# 当前项目目标（2026-10-10重写）

用户要求“重写目标 推进”。应用持续目标已经重新创建为active；本文件、PROJECT_PROGRESS.json与本轮任务登记共同定义当前工作。旧科学结果及原始数据保留，旧暂停或阶段结束不表示研究完成。

## 目标与当前基线

完成EJ-276D完整仿真波形到实测响应的无配对神经转换，保留源事件关联，交付可运行、可复现且结论与证据一致的方法。当前开发比较基线为R28 FRESH_FINAL第2950步；这是已比较完整波形指标的当前最佳候选，完整响应验收尚未通过。原TRAIN内部选模规则不因这一开发比较而追溯改变。R29有部分PSD收益，但完整波形退化，该单次配方关闭。

## 当前执行路径

1. 先完成R30保存输出诊断：区分全局/观察电荷条件平均波形、逐点标准差及剩余分布结构能够解释的差距；相同处理的REAL-REAL作为参照。参数只用固定分类器训练行拟合，固定留出行评估。诊断变换需要域身份，不能当作可部署波形转换器。
2. 根据诊断选择一个主要问题，另行登记一次有界训练或验证。训练改动一次只回答一个问题，比较相同数据与预算的基线，并明确继续/关闭规则。
3. 对有证据支持的候选冻结方法，完成既定任务效用与适用范围验证、模型复现和最终研究交付。负结果保留；单轮封存不等于项目完成。

## 方法与共同验收

- 输入是完整480点源仿真波形，两域无逐事件配对；不构造随机配对逐点真值，不以目标域随机自生成替代源驱动转换。
- 源父事件、沉积组成、电荷与PSD相关信息共同验证；纯沉积组成与入射粒子真值分开，unknown/重叠与QC/支持失败保留。
- 完整及局部波形、电荷-PSD联合分布、条件峰位/峰宽、噪声/时间相关、源内容保真与同处理REAL-REAL参照共同决定是否升级；一个指标改善或波形视觉变平滑不足以通过。
- 只在科学证据支持的范围内给结论；单次采集、历史开发暴露和高置信伪标签边界延续，不宣称物理等效、唯一真实映射或独立采集泛化。
- TRAIN用于响应训练和内部选择，DEV用于开发判断，已暴露EVAL只允许候选冻结后的明确回顾性报告，不反复用于选参。复用已审来源，不重复完整来源审计。

## 执行与交付

根负责方向、预算登记和关键结果判断；既有GPT-6.1-Sol-Max执行器连续实现、运行、必要自检、绘图、报告和封存，无子步骤审批、不新增审核代理。每轮记录问题、实际结论、未知项及下一轮理由，不做无边界参数扫描。

本轮R30在本机执行已有输出诊断，详见tasks/R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001.md及tasks/R30_RESUME_ADDENDUM_20261010_001.md。后续训练可用已恢复服务器，必须先确认资源和现有作业，在本项目新的独立目录按新登记执行。一次一个科学作业，默认CPU4线程、实际进程RSS4GiB、GPU分配4GiB；每轮登记科学/墙钟/文件量预算。不修改共享环境和他人作业，未知状态先确认，不重复提交。

本机数据、代码、日志、缓存和临时文件只写E/D。根以明确路径维护私有Git，代码/配置/聚合结果/关键图提交；模型和逐事件数据留E/D，D盘按正常复制备份，迁移不新增内容一致性测试。保留原始数据和历史产物。

最终交付权重、推理入口、可重现输入输出、直观仿真/重建/同条件实测参照图、完整评价与任务效用/适用范围报告，以及明确的失败和未知项。只有项目实质验收和交付完成才能标记目标完成；必要外部材料缺失时说明具体缺口及影响，不虚构通过。
"""
write("PROJECT_OBJECTIVE.md",objective)
p.update(status="project_goal_active",current_stage_progress="resumed_registered_for_executor",
    current_job={"stage":STAGE,"status":"resumed_local_diagnostic_preparation","scientific_jobs":0,"execution_location":"local"},
    current_stage_registration=REG_REL+"/RECEIPT.json",current_engineering_addendum=receipt["task_addendum"],
    stage_deadline_UTC=receipt["stage_deadline_UTC"],current_stage_deadline_UTC=receipt["stage_deadline_UTC"],
    executor_seal_deadline_UTC=receipt["executor_seal_deadline_UTC"],updated_at_utc=NOW.isoformat(),
    latest_user_steering="重写目标 推进",latest_user_steering_receipt=receipt["goal_receipt"],
    goal_registration_receipt=receipt["goal_receipt"],current_plan_integrity_reference="PROJECT_PROGRESS.json",
    baseline_for_next_response_trial={"stage":"R28_FRESH_PROJECTION_TRANSLATION_001","endpoint":"FRESH_FINAL","step":2950,
    "role":"development_reference_not_scientifically_accepted"},
    latest_permission_status="Explicit goal rewrite and proceed; existing executor continues bounded R30 without substep approval.",
    next_action="Existing executor completes unchanged R30 within renewed wall clock; root reviews diagnosis and registers one targeted next experiment.")
p["execution_location"]["subsequent_training_policy"]="Server reported restored; verify resources and own job state before a separately registered training stage. R30 remains local."
save("PROGRAM.json",p)
progress=read("PROJECT_PROGRESS.json")
progress.update(updated_UTC=NOW.isoformat(),application_goal_status="active",project_status="active_not_complete",
    current_activity="R30_resumed_saved_output_moment_diagnostic",latest_user_steering="重写目标 推进",
    latest_user_steering_receipt=receipt["goal_receipt"],goal_registration_receipt=receipt["goal_receipt"],
    current_engineering_addendum=receipt["task_addendum"],
    current_direction="R28 baseline; complete R30 mean/scale diagnostic, then register one evidence-directed change under a matched comparison and joint full-response acceptance.")
save("PROJECT_PROGRESS.json",progress)
exp=read("experiments.json")
rows=[row for row in exp["experiments"] if row["id"]==STAGE]
assert len(rows)==1
rows[0].update(status="resumed_registered_for_executor",resume_registration=REG_REL+"/RECEIPT.json",
    resumed_UTC=NOW.isoformat(),resume_reason="Explicit user goal rewrite after status/preview; zero prior R30 science.",
    task_addendum=receipt["task_addendum"],project_goal_status="active")
save("experiments.json",exp)
intro = "# 修订056：R28基线、先诊断再针对性训练\n\n2026-10-10用户要求“重写目标 推进”。项目目标已重新创建为active，验收见PROJECT_OBJECTIVE.md。R30尚未启动，按原科学范围恢复执行，仅以tasks/R30_RESUME_ADDENDUM_20261010_001.md更新墙钟起点。根随后据诊断只登记一个主要问题；完整波形、PSD、源保真和REAL-REAL共同决定模型升级。R28当前为开发比较基线，R29保持负结果。后续训练可安排已恢复服务器，先确认资源/作业并登记独立阶段；本轮诊断仍在本机完成。\n\n"
write("PLAN.md",intro+(ROOT/"PLAN.md").read_text(encoding="utf-8-sig"))
ag="# 2026-10-10当前执行补充\n\n用户明确要求“重写目标 推进”。当前目标见PROJECT_OBJECTIVE.md；先以本机保存输出完成R30原范围诊断，使用tasks/R30_RESUME_ADDENDUM_20261010_001.md中的新墙钟预算。服务器已由用户报告恢复，后续训练可在确认资源/已有作业后登记本项目独立目录执行；本条覆盖旧的临时后续全本机安排，但不改变R30的local-only范围。根保持方向和Git，既有执行器连续施工，不新派代理。当前基线R28，按完整响应和源保真共同验收。\n\n"
write("AGENTS.md",ag+(ROOT/"AGENTS.md").read_text(encoding="utf-8-sig"))
print(json.dumps(receipt,ensure_ascii=True))
