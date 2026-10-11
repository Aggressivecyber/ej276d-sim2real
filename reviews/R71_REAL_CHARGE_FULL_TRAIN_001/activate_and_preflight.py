"""Record the new user-authorized goal and inspect native server resources."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import subprocess
import sys
P = Path(__file__).resolve().parent
R = P.parents[1]
sys.path.insert(0, str(R / 'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard = apply(P / 'activation_runtime')
now = datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    old = path.read_bytes() if path.exists() else b''
    nl = b'\r\n' if old.count(b'\r\n') > old.count(b'\n') // 2 else b'\n'
    trailer = old[len(old.rstrip(b'\r\n')):] or nl
    path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2).encode().replace(b'\n', nl) + trailer)


goal = read(P / 'GOAL_CREATION_RECEIPT.json')['result']['goal']
assert goal['status'] == 'active'
assert read(R / 'CURRENT_ACCEPTANCE.json')['version'] == 5
snapshot = P / 'before_goal_activation'
assert not snapshot.exists()
snapshot.mkdir()
for name in ('AGENTS.md', 'STATUS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'LONG_TERM_GOAL_20261010.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json'):
    shutil.copy2(R / name, snapshot / name)
prefix = '''# 当前执行：v5 长期目标已激活，优先数小时内完成全量

用户要求“设置目标推进 如果全量在数个小时能跑完 就直接全量”。应用目标已按 CURRENT_ACCEPTANCE.json 指向的 v5 实际建立为 active，无 token 预算。本条覆盖下文 paused 与仅静态准备状态。实测电荷分布作为目标，源父事件与权重、同处理 REAL、其余联合验收和历史负结果保持。

R71 准备公共条件电荷响应头的完整 TRAIN 训练：全部 6593 SOURCE 与 21047 REAL 参与，先测实际吞吐，以保守预计 4 小时为本次“数个小时”的执行判断；能满足则直接跑预登记全量固定终点。此项电荷响应训练不代替最终完整波形 A–F。全部数值工作先登记，一次一个科学作业；不恢复旧 R65。主助手统筹，既有 gpt-6.1-sol/max 辅助实现和静态审核；服务器 CPU12/RSS48GiB/GPU10GiB，本机共享 CPU4/RSS4GiB，新增本地文件只写 E/D。

当前准备入口：reviews/R71_REAL_CHARGE_FULL_TRAIN_001/；目标实际回执为 GOAL_CREATION_RECEIPT.json。没有降低科学门槛或把耗时估计视为模型成功。

## 以下为历史记录

'''
for name in ('AGENTS.md', 'STATUS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'LONG_TERM_GOAL_20261010.md'):
    old = (R / name).read_bytes()
    nl = b'\r\n' if old.count(b'\r\n') > old.count(b'\n') // 2 else b'\n'
    (R / name).write_bytes(prefix.encode().replace(b'\n', nl) + old)
for name in ('PROGRAM.json', 'PROJECT_PROGRESS.json'):
    data = read(R / name)
    assert data['active_scientific_jobs'] == 0
    data.update({
        'updated_UTC': now, 'updated_at_utc': now, 'application_goal_status': 'active',
        'application_goal_status_observed_UTC': read(P / 'GOAL_CREATION_RECEIPT.json')['UTC'],
        'application_goal_status_note': 'New v5 goal created on explicit user request; authoritative receipt in R71.',
        'goal_registration_receipt': 'reviews/' + P.name + '/GOAL_CREATION_RECEIPT.json',
        'goal_status_receipt': 'reviews/' + P.name + '/GOAL_CREATION_RECEIPT.json',
        'project_status': 'active_v5_model_incomplete', 'status': 'R71_full_TRAIN_method_preparation',
        'current_stage': P.name, 'current_stage_scope': 'Common conditional charge-response full TRAIN endpoint after engineering cost measurement',
        'current_activity': 'R71_implementation_registration_and_cost_preparation',
        'current_task': None, 'current_stage_registration': None, 'numeric_execution_released': False,
        'latest_user_steering': '设置目标推进 如果全量在数个小时能跑完 就直接全量',
        'latest_user_steering_receipt': 'reviews/' + P.name + '/GOAL_CREATION_RECEIPT.json',
        'latest_permission_status': 'Full legal TRAIN authorized directly if measured projected endpoint fits several hours; no stepwise approval',
        'current_priority': 'Run full current-method TRAIN endpoint when conservative measured estimate is at most 4 hours',
        'current_direction': 'Common learned charge response preserving real charge law; all original source parents retained',
        'next_action': 'Finish one fixed implementation, register cost measurement and full endpoint, launch one supervised worker',
        'next_prepared_task': P.name, 'model_goal_complete': False,
        'goal_status_note': 'Active v5 goal; charge-component training alone cannot complete A-F.',
        'execution_policy': 'Preregister engineering timing and fixed full-data endpoint; run full if conservative estimate <=4 hours',
        'further_scientific_execution_authorized': True,
        'execution_plan': 'PLAN.md', 'current_route_decision': 'reviews/' + P.name + '/GOAL_CREATION_RECEIPT.json',
    })
    write(R / name, data)

code = r'''
from pathlib import Path
from datetime import datetime,timezone
import os,json,shutil,subprocess
os.sched_setaffinity(0,{11})
base=Path('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305')
processes=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit() or int(d.name)==os.getpid():continue
 try:
  argv=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf8','replace')
  cwd=str((d/'cwd').resolve())
  if 'python' in argv and (str(base) in argv or cwd.startswith(str(base))):
   s=(d/'stat').read_text().rsplit(')',1)[1].split()
   processes.append(dict(PID=int(d.name),state=s[0],start_ticks=int(s[19]),cwd=cwd,argv=argv,affinity=sorted(os.sched_getaffinity(int(d.name)))))
 except(OSError,PermissionError):pass
mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),project_processes=processes,MemAvailable_bytes=mem['MemAvailable'],GPU=gpu,GPU_compute_jobs=jobs,disk_free_bytes=shutil.disk_usage(base).free,R69_native_receipt_exists=(base/'R69_ROOT_20261011_082144_UTC/pilot.py.NATIVE_RECEIPT.json').exists())))
'''
run = subprocess.run(['ssh', '-T', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                      'rylan@100.82.231.102', '/usr/bin/python3 -B -'], input=code,
                     text=True, capture_output=True, encoding='utf8', timeout=30)
(P / 'PREFLIGHT.stdout.log').write_text(run.stdout, encoding='utf8')
(P / 'PREFLIGHT.stderr.log').write_text(run.stderr, encoding='utf8')
assert run.returncode == 0, run.stderr
result = json.loads(run.stdout)
result.update(local_E_free_bytes=shutil.disk_usage('E:/').free, local_D_free_bytes=shutil.disk_usage('D:/').free,
              local_guard={**guard, **memory()}, science_arrays_read=False, scientific_job_launched=False)
write(P / 'NATIVE_PREFLIGHT.json', result)
print(json.dumps(result, ensure_ascii=False, indent=2))
