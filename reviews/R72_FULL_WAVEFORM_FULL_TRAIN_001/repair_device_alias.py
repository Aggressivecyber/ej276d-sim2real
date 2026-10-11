"""Preserve the invalid pre-data attempt, then register one exact device repair."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'device_repair_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
native=read(S/'train_full.py.NATIVE_RECEIPT.json');failure=read(S/'FAILURE.json')
assert native['actual_worker_native_exit']==1
assert failure['error']=="TypeError('W1 candidate must be a tensor on the common device')"
assert 'run_checks' in failure['traceback'] and not(S/'DATA_RECEIPT.json').exists()and not(S/'COST_ADMISSION.json').exists()
assert read(P/'COLLECTION_RECEIPT.json')['all_file_SHA_verified']and read(R/'jobs.json')['active_scientific_jobs']==0
snapshot=P/'attempt_01_predata_device_alias';assert not snapshot.exists();snapshot.mkdir()
manifest=[]
def preserve(src,relative):
    dst=snapshot/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    digest=sha(dst);assert digest==sha(src);manifest.append(dict(path=str(relative),sha256=digest,bytes=src.stat().st_size))
for f in S.iterdir():
    if f.is_file():preserve(f,Path('work')/f.name)
for f in P.iterdir():
    if f.is_file():preserve(f,Path('review')/f.name)
for name in ['AGENTS.md','STATUS.md','PLAN.md','PROGRAM.json','PROJECT_PROGRESS.json','experiments.json','jobs.json']:
    preserve(R/name,Path('governance')/name)
task=R/'tasks'/(P.name+'.md');preserve(task,Path('task')/task.name)
write(snapshot/'PRESERVED_SHA256.json',manifest)
oldreg=read(P/'REGISTRATION.json');C=read(S/'config.json');oldremote=C['remote_root'];C['remote_root']=oldremote+'_ENGINEERING_REPAIR_02'
source=S/'full_waveform_loss.py';before=source.read_text('utf8')
needle='        self.device = torch.device(device)\n'
replacement=needle+'        if self.device.type == "cuda" and self.device.index is None:\n            self.device = torch.device("cuda", torch.cuda.current_device())\n'
assert before.count(needle)==1;source.write_text(before.replace(needle,replacement),'utf8')
old_binding=read(S/'FILE_BINDING.json');immutable={row['name']for row in old_binding['files']}|{'FILE_BINDING.json'}
removed=[]
for f in S.iterdir():
    if f.is_file()and f.name not in immutable:
        assert f.resolve().parent==S.resolve()and sha(f)==sha(snapshot/'work'/f.name)
        removed.append(f.name);f.unlink()
for name in ['REGISTRATION.json','LAUNCH_RECEIPT.json','LATEST_NATIVE_OBSERVATION.json','COLLECTION_RECEIPT.json','R72_FROZEN_INSTALL.tar','R72_TERMINAL_OUTPUT.tar']:
    f=P/name
    if f.exists():
        assert f.resolve().parent==P.resolve()and sha(f)==sha(snapshot/'review'/name);f.unlink()
C['engineering_attempt']=2;C['previous_invalid_predata_attempt']=dict(remote_root=oldremote,native_exit=1,scientific_arrays_decoded=0,engineering_updates=0,main_updates=0,
    preserved_local=str(snapshot.relative_to(R)).replace('\\','/'))
write(S/'config.json',C)
note='''\n## 注册后的工程修复02\n\n首个受控进程在CUDA人工W1检查中将torch.device("cuda")与实际Tensor设备cuda:0直接比较，触发类型检查错误；原生exit1，尚未读取科学数组、工程更新0、主训练0。完整初次源码、输入绑定、原生输出和哈希在reviews/R72_FULL_WAVEFORM_FULL_TRAIN_001/attempt_01_predata_device_alias/保存。修复仅将未指定序号的CUDA别名解析为当前设备；科学公式、架构、人口、随机种子、终点和cost规则不变。修复后新远端目录重新执行全部CPU/CUDA人工检查，通过后才读TRAIN。\n'''
task.write_text(task.read_text('utf8')+note,'utf8');shutil.copy2(task,S/'PROTOCOL.md')
for f in S.glob('*.py'):compile(f.read_text('utf8'),str(f),'exec')
now=datetime.now(timezone.utc).isoformat()
write(S/'FILE_BINDING.json',dict(UTC=now,files=[dict(name=f.name,path=C['remote_root']+'/'+f.name,sha256=sha(f),bytes=f.stat().st_size)
    for f in sorted(S.iterdir())if f.is_file()and f.name!='FILE_BINDING.json'],scope='same scientific method with exact CUDA device alias resolution repair'))
repair=dict(UTC=now,previous_remote_root=oldremote,previous_actual_worker_PID=native['actual_worker_PID'],previous_native_exit=1,
    previous_scientific_arrays_decoded=0,previous_engineering_updates=0,previous_main_updates=0,
    previous_evidence_root=str(snapshot),preserved_files=len(manifest),preserved_SHA256_all_verified=True,
    repair='UniformW1Cache resolves unspecified CUDA index to current device before exact tensor-device validation',
    unchanged_scientific_method=True,new_remote_root=C['remote_root'],removed_current_terminal_copies=removed,
    source_before_SHA256=sha(snapshot/'work/full_waveform_loss.py'),source_after_SHA256=sha(source),guard={**guard,**memory()})
write(P/'ENGINEERING_REPAIR_02.json',repair)
for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
    d=read(R/name);j=next(v for v in d[key]if v.get('id',v.get('stage'))==P.name)
    j.update(status='registered_engineering_repair02_not_started',remote_root=C['remote_root'],current_job=None,engineering_attempt=2,
      prior_attempts=[dict(remote_root=oldremote,native_exit=1,status='INVALID_PRE_DATA_DEVICE_ALIAS',scientific_arrays_decoded=0,main_updates=0)],updated_UTC=now)
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=read(R/name);d.update(updated_UTC=now,status='R72_registered_engineering_repair02_not_started',current_job=None,active_scientific_jobs=0,
      current_activity='repeat_artificial_gate_after_CUDA_device_alias_fix',current_stage_progress='main0_engineering0_scientific_arrays0',
      engineering_repair_receipt='reviews/'+P.name+'/ENGINEERING_REPAIR_02.json');write(R/name,d)
prefix='''# 当前执行：R72 CUDA人工接口错误已封存，工程修复02待启动

首轮原生exit1发生在CUDA人工W1设备别名检查，科学数组0、工程更新0、主训练0。已保存全部原始源码/绑定/原生输出及SHA；仅修复cuda别名到cuda:0的匹配，科学方法/人口/50000终点/4小时规则不变。新注册目录重做全部CPU/CUDA人工检查后才读TRAIN；v5长期目标active。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:(R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
newreg={**oldreg,'UTC':now,'engineering_attempt':2,'previous_registration':str((snapshot/'review/REGISTRATION.json').relative_to(R)).replace('\\','/'),
    'repair_receipt':'reviews/'+P.name+'/ENGINEERING_REPAIR_02.json'}
paths=[R/v['path']for v in oldreg['local_files']]+[P/'repair_device_alias.py',P/'ENGINEERING_REPAIR_02.json']
newreg['local_files']=[dict(path=f.relative_to(R).as_posix(),sha256=sha(f),bytes=f.stat().st_size)for f in paths]
write(P/'REGISTRATION.json',newreg)
print(json.dumps(repair,indent=2),flush=True)
