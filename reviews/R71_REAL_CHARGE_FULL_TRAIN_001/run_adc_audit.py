"""Register and launch one original-output quantizer audit, without model inference."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,tarfile,shutil
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name;A=P/'adc_representation_audit'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'adc_control_runtime')
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((R/'jobs.json').read_text('utf8'))['active_scientific_jobs']==0
assert json.loads((S/'train_full.py.NATIVE_RECEIPT.json').read_text('utf8'))['actual_worker_native_exit']==0
A.mkdir();old=json.loads((S/'config.json').read_text());remote=old['remote_root']+'_ADC_AUDIT'
cfg={**old,'R71_root':old['remote_root'],'remote_root':remote,'CPU_threads':1,'exact_CPU_affinity':[11],
    'allowed_scientific_entries':['adc_audit.py'],'GPU_scientific_entries':['adc_audit.py'],'stage':P.name+'_ADC_AUDIT'}
write(A/'config.json',cfg)
for name in ['linux_supervise.py','linux_owned_worker.py']:shutil.copy2(S/name,A/name)
shutil.copy2(P/'adc_audit.py',A/'adc_audit.py')
files=[dict(name=f.name,sha256=sha(f))for f in sorted(A.iterdir())if f.is_file()]
write(A/'REGISTRATION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),source_files=files,
    parent_stage=P.name,source_preADC_files=['LEARNED_SOURCE_PRIVATE.npz','LEARNED_REAL_PRIVATE.npz'],
    fixed_selection='first64 each',purpose='Check actual integer ADC codes on NumPy versus frozen CUDA ADC operator; raw float64 voltage equality was misnamed in the main diagnostic',
    rows=128,model_inference_rows=0,optimizer_updates=0,original_metrics_unchanged=True,
    input_hashes={n:sha(S/n)for n in ['LEARNED_SOURCE_PRIVATE.npz','LEARNED_REAL_PRIVATE.npz','physical_operators.py']},
    CPU=1,RSS_limit_bytes=cfg['RSS_limit_bytes'],GPU_limit_bytes=cfg['GPU_limit_bytes']))
package=A/'AUDIT_INSTALL.tar'
with tarfile.open(package,'w')as t:
    for f in sorted(A.iterdir()):
        if f.is_file()and f!=package:t.add(f,arcname=f.name,recursive=False)
def ssh(code):
    r=subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
    assert r.returncode==0,r.stderr
    return json.loads(r.stdout)
ssh('from pathlib import Path\nimport os,json\nos.sched_setaffinity(0,{11})\nPath('+repr(remote)+').mkdir()\nprint("{}")')
cp=subprocess.run(['scp','-q',str(package),'rylan@100.82.231.102:'+remote+'/AUDIT_INSTALL.tar'],capture_output=True,text=True,timeout=30);assert cp.returncode==0,cp.stderr
code='from pathlib import Path\nimport os,json,hashlib,tarfile,subprocess\nos.sched_setaffinity(0,{11})\np=Path('+repr(remote)+')\n'+r'''
with tarfile.open(p/'AUDIT_INSTALL.tar')as t:t.extractall(p,filter='data')
cfg=json.loads((p/'config.json').read_text());reg=json.loads((p/'REGISTRATION.json').read_text())
for item in reg['source_files']:assert hashlib.sha256((p/item['name']).read_bytes()).hexdigest()==item['sha256']
for name,digest in reg['input_hashes'].items():assert hashlib.sha256((Path(cfg['R71_root'])/name).read_bytes()).hexdigest()==digest
with(p/'SUPERVISOR.stdout.log').open('wb')as out,(p/'SUPERVISOR.stderr.log').open('wb')as err:
 child=subprocess.Popen([cfg['server_python'],'-B',str(p/'linux_supervise.py'),'adc_audit.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
print(json.dumps(dict(PID=child.pid,remote_root=str(p))))
'''
launched=ssh(code);launched['UTC']=datetime.now(timezone.utc).isoformat();write(A/'LAUNCH.json',launched)
jobs=json.loads((R/'jobs.json').read_text('utf8'));jobs['jobs'].append(dict(stage=cfg['stage'],parent=P.name,status='running',remote_root=remote,
    registration=str((A/'REGISTRATION.json').relative_to(R)),main_scientific_worker_complete=True,scope='128row ADC-only numerical audit',supervisor_PID=launched['PID']))
jobs.update(active_scientific_jobs=1,active_remote_scientific_jobs=1,active_local_scientific_jobs=0);write(R/'jobs.json',jobs)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=json.loads((R/name).read_text('utf8'));d.update(active_scientific_jobs=1,current_activity='R71_supplementary_ADC_representation_audit',current_job=launched);write(R/name,d)
print(json.dumps(launched),flush=True)
