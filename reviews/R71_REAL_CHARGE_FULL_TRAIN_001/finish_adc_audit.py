from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
P=Path(__file__).resolve().parent;R=P.parents[1];A=P/'adc_representation_audit'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'adc_fetch_runtime');remote=json.loads((A/'LAUNCH.json').read_text())['remote_root']
code='from pathlib import Path\nimport os,json\nos.sched_setaffinity(0,{11})\np=Path('+repr(remote)+')\n'+r'''
native=json.loads((p/'adc_audit.py.NATIVE_RECEIPT.json').read_text())
result=json.loads((p/'ADC_REPRESENTATION_AUDIT.json').read_text())
assert native['actual_worker_native_exit']==0 and native['termination_reason']is None
assert not Path('/proc/'+str(native['actual_worker_PID'])).exists()
files={f.name:f.read_text()for f in p.iterdir()if f.is_file()and f.suffix in {'.json','.log'}and f.name not in ['config.json','REGISTRATION.json']}
print(json.dumps(dict(native=native,result=result,files=files)))
'''
run=subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
assert run.returncode==0,run.stderr
data=json.loads(run.stdout)
for name,content in data.pop('files').items():(A/name).write_text(content,'utf8')
data.update(UTC=datetime.now(timezone.utc).isoformat(),guard={**guard,**memory()})
(A/'ROOT_ADC_AUDIT_RECEIPT.json').write_text(json.dumps(data,indent=2)+'\n','utf8')
jobs=json.loads((R/'jobs.json').read_text('utf8'));row=next(x for x in jobs['jobs']if x.get('stage')==P.name+'_ADC_AUDIT')
row.update(status='completed',native_exit=0,receipt=str((A/'ROOT_ADC_AUDIT_RECEIPT.json').relative_to(R)),native=data['native'])
jobs.update(active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0)
(R/'jobs.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2)+'\n','utf8')
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    d=json.loads((R/name).read_text('utf8'));d.update(active_scientific_jobs=0,current_activity='R71_root_review_and_delivery',current_job=None)
    (R/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf8')
print(json.dumps(dict(native_exit=data['native']['actual_worker_native_exit'],result=data['result'],native_peak_RSS_bytes=data['native']['job_tree_RSS_peak_bytes'])))
