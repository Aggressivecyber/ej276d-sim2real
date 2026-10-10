"""Ordinary small-file return of new R48 outputs, without migration tests."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,os,time,shlex

R=Path(__file__).resolve().parent
C=json.loads((R/'config.json').read_text('utf8'))
remote=C['remote_root'];python=C['server_python'];start=time.monotonic()
program=("from pathlib import Path;import json;"
         f"r=Path({remote!r});"
         "files=[{'name':p.relative_to(r).as_posix(),'size':p.stat().st_size}"
         " for p in r.rglob('*') if p.is_file()];"
         "print(json.dumps({'files':files,'total_file_bytes':sum(x['size'] for x in files)}))")
args=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',
      'rylan@100.82.231.102',python+' -B -c '+shlex.quote(program)]
q=subprocess.run(args,capture_output=True,text=True,timeout=60)
if q.returncode:raise RuntimeError({'inventory_exit':q.returncode,'stderr':q.stderr})
inv=json.loads(q.stdout)
assert inv['total_file_bytes']<=167772160,('remote stage exceeded160MiB',inv['total_file_bytes'])
(R/'REMOTE_OUTPUT_INVENTORY.json').write_bytes((json.dumps(inv,indent=2)+'\n').encode())
sources=set(C['arms'])
arms=sources|{'GUIDED_FIXED_NOISE_REAL_CONTROL','GUIDED_COMPLETION_REAL_CONTROL'}
record={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
        'remote_total_file_bytes':inv['total_file_bytes'],'inventory_is_stat_only':True,
        'normal_transfer_no_content_equivalence_test':True,'operations':[]}
for item in sorted(inv['files'],key=lambda x:x['name']):
    name=item['name'];parts=Path(name).parts
    if not parts or '..'in parts:raise ValueError(name)
    output=parts[0]in arms or parts[0]=='results'
    # Training models/RNG already returned at their terminal milestones.
    if output and (R/name).exists() and parts[0]in sources and len(parts)>1:
        continue
    metadata=len(parts)==1 and (name.endswith('.json')or name.endswith('.log')or name.endswith('.used'))
    if not output and not metadata:continue
    if len(parts)==1 and name in {'config.json','config_cpu.json','PRETRAIN_FREEZE.json','SERVER_PREFLIGHT.json','EXECUTOR_ACK.json'}:continue
    dest=R/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if sum(p.stat().st_size for p in R.rglob('*')if p.is_file())+item['size']>167772160:
        raise RuntimeError(('local stage160MiB reservation failed',name,item['size']))
    p=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',
                      'rylan@100.82.231.102:'+remote+'/'+name,str(dest)],
                     capture_output=True,text=True,timeout=90)
    record['operations'].append({'file':name,'native_exit':p.returncode,
                                 'bytes':dest.stat().st_size if dest.exists()else None,'stderr':p.stderr[-1000:]})
    (R/'OUTPUT_RETURN_RECEIPT.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
    if p.returncode:raise RuntimeError(record['operations'][-1])
record.update(elapsed_s=time.monotonic()-start,
              local_total_file_bytes=sum(p.stat().st_size for p in R.rglob('*')if p.is_file()))
(R/'OUTPUT_RETURN_RECEIPT.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in record.items()if k!='operations'}|{'files_returned':len(record['operations'])}),flush=True)
