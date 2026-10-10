from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,sys
R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
C=json.loads((S/'config.json').read_text('utf8'));host='rylan@100.82.231.102';remote=C['remote_root'];events=[]
required=sys.argv[1]
assert required in ['train_final.py','numerical_check.py','generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py']
def call(args,code=None):
    p=subprocess.run(args,input=code,text=True,capture_output=True,timeout=180)
    events.append({'argv':args,'native_exit':p.returncode,'stderr':p.stderr})
    if p.returncode:raise RuntimeError(events[-1])
    return p.stdout
code='from pathlib import Path\nimport json\np=Path('+repr(remote)+')\n'+'''
a=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text());assert a.get('actual_worker_native_exit')==0 and a['termination_reason']is None
assert not Path('/proc',str(a['actual_worker_PID'])).exists()
n=json.loads((p/'''+repr(required+'.NATIVE_RECEIPT.json')+''').read_text());assert n['actual_worker_native_exit']==0 and n['termination_reason']is None
print(json.dumps([str(f.relative_to(p))for f in sorted(p.rglob('*'))if f.is_file()and 'cache'not in f.relative_to(p).parts]))
'''
files=json.loads(call(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -'],code))
protected={n for n in files if Path(n).suffix in ['.py','.md'] or Path(n).name in ['config.json','config_post.json','config_reporting.json','INPUT_BINDING.json','POSTPROCESS_PREPARATION.json','POSTPROCESS_ROOT_REVIEW.json']}
selected=[n for n in files if n not in protected or not(S/n).exists()]
for parent in sorted(set(str(Path(n).parent)for n in selected)):
    dest=S/parent;dest.mkdir(parents=True,exist_ok=True)
    names=[n for n in selected if str(Path(n).parent)==parent]
    assert all('..'not in Path(n).parts and not Path(n).is_absolute()for n in names)
    call(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',*[host+':'+remote+'/'+n for n in names],str(dest)+'/'])
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'required_terminal_entry':required,'events':events,'copied':selected,'normal_transfer':True,'migration_consistency_test':False}
(S/('FETCH_'+required+'.json')).write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps({'files_copied':len(selected),'bytes':sum((S/n).stat().st_size for n in selected),'required_terminal_entry':required},indent=2))
