from pathlib import Path
import json,ast,tarfile,hashlib,os
from datetime import datetime,timezone
r=Path(__file__).resolve().parent;c=json.loads((r/'config.json').read_text('utf8'));pre=json.loads((r/'SERVER_PREFLIGHT.json').read_text('utf-8-sig'))
assert pre['MemAvailable_bytes']>=60129542144 and pre['GPU_compute_jobs']=='' and not pre['own_project_jobs'] and not pre['remote_leaf_exists']
assert int(pre['GPU'].split(',')[2].strip())*1048576>=11811160064 and pre['disk_free_bytes']>8589934592+2147483648
remote=c.copy();old='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R43_REMOTE_20261009_233131_UTC'
remote['R43_root']=old;remote['TRAIN_path']=pre['R43_config_paths']['training_input_paths']['TRAIN_PRIVATE.npz']
for k in ['consumed_R26_stream','consumed_R26_receipt','actual_training_random_states','actual_training_receipt']:remote[k]=pre['R43_config_paths'][k]
for k,n in [('weight_path','SCALAR_TIME_FLOW_FINAL/FINAL_FLOW.pt'),('constants_path','UNIT_CHARGE_CONSTANTS.npz'),('coupling_path','SCALAR_TIME_FLOW_FINAL/ACTUAL_CFM_STREAM_PRIVATE.npz'),('train_receipt_path','SCALAR_TIME_FLOW_FINAL/TRAINING_RECEIPT.json')]:remote[k]=old+'/'+n
(r/'config_remote.json').write_text(json.dumps(remote,indent=2)+'\n','utf8')
g=(r/'linux_supervise_r43_original.py').read_text('utf8')
g=g.replace("['train_server.py','train_recover.py']","['diagnose_frozen.py']").replace("os.environ[k]='4'","os.environ[k]=str(cfg['CPU_threads'])").replace("time.time()-300,900)","time.time()-120)")
pos=g.index('\nstart=time.monotonic();row=')
extra='''
def process_tree(pid):
    links={}
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            fields=(d/'stat').read_text().rsplit(')',1)[1].split();links[int(d.name)]=int(fields[1])
        except(OSError,ValueError,IndexError):pass
    seen={pid};changed=True
    while changed:
        new={p for p,parent in links.items()if parent in seen};changed=not new.issubset(seen);seen.update(new)
    return seen
def tree_rss(pid):
    total=0;members=[]
    for p0 in process_tree(pid):
        try:
            lines=Path('/proc/'+str(p0)+'/status').read_text().splitlines()
            value=next(int(l.split()[1])*1024 for l in lines if l.startswith('VmRSS:'))
            total+=value;members.append(p0)
        except(OSError,StopIteration):
            if p0==pid:raise
    return total,members
last_GPU_query=0.
def gpu_used(pid):
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
    members=process_tree(pid);total=0
    for l in text.splitlines():
        if not l.strip():continue
        parts=l.split(',')
        if int(parts[0])in members:total+=int(parts[1])*1048576
    return total
'''
g=g[:pos]+extra+g[pos:]
g=g.replace("'handshake_before_imports':False};reason=None","'handshake_before_imports':False,'job_tree_RSS_peak_bytes':0,'GPU_process_memory_peak_bytes':0,'GPU_query_failures':[]};reason=None")
g=g.replace("if row['actual_worker_RSS_peak_bytes']>cfg['RSS_limit_bytes']:reason='actual_worker_RSS_limit'","tree,treeids=tree_rss(p.pid);row['job_tree_RSS_peak_bytes']=max(row['job_tree_RSS_peak_bytes'],tree);row['last_job_tree_PIDs']=treeids\n                    if tree>cfg['RSS_limit_bytes']:reason='job_tree_RSS_limit'\n                    if time.monotonic()-last_GPU_query>=2:\n                        last_GPU_query=time.monotonic()\n                        try:\n                            used=gpu_used(p.pid);row['GPU_process_memory_peak_bytes']=max(row['GPU_process_memory_peak_bytes'],used)\n                            if used>cfg['GPU_limit_bytes']:reason='GPU_process_memory_limit'\n                        except BaseException as e:row['GPU_query_failures'].append({'UTC':utc(),'error':repr(e)})")
ast.parse(g);(r/'linux_supervise.py').write_text(g,'utf8')
# Local auxiliary supervision reuses the actual-PID handshake and includes remote elapsed costs.
local=(r/'supervise.py').read_text('utf8').replace("entry=sys.argv[1];python=sys.argv[2]","cfg['RSS_limit_bytes']=4294967296;entry=sys.argv[1];python=sys.argv[2]")
local=local.replace("cfg['science_budget']-sum(r['elapsed_s']for r in history)","cfg['science_budget']-sum(r['elapsed_s']for r in history)-sum(r['elapsed_s']for r in json.loads((ROOT/'REMOTE_RUN_HISTORY.json').read_text('utf8')))")
ast.parse(local);(r/'supervise.py').write_text(local,'utf8')
names=['diagnose_frozen.py','network.py','source_flow_core.py','reorder_core.py','linux_supervise.py','linux_owned_worker.py']
with tarfile.open(r/'UPLOAD.tar','w',format=tarfile.USTAR_FORMAT)as t:
    for n in names:t.add(r/n,arcname=n)
    t.add(r/'config_remote.json',arcname='config.json')
binding={n:{'bytes':(r/n).stat().st_size,'sha256':hashlib.sha256((r/n).read_bytes()).hexdigest()}for n in [*names,'config_remote.json','supervise.py','owned_worker.py','config.json','PROTOCOL.md']}
(r/'PRE_EXECUTION.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'forward_rows_before_launch':0,'scientific_definition_unchanged':True,'resource_preflight':pre,'source_bindings':binding,'normal_transfer_no_content_tests':True},indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'AST_pass':True,'UPLOAD_bytes':(r/'UPLOAD.tar').stat().st_size,'unique_remote':c['remote_root'],'forward_rows':0}))
