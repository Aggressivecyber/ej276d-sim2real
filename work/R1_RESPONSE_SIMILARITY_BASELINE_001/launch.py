from __future__ import annotations
import os,sys,pathlib,json,datetime,time,subprocess,ctypes
from ctypes import wintypes
ROOT=pathlib.Path(__file__).resolve().parent
cfg=json.loads((ROOT/'config.json').read_text());history=ROOT/'RUN_HISTORY.json';past=json.loads(history.read_text()) if history.exists() else []
numeric_used=sum(r['wall_s'] for r in past);now=datetime.datetime.now(datetime.timezone.utc);deadline=datetime.datetime.fromisoformat(cfg['deadline_UTC']);timeout=min(1200-numeric_used,(deadline-now).total_seconds()-60)
if timeout<=0:raise RuntimeError('R1 stage/numeric budget exhausted')
run_name=f'run_{len(past)+1:03d}';folder=ROOT/run_name;folder.mkdir(exist_ok=False)
env=os.environ.copy();env['R1_RUN_DIR']=run_name
for k in ['TEMP','TMP']:env[k]='E:/GAN/research/sim2real_two_papers_20261008_0305/.runtime-cache/tmp'
env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUTF8']='1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[k]='4'
class PMC(ctypes.Structure):
 _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
getinfo=ctypes.windll.psapi.GetProcessMemoryInfo;getinfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
start=datetime.datetime.now(datetime.timezone.utc);clock=time.monotonic();peak=None;memory_probe_failures=0;reason=None
with (folder/'stdout.log').open('wb') as out,(folder/'stderr.log').open('wb') as err:
 p=subprocess.Popen([sys.executable,'-B',str(ROOT/'evaluate.py')],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
 print(json.dumps({'kind':'R1_numerical_started','supervisor_PID':os.getpid(),'worker_PID':p.pid,'start_UTC':start.isoformat(),'timeout_s':timeout,'run':run_name}),flush=True)
 while p.poll() is None:
  c=PMC();c.cb=ctypes.sizeof(c)
  if getinfo(wintypes.HANDLE(int(p._handle)),ctypes.byref(c),c.cb):peak=max(peak or 0,int(c.PeakWorkingSetSize))
  else:memory_probe_failures+=1
  size=sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file())
  if time.monotonic()-clock>timeout:reason='numeric/stage deadline'
  if peak is not None and peak>cfg['RSS_limit_bytes']:reason='RSS4GiB limit'
  if size>cfg['output_limit_bytes']:reason='output512MiB limit'
  if reason:p.kill();break
  time.sleep(.25)
 p.wait(timeout=5)
receipt={'supervisor_PID':os.getpid(),'worker_PID':p.pid,'worker_native_exit':p.returncode,'start_UTC':start.isoformat(),'stop_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_s':time.monotonic()-clock,'numeric_cumulative_wall_s':numeric_used+time.monotonic()-clock,'peak_RSS_observed':peak,'memory_probe_failures':memory_probe_failures,'RSS_measurement_status':'known' if peak is not None else 'unknown','reason':reason,'run':run_name,'threads':4,'command':[sys.executable,'-B',str(ROOT/'evaluate.py')],'EVAL_read':0,'SSH':0,'new_response_training':0}
past.append(receipt);history.write_text(json.dumps(past,indent=2)+'\n');(folder/'NATIVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
sys.exit(0 if p.returncode==0 and reason is None else 1)
