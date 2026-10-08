from __future__ import annotations
import os,sys,pathlib,json,datetime,time,subprocess,ctypes
from ctypes import wintypes
ROOT=pathlib.Path(__file__).resolve().parent
cfg=json.loads((ROOT/'config.json').read_text());history=ROOT/'RUN_HISTORY.json';past=json.loads(history.read_text()) if history.exists() else []
used=sum(r['wall_s'] for r in past);now=datetime.datetime.now(datetime.timezone.utc)
timeout=min(cfg['numeric_wall_limit_s']-used,(datetime.datetime.fromisoformat(cfg['deadline_UTC'])-now).total_seconds()-60)
if timeout<=0:raise RuntimeError('stage/numeric exhausted')
folder=ROOT/('numeric_'+str(len(past)+1));folder.mkdir(exist_ok=False)
env=os.environ.copy()
for k in ['TEMP','TMP']:env[k]='E:/GAN/research/sim2real_two_papers_20261008_0305/.runtime-cache/tmp'
env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUTF8']='1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[k]='4'
class FullPMC(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
getinfo=ctypes.windll.psapi.GetProcessMemoryInfo;getinfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(FullPMC),wintypes.DWORD]
start=datetime.datetime.now(datetime.timezone.utc);clock=time.monotonic();peak=None;failures=0;reason=None
with (folder/'stdout.log').open('wb') as out,(folder/'stderr.log').open('wb') as err:
    p=subprocess.Popen([sys.executable,'-B',str(ROOT/('replay.py' if os.environ.get('R5_MODE')=='replay' else 'run.py'))],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    print(json.dumps({'kind':'R5_utility_started','supervisor_PID':os.getpid(),'worker_PID':p.pid,'start_UTC':start.isoformat(),'timeout_s':timeout}),flush=True)
    while p.poll() is None:
        c=FullPMC();c.cb=ctypes.sizeof(c)
        if getinfo(wintypes.HANDLE(int(p._handle)),ctypes.byref(c),c.cb):peak=max(peak or 0,int(c.PeakWorkingSetSize))
        else:failures+=1
        size=sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file())
        if time.monotonic()-clock>timeout:reason='numeric/stage limit'
        if peak and peak>cfg['RSS_limit_bytes']:reason='RSS4GiB'
        if size>cfg['output_limit_bytes']:reason='output512MiB'
        if reason:p.kill();break
        time.sleep(.25)
    p.wait(timeout=5)
receipt={'kind':'prefix_reconstruction_utility','supervisor_PID':os.getpid(),'worker_PID':p.pid,'worker_native_exit':p.returncode,
         'start_UTC':start.isoformat(),'stop_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'wall_s':time.monotonic()-clock,'numeric_cumulative_wall_s':used+time.monotonic()-clock,
         'peak_RSS_observed':peak,'memory_probe_failures':failures,'reason':reason,'threads':4,'response_training':0,'SSH':0}
past.append(receipt);history.write_text(json.dumps(past,indent=2)+'\n');(folder/'NATIVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True);sys.exit(0 if p.returncode==0 and reason is None else 1)
