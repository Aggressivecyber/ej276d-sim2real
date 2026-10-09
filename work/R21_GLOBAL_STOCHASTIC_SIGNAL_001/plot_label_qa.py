from __future__ import annotations
import os,time,pathlib,json,datetime,shutil,ctypes
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parent
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m=json.loads((ROOT/'metrics.json').read_text('utf8'));valley=json.loads((ROOT/'VALLEY_DEFINITION.json').read_text('utf8'))
out=ROOT/'results/figures';old=ROOT/'results/figures_pre_label_qa';old.mkdir(exist_ok=True)
shutil.copyfile(out/'style_only_parent_PSD.png',old/'style_only_parent_PSD.png')
# Replot the already saved 1024 PSD values; no waveform inference, feature
# computation, optimizer or density/classifier fit occurs.
with np.load(ROOT/'results/style_only32_REPEAT_PRIVATE.npz',allow_pickle=False)as a:
    p=a['PSD'].reshape(32,32);valid=a['valid'].reshape(32,32);cat=a['parent_composition']
rows=m['random_decomposition']['groups']['style_only32']['rows']
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,constrained_layout=True)
for ax,kind in zip(axes,['pure_EE','pure_had']):
    for j,i in enumerate(np.flatnonzero(cat==kind)):
        ax.scatter(np.full(np.sum(valid[i]),j),p[i][valid[i]],s=8,alpha=.5,color='tab:purple')
        b=rows[i]['source_Qbin_1based'];de=valley['bins'][b-1]if b is not None else None
        if de and de['valid']:
            ax.plot([j-.3,j+.3],[de['low'],de['low']],color='gray',alpha=.5)
            ax.plot([j-.3,j+.3],[de['high'],de['high']],color='gray',alpha=.5)
    ax.set_title(kind+': 16 fixed sources x32 styles\nold z/white fixed; gray = frozen source-Q valley')
    ax.set_xlabel('prespecified parent order (not a particle label)');ax.grid(alpha=.15)
axes[0].set_ylabel('signed PSD');fig.savefig(out/'style_only_parent_PSD.png',dpi=160,bbox_inches='tight');plt.close(fig)
class Mem(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t)for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
gm=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo;gm.argtypes=[ctypes.c_void_p,ctypes.POINTER(Mem),ctypes.c_ulong];gm.restype=ctypes.c_int
mem=Mem();mem.cb=ctypes.sizeof(mem);ok=gm(kernel.GetCurrentProcess(),ctypes.byref(mem),mem.cb)
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'operation':'saved PSD figure title precision only: own-Q -> source-Q','old_figure_preserved':True,'fits':0,'generation':0,'feature_recomputation':0,'PSD_values_reopened':1024,'RSS_peak_bytes':mem.PeakWorkingSetSize if ok else None,'scientific_cumulative_s':json.loads((ROOT/'LOCAL_NATIVE_RECEIPT.json').read_text())['numeric_cumulative_s']+time.monotonic()-START}
(ROOT/'PLOT_LABEL_QA_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8');print(json.dumps(receipt))
assert ok and mem.PeakWorkingSetSize<=4*1024**3 and receipt['scientific_cumulative_s']<=1800
