from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,ctypes
ROOT=pathlib.Path(__file__).resolve().parent;start=time.monotonic();os.environ['MPLCONFIGDIR']=str(ROOT/'cache/local_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
folder=ROOT/'results/figures';dest=folder/'conditional_PSD_CDF.png';original=folder/'original_conditional_PSD_CDF.png'
if not original.exists():original.write_bytes(dest.read_bytes())
m=json.loads((ROOT/'metrics.json').read_text());colors={'REAL':'black','STEP0':'#777777','NLL_CONT':'#0072b2','NLL_PSD':'#d55e00'};names=list(colors);plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
with np.load(ROOT/'results/PSD_SECTIONS_PRIVATE.npz',allow_pickle=False)as z:
    fig,axes=plt.subplots(2,4,figsize=(20,10),sharex=True,sharey=True,layout='constrained')
    for i,ax in enumerate(axes.ravel()):
        for n in names:
            a=z[n+'__bin'+str(i+1)+'__PSD_sorted'];ax.plot(a,np.arange(1,len(a)+1)/len(a)if len(a)else [],label=n,color=colors[n])
        v=m['PSD_main']['tables']['REAL']['bins'][i]['valley_definition'];ax.axvspan(v['low'],v['high'],color='gray',alpha=.12);counts=[m['PSD_main']['tables'][n]['bins'][i]['N']for n in names];ax.set_title('Frozen bin '+str(i+1)+'\nN: REAL '+str(counts[0])+', STEP0 '+str(counts[1])+'\nCONT '+str(counts[2])+', PSD '+str(counts[3]),fontsize=9);ax.set_xlabel('Full finite signed PSD');ax.set_ylabel('Empirical CDF');ax.legend(fontsize=7)
    fig.suptitle('All finite post-ADC PSD retained; gray valley fixed by old TRAIN fit');fig.savefig(dest,dpi=140,bbox_inches='tight');plt.close(fig)
rss=None;error=None
try:
    class PMC(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t)for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    kernel=ctypes.WinDLL('kernel32');psapi=ctypes.WinDLL('psapi');kernel.GetCurrentProcess.restype=ctypes.c_void_p;psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(PMC),ctypes.c_ulong];p=PMC();p.cb=ctypes.sizeof(p)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(p),p.cb):raise ctypes.WinError()
    rss=int(p.PeakWorkingSetSize)
except BaseException as ex:error=repr(ex)
row={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'status':'complete','only_saved_CDF_plot_redrawn':True,'scientific_metric_recompute':0,'fits':0,'response_generation':0,'RSS_peak_bytes':rss,'RSS_probe_error':error,'reason':'initial adjacent long titles overlap; compact multiline counts','original_preserved':str(original),'output':str(dest)};(ROOT/'CDF_LAYOUT_RECEIPT.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
