from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,shutil,hashlib,ctypes
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];S=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'
assert S.drive.upper()in ['E:','D:']
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
for n in ['tmp','matplotlib']:(S/'cache'/n).mkdir(parents=True,exist_ok=True)
os.environ.update(TMP=str(S/'cache/tmp'),TEMP=str(S/'cache/tmp'),TMPDIR=str(S/'cache/tmp'),MPLCONFIGDIR=str(S/'cache/matplotlib'))
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
h=kernel.GetCurrentProcess();allowed=ctypes.c_size_t();system=ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(h,ctypes.byref(allowed),ctypes.byref(system))
cpus=[i for i in range(ctypes.sizeof(ctypes.c_size_t)*8)if allowed.value&(1<<i)][:4]
assert kernel.SetProcessAffinityMask(h,sum(1<<i for i in cpus))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
M=json.loads((S/'METRICS_COMPACT.json').read_text('utf8'))
arms=['DIRECT_DISTRIBUTION_RESPONSE','DIRECT_DISTRIBUTION_REAL_CONTROL']
data=[M['explicit_constraint_projection'][n]for n in arms]
assert all(d['N']==2000 and d['range_active_events']==0 and d['rail_sample_total']==0 and d['fields']['constraint_projection_RMS_V']['quantiles'][-1]==0 for d in data)
out=S/'figures/explicit_constraint_projection.png';D=R/'reviews/R59_FIGURE_REPAIR_001'
D.mkdir(exist_ok=False);original=D/'explicit_constraint_projection.original.png';shutil.copy2(out,original)
plt.rcParams.update({'font.size':12,'axes.titlesize':15})
fig,axes=plt.subplots(1,2,figsize=(14,5),gridspec_kw={'width_ratios':[1,1.4]})
colors=['#0d6dcc','#e97905'];labels=['SOURCE','Same-treated REAL']
axes[0].bar(labels,[2000,2000],color=colors,width=.55)
for i in range(2):axes[0].text(i,2040,'2,000 / 2,000',ha='center',va='bottom',fontsize=13)
axes[0].set(ylim=(0,2350),ylabel='Events with zero projection correction',title='Saved main outputs: exact zero corrections')
axes[0].spines[['top','right']].set_visible(False)
axes[1].axis('off')
table=axes[1].table(cellText=[['Range-active events','0 / 2000','0 / 2000'],['Maximum RMS correction (mV)','0','0'],['Samples on voltage rails','0','0']],
 colLabels=['Recorded quantity','SOURCE','Same-treated REAL'],colWidths=[.56,.20,.24],cellLoc='center',bbox=[0,.51,1,.41])
table.auto_set_font_size(False);table.set_fontsize(11)
axes[1].text(0,.39,'The box-charge projection remains part of the model.\nThese zero counts describe the 4,000 saved main outputs.',va='top',fontsize=12)
trace=M['training_audit']['constraint_loss_trace_summary']
frac=trace['SOURCE_global_range_projection_fraction']['mean']
axes[1].text(0,.20,f'TRAIN update-mean SOURCE range activation: {100*frac:.2f}%.\nThis is a weighted minibatch training trace,\nnot an exhaustive frozen-TRAIN evaluation.',va='top',fontsize=12)
fig.suptitle('Explicit constraint projection: zero main-output effect shown as counts',fontsize=16)
fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(out,dpi=140);plt.close(fig)
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'CPU_affinity':cpus,
 'reason':'The original all-zero correction histogram rendered as an empty plot with negative automatic axes; replace with explicit saved counts.',
 'source':'METRICS_COMPACT.json, existing projection and training-audit summaries only','scientific_recomputation':False,
 'original_figure':str(original.relative_to(R)),'original_SHA256':hashlib.sha256(original.read_bytes()).hexdigest(),
 'revised_figure':str(out.relative_to(R)),'revised_SHA256':hashlib.sha256(out.read_bytes()).hexdigest(),
 'frozen_science_code_weights_metrics_modified':False,'visual_review_pending':True}
(D/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt,indent=2))
