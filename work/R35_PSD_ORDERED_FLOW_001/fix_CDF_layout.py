import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[key]='4'
from pathlib import Path
import json,time,datetime,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/figures';START=time.monotonic()
original=OUT/'conditional_psd_CDF.png';archive=OUT/'conditional_psd_CDF_layout_initial.png'
if archive.exists():raise RuntimeError('layout recovery already used')
shutil.copyfile(original,archive)
m=json.loads((ROOT/'metrics.json').read_text('utf8'))
names=['REAL','BASE','INDEPENDENT_CFM','OT_CFM','PSD_ORDERED_CFM']
labels=['REAL DEV','Frozen base','Independent','OT','PSD ordered'];colors=['#111111','#888888','#7eb1d8','#e9b887','#1265a8'];styles=['-','--',':',':','-']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
with np.load(ROOT/'results/PSD_SECTIONS_PRIVATE.npz',allow_pickle=False)as z:
    fig,ax=plt.subplots(2,4,figsize=(14,6.7),sharex=True,sharey=True)
    for i,a in enumerate(ax.flat):
        counts=[]
        for name,label,color,style in zip(names,labels,colors,styles):
            x=z[name+'__bin'+str(i+1)+'__PSD'];counts.append(len(x))
            if len(x):a.plot(x,np.arange(1,len(x)+1)/len(x),color=color,ls=style,lw=1.1,label=label)
        a.set(title=f'Observed-Q bin {i+1}',xlim=(-.1,.6),ylim=(0,1));a.grid(alpha=.12)
        a.text(.02,.03,'N '+ '/'.join(map(str,counts)),transform=a.transAxes,fontsize=7)
        if not all(n>=40 for n in counts):a.text(.02,.93,'Outside common N>=40',transform=a.transAxes,fontsize=7,color='#555')
    for a in ax[1]:a.set_xlabel('signed PSD')
    for a in ax[:,0]:a.set_ylabel('CDF')
    fig.legend(*ax[0,0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False)
    fig.tight_layout(rect=(0,.065,1,1));fig.savefig(original,dpi=150,bbox_inches='tight');plt.close(fig)
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'only_layout_changed':True,'new_fit_generation':0,'old_plot_retained':archive.name,'plot':original.name,'sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'old_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'reason':'move legend outside panels to expose first-panel counts'}
(ROOT/'CDF_LAYOUT_RECOVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8');print(json.dumps(receipt),flush=True)
