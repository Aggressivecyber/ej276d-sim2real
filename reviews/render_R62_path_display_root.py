"""Display-only nonnegative y limits; original figure and scientific files retained."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,hashlib,ctypes
R=Path(__file__).resolve().parents[1];S=R/'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001'
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='4'
os.environ.update(MPLCONFIGDIR=str(S/'cache/matplotlib'),TEMP=str(S/'cache/tmp'),TMP=str(S/'cache/tmp'),PYTHONDONTWRITEBYTECODE='1')
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t];assert kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),15)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(p.read_text('utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected={p.name:sha(p)for p in S.iterdir()if p.is_file()and p.suffix in ['.json','.npz','.py','.md']}
with np.load(S/'PATH_STATISTICS_PRIVATE.npz',allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
fig,axes=plt.subplots(2,2,figsize=(14,9))
colors=['#607d8b','#f5a623','#cf4c54','#825cc7','#126ab3'];stages=['input','noise_initial','neural_raw','projected_f32','canonical_ADC'];labels=['input','noise initial','neural raw','projected float32','canonical ADC']
for ax,name in zip(axes.flat,['V480','U480','DU479','D2U478']):
    for stage,col,label in zip(stages,colors,labels):ax.plot(data['SOURCE_'+stage+'_all_'+name+'_W1'],color=col,label=label,lw=1.1)
    ax.plot(data['REAL_control_canonical_ADC_all_'+name+'_W1'],color='#228b65',ls='--',label='same-treated REAL',lw=1)
    ax.set(title=name+' coordinate W1',xlabel='sample / difference index',ylabel='V'if name=='V480'else'normalized units')
    for k in [100,150,400]:ax.axvline(k,color='#cccccc',lw=.7)
    ax.set_yscale('symlog',linthresh=1e-6);ax.set_ylim(bottom=0);ax.grid(alpha=.15)
axes[0,0].legend(fontsize=8,ncol=2)
fig.suptitle('Complete waveform residuals along the frozen response path',fontsize=14)
fig.text(.5,.012,'Frozen R61 on original TRAIN parents | descriptive diagnostics; no A-F acceptance claim',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.035,1,.955])
file=S/'figures/complete_path_coordinate_W1.display.png';fig.savefig(file,facecolor='white');plt.close(fig)
assert all(sha(S/name)==digest for name,digest in protected.items())
v={'UTC':datetime.now(timezone.utc).isoformat(),'scope':'Set nonnegative lower y limits for the nonnegative W1 statistic; plot data unchanged',
 'original_path':'figures/complete_path_coordinate_W1.png','original_SHA256':sha(S/'figures/complete_path_coordinate_W1.png'),
 'display_path':'figures/'+file.name,'display_SHA256':sha(file),'script_SHA256':sha(Path(__file__)),
 'all_original_scientific_and_render_files_unchanged':True,'new_data_access_model_forwards_or_fits':0,'CPU_affinity':[0,1,2,3],'visual_QA_pending':True}
(S/'DISPLAY_ONLY_RECEIPT.json').write_bytes((json.dumps(v,indent=2)+'\n').encode());print(json.dumps(v,indent=2))
