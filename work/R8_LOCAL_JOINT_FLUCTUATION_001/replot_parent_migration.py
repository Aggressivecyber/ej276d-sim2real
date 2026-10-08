from __future__ import annotations
import os,pathlib,json,time,datetime,hashlib,shutil
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic()
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m=json.loads((ROOT/'metrics.json').read_text(encoding='utf-8'));p=ROOT/'results/figures/parent_migration.png'
old=ROOT/'results/figures_layout_before_fix';old.mkdir(exist_ok=False);shutil.copyfile(p,old/p.name);oldsha=hashlib.sha256(p.read_bytes()).hexdigest()
fig=plt.figure(figsize=(17,5.3),layout='constrained');gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,.055])
axes=[fig.add_subplot(gs[0,i]) for i in range(3)];cax=fig.add_subplot(gs[0,3])
for n,ax in zip(['R3B','JOINT','JOINT_SIGNAL'],axes):
    counts=np.array(m['source_nominal_condition'][n]['migration_rows_nominal_cols_output_q1_q2_q3_q4_invalid']);frac=counts/np.maximum(counts.sum(axis=1,keepdims=True),1)
    im=ax.imshow(frac,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(4):
        for j in range(5):ax.text(j,i,str(counts[i,j]),ha='center',va='center',color='white' if frac[i,j]>.6 else 'black',fontsize=11)
    ax.set_xticks(range(5),['q1','q2','q3','q4','invalid']);ax.set_yticks(range(4),['q1','q2','q3','q4']);ax.set_xlabel('Observed output charge region');ax.set_ylabel('G4 nominal source charge region');ax.set_title(n)
fig.colorbar(im,cax=cax,label='Within-source-group fraction');fig.suptitle('All 2000 parents retained; source charge is a calibrated proxy, not incident true energy')
fig.savefig(p,dpi=150,bbox_inches='tight');plt.close(fig)
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'reason':'old colorbar overlapped JOINT_SIGNAL q4 counts; dedicated fourth gridspec column fixes it','input':'saved metrics.json counts only','old_plot_sha256':oldsha,'new_plot_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'old_plot_preserved':str((old/p.name).relative_to(ROOT)),'array_body_decode':0,'fit':0,'metric_recomputation':0,'scientific_code_change':0}
(ROOT/'LAYOUT_REPAIR_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True)
