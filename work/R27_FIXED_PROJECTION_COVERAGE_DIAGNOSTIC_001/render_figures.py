from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
import json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
M=json.loads((ROOT/'metrics.json').read_bytes());CFG=json.loads((ROOT/'config.json').read_bytes());START=time.monotonic()
NAMES=CFG['candidates'];COMPARE=[*NAMES,'TRAIN_REFERENCE'];LABELS=['D1 selected','HOMO selected','HOMO final','D1 final','TRAIN ref']
SHORT={**dict(zip(COMPARE,LABELS)),'REAL':'REAL DEV'}
COLORS=dict(zip(COMPARE,['#3b6fa8','#e07d24','#2e9263','#a04c98','#333333']));COLORS['REAL']='#111111'
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'figure.dpi':100})
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    print('saved '+name,flush=True)
def read(path):
    with np.load(path,allow_pickle=False) as z:return{k:z[k] for k in z.files}

space=M['A_space'];sv=np.asarray(space['singular_values']);p=read(OUT/'PROJECTION_SPACE_PRIVATE.npz')
fig,ax=plt.subplots(1,3,figsize=(13.8,3.6))
ax[0].semilogy(np.arange(1,len(sv)+1),sv,color='#3b6fa8');ax[0].axhline(1e-10*sv[0],ls=':',color='black')
ax[0].set(xlabel='Unit-normalized C singular-value index',ylabel='Singular value',title='390 columns / rank390 / complement90')
ax[1].plot(np.arange(480)-120,np.diag(p['Pseen']),label='Pseen diagonal');ax[1].plot(np.arange(480)-120,np.diag(p['Pcomplement']),label='Pcomplement diagonal')
ax[1].set(xlabel='Saved time -120 [ns]',ylabel='Projector diagonal',title='Explicit first100 coordinates covered');ax[1].legend()
im=ax[2].imshow(p['Pcomplement'],aspect='auto',origin='lower',cmap='coolwarm',vmin=-.2,vmax=.2)
ax[2].set(xlabel='Time coordinate',ylabel='Time coordinate',title='Complement map; not whole-loss nullspace');fig.colorbar(im,ax=ax[2],shrink=.85)
fig.tight_layout();save(fig,'projection_space')

A=M['A_coverage'];fig,ax=plt.subplots(1,3,figsize=(14.2,4.1),gridspec_kw={'width_ratios':[1,1,1.7]})
x=np.arange(5);seen=[A[n]['observed_Q'][0]['seen_fraction'] for n in COMPARE];null=[A[n]['observed_Q'][0]['complement_fraction'] for n in COMPARE]
ax[0].bar(x,seen,label='seen',color='#3b6fa8');ax[0].bar(x,null,bottom=seen,label='complement',color='#e07d24')
ax[0].set_xticks(x,LABELS,rotation=45,ha='right');ax[0].set(ylabel='Fraction of mean-residual squared norm',ylim=(0,1),title='Whole unweighted populations');ax[0].legend()
ax[1].bar(x,[A[n]['observed_Q'][0]['mean_difference_sqnorm'] for n in COMPARE],color=[COLORS[n] for n in COMPARE]);ax[1].set_yscale('log')
ax[1].set_xticks(x,LABELS,rotation=45,ha='right');ax[1].set(ylabel='Mean u difference squared norm',title='Magnitude; population priors differ')
values=np.full((5,8),np.nan)
for j,n in enumerate(COMPARE):
    for i,r in enumerate(A[n]['observed_Q'][1:]):
        if r['qualified']:values[j,i]=r['complement_fraction']
im=ax[2].imshow(np.ma.masked_invalid(values),vmin=0,vmax=.6,cmap='YlOrRd',aspect='auto')
ax[2].set_xticks(np.arange(8),np.arange(1,9));ax[2].set_yticks(np.arange(5),LABELS);ax[2].set(xlabel='Observed Q-bin',title='Complement fraction; each domain N>=40')
for j,n in enumerate(COMPARE):
    for i,r in enumerate(A[n]['observed_Q'][1:]):
        txt=f'{values[j,i]:.2f}' if r['qualified'] else f'n/a\nN={r["QC_n"]}'
        ax[2].text(i,j,txt,ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=ax[2],shrink=.85);fig.tight_layout();save(fig,'mean_residual_coverage')

temporal=read(OUT/'TEMPORAL_DIAGNOSTICS_PRIVATE.npz');t=np.arange(480)-120
fig,axes=plt.subplots(3,3,figsize=(13.8,9),sharex=True)
for row,view in enumerate(['delta_u','delta_seen','delta_complement']):
    for col,b in enumerate([2,3,4]):
        ax=axes[row,col]
        for n in COMPARE:
            ax.plot(t,temporal[n+'__observed_Q__bin'+str(b)+'__'+view],color=COLORS[n],label=SHORT[n],lw=1,ls='--' if n=='TRAIN_REFERENCE' else '-')
        ax.axhline(0,color='gray',lw=.6);ax.set_xlim(0,300)
        ax.set_title('Observed Q-bin'+str(b))
        if col==0:ax.set_ylabel({'delta_u':'Mean residual u','delta_seen':'Seen mean residual','delta_complement':'Complement mean residual'}[view])
        if row==2:ax.set_xlabel('Saved time -120 [ns]')
axes[0,0].legend(fontsize=7,ncol=2)
fig.suptitle('Candidate minus REAL conditional means; distinct populations in the same observed-Q interval')
fig.tight_layout(rect=[0,0,1,.97]);save(fig,'conditional_residuals')

fig,axes=plt.subplots(3,2,figsize=(12,8.5),sharex=True)
for row,b in enumerate([2,3,4]):
    for col,stat in enumerate(['mean_u','median_u']):
        ax=axes[row,col]
        for n in ['REAL',*NAMES]:
            ax.plot(t,temporal[n+'__observed_Q__bin'+str(b)+'__'+stat],color=COLORS[n],lw=1.05,label=SHORT[n])
        ax.set_xlim(0,300);ax.set_title('Q-bin'+str(b)+' / '+('mean' if col==0 else 'median'))
        ax.set_ylabel('Own-total-peak normalized u')
        if row==2:ax.set_xlabel('Saved time -120 [ns]')
axes[0,0].legend(fontsize=7)
fig.suptitle('Saved post-ADC populations; conditional means and medians do not imply distribution equivalence')
fig.tight_layout(rect=[0,0,1,.97]);save(fig,'conditional_mean_median')

B=M['B_projections'];groups=CFG['projection_draw_order']
fig,axes=plt.subplots(1,3,figsize=(14.3,4.2))
for ax,g in zip(axes,groups):
    s=B['common_support_summary'][g]
    for i,n in enumerate(COMPARE):
        r=s['domains'][n];ax.plot(i-.08,r['equal_mean_on_common_bins']['ORIGINAL'],'o',color='#333333',label='Original128' if i==0 else None)
        lo,hi=r['new_block_range'];ax.vlines(i+.08,lo,hi,color='#2584a4',lw=2)
        ax.plot(i+.08,r['new_block_mean'],'D',color='#2584a4',label='4 new blocks: mean + range' if i==0 else None)
    ax.set_xticks(np.arange(5),LABELS,rotation=45,ha='right');ax.set_ylabel('TRAIN-scaled projection quantile W1')
    ax.set_title(g+' / common bins'+','.join(map(str,s['common_qualified_bin_ids_1based'])))
axes[0].legend(fontsize=7)
fig.suptitle('Equal weight across identical supported Q-bins; new-block range is not a confidence interval')
fig.tight_layout(rect=[0,0,1,.95]);save(fig,'projection_blocks_common')

fig,axes=plt.subplots(3,4,figsize=(14.2,9.0),sharex=True)
newlabels=[k for k in B['blocks'] if k!='ORIGINAL'];x=np.arange(1,9)
for row,g in enumerate(groups):
    for col,n in enumerate(NAMES):
        ax=axes[row,col];old=B['blocks']['ORIGINAL'][g][n]['bins'];vals=np.array([[r['W1_quantile_direction_mean'] for r in B['blocks'][b][g][n]['bins']] for b in newlabels],float)
        original=np.array([r['W1_quantile_direction_mean'] for r in old],float)
        ax.fill_between(x,vals.min(0),vals.max(0),color='#2584a4',alpha=.18,label='4 new-block range')
        ax.plot(x,vals.mean(0),'-D',color='#2584a4',ms=3,label='New-block mean');ax.plot(x,original,'-o',color='#333333',ms=3,label='Original128')
        rr=np.array([r['W1_quantile_direction_mean'] for r in B['blocks']['ORIGINAL'][g]['TRAIN_REFERENCE']['bins']]);ax.plot(x,rr,':',color='#888888',label='RR original')
        bad=[r['bin_id_1based'] for r in old if not r['qualified']]
        if bad:
            for k in bad:ax.axvspan(k-.18,k+.18,color='#dddddd',alpha=.6)
            ax.text(.98,.98,'Gray: N<40',transform=ax.transAxes,ha='right',va='top',fontsize=7)
        ax.set_xticks(x);ax.set_title(SHORT[n]+' / '+g);ax.set_ylim(bottom=0)
        if col==0:ax.set_ylabel('TRAIN-scaled quantile W1')
        if row==2:ax.set_xlabel('Observed Q-bin')
axes[0,0].legend(fontsize=7)
fig.suptitle('All eight observed-Q bins retained; unsupported intervals are descriptive only')
fig.tight_layout(rect=[0,0,1,.97]);save(fig,'projection_blocks_by_charge')

C=M['C_classifiers']['results'];views=CFG['C2ST_views'];viewlabels=['u480','seen480','complement480'];viewcolors=['#333333','#3b6fa8','#e07d24']
fig,axes=plt.subplots(1,3,figsize=(14.1,4.3))
for j,(view,label,color) in enumerate(zip(views,viewlabels,viewcolors)):
    xx=np.arange(5)+(j-1)*.22
    for k,n in enumerate(COMPARE):
        r=C[n][view];axes[0].plot(xx[k],r['training_balanced_accuracy'],'o',color=color,label=label if k==0 else None)
        lo,hi=r['heldout_CI']['balanced_accuracy_95CI'];axes[1].vlines(xx[k],lo,hi,color=color);axes[1].plot(xx[k],r['balanced_accuracy'],'o',color=color)
        lo,hi=r['heldout_CI']['AUC_95CI'];axes[2].vlines(xx[k],lo,hi,color=color);axes[2].plot(xx[k],r['AUC'],'o',color=color)
for ax,title in zip(axes,['Training BA','Holdout BA + group95%CI','Holdout AUC + group95%CI']):
    ax.axhline(.5,color='gray',ls=':');ax.set_xticks(np.arange(5),LABELS,rotation=45,ha='right');ax.set_ylim(.42,1.03);ax.set_title(title)
axes[0].legend()
fig.suptitle('15 new fixed HGB fits / 600+250 per domain / pure480 time views, no13 appended scalars')
fig.tight_layout(rect=[0,0,1,.95]);save(fig,'temporal_c2st')

branch=read(OUT/'BRANCH_DECOMPOSITION_PRIVATE.npz');fig,axes=plt.subplots(3,2,figsize=(12,8.4),sharex=True)
for row,b in enumerate([2,3,4]):
    for col,n in enumerate(['HOMO_SELECTED','HOMO_FINAL']):
        ax=axes[row,col];prefix=n+'__observed_Q__bin'+str(b)+'__'
        ax.plot(t,branch[prefix+'signal'],color='#3b6fa8',label='B(signal)/total p')
        ax.plot(t,branch[prefix+'electronic'],color='#e07d24',label='B(electronic)/total p')
        ax.plot(t,branch[prefix+'total'],color='#2e9263',label='preADC total',lw=1.2)
        ax.plot(t,temporal['REAL__observed_Q__bin'+str(b)+'__mean_u'],color='black',ls='--',label='REAL postADC mean')
        ax.set_xlim(0,300);ax.set_title(SHORT[n]+' / Q-bin'+str(b));ax.set_ylabel('Shared total-peak normalized mean')
        if row==2:ax.set_xlabel('Saved time -120 [ns]')
axes[0,0].legend(fontsize=7)
fig.suptitle('Functional additive branches share each total output p; neither is a conditional mean or independent physical noise')
fig.tight_layout(rect=[0,0,1,.97]);save(fig,'branch_conditional_means')

D=M['D_branches']['joint8'];fig,axes=plt.subplots(3,2,figsize=(12.5,9))
cats=['pure_EE','pure_had','mixed'];fields=['signal_variance','electronic_variance','twice_covariance','total_variance'];colors=['#3b6fa8','#e07d24','#a04c98','#2e9263']
for row,cat in enumerate(cats):
    for col,(view,gate,factor,units) in enumerate([('raw_baseline_subtracted_V','postpeak',1e6,'mV²'),('normalized_complement','all',1e4,'u² ×1e4')]):
        ax=axes[row,col]
        for j,(field,color) in enumerate(zip(fields,colors)):
            vals=[D[n][view][cat]['gates'][gate][field]['mean']*factor for n in NAMES]
            ax.bar(np.arange(4)+(j-1.5)*.18,vals,width=.18,label=field.replace('_',' '),color=color)
        ax.axhline(0,color='gray',lw=.6);ax.set_xticks(np.arange(4),LABELS[:4],rotation=25,ha='right')
        ax.set_title(cat+' / '+('postpeak150:220 rawV' if col==0 else 'complement all480'))
        ax.set_ylabel('Mean parent sample variance ['+units+']')
axes[0,0].legend(fontsize=7,ncol=2)
fig.suptitle('Saved128 parents ×8 repeats, ddof1; negative cross term retained, repeats are not independent REAL events')
fig.tight_layout(rect=[0,0,1,.97]);save(fig,'joint_variance_covariance')

files={q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in FIG.glob('*.png')}
receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'figures':files,
         'figures_count':len(files),'fit_generation_updates':0,'source':'existing saved R27 metrics/derived arrays only'}
(ROOT/'PLOT_RECEIPT.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
print(json.dumps(receipt))
