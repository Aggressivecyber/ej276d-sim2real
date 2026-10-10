"""Plots and one metadata clarification from saved R47 diagnostic outputs only."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;OUT=R/'results';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
START=time.monotonic()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def put(p,x):Path(p).write_bytes((json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
raw=R/'METRICS_COMPACT.json';original=R/'METRICS_COMPACT_EXECUTED.json'
if original.exists():raise RuntimeError('metadata clarification already executed')
shutil.copyfile(raw,original);m=read(raw)
for slot,row in m['classifiers'].items():
    row.update(candidate_domain_column=row['fixed_R26_phase'],training_phase=0,holdout_phase=1,
      legacy_fixed_R26_phase_field_semantics='original value was candidate domain column, not phase; retained')
put(raw,m)
put(R/'METADATA_CORRECTION_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
 'type':'metadata only, no new HGB/statistics/inference','original':'METRICS_COMPACT_EXECUTED.json',
 'original_sha256':sha(original),'corrected':'METRICS_COMPACT.json','corrected_sha256':sha(raw),
 'fields_added':['candidate_domain_column','training_phase=0','holdout_phase=1','legacy_fixed_R26_phase_field_semantics'],
 'original_metrics_json_and_numeric_source_retained':True,'candidate_domain_column_0_SOURCE_2_REFERENCE':True})
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
colors={'RAW':'#30343b','INITIAL_PREADC':'#cc8231','INITIAL_POSTADC':'#c75a10','SHORT':'#2878b5','LONG':'#8a4ca5','BASE':'#258570'}
population_colors=['#2b78b8','#b49c5a','#81469b','#27956e','#292929']
figures=[]
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig);figures.append(name+'.png')
def ecdf(ax,v,label,color,style='-'):
    a=np.asarray(v,dtype='f8');x=np.sort(a[np.isfinite(a)])
    if len(x):ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=label,color=color,ls=style,lw=1.6)
    else:ax.plot([],[],label=label+' (undefined)',color=color,ls=style)
with np.load(OUT/'POPULATION_FEATURES_PRIVATE.npz',allow_pickle=False)as z:pop={k:z[k]for k in z.files if k.endswith('__Qtotal_mVns')}
fig,ax=plt.subplots(1,2,figsize=(15,5.7));names=list(m['population']);short_labels=['Source input','Source once ADC','LONG source','REAL TRAIN reference','REAL DEV']
for name,label,col in zip(names,short_labels,population_colors):
    ecdf(ax[0],pop[name+'__Qtotal_mVns'],label,col)
ax[0].set_xscale('log');ax[0].set_xlabel('Signed charge [mV ns]');ax[0].set_ylabel('All finite-row CDF');ax[0].legend(fontsize=9)
categories=['below']+[str(i+1)for i in range(8)]+['above'];xx=np.arange(10)
for j,(name,label,col)in enumerate(zip(names,short_labels,population_colors)):
    d=m['population'][name]['counts'];counts=[d['below_lower_support']]+d['bin_counts_all_rows']+[d['above_upper_support']]
    ax[1].bar(xx+(j-2)*.16,np.array(counts)/d['total_rows'],width=.15,color=col,label=label)
ax[1].set_xticks(xx,categories);ax[1].set_xlabel('Own observed charge bin; outside support retained');ax[1].set_ylabel('Fraction of all 2000 rows')
ax[1].legend(fontsize=8,ncol=2);fig.suptitle('Unweighted charge populations: same parents and original fixed eight bins');fig.tight_layout();save(fig,'charge_population')

fig,axes=plt.subplots(1,2,figsize=(12,5.8));keys=['CHARGE_SOURCE','CHARGE_LONG','CHARGE_RAW_REFERENCE'];labels=['SOURCE input','LONG output','RAW reference']
for k,(metric,ci_key)in enumerate([('balanced_accuracy','balanced_accuracy_95CI'),('AUC','AUC_95CI')]):
    ax=axes[k];xx=np.arange(3);point=np.array([m['classifiers'][s][metric]for s in keys]);ci=np.array([m['classifiers'][s]['heldout_CI'][ci_key]for s in keys])
    trainkey='training_balanced_accuracy'if k==0 else'training_AUC';train=[m['classifiers'][s][trainkey]for s in keys]
    ax.errorbar(xx,point,yerr=np.maximum(0,np.vstack([point-ci[:,0],ci[:,1]-point])),fmt='o',color='#266ca8',capsize=5,label='Holdout + 95% group CI')
    ax.plot(xx,train,'x',color='#ab5d28',label='TRAIN score');ax.axhline(.5,color='gray',ls='--');ax.set_ylim(.38,.79)
    if k==0:ax.axhspan(.4,.6,color='gray',alpha=.08);ax.axhline(.6,color='gray',ls=':')
    ax.set_xticks(xx,labels);ax.set_ylabel('Balanced accuracy'if k==0 else'AUC');ax.legend(loc='upper left',fontsize=8)
    for i,s in enumerate(keys):
        if k==0:ax.text(i,.397,'uncertain'if m['classifiers'][s]['population_diagnostic']=='uncertain'else'not detected',ha='center',fontsize=9)
fig.suptitle('Fixed charge-only diagnostic: 600 TRAIN + 250 holdout per domain\nCI crossing 0.60 is uncertain; not a Q–PSD evaluator lower bound');fig.tight_layout();save(fig,'charge_only_C2ST')

with np.load(OUT/'BASELINE_FEATURES_PRIVATE.npz',allow_pickle=False)as z:bf={k:z[k]for k in z.files}
fig,axes=plt.subplots(2,2,figsize=(14,10),sharey=True)
for row,label in enumerate(['SOURCE','REAL_REFERENCE']):
    for step in ['RAW','INITIAL_PREADC','INITIAL_POSTADC','SHORT','LONG','BASE']:
        if step not in m['baseline_paths'][label]:continue
        for col,key in enumerate(['baseline_RMS_V','ACF1']):
            v=bf[label+'__'+step+'__'+key];ecdf(axes[row,col],v*(1000 if col==0 else 1),step.replace('_',' '),colors[step],':'if step in ['INITIAL_PREADC','BASE']else'-')
    for col,key in enumerate(['baseline_RMS_V','ACF1']):
        ecdf(axes[row,col],bf['REAL_DEV__'+key]*(1000 if col==0 else 1),'REAL DEV','#168335','--')
        axes[row,col].set_title('SOURCE input path'if row==0 else'REAL TRAIN-reference path');axes[row,col].set_xlabel('Baseline sample RMS [mV]'if col==0 else'Baseline ACF1; full100 denominator')
        axes[row,col].set_ylabel('Finite-row CDF');axes[row,col].legend(fontsize=8)
fig.suptitle('RAW → initial projected Gaussian → saved endpoints\nAll rows retained; SOURCE RAW has zero variance / undefined ACF1');fig.tight_layout();save(fig,'baseline_paths')

fig,axes=plt.subplots(2,2,figsize=(14,11));armorder=['SHORT_2950_HEUN64','LONG_50000_HEUN64','SHORT_2950_REAL_CONTROL','LONG_50000_REAL_CONTROL']
terms=['Var_S','Var_G','Var_F','Var_R','2Cov_S_G','2Cov_S_F','2Cov_S_R','2Cov_G_F','2Cov_G_R','2Cov_F_R']
for arm,ax in zip(armorder,axes.ravel()):
    st=m['variance_decomposition'][arm];v=np.array([st['term_summary'][k]['mean']for k in terms])*1e6
    labels=[k.replace('Var_','Var ').replace('2Cov_','2Cov ').replace('_',',')for k in terms]
    ax.barh(np.arange(10),v,color=['#2878b5'if x>=0 else'#cb654c'for x in v]);ax.axvline(0,color='gray',lw=.7)
    actual=st['term_summary']['actual_preADC_baseline_variance_V2']['mean']*1e6
    ax.scatter([actual],[10],color='#1c1c1c',marker='D',label='Actual total');ax.set_yticks(np.arange(11),labels+['Total']);ax.invert_yaxis()
    ax.set_title(('SOURCE 'if 'HEUN64'in arm else'REAL ')+('SHORT'if arm.startswith('SHORT')else'LONG'))
    ax.set_xlabel('Mean baseline contribution [mV²]; signed cross terms');ax.legend(fontsize=8)
    ax.text(.99,.02,f"Algebra max error {st['variance_sum_maxabs_error_V2']:.2e} V²",transform=ax.transAxes,ha='right',fontsize=8)
fig.suptitle('Four functional variances + all six 2Cov terms = actual preADC variance\nEach baseline100 row centered; denominator99. No physical independence or causal deletion claim.');fig.tight_layout();save(fig,'baseline_variance_components')

fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
new=m['classifiers'];old=m['old_baseline_C2ST_reused']
pools=[('SOURCE initial/old outputs',[('Initial',new['BASELINE_SOURCE_INITIAL']),('BASE',old['BASE']),('SHORT',old['SHORT_2950_HEUN64']),('LONG',old['LONG_50000_HEUN64'])]),
       ('REAL reference initial/old controls',[('RAW RR',m['RAW_RR_baseline_reused']),('Initial',new['BASELINE_REAL_INITIAL']),('BASE',old['BASE_REAL_CONTROL']),('SHORT',old['SHORT_2950_REAL_CONTROL']),('LONG',old['LONG_50000_REAL_CONTROL'])])]
for ax,(title,items)in zip(axes,pools):
    point=np.array([a.get('balanced_accuracy',np.nan)for _,a in items]);ci=np.array([a.get('heldout_CI',{}).get('balanced_accuracy_95CI',[np.nan,np.nan])for _,a in items]);xx=np.arange(len(items))
    ax.errorbar(xx,point,yerr=np.maximum(0,np.vstack([point-ci[:,0],ci[:,1]-point])),fmt='o',color='#266ca8',capsize=5)
    ax.set_xticks(xx,[n for n,_ in items]);ax.axhline(.5,color='gray',ls='--');ax.set_ylim(.4,1.03);ax.set_ylabel('Baseline-local holdout BA');ax.set_title(title)
p=m['REAL_initial_minus_RAW_RR_paired'];text='Paired REAL initial minus RAW RR: unavailable'
if p['defined']:text=f"Paired REAL initial − RAW RR: ΔBA {p['BA_delta_first_minus_second']:+.3f}, 95% CI [{p['BA_delta_95CI'][0]:+.3f}, {p['BA_delta_95CI'][1]:+.3f}]"
fig.suptitle('Initial baseline classification with old predictions reused\nLocal view is amplitude-normalized; RMS mismatch is reported separately');fig.text(.5,.012,text,ha='center',fontsize=10);fig.tight_layout(rect=[0,.045,1,.93]);save(fig,'baseline_diagnostic_C2ST')

fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
for row,label in enumerate(['SOURCE','REAL_REFERENCE']):
    for step in ['RAW','INITIAL_POSTADC','SHORT','LONG','BASE']:
        if step not in m['baseline_paths'][label]:continue
        rows=m['baseline_paths'][label][step]['own_observed_Q_bins']
        for col,key in enumerate(['baseline_RMS_V','ACF1']):
            v=np.array([a['fields'][key]['distance_vs_REAL_same_observed_Q']['W1']if a['fields'][key]['distance_vs_REAL_same_observed_Q']['W1']is not None else np.nan for a in rows])
            axes[row,col].plot(np.arange(1,9),v*(1000 if col==0 else 1),'o-',color=colors[step],label=step.replace('_',' '),ms=4)
    for col in range(2):
        axes[row,col].set_title(label.replace('_',' '));axes[row,col].set_xlabel('Own observed Q-bin');axes[row,col].set_ylabel('RMS W1 versus REAL [mV]'if col==0 else'ACF1 W1 versus REAL');axes[row,col].set_xticks(np.arange(1,9));axes[row,col].legend(fontsize=8)
fig.suptitle('Fixed eight-bin noise descriptions: original failed / undefined support retained\nOwn-domain charge bins are not matched event populations or causal interventions');fig.tight_layout();save(fig,'noise_by_charge')
put(R/'PLOT_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_body_s':time.monotonic()-START,
 'figures':figures,'new_HGB_EM_training_ODE':0,'source_data':'R47 saved compact/private summaries only','metadata_only_clarification':'METADATA_CORRECTION_RECEIPT.json'})
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'figures':figures,'no_new_fits':True}),flush=True)
