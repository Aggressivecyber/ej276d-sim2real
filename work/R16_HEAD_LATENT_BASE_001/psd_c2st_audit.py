from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime
ROOT=pathlib.Path(__file__).resolve().parent;os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');START=time.monotonic()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import evaluation as ev
m=json.loads((ROOT/'metrics.json').read_text());f=np.load(ROOT/'results/FEATURES_PRIVATE.npz',allow_pickle=False);p=np.load(ROOT/'results/C2ST_PRIVATE.npz',allow_pickle=False);private={};summary={};colors={'REAL':'black','STANDARD':'#6c4c9b'}
view='Q_PSD_2D';pr=p['STANDARD__'+view+'__probability'];labels=p['STANDARD__'+view+'__label'];assert len(pr)==500 and np.array_equal(labels,np.r_[np.ones(250),np.zeros(250)])
selected={};allq=[]
for phase in [0,1]:
    for domain,j in [('STANDARD',0),('REAL',1)]:
        ix=p['phase'+str(phase)+'_'+str(j)];assert len(ix)==(600 if phase==0 else 250) and len(np.unique(ix))==len(ix);q=f[domain+'__Qtotal_mVns'][ix];psd=f[domain+'__PSD'][ix];valid=f[domain+'__valid'][ix];assert valid.all();matrix=np.column_stack([q,psd]);assert np.isfinite(matrix).all();selected[(phase,domain)]=(q,psd);allq.append(q);key=('train' if phase==0 else 'holdout')+'__'+domain;summary[key]={'N':len(ix),'all_original_positions_QC_valid':True,'original_position_min':int(ix.min()),'original_position_max':int(ix.max()),'Qtotal_mVns':ev.summary(q),'signed_PSD':ev.summary(psd),'whole_domain_Qtotal_mVns':ev.summary(f[domain+'__Qtotal_mVns'][f[domain+'__valid']]),'whole_domain_PSD':ev.summary(f[domain+'__PSD'][f[domain+'__valid']]),'regions':{str(r+1):int(np.sum(f[domain+'__region'][ix]==r))for r in range(4)}};private[key+'__original_positions']=ix;private[key+'__raw_columns_Q_mVns_PSD']=matrix
np.savez_compressed(ROOT/'results/PSD_C2ST_AUDIT_PRIVATE.npz',**private)
predsummary={d:{'N':int((labels==y).sum()),'probability_candidate':ev.summary(pr[labels==y]),'fraction_predict_candidate':float(np.mean(pr[labels==y]>=.5))}for d,y in [('STANDARD',1),('REAL',0)]}
metrics=m['C2ST']['results']['STANDARD'][view];assert metrics['features']==2 and metrics['train_each_domain']==600 and metrics['holdout_each_domain']==250
fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained');qlim=(min(x.min()for x in allq)*.92,max(x.max()for x in allq)*1.08)
for phase in [0,1]:
    for j,domain in enumerate(['REAL','STANDARD']):
        ax=axes[phase,j];q,psd=selected[(phase,domain)];ax.scatter(q,psd,s=9,alpha=.5,color=colors[domain]);ax.set_xscale('log');ax.set_xlim(qlim);ax.set_ylim(0,.6);ax.xaxis.set_major_locator(ticker.LogLocator(base=10,numticks=4));ax.xaxis.set_minor_formatter(ticker.NullFormatter());outside=int(np.count_nonzero((psd<0)|(psd>.6)));ax.set_title(domain+' '+('TRAIN600' if phase==0 else 'HOLDOUT250')+';outside='+str(outside));ax.set_xlabel('Observed Qtotal [mV ns]');ax.set_ylabel('Signed PSD')
    ax=axes[phase,2]
    for domain in ['REAL','STANDARD']:
        _,psd=selected[(phase,domain)];ax.hist(psd,bins=np.linspace(0,.6,41),density=True,histtype='step',color=colors[domain],label=domain)
    ax.set_title(('TRAIN' if phase==0 else 'HOLDOUT')+' marginal PSD;fixed0–0.6 display');ax.set_xlabel('Signed PSD');ax.set_ylabel('Histogram density');ax.legend()
fig.suptitle('Exact original C2ST selections, common axes; classifier input columns are raw(Qtotal_mVns,PSD)');fig.savefig(ROOT/'results/figures/psd_c2st_selection_audit.png',dpi=140,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
for domain,y in [('REAL',0),('STANDARD',1)]:ax.hist(pr[labels==y],bins=np.linspace(0,1,31),density=True,histtype='step',label=domain,color=colors[domain])
ax.axvline(.5,color='gray',ls='--');ax.set_xlim(0,1);ax.set_xlabel('Heldout predicted candidate probability');ax.set_ylabel('Density');ax.set_title('Saved Q/PSD classifier predictions: BA='+str(metrics['balanced_accuracy'])+' AUC='+str(metrics['AUC']));ax.legend();fig.savefig(ROOT/'results/figures/psd_c2st_prediction_audit.png',dpi=140,bbox_inches='tight');plt.close(fig)
result={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_model_fits':0,'new_density_fits':0,'elapsed_s':time.monotonic()-START,'RSS_at_stop_bytes':None,'RSS_peak_bytes':None,'matrix_source':'saved FEATURES_PRIVATE scalar Qtotal_mVns and signedPSD; current unchanged diagnostic_core.matrix matches column order','actual_original_transformed_train_matrix_saved':False,'runtime_evidence':'original two-feature fit receipt and saved heldout probabilities; raw matrices reconstructed from saved inputs, not a new classifier run','selections':summary,'heldout_predictions':predsummary,'saved_HGB_result':metrics,'interpretation':'fixed HGB did not discriminate this same-N heldout sample; finite test power is not a PSD equivalence result. Visual conditional two-component structure and its qualified support are separate evidence.','migration_test':False,'threads':4}
(ROOT/'PSD_C2ST_SELECTION_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'PID':os.getpid(),'elapsed_s':result['elapsed_s'],'new_fits':0,'predictions':predsummary,'BA':metrics['balanced_accuracy'],'AUC':metrics['AUC']}))
