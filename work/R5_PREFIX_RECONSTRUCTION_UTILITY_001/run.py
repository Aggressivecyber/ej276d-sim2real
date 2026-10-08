from __future__ import annotations
import os,pathlib,json,time,datetime,hashlib,sys,io,traceback,importlib.util
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parent.parent
OUT=ROOT/'results';FIG=OUT/'figures';OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl-cache');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'joblib-cache')
START=time.monotonic()
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import evaluation as ev
from sim_generator import pulse_rate,spectrum_noise,digitize,_normalize_waveform,_window_waveform,constant_fraction_tp0,YIELD,QUENCH,CAL_A,CAL_B
CFG=json.loads((ROOT/'config.json').read_text());ev.OUT=OUT;ev.FIG=FIG
torch.manual_seed(CFG['seed']);np.random.seed(CFG['seed'])
INPUTS=[]
def write(n,x):ev.write_json(n,x)
def sha_file(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(p,expected=None):
    b=pathlib.Path(p).read_bytes();h=hashlib.sha256(b).hexdigest()
    if expected and h!=expected:raise RuntimeError('frozen input SHA '+str(p))
    INPUTS.append({'path':str(p),'bytes':len(b),'sha256':h})
    return np.load(io.BytesIO(b),allow_pickle=False)
def order(ids,prefix=None):
    return np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256(((prefix+'|') if prefix else '') .encode()+str(ids[i]).encode()).digest()),dtype='i8')
def prefix_features(w):
    v=np.asarray(w[:,:150],dtype='f8');b=v[:,:100].mean(axis=1);x=v-b[:,None]
    peak=x[:,100:150].max(axis=1);rms=x[:,:100].std(axis=1,ddof=1)
    p=np.maximum(peak,CFG['prefix_peak_floor_V']);s=np.maximum(rms,CFG['prefix_RMS_floor_V'])
    features=np.column_stack([x[:,100:150]/p[:,None],np.log(p),np.log(s)])
    return features,{'prefix_nonpositive_peak':int(np.sum(peak<=0)),'prefix_peak_floor':int(np.sum(peak<CFG['prefix_peak_floor_V'])),'prefix_RMS_floor':int(np.sum(rms<CFG['prefix_RMS_floor_V'])),'uses_samples_after150':False}
def targets(w):
    d=ev.features(w);y=np.column_stack([d['features']['PSD'],np.log(d['features']['Qtotal_mVns'])]);return d,y
def segment(p,dtype,first,n):
    dt=np.dtype(dtype)
    with pathlib.Path(p).open('rb') as f:f.seek(first*dt.itemsize);b=f.read(n*dt.itemsize)
    if len(b)!=n*dt.itemsize:raise RuntimeError('bound training segment')
    INPUTS.append({'path':str(p),'offset':first*dt.itemsize,'bytes':len(b),'SHA256_segment':hashlib.sha256(b).hexdigest()})
    return np.frombuffer(b,dtype=dt).copy()
def synthetic(c,q,ee,had,ids):
    g=torch.Generator().manual_seed(CFG['seed']);before=g.get_state().clone();z=torch.randn((2000,8),generator=g);white=torch.randn((2000,512),generator=g)
    torch.save({'seed':CFG['seed'],'generator_before_draw':before,'generator_after_draw':g.get_state(),'n':2000,'chunk':256,'global_before_constructor':torch.get_rng_state()},ROOT/'GENERATION_RANDOM_STATE.pt')
    np.savez_compressed(ROOT/'generation_inputs.npz',G4_condition=c,Qsource_Vns=q,edep_ee_MeV=ee,edep_hadron_MeV=had,parent_ids=ids,latent_z=z.numpy(),white=white.numpy())
    outputs={};meta={};recipe=json.loads((ROOT/'SIM_RECIPE.json').read_text());spectrum=recipe['spectrum'];rng=np.random.default_rng(CFG['seed']);sim=[];clips=0;pads=0
    t=np.arange(512,dtype='f8')
    for e,h in zip(ee,had):
        rate=pulse_rate(YIELD*float(e),YIELD*float(h)*QUENCH,t,128.)
        charge=(float(e)+.15*float(h)-CAL_B)/CAL_A
        v=(rate*charge/rate.sum()+spectrum_noise(512,recipe['noise_sigma_mV'],rng,spectrum['frequencies_mhz'],spectrum['shape_psd_per_mhz'],1000))/1000
        clips+=int(np.any((v<-.5)|(v>.5)));qv=digitize(v).astype('f4');norm=_normalize_waveform(qv);tp=constant_fraction_tp0(norm,.5);left=tp-120;pads+=int(left<0 or left+480>512)
        sim.append(_window_waveform(qv,tp,480,120))
    outputs['SIM']=np.stack(sim);meta['SIM']={'clipping':clips,'padding':pads,'historical_noise_REAL_count':79793,'fixed_spectrum_receipt_SHA256':recipe['noise_spectrum_provenance_sha256'],'new_seed':CFG['seed']}
    for name,stage,module_name in [('R2','R2_SMOOTH_STOCHASTIC_RESPONSE_001','model_R2'),('R3','R3_FLEXIBLE_TEMPORAL_RESPONSE_001','model_flexible'),('R3B','R3B_ENERGY_CONDITIONAL_WIDTH_001','model_flexible')]:
        module=__import__(module_name);p=BASE/'work'/stage/'model.pt'
        frozen=json.loads((ROOT/'READINESS.json').read_text())['frozen_bindings'];authority=next(s for s in frozen if pathlib.Path(s['path'])==p)
        checkpoint=p.read_bytes()
        if len(checkpoint)!=authority['bytes'] or hashlib.sha256(checkpoint).hexdigest()!=authority['sha256']:raise RuntimeError('checkpoint drift')
        # These are SHA-bound checkpoints produced by this project; their constructors contain NumPy scalars.
        pack=torch.load(io.BytesIO(checkpoint),map_location='cpu',weights_only=False)
        model=module.SmoothResponse(**pack['constructor']);model.load_state_dict(pack['state_dict']);model.eval();waves=[];params=[]
        with torch.no_grad():
            for i in range(0,2000,256):
                v,pm=model(torch.tensor(c[i:i+256]),torch.tensor(q[i:i+256]),z[i:i+256],white[i:i+256]);waves.append(v.numpy());params.append(pm)
        outputs[name]=np.concatenate(waves);meta[name]={'model_path':str(p),'model_SHA256':sha_file(p),'response_fit':0,'clipping':sum(int(a['quantization_clip'].sum()) for a in params),'padding':sum(int(a['padding'].sum()) for a in params)}
    np.savez_compressed(ROOT/'synthetic_waveforms.npz',parent_ids=ids,**{n+'_V':v for n,v in outputs.items()})
    return outputs,meta
def error_summary(truth,prediction,idx=None):
    if idx is None:idx=np.ones(len(truth),dtype=bool)
    t=truth[idx];p=prediction[idx];e=p-t
    if not len(t):return {'N':0,'MAE':None,'RMSE':None,'median_abs':None,'q95_abs':None}
    if not np.all(np.isfinite(e)):return {'N':len(t),'nonfinite_prediction':int((~np.isfinite(e)).sum()),'MAE':None,'RMSE':None,'median_abs':None,'q95_abs':None}
    return {'N':len(t),'MAE':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),'median_abs':float(np.median(np.abs(e))),'q95_abs':float(np.quantile(np.abs(e),.95))}
def evaluate(prediction,truth,charge,region,cut):
    result={}
    for r in [None,0,1,2,3]:
        key='overall' if r is None else 'q'+str(r+1);mask=np.ones(len(truth),bool) if r is None else region==r
        result[key]={'PSD':error_summary(truth[:,0],prediction[:,0],mask),'Qtotal_mVns':error_summary(charge,np.exp(prediction[:,1]),mask),'logQ':error_summary(truth[:,1],prediction[:,1],mask)}
    margin=np.abs(truth[:,0]-cut);err=np.abs(prediction[:,0]-truth[:,0]);flip=(prediction[:,0]<=cut)!=(truth[:,0]<=cut)
    edges=np.r_[CFG['margin_edges_PSD'],np.inf];bins=np.searchsorted(edges,margin,side='right')-1
    strata=[]
    for i in range(len(edges)-1):
        m=bins==i;strata.append({'margin_low':float(edges[i]),'margin_high':float(edges[i+1]) if np.isfinite(edges[i+1]) else None,'N':int(m.sum()),'PSD_MAE':float(err[m].mean()) if m.any() else None,'side_disagreement_rate':float(flip[m].mean()) if m.any() else None})
    guaranteed=err<margin;violations=int(np.sum(guaranteed&flip));assert violations==0
    result['cut']={'value':cut,'side_disagreement_rate':float(flip.mean()),'margin_strata':strata,'strict_error_below_margin_count':int(guaranteed.sum()),'guarantee_violations':violations,'not_particle_truth':True}
    return result
def bootstrap_differences(pred,keys,truth,charge,groups):
    uniq,inv=np.unique(groups,return_inverse=True);rng=np.random.default_rng(CFG['bootstrap_seed']);draw=np.array([np.bincount(rng.integers(len(uniq),size=len(uniq)),minlength=len(uniq)) for _ in range(1000)])
    count=np.bincount(inv,minlength=len(uniq));den=draw@count;out={}
    def agg(v):return np.bincount(inv,weights=v,minlength=len(uniq))
    for budget in [256,1024]:
        for model in ['Ridge','HGB']:
            basekey=f'b{budget}__REAL__{model}';base=pred[basekey]
            for arm in ['SIM','R2','R3','R3B']:
                key=f'b{budget}__REAL+{arm}__{model}';p=pred[key];rows={}
                for name,t,a,b in [('PSD',truth[:,0],p[:,0],base[:,0]),('Qtotal_mVns',charge,np.exp(p[:,1]),np.exp(base[:,1])),('logQ',truth[:,1],p[:,1],base[:,1])]:
                    delta=np.abs(a-t)-np.abs(b-t);mae=(draw@agg(delta))/den
                    rmse=np.sqrt((draw@agg((a-t)**2))/den)-np.sqrt((draw@agg((b-t)**2))/den)
                    rows[name]={'paired_MAE_difference':float(delta.mean()),'MAE_difference_95CI':np.quantile(mae,[.025,.975]).tolist(),
                                'RMSE_difference':float(np.sqrt(np.mean((a-t)**2))-np.sqrt(np.mean((b-t)**2))),'RMSE_difference_95CI':np.quantile(rmse,[.025,.975]).tolist(),
                                'negative_favors_mixed_arm':True}
                out[key]={'versus':basekey,'artificial_groups':len(uniq),'replicates':1000,'conditional_on_saved_estimators':True,'not_acquisition_or_generator_training_uncertainty':True,'targets':rows}
    write('PAIRED_GROUP_BOOTSTRAP.json',out);return out
def plots(scores,pred,truth,charge,region,cut,bootstrap):
    plt=ev.plt;colors={'REAL':'black','REAL+SIM':'#2878b5','REAL+R2':'#bb9b48','REAL+R3':'#bda0cb','REAL+R3B':'#e65c24','REAL_FULL':'#29a060'}
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for i,model in enumerate(['Ridge','HGB']):
        for j,target in enumerate(['PSD','Qtotal_mVns']):
            ax=axes[i,j]
            for arm in CFG['arms']:
                yy=[scores[f'b{b}__{arm}__{model}']['overall'][target]['MAE'] for b in [256,1024]]
                ax.plot([256,1024],yy,'o-',label=arm,color=colors[arm])
            ax.axhline(scores[f'b2592__REAL_FULL__{model}']['overall'][target]['MAE'],color=colors['REAL_FULL'],ls='--',label='REAL-only2592')
            ax.set_xlabel('Task REAL fit events');ax.set_ylabel(target+' MAE');ax.set_title(model);ax.legend(fontsize=8)
    fig.suptitle('All fixed arms; generators already use REAL training/calibration/development and SIM old-noise prior')
    fig.tight_layout();ev.savefig(fig,'prefix_errors')
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for i,model in enumerate(['Ridge','HGB']):
        for j,budget in enumerate([256,1024]):
            ax=axes[i,j]
            for arm in CFG['arms']+['REAL_FULL']:
                key=f'b{2592 if arm=="REAL_FULL" else budget}__{arm}__{model}';err=np.sort(np.abs(pred[key][:,0]-truth[:,0]));ax.plot(err,np.arange(1,len(err)+1)/len(err),label=arm,color=colors[arm])
            ax.set_xlabel('Allowed absolute signed PSD error');ax.set_ylabel('Coverage of all EVAL QC');ax.set_title(f'{model},task REAL={budget}');ax.legend(fontsize=8);ax.set_xscale('symlog',linthresh=.001)
    fig.suptitle('Complete tolerance–coverage curves; no assumed universal success threshold')
    fig.tight_layout();ev.savefig(fig,'tolerance_coverage')
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for i,model in enumerate(['Ridge','HGB']):
        for j,budget in enumerate([256,1024]):
            ax=axes[i,j]
            for arm in CFG['arms']:
                key=f'b{budget}__{arm}__{model}';s=scores[key]['cut']['margin_strata'];x=np.arange(len(s));ax.plot(x,[r['side_disagreement_rate'] for r in s],'o-',label=arm,color=colors[arm])
            ax.set_xticks(np.arange(6),['<.005','.005–.01','.01–.02','.02–.05','.05–.1','≥.1'],rotation=20);ax.set_xlabel('|true PSD - frozen CAL cut|');ax.set_ylabel('Cut-side disagreement');ax.set_title(f'{model},REAL={budget}');ax.legend(fontsize=8)
    fig.suptitle('Pre-aligned prefix inference; cut sides are diagnostic, not particle labels')
    fig.tight_layout();ev.savefig(fig,'cut_margin_flip')
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for i,model in enumerate(['Ridge','HGB']):
        for j,target in enumerate(['PSD','Qtotal_mVns']):
            ax=axes[i,j]
            for arm in CFG['arms']+['REAL_FULL']:
                key=f'b{2592 if arm=="REAL_FULL" else 1024}__{arm}__{model}';ax.plot(np.arange(1,5),[scores[key]['q'+str(r+1)][target]['MAE'] for r in range(4)],'o-',color=colors[arm],label=arm)
            ax.set_xticks([1,2,3,4]);ax.set_xlabel('Frozen CAL true-charge region');ax.set_ylabel(target+' MAE');ax.set_title(f'{model},task REAL1024 orfull2592');ax.legend(fontsize=8)
    fig.tight_layout();ev.savefig(fig,'regional_errors')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for j,model in enumerate(['Ridge','HGB']):
        ax=axes[j];names=[f'b{b}__REAL+{a}__{model}' for b in [256,1024] for a in ['SIM','R2','R3','R3B']]
        for i,key in enumerate(names):
            r=bootstrap[key]['targets']['PSD'];v=r['paired_MAE_difference'];lo,hi=r['MAE_difference_95CI'];ax.errorbar(i,v,yerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=colors[key.split('__')[1]],capsize=3)
        ax.axhline(0,color='gray',ls='--');ax.set_xticks(range(8),[k.split('__')[0]+' '+k.split('__')[1] for k in names],rotation=40,ha='right');ax.set_ylabel('Paired PSD MAE difference vs same-budget REAL-only');ax.set_title(model)
    fig.suptitle('1000 artificial-group resamples; negative favors mixed arm; fixed estimators')
    fig.tight_layout();ev.savefig(fig,'paired_bootstrap')
def main():
    ev.log(f'R5 worker PID={os.getpid()},local CPU4;44 task fits,no response refit')
    ready=json.loads((ROOT/'READINESS.json').read_text())
    for s in ready['execution_sources']:
        if sha_file(s['path'])!=s['sha256']:raise RuntimeError('execution source drift')
    k=BASE/'work/P1K_REAL_REFERENCE_AND_REPEATABILITY_001';data=k/'A2_ENGINEERING_001/run_001'
    train=segment(data/'window_V_f32.bin','<f4',2000*480,3200*480).reshape(3200,480)
    sourceqc=segment(data/'QC_reason_u16.bin','<u2',2000,3200);groups=segment(k/'selection_group.bin','<u2',2000,3200)
    orig=segment(k/'selection_original_index.bin','<u4',2000,3200);roles=segment(k/'selection_role.bin','u1',2000,3200)
    dig=segment(k/'selection_source_id_sha256.bin','u1',2000*32,3200*32).reshape(3200,32)
    ids=np.array(['data:source:'+b.tobytes().hex() for b in dig])
    assert np.all(roles==1) and np.array_equal(orig//50,groups)
    d,y=targets(train);good=d['valid']&(sourceqc==0)
    oldroles=json.loads((BASE/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/results/DATA_ROLES.json').read_text());vg=oldroles['REAL_internal_validation_artificial_groups']
    fit=np.flatnonzero(good&~np.isin(groups,vg));val=np.flatnonzero(good&np.isin(groups,vg));fit=fit[order(ids[fit])]
    assert len(fit)==2592 and len(val)==599 and not set(groups[fit])&set(groups[val])
    x,prefix_counts=prefix_features(train)
    cache=BASE/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/data/G4_TRAIN_CACHE.npz'
    with load(cache,'117152db28a16fcfc9a83012cc863e32f2d73bdef77bbf7b5f8e977c3e64818b') as z:
        parent=z['event_ids'];selected=order(parent,'R5-synth-v1')[:2000]
        c=z['condition'][selected];q=z['Qsource_Vns'][selected];ee=z['edep_ee_MeV'][selected];had=z['edep_hadron_MeV'][selected];sids=parent[selected]
    synthetic_waves,generation=synthetic(c,q,ee,had,sids)
    sx={};sy={};synthetic_qc={}
    for name,w in synthetic_waves.items():
        d0,y0=targets(w);x0,c0=prefix_features(w);sx[name]=x0[d0['valid']];sy[name]=y0[d0['valid']]
        synthetic_qc[name]={'fixed_parent_N':2000,'QC':int(d0['valid'].sum()),'invalid':int((~d0['valid']).sum()),'prefix':c0}
    models={};fit_receipts=[];validation={};scalers={};fitted_count=0
    for budget in [256,1024,2592]:
        ii=fit[:budget];mean=x[ii].mean(axis=0);std=x[ii].std(axis=0,ddof=0);std=np.where(std>0,std,1.)
        ym=y[ii].mean(axis=0);ys=y[ii].std(axis=0,ddof=0);ys=np.where(ys>0,ys,1.)
        scalers[str(budget)]={'X_mean':mean,'X_std':std,'Y_mean':ym,'Y_std':ys,'REAL_only_n':budget}
        for arm in (CFG['arms'] if budget!=2592 else ['REAL_FULL']):
            xx=x[ii];yy=y[ii];weight=np.full(budget,1/budget)
            if arm.startswith('REAL+'):
                candidate=arm.split('+')[1];xx=np.vstack([xx,sx[candidate]]);yy=np.vstack([yy,sy[candidate]])
                weight=np.r_[np.full(budget,.5/budget),np.full(len(sx[candidate]),.5/len(sx[candidate]))]
            xx=(xx-mean)/std;yy=(yy-ym)/ys
            for kind in ['Ridge','HGB']:
                key=f'b{budget}__{arm}__{kind}';pair=[]
                for target in range(2):
                    estimator=Ridge(**CFG['Ridge']) if kind=='Ridge' else HistGradientBoostingRegressor(**CFG['HGB'])
                    begin=time.monotonic();estimator.fit(xx,yy[:,target],sample_weight=weight);fitted_count+=1;pair.append(estimator)
                    fit_receipts.append({'key':key,'target':CFG['targets'][target],'fit_s':time.monotonic()-begin,'REAL_N':budget,'synthetic_N':len(xx)-budget,'REAL_groups':len(np.unique(groups[ii])),'sample_weight_sum':float(weight.sum()),'fixed_hyperparameters':True})
                models[key]={'models':pair,'scaler':scalers[str(budget)]}
                vp=np.column_stack([a.predict((x[val]-mean)/std)*ys[j]+ym[j] for j,a in enumerate(pair)])
                validation[key]={'PSD':error_summary(y[val,0],vp[:,0]),'logQ':error_summary(y[val,1],vp[:,1]),'not_used_for_model_or_hyperparameter_selection':True}
                ev.log(f'R5 fitted {key};fits={fitted_count}/44')
    assert fitted_count==44
    joblib.dump(models,ROOT/'task_models.joblib',compress=3)
    write('TASK_TRAIN_IDENTITIES.json',{'nested_REAL_source_ids':ids[fit],'nested_original_index':orig[fit],'nested_groups':groups[fit],
          'TRAIN_validation_source_ids':ids[val],'TRAIN_validation_groups':groups[val],'G4_selected_parent_ids':sids,
          'subset_group_counts':{str(b):len(np.unique(groups[fit[:b]])) for b in [256,1024,2592]},'all_fit_roles1':True,'validation599_not_fit':True})
    write('MODELS_FROZEN_BEFORE_EVAL.json',{'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),'fit_count':44,
          'model_path':str(ROOT/'task_models.joblib'),'model_sha256':sha_file(ROOT/'task_models.joblib'),'config_sha256':sha_file(ROOT/'config.json'),
          'task_sha256':CFG['task_SHA256'],'fit_identity_sha256':hashlib.sha256((OUT/'TASK_TRAIN_IDENTITIES.json').read_bytes()).hexdigest(),
          'EVAL_values_loaded_in_this_process':0,'response_model_fit':0,'TRAIN_validation_check_only':validation})
    # EVAL opens only after all task estimators, identities and freeze receipt exist.
    ep=BASE/'work/R4_FROZEN_HELDOUT_RESPONSE_001/EVAL_EXTRACT.npz'
    with load(ep,'dc5dfa4e4f0ab3fb419d91a05edf1831e5fc918465f1aa0945c5ac8a23ea91d4') as z:
        ew=z['window_V_f32'];eqc=z['QC_reason_u16'];eorig=z['original_index']
    ed,ey=targets(ew);valid=ed['valid']&(eqc==0);ew=ew[valid];ey=ey[valid];eorig=eorig[valid];egroups=eorig//50
    assert len(ew)==7583
    ex,ecount=prefix_features(ew);charge=np.exp(ey[:,1]);ref=json.loads((data/'STATISTICS.json').read_text());edges=ref['CAL_charge_quantile_edges_mV_ns'];region=np.searchsorted(edges,charge,side='right');cut=ref['slots'][0]['cut']
    prediction={};scores={}
    for key,pack in models.items():
        s=pack['scaler'];xp=(ex-s['X_mean'])/s['X_std'];p=np.column_stack([model.predict(xp)*s['Y_std'][j]+s['Y_mean'][j] for j,model in enumerate(pack['models'])]);prediction[key]=p;scores[key]=evaluate(p,ey,charge,region,cut)
    np.savez_compressed(ROOT/'predictions.npz',truth_PSD_logQ=ey,original_index=eorig,group=egroups,charge_region=region,**prediction)
    np.savez_compressed(ROOT/'REPLAY_INPUTS.npz',prefix_features=ex[:64],**{key:p[:64] for key,p in prediction.items()})
    boot=bootstrap_differences(prediction,list(models),ey,charge,egroups)
    metric={'status':'complete','fit_count':44,'response_new_fit':0,'SSH':0,'EVAL_QC':7583,'EVAL_artificial_groups':len(np.unique(egroups)),
            'results':scores,'paired_group_bootstrap':boot,'TRAIN_validation599':validation,'fit_receipts':fit_receipts,'generation':generation,
            'synthetic_QC':synthetic_qc,'prefix_floor_counts':{'REAL_TRAIN3200':prefix_counts,'EVAL':ecount},'CAL_global_cut':cut,'CAL_charge_edges_mVns':edges,
            'CAL_q4_reference_stays_failed':True,'data_cost':{'learned_generators_REAL_TRAIN_QC':3191,'reference_CAL_QC':1594,'DEV_used_for_method_development':2000,
            'original_SIM_historical_noise_REAL_used':79793,'task_train_budgets':[256,1024],'full_REAL_task_fit':2592,'EVAL_real_QC':7583,
            'these_are_not_new_independent_acquisitions':'all one acquisition;cost unions not recalculated;no claimonly256/1024 acquired'},
            'worker_PID':os.getpid(),'seed':CFG['seed'],'elapsed_before_plots_s':time.monotonic()-START,'no_EVAL_selection':True,'online_trigger_or_FPGA_validation':False}
    write('metrics.json',metric);write('INPUTS.json',INPUTS);plots(scores,prediction,ey,charge,region,cut,boot)
    metric['numeric_elapsed_s']=time.monotonic()-START;write('metrics.json',metric);(ROOT/'metrics.json').write_text((OUT/'metrics.json').read_text(),encoding='utf-8')
    ev.log('R5 complete '+str(metric['numeric_elapsed_s'])+'s')
if __name__=='__main__':
    try:main()
    except BaseException as e:write('FAILURE.json',{'worker_PID':os.getpid(),'elapsed_s':time.monotonic()-START,'error':repr(e),'traceback':traceback.format_exc()});raise
