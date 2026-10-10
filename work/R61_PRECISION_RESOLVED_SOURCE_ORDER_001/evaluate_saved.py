from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[key]='12'
import pathlib,json,time,datetime,traceback,hashlib,platform
import numpy as np,torch,sklearn
from scipy.stats import wasserstein_distance,ks_2samp
from sklearn.metrics import roc_auc_score
import evaluation as ev,diagnostic_core as dc,assessment as aa,energy_width as ew,data_support as ds
from model import adc
import flow_diagnostics as fd
import endpoint_common as common
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);START=time.monotonic()
CFG=json.loads((ROOT/'config_post.json').read_text('utf8'));ARMS=CFG['endpoints'];SOURCE_ARMS=CFG['arms'];REAL_ARMS=CFG['REAL_arms']
CONTROLS=['BASE','INDEPENDENT_CFM'];NAMES=[*CONTROLS,'BASE_REAL_CONTROL',*ARMS];GROUPS=['shape480','diff479','baseline100_V']
VALLEY=json.loads((pathlib.Path(CFG['frozen_eval_input_root'])/'VALLEY_DEFINITION.json').read_text('utf8'));EDGES=np.asarray(json.loads(pathlib.Path(CFG['local_input_paths']['width_definition']).read_text('utf8'))['edges_logQ'])
COUNTER={'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'repeat_noise_rows':0,
    'source_probe_rows':0,'extra_guide_probe_rows':0,'EVAL_read':0,'CAL_waveform_decode':0,'response_updates':0,'completed_projection_group_blocks':0}
torch.set_num_threads(12);torch.set_num_interop_threads(12);torch.use_deterministic_algorithms(True)
def log(v):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),v,flush=True)
def write(n,v):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(ev.safe(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def progress():write('EVALUATION_PROGRESS.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counters':COUNTER})
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def qb(q):
    z=np.log(np.maximum(q,1e-30));b=np.searchsorted(EDGES,z,side='right')-1;b[z==EDGES[-1]]=7;return b
for mod in [common]:
    for n in ['ROOT','OUT','CFG','VALLEY','EDGES','COUNTER','NAMES','GROUPS','ev','dc','aa','ew','ds','adc','load','qb','write','progress',
        'log','roc_auc_score','torch','np','pathlib','json','hashlib','wasserstein_distance','ks_2samp']:setattr(mod,n,globals()[n])
def attach(d):
    s=d['V'].astype('f8')-d['V'].astype('f8')[:,:100].mean(1,keepdims=True);p=np.maximum(s.max(1,keepdims=True),.0025)
    d.update(s=s,p=p,u=s/p,q=d['features']['Qtotal_mVns'],b=qb(d['features']['Qtotal_mVns']));return d


def c2st(domains,old,real):
    saved=load(pathlib.Path(CFG['R26_root'])/'results/C2ST_PRIVATE.npz',keys=[f'phase{p}_{j}'for p in[0,1]for j in range(3)])
    selected={p:tuple(saved[f'phase{p}_{j}']for j in range(3))for p in[0,1]}
    histroot=pathlib.Path(CFG['canonical_R51_root'])
    hist=json.loads((histroot/'CANONICAL_C2ST.json').read_text('utf8'))
    hp=load(histroot/'CANONICAL_PREDICTIONS_PRIVATE.npz')
    names={'BASE':'BASE_SOURCE','BASE_REAL_CONTROL':'BASE_REAL','RAW_REAL_REAL':'RAW_REFERENCE',
           'R49_COMPLETION_SOURCE':'R49_COMPLETION_SOURCE','R49_COMPLETION_REAL':'R49_COMPLETION_REAL'}
    views=CFG['C2ST_views'];result={};paired={};private={}
    r59root=pathlib.Path(CFG['historical_R59_root'])
    r59metrics=json.loads((r59root/'metrics.json').read_text('utf8'))
    r59pred=load(r59root/'results/C2ST_PRIVATE.npz')
    assert all(np.array_equal(saved[f'phase{p}_{j}'],r59pred[f'phase{p}_{j}']) for p in [0,1] for j in range(3))
    r59names={'R59_SOURCE':'DIRECT_DISTRIBUTION_RESPONSE','R59_REAL_CONTROL':'DIRECT_DISTRIBUTION_REAL_CONTROL'}
    def r59_prediction(name,view):
        tag=r59names[name]+'__'+view+'__'
        return {k:r59pred[tag+k] for k in ['probability','label','cluster']}
    for name,oldname in r59names.items():
        result[name]={v:dict(r59metrics['C2ST']['results'][oldname][v],reused_from=str(r59root/'metrics.json'),new_fit=False) for v in views}
    def old_prediction(name,view):
        tag=names[name]+'__'+view+'__'
        return {k:hp[tag+k] for k in ['probability','label','cluster']}
    for name,oldname in names.items():
        result[name]={}
        for view in views:
            row=dict(hist['results'][oldname+'__'+view])
            row.update(balanced_accuracy=row['BA'],heldout_CI={'balanced_accuracy_95CI':row['BA_95CI']},
                       reused_from=str(histroot/'CANONICAL_C2ST.json'),new_fit=False)
            result[name][view]=row
    paired['BASE_REAL_CONTROL']={v:dict(aa.paired_bootstrap(old_prediction('BASE_REAL_CONTROL',v),old_prediction('RAW_REAL_REAL',v),CFG),reference='RAW_REAL_REAL') for v in views}
    previous=dc.matrix
    def matrix(d,view):
        return d['u'].copy() if view=='normalized_u480' else previous(d,view)
    dc.matrix=matrix
    for name in ARMS:
        d=dict(domains[name]);r=dict(domains['REAL']);j=2 if name in REAL_ARMS else 0
        tr,te=selected[0][j],selected[1][j];rt,re=selected[0][1],selected[1][1]
        assert not set(d['groups'][tr])&set(d['groups'][te]) and not set(r['groups'][rt])&set(r['groups'][re])
        r['cluster']=np.array(['R|'+str(g)for g in r['groups']])
        if name in REAL_ARMS:d['cluster']=np.array(['R|'+str(g)for g in d['groups']])
        result[name]={};paired[name]={}
        for view in views:
            slot=name+'__'+view
            def persist(row,p):
                np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**p)
                write('results/C_'+slot+'.json',row);progress()
            dc.PERSIST_HGB_CALLBACK=persist
            row,p=dc.fit_one(d,r,(tr,te),(rt,re),view,CFG,COUNTER)
            row.update(original_fixed_rows_retained=True,canonical_ADC_representation=True,
                candidate_train_QC_n=int(d['valid'][tr].sum()),candidate_holdout_QC_n=int(d['valid'][te].sum()))
            persist(row,p);result[name][view]=row
            private.update({slot+'__'+k:p[k]for k in ['probability','label','cluster']})
            refname='RAW_REAL_REAL' if name in REAL_ARMS else 'BASE'
            paired[name][view]=dict(aa.paired_bootstrap(p,old_prediction(refname,view),CFG),reference=refname)
            write('C2ST_RUNNING.json',{'results':result,'paired':paired,'counters':COUNTER})
            log('C2ST '+slot+' BA '+str(row['balanced_accuracy']))
    same_treated={};historical_delta={};r59_delta={}
    for name in ARMS:
        histname='R49_COMPLETION_REAL' if name in REAL_ARMS else 'R49_COMPLETION_SOURCE'
        historical_delta[name]={}
        r59_delta[name]={}
        if name in REAL_ARMS:same_treated[name]={}
        for view in views:
            pp={k:private[name+'__'+view+'__'+k]for k in ['probability','label','cluster']}
            historical_delta[name][view]=dict(aa.paired_bootstrap(pp,old_prediction(histname,view),CFG),reference=histname)
            previous_name='R59_REAL_CONTROL' if name in REAL_ARMS else 'R59_SOURCE'
            r59_delta[name][view]=dict(aa.paired_bootstrap(pp,r59_prediction(previous_name,view),CFG),reference=previous_name)
            if name in REAL_ARMS:
                same_treated[name][view]=dict(aa.paired_bootstrap(pp,old_prediction('BASE_REAL_CONTROL',view),CFG),reference='BASE_REAL_CONTROL')
    write('SAME_TREATED_REAL_CONTROL_PAIRED.json',same_treated)
    write('PAIRED_R61_MINUS_R59.json',{'results':r59_delta,'same_fixed_selection':True,'same_GPU_platform':True,'single_factor_ablation':json.loads((ROOT/'TRAINING_FROZEN_AUDIT.json').read_text())['pass'],'single_training_seed':True,'training_seed_variability_not_estimated':True,'no_new_control_fits':True,'conditional_on_fitted_classifiers':True})
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{f'phase{p}_{j}':a for p,v in selected.items()for j,a in enumerate(v)})
    return {'results':result,'paired':paired,'same_treated_REAL_control_paired':same_treated,
        'R61_minus_frozen_R49_completion':historical_delta,'R61_minus_frozen_R59':r59_delta,'original_fixed_selection_no_resplit':True,
        'fixed_classifier_CI_not_response_training_uncertainty':True,'historical_control_fits_reused_from_R51':True,
        'all_five_original_acceptance_views':True,'new_fits':COUNTER['HGB_attempts'],
        'REAL_REAL_reference':'same-acquisition disjoint TRAIN reference/DEV artificial groups; not physical equivalence'}

def widths(domains,old):
    tables={n:old['energy_conditional_width']['tables'][n]for n in ['REAL','SOURCE_CLEAN','BASE_REAL_CONTROL',*CONTROLS]};grid=np.linspace(-.1,.6,129)
    for name in ARMS:
        d=domains[name];good=d['valid'];x=d['features']['PSD'][good];q=d['features']['Qtotal_mVns'][good];z,b=ew.bin_indices(q,EDGES);weights=np.ones(len(x),'f8');rows=[]
        for k in range(8):
            choose=b==k;finite=np.isfinite(x[choose])&np.isfinite(z[choose]);attempt=int(finite.sum()>=40)
            assert COUNTER['density_EM_attempts']+attempt<=16 and COUNTER['density_description_slots']<16
            COUNTER['density_EM_attempts']+=attempt;COUNTER['density_description_slots']+=1;progress()
            row=ew.fit_slice(x[choose],z[choose],weights[choose]);assert row['fit_attempts']==attempt
            row.update(bin=k,logQ_edges=EDGES[k:k+2].tolist(),charge_edges_mVns=np.exp(EDGES[k:k+2]).tolist(),
                actual_charge_range_mVns=[float(q[choose].min()),float(q[choose].max())]if choose.any()else None)
            row['histogram_counts']=np.histogram(x[choose],bins=grid)[0].tolist();row['histogram_PSD_outside_grid_count']=int(np.sum(choose&((x<grid[0])|(x>grid[-1]))))
            write('results/WIDTH_'+name+'_bin'+str(k+1)+'.json',row);rows.append(row)
        tables[name]={'N':float(len(x)),'below_logQ_range':int((b<0).sum()),'above_logQ_range':int((b>7).sum()),
            'PSD_below_minus0_1':int((x<-.1).sum()),'PSD_above0_6':int((x>.6).sum()),'slices':rows,'histogram_edges_PSD':grid.tolist()}
        write('results/WIDTH_'+name+'_TABLE.json',tables[name])
    errors={}
    for label,names in [(arm+'_vs_BASE',['BASE',arm])for arm in SOURCE_ARMS]+[('SOURCE_and_BASE',['BASE',*SOURCE_ARMS]),('all_new_and_BASE',['BASE',*ARMS])]:
        rows=[]
        for k in[0,1]:
            for field in['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=list(range(1,7));missing={n:[i+1 for i in bins if not(tables[n]['slices'][i]['valid']and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field]is not None)]for n in ['REAL',*names]}
                rows.append({'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'fixed_required_bins_not_candidate_intersection':True,'failed_required_bins_by_domain':missing,
                    'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field])for i in bins]))if not missing['REAL']and not missing[n]else None for n in names}})
        errors[label]=rows
    return{'tables':tables,'common_support_errors':errors,'edges_logQ':EDGES,'old_fits_reused':True,'all_slots_immediately_saved':True,'density_components_not_particle_truth':True}

def run():
    assert json.loads((ROOT/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))['both_pass']
    old=json.loads((pathlib.Path(CFG['R39_root'])/'metrics.json').read_text('utf8'));ref=json.loads((pathlib.Path(CFG['frozen_eval_input_root'])/'CAL_REFERENCE_METADATA.json').read_text('utf8'))
    actual=load(pathlib.Path(CFG['frozen_eval_input_root'])/'ACTUAL_GENERATION_INPUTS.npz');real=load(pathlib.Path(CFG['frozen_eval_input_root'])/'REAL_DEV_AND_REFERENCE_PRIVATE.npz');meta=load(pathlib.Path(CFG['frozen_eval_input_root'])/'DEV_SOURCE_METADATA.npz')
    ids=actual['source_ids'];source=actual['source_waveforms_V'];comp=meta['composition'];pdg=meta['pdg_inferred'];assert np.array_equal(ids,meta['source_ids'])
    source_adc,_=adc(source)
    domains={'REAL':aa.data(real['real_V'],real['real_ids'],real['real_groups'],'REAL',ref),
        'SOURCE_INPUT':aa.data(source,ids,np.arange(2000),'G4_SOURCE',ref,adc_domain=False),'SOURCE_CLEAN':aa.data(source_adc,ids,np.arange(2000),'G4_SOURCE',ref)}
    domains['REAL']['valid']&=real['real_QC_valid'];rr=aa.data(real['reference_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
    for n,path in [('BASE',pathlib.Path(CFG['R32_root'])/'CONTROL_FINAL/generated.npz'),
        ('INDEPENDENT_CFM',pathlib.Path(CFG['R33_root'])/'INDEPENDENT_CFM/generated.npz')]+[(n,ROOT/n/'generated.npz')for n in SOURCE_ARMS]:
        a=load(path,keys=['waveforms_V','source_ids']);assert np.array_equal(a['source_ids'],ids)
        domains[n]=aa.data(a['waveforms_V'],ids,np.arange(2000),'G4_SOURCE',ref)
    for name in REAL_ARMS:
        path=pathlib.Path(CFG['old_BASE_REAL_generated'])if name=='BASE_REAL_CONTROL'else ROOT/name/'generated.npz'
        a=load(path,keys=['waveforms_V','reference_ids','reference_groups'])
        assert np.array_equal(a['reference_ids'],real['reference_ids'])and np.array_equal(a['reference_groups'],real['reference_groups'])
        domains[name]=aa.data(a['waveforms_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
    generation=json.loads((ROOT/'GENERATION_RECEIPT.json').read_text('utf8'));COUNTER.update({k:generation['counter'][k]for k in ['main_generation_rows','repeat_noise_rows','source_probe_rows','extra_guide_probe_rows']})
    domains['TRAIN_REFERENCE']=rr
    actual_bins=qb(domains['SOURCE_INPUT']['features']['Qtotal_mVns'])
    # This is the historical R49 denominator, not the new canonical ADC feature view.
    reference_bins=qb(ev.features(source_adc)['features']['Qtotal_mVns'])
    assert [int(np.sum((comp=='pure_had')&(actual_bins==b)))for b in[2,3]]==[213,181]
    assert [int(np.sum((comp=='pure_had')&(reference_bins==b)))for b in[2,3]]==[207,182]
    stats,distance=ev.characterize(domains,ref);psd=common.psd_metrics(domains)
    for d in domains.values():attach(d)
    cs=c2st(domains,old,real);width=widths(domains,old)
    md={};mp={}
    source_domains={n:domains[n]for n in ['REAL','SOURCE_CLEAN','SOURCE_INPUT',*CONTROLS,*SOURCE_ARMS]}
    composition=fd.source_tables(source_domains,actual_bins,reference_bins,comp,VALLEY,EDGES)
    association={n:ds.association({'SOURCE_CLEAN':domains['SOURCE_INPUT'],'TRANSLATED':domains[n]},comp,pdg,EDGES)for n in [*CONTROLS,*SOURCE_ARMS]}
    historical={n:ds.association({'SOURCE_CLEAN':domains['SOURCE_CLEAN'],'TRANSLATED':domains[n]},comp,pdg,EDGES)for n in [*CONTROLS,*SOURCE_ARMS]}
    local={n:dc.native_statistics(d,ref)for n,d in domains.items()};feat={};waves={};raw={};raw_arrays={}
    for n,d in domains.items():
        feat.update({n+'__'+k:v for k,v in{**d['features'],**d['extra']}.items()});feat[n+'__valid']=d['valid'];feat[n+'__region']=d['region'];waves[n]=d['V'].astype('f4')
        if n in ['SOURCE_INPUT','SOURCE_CLEAN']:continue
        raw[n]=[]
        for k in[None,*range(8)]:
            use=d['valid']if k is None else d['valid']&(d['b']==k);target=domains['REAL']['valid']if k is None else domains['REAL']['valid']&(domains['REAL']['b']==k)
            row={'bin_id_1based':None if k is None else k+1,'N':int(use.sum()),'REAL_N':int(target.sum()),'qualified':int(use.sum())>=40 and int(target.sum())>=40}
            if use.any()and target.any():
                mv=d['V'][use].mean(0);rv=domains['REAL']['V'][target].mean(0);row['mean_full480_MSE_V2']=float(np.mean((mv-rv)**2));raw_arrays[n+'__'+str(k)+'__mean_V']=mv
                if use.sum()>1 and target.sum()>1:
                    sv=d['V'][use].std(0,ddof=1);rs=domains['REAL']['V'][target].std(0,ddof=1);row['SD_full480_RMS_difference_V']=float(np.sqrt(np.mean((sv-rs)**2)));raw_arrays[n+'__'+str(k)+'__SD_V']=sv
            raw[n].append(row)
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**feat);write('PLOT_WAVEFORM_REFERENCES.json',{'new':{n:str(ROOT/n/'generated.npz') for n in ARMS},'old_input_root':CFG['frozen_eval_input_root'],'old_outputs_referenced_not_copied':True})
    np.savez_compressed(OUT/'RAW_VOLTAGE_MOMENTS_PRIVATE.npz',**raw_arrays);write('MEAN_SD_METRICS.json',md)
    boundary=fd.boundary_metrics(domains,OUT);write('BOUNDARY_METRICS.json',boundary)
    gps=json.loads((ROOT/'GUIDE_PERTURBATION_COMPLETE_SUMMARY.json').read_text('utf8'))

    cv={'not_requested':True,'new_full_covariance_arrays_saved':0};coverage={};projection={}
    metrics={'covariance_description':cv,'status':'complete','training_scope':'R61 sole final50000 precision-order ablation; same R59 initialization, six consumed streams, parents and frozen constants','new_training_runs':1,'new_response_updates':50000,'training':{a:json.loads((pathlib.Path(CFG['historical_training_root'])/a/'TRAINING_RECEIPT.json').read_text('utf8'))for a in SOURCE_ARMS},
        'data_roles':json.loads((pathlib.Path(CFG['frozen_eval_input_root'])/'DATA_ROLES.json').read_text('utf8')),'domains':stats,'domain_vs_REAL':distance,'local':local,'PSD_main':psd,
        'energy_conditional_width':width,'source_composition':composition,'source_association_actual_preADC_input':association,
        'source_association_historical_postADC_reference':historical,'source_probes':json.loads((ROOT/'SOURCE_PROBES_SUMMARY.json').read_text('utf8')),
        'random_decomposition':json.loads((ROOT/'JOINT8_SUMMARY.json').read_text('utf8')),'guide_probe':gps,'C2ST':cs,'generation':generation['domains'],
        'boundary_differences':boundary,'projection_coverage':coverage,'projection_distances':projection,'explicit_constraint_projection':json.loads((ROOT/'CONSTRAINT_PROJECTION_SUMMARY.json').read_text('utf8')),'conditional_mean_SD':md,'raw_voltage_moments':raw,
        'counters':COUNTER,'primary_DEV_population_reweighted':False,'EVAL_read':0,'old_reference_fits_and_RAW_RR_reused':True,
        'one_new_same_treated_REAL_control':True,'canonical_ADC_features_before_all_new_statistics':True,'raw_SOURCE_INPUT_charge_and_PSD_preserved':True,'RAW_RR_same_acquisition_reference_reused':True,'final_physical_signal_electronic_split':'unknown',
        'versions':{'Python':platform.python_version(),'torch':torch.__version__,'NumPy':np.__version__,'sklearn':sklearn.__version__}}
    metrics['historical_SOURCE_CLEAN_bin_definition']='original R49 float32 postADC charge retained for legacy denominators; primary bins use raw SOURCE input'
    metrics['paired_R61_minus_historical_R49_completion']=cs['R61_minus_frozen_R49_completion']
    metrics['paired_R61_minus_frozen_R59']=cs['R61_minus_frozen_R59']
    write('metrics.json',metrics)
    assert COUNTER['HGB_attempts']==10 and COUNTER['density_description_slots']==16 and COUNTER['density_EM_attempts']<=16
    write('EVALUATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete'})
    log('R61 saved evaluation complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:
    write('EVALUATION_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
