from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[key]='4'
import pathlib,json,time,datetime,traceback,hashlib,platform
import numpy as np,torch,sklearn
from scipy.stats import wasserstein_distance,ks_2samp
from sklearn.metrics import roc_auc_score
import evaluation as ev,diagnostic_core as dc,assessment as aa,energy_width as ew,data_support as ds
from model import adc
import flow_diagnostics as fd
from mean_description import describe as describe_means
import endpoint_common as common,temporal_common as temporal
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);START=time.monotonic()
CFG=json.loads((ROOT/'config.json').read_text('utf8'));ARMS=CFG['endpoints'];SOURCE_ARM='CONDITIONAL_DIFFUSION_EDIT';REAL_ARM='REAL_GUIDE_EDIT'
CONTROLS=['BASE','INDEPENDENT_CFM','OT_CFM','PSD_ORDERED_CFM'];NAMES=[*CONTROLS,*ARMS];GROUPS=['shape480','diff479','baseline100_V']
VALLEY=json.loads((ROOT/'VALLEY_DEFINITION.json').read_text('utf8'));EDGES=np.asarray(json.loads(pathlib.Path(CFG['local_input_paths']['width_definition']).read_text('utf8'))['edges_logQ'])
COUNTER={'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'repeat_noise_rows':0,
    'source_probe_rows':0,'extra_guide_probe_rows':0,'EVAL_read':0,'CAL_waveform_decode':0,'response_updates':0,'completed_projection_group_blocks':0}
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
def log(v):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),v,flush=True)
def write(n,v):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(ev.safe(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def progress():write('EVALUATION_PROGRESS.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counters':COUNTER})
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def qb(q):
    z=np.log(np.maximum(q,1e-30));b=np.searchsorted(EDGES,z,side='right')-1;b[z==EDGES[-1]]=7;return b
for mod in [common,temporal]:
    for n in ['ROOT','OUT','CFG','VALLEY','EDGES','COUNTER','NAMES','GROUPS','ev','dc','aa','ew','ds','adc','load','qb','write','progress',
        'log','roc_auc_score','torch','np','pathlib','json','hashlib','wasserstein_distance','ks_2samp']:setattr(mod,n,globals()[n])
def attach(d):
    s=d['V'].astype('f8')-d['V'].astype('f8')[:,:100].mean(1,keepdims=True);p=np.maximum(s.max(1,keepdims=True),.0025)
    d.update(s=s,p=p,u=s/p,q=d['features']['Qtotal_mVns'],b=qb(d['features']['Qtotal_mVns']));return d

def raw_RR(views,selected,real):
    r7=ROOT.parent/'R7_RESPONSE_RESIDUAL_DIAGNOSIS_001'
    sel=json.loads((r7/'results/DIAGNOSTIC_SELECTIONS.json').read_text('utf8'))['indices']
    match={k:bool(np.array_equal(np.asarray(sel[name][part]),selected[p][j]))for k,name,part,p,j in
        [('reference_train','TRAIN_REFERENCE','train',0,2),('reference_holdout','TRAIN_REFERENCE','holdout',1,2),
         ('REAL_train','REAL','train',0,1),('REAL_holdout','REAL','holdout',1,1)]}
    assert all(match.values()),('RAW_RR_fixed_selection_conflict',match)
    p7=load(r7/'results/C2ST_HOLDOUT_PREDICTIONS.npz',keys=[f'native|{v}|REAL_TRAIN_vs_DEV__{k}'for v in CFG['C2ST_views']for k in ['probability','label','cluster']])
    result={};pred={};metrics7=json.loads((r7/'metrics.json').read_text('utf8'))['C2ST']
    expected_labels=np.r_[np.ones(len(selected[1][2])),np.zeros(len(selected[1][1]))]
    expected_groups=np.r_[real['reference_groups'][selected[1][2]],real['real_groups'][selected[1][1]]]
    canonical=np.array(['R|'+str(g)for g in expected_groups])
    for view in views:
        if view in CFG['C2ST_views']:
            key=f'native|{view}|REAL_TRAIN_vs_DEV';pp={k:p7[key+'__'+k]for k in ['probability','label','cluster']};row=metrics7[key]
        else:
            rrroot=pathlib.Path(CFG['R27_root'])/'results';pp=load(rrroot/('TRAIN_REFERENCE__'+view+'_PREDICTIONS_PRIVATE.npz'),keys=['probability','label','cluster'])
            row=json.loads((rrroot/('C_TRAIN_REFERENCE__'+view+'.json')).read_text('utf8'))
        assert np.array_equal(pp['label'],expected_labels)
        actual_suffix=np.array([str(g).split('|',1)[1]for g in pp['cluster']])
        assert np.array_equal(actual_suffix,np.array([str(g)for g in expected_groups])),('RAW_RR_cluster_order',view)
        original_cluster=pp['cluster'].copy();pp['cluster']=canonical.copy();result[view]=row;pred[view]=pp
        np.savez_compressed(OUT/('RAW_RR__'+view+'_REUSED_PREDICTIONS_PRIVATE.npz'),**pp,original_namespace_cluster=original_cluster)
    write('RAW_RR_REUSE_BINDING.json',{'original_R7_R26_phase_rows_equal':match,'original_event_group_order_checked':True,
        'canonical_R_namespace_preserves_actual_group_identifiers':True,'old_predictions_no_fit':True,
        'historical7_source':str(r7),'pure480_views_source':CFG['R27_root'],'fixed_classifier_bootstrap_seed':CFG['bootstrap_seed']})
    return result,pred

def c2st(domains,old,real):
    saved=load(pathlib.Path(CFG['R26_root'])/'results/C2ST_PRIVATE.npz',keys=[f'phase{p}_{j}'for p in[0,1]for j in range(3)])
    selected={p:tuple(saved[f'phase{p}_{j}']for j in range(3))for p in[0,1]}
    views=[*CFG['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480']
    rrrows,rrpred=raw_RR(views,selected,real)
    result={n:old['C2ST']['results'][n]for n in CONTROLS};result['RAW_REAL_REAL']=rrrows;private={};paired={}
    previous_matrix=dc.matrix
    def matrix(d,v):
        key={'normalized_u480':'u','seen_reconstruction480':'seen','complement_reconstruction480':'complement'}.get(v)
        return d[key].copy()if key else previous_matrix(d,v)
    dc.matrix=matrix
    for name in ARMS:
        result[name]={};r=domains['REAL'];d=domains[name];j=2 if name==REAL_ARM else 0
        tr,te=selected[0][j],selected[1][j];rt,re=selected[0][1],selected[1][1]
        assert not set(d['groups'][tr])&set(d['groups'][te])and not set(r['groups'][rt])&set(r['groups'][re])
        if name==REAL_ARM:
            d=dict(d);r=dict(r);d['cluster']=np.array(['R|'+str(g)for g in d['groups']]);r['cluster']=np.array(['R|'+str(g)for g in r['groups']])
        paired[name]={}
        for v in views:
            slot=name+'__'+v
            def persist(row,p):
                np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**p);write('results/C_'+slot+'.json',row);progress()
            dc.PERSIST_HGB_CALLBACK=persist
            row,p=dc.fit_one(d,r,(tr,te),(rt,re),v,CFG,COUNTER)
            row.update(original_fixed_rows_retained=True,candidate_train_QC_n=int(d['valid'][tr].sum()),candidate_holdout_QC_n=int(d['valid'][te].sum()))
            persist(row,p);result[name][v]=row;private.update({slot+'__'+k:p[k]for k in['probability','label','cluster']})
            if name==REAL_ARM:ref=rrpred[v];refname='RAW_REAL_REAL'
            else:ref=load(pathlib.Path(CFG['R32_root'])/'results'/('CONTROL_FINAL__'+v+'_PREDICTIONS_PRIVATE.npz'),keys=['probability','label','cluster']);refname='BASE'
            paired[name][v]=aa.paired_bootstrap(p,ref,CFG);paired[name][v]['reference']=refname
            write('C2ST_RUNNING.json',{'results':result,'paired':paired,'counters':COUNTER});log('C2ST '+slot+' BA '+str(row['balanced_accuracy']))
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{f'phase{p}_{j}':a for p,v in selected.items()for j,a in enumerate(v)})
    return{'results':result,'paired':paired,'original_fixed_selection_no_resplit':True,'source_and_REAL_control_use_corresponding_phase0_2_rows':True,
        'fixed_classifier_CI_not_response_training_uncertainty':True,'raw_RR_refs_reused_no_fit':True,
        'REAL_REAL_reference':'same-acquisition disjoint TRAIN reference/DEV artificial groups; not physical equivalence'}

def widths(domains,old):
    tables={n:old['energy_conditional_width']['tables'][n]for n in ['REAL','SOURCE_CLEAN',*CONTROLS]};grid=np.linspace(-.1,.6,129)
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
    for label,names in [('DIFFUSION_vs_BASE',['BASE',SOURCE_ARM]),('DIFFUSION_vs_ORDERED',['PSD_ORDERED_CFM',SOURCE_ARM]),('all_new_and_BASE',['BASE',*ARMS])]:
        rows=[]
        for k in[0,1]:
            for field in['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8)if all(tables[n]['slices'][i]['valid']and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field]is not None for n in['REAL',*names])]
                rows.append({'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],
                    'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field])for i in bins]))if bins else None for n in names}})
        errors[label]=rows
    return{'tables':tables,'common_support_errors':errors,'edges_logQ':EDGES,'old_fits_reused':True,'all_slots_immediately_saved':True,'density_components_not_particle_truth':True}

def run():
    old=json.loads((pathlib.Path(CFG['R35_root'])/'metrics.json').read_text('utf8'));ref=json.loads((ROOT/'CAL_REFERENCE_METADATA.json').read_text('utf8'))
    actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');real=load(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz');meta=load(ROOT/'DEV_SOURCE_METADATA.npz')
    ids=actual['source_ids'];source=actual['source_waveforms_V'];comp=meta['composition'];pdg=meta['pdg_inferred'];assert np.array_equal(ids,meta['source_ids'])
    source_adc,_=adc(source)
    domains={'REAL':aa.data(real['real_V'],real['real_ids'],real['real_groups'],'REAL',ref),
        'SOURCE_INPUT':aa.data(source,ids,np.arange(2000),'G4_SOURCE',ref),'SOURCE_CLEAN':aa.data(source_adc,ids,np.arange(2000),'G4_SOURCE',ref)}
    domains['REAL']['valid']&=real['real_QC_valid'];rr=aa.data(real['reference_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
    for n,path in [('BASE',pathlib.Path(CFG['R32_root'])/'CONTROL_FINAL/generated.npz'),
        ('INDEPENDENT_CFM',pathlib.Path(CFG['R33_root'])/'INDEPENDENT_CFM/generated.npz'),('OT_CFM',pathlib.Path(CFG['R33_root'])/'OT_CFM/generated.npz'),
        ('PSD_ORDERED_CFM',pathlib.Path(CFG['R35_root'])/'PSD_ORDERED_CFM/generated.npz'),(SOURCE_ARM,ROOT/SOURCE_ARM/'generated.npz')]:
        a=load(path,keys=['waveforms_V','source_ids']);assert np.array_equal(a['source_ids'],ids)
        domains[n]=aa.data(a['waveforms_V'],ids,np.arange(2000),'G4_SOURCE',ref)
    a=load(ROOT/REAL_ARM/'generated.npz',keys=['waveforms_V','reference_ids','reference_groups'])
    assert np.array_equal(a['reference_ids'],real['reference_ids'])and np.array_equal(a['reference_groups'],real['reference_groups'])
    domains[REAL_ARM]=aa.data(a['waveforms_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
    generation=json.loads((ROOT/'GENERATION_RECEIPT.json').read_text('utf8'));COUNTER.update({k:generation['counter'][k]for k in ['main_generation_rows','repeat_noise_rows','source_probe_rows','extra_guide_probe_rows']})
    domains['TRAIN_REFERENCE']=rr
    actual_bins=qb(domains['SOURCE_INPUT']['features']['Qtotal_mVns']);reference_bins=qb(domains['SOURCE_CLEAN']['features']['Qtotal_mVns'])
    assert [int(np.sum((comp=='pure_had')&(actual_bins==b)))for b in[2,3]]==[213,181]
    assert [int(np.sum((comp=='pure_had')&(reference_bins==b)))for b in[2,3]]==[207,182]
    stats,distance=ev.characterize(domains,ref);psd=common.psd_metrics(domains)
    for d in domains.values():attach(d)
    maps=load(pathlib.Path(CFG['R27_root'])/'results/PROJECTION_SPACE_PRIVATE.npz',keys=['Pseen','Pcomplement']);seen=maps['Pseen'];complement=maps['Pcomplement']
    for d in domains.values():d['seen']=d['u']@seen;d['complement']=d['u']@complement
    cs=c2st(domains,old,real);width=widths(domains,old)
    tdomains={n:domains[n]for n in ['REAL',*NAMES,'TRAIN_REFERENCE']};CFG['source_condition_domains']=[*CONTROLS,SOURCE_ARM]
    constants=load(CFG['local_training_input_paths']['FINE_CONSTANTS.npz'])
    coverage=temporal.means_coverage(tdomains,seen,complement,actual_bins);projection=temporal.fine_distances(tdomains,constants)
    oldmu=load(pathlib.Path(CFG['R32_root'])/'MEAN_TARGETS_PRIVATE.npz',keys=['mu'])['mu'];md,mp=describe_means(tdomains,[*CONTROLS,SOURCE_ARM],actual_bins,oldmu)
    md['source_Q_definition']='actual unADC NN input';np.savez_compressed(OUT/'MEAN_SD_DESCRIPTIONS_PRIVATE.npz',**mp)
    source_domains={n:domains[n]for n in ['REAL','SOURCE_CLEAN','SOURCE_INPUT',*CONTROLS,SOURCE_ARM]}
    composition=fd.source_tables(source_domains,actual_bins,reference_bins,comp,VALLEY,EDGES)
    association={n:ds.association({'SOURCE_CLEAN':domains['SOURCE_INPUT'],'TRANSLATED':domains[n]},comp,pdg,EDGES)for n in [*CONTROLS,SOURCE_ARM]}
    historical={n:ds.association({'SOURCE_CLEAN':domains['SOURCE_CLEAN'],'TRANSLATED':domains[n]},comp,pdg,EDGES)for n in [*CONTROLS,SOURCE_ARM]}
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
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**feat);np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**waves)
    np.savez_compressed(OUT/'RAW_VOLTAGE_MOMENTS_PRIVATE.npz',**raw_arrays);write('MEAN_SD_METRICS.json',md)
    boundary=fd.boundary_metrics(domains,OUT);write('BOUNDARY_METRICS.json',boundary)
    gp=load(OUT/'GUIDE_PERTURBATION_PRIVATE.npz',keys=['original_guide_V','actual_perturbed_guide_f32_V'])
    def charges(x):
        s=x.astype('f8')-x.astype('f8')[:,:100].mean(1,keepdims=True);return s[:,100:150].sum(1)*1000.,s[:,100:400].sum(1)*1000.
    qf0,qt0=charges(gp['original_guide_V']);qf1,qt1=charges(gp['actual_perturbed_guide_f32_V'])
    gps=json.loads((ROOT/'GUIDE_PERTURBATION_SUMMARY.json').read_text('utf8'));gps.update(actual_f32_Qfast_mVns_change=ev.summary(qf1-qf0),
        Qfast_exact_unchanged=bool(np.array_equal(qf0,qf1)),saved_array_description_no_new_generation=True)
    write('GUIDE_PERTURBATION_COMPLETE_SUMMARY.json',gps)
    metrics={'status':'complete','training':json.loads((ROOT/SOURCE_ARM/'TRAINING_RECEIPT.json').read_text('utf8')),
        'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text('utf8')),'domains':stats,'domain_vs_REAL':distance,'local':local,'PSD_main':psd,
        'energy_conditional_width':width,'source_composition':composition,'source_association_actual_preADC_input':association,
        'source_association_historical_postADC_reference':historical,'source_probes':json.loads((ROOT/'SOURCE_PROBES_SUMMARY.json').read_text('utf8')),
        'random_decomposition':json.loads((ROOT/'JOINT8_SUMMARY.json').read_text('utf8')),'guide_probe':gps,'C2ST':cs,'generation':generation['domains'],
        'boundary_differences':boundary,'projection_coverage':coverage,'projection_distances':projection,'conditional_mean_SD':md,'raw_voltage_moments':raw,
        'counters':COUNTER,'primary_DEV_population_reweighted':False,'EVAL_read':0,'old_reference_fits_and_RAW_RR_reused':True,
        'REAL_control_seen_TRAIN_reference_not_unseen_acquisition':True,'final_physical_signal_electronic_split':'unknown',
        'versions':{'Python':platform.python_version(),'torch':torch.__version__,'NumPy':np.__version__,'sklearn':sklearn.__version__}}
    write('metrics.json',metrics)
    assert COUNTER['HGB_attempts']==20 and COUNTER['density_description_slots']==16 and COUNTER['density_EM_attempts']<=16
    write('EVALUATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete'})
    log('R36 saved evaluation complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:
    write('EVALUATION_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
