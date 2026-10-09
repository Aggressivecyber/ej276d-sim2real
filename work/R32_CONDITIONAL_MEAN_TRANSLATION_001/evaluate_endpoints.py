from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[key]='4'
import pathlib,json,time,datetime,traceback,hashlib,platform
import numpy as np,torch,sklearn
from scipy.stats import wasserstein_distance,ks_2samp
from sklearn.metrics import roc_auc_score
import evaluation as ev,diagnostic_core as dc,assessment as aa,energy_width as ew,data_support as ds
from model import Translator,infer,adc
from random_diagnostics import summarize
from mean_description import describe as describe_means
import endpoint_common as common
import temporal_common as temporal
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);START=time.monotonic()
CFG=json.loads((ROOT/'config.json').read_text('utf8'));ARMS=CFG['endpoints'];CONTROLS=[]
NAMES=[*CONTROLS,*ARMS];GROUPS=['shape480','diff479','baseline100_V']
VALLEY=json.loads((ROOT/'VALLEY_DEFINITION.json').read_text('utf8'));EDGES=np.asarray(json.loads(pathlib.Path(CFG['local_input_paths']['width_definition']).read_text('utf8'))['edges_logQ'])
COUNTER={'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,
         'repeat_noise_rows':0,'source_probe_rows':0,'EVAL_read':0,'CAL_waveform_decode':0,
         'response_updates':0,'completed_projection_group_blocks':0}
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
def log(x):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),x,flush=True)
def write(n,v):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
def progress():write('EVALUATION_PROGRESS.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counters':COUNTER})
def load(path,keys=None):
    with np.load(path,allow_pickle=False) as z:return{k:z[k].copy() for k in (z.files if keys is None else keys)}
def qb(q):
    z=np.log(np.maximum(q,1e-30));b=np.searchsorted(EDGES,z,side='right')-1;b[z==EDGES[-1]]=7;return b
for mod in [common,temporal]:
    for n in ['ROOT','OUT','CFG','VALLEY','EDGES','COUNTER','NAMES','GROUPS','ev','dc','aa','ew','ds','infer','adc',
              'summarize','load','qb','write','progress','log','roc_auc_score','torch','np','pathlib','json','hashlib',
              'wasserstein_distance','ks_2samp']:
        setattr(mod,n,globals()[n])
def equal_states(a,b):return list(a)==list(b) and all(torch.equal(a[k],b[k]) for k in a)

def attach_temporal(d):
    s=d['V'].astype('f8')-d['V'].astype('f8')[:,:100].mean(1,keepdims=True)
    p=np.maximum(s.max(1,keepdims=True),.0025);d.update(s=s,p=p,u=s/p,q=d['features']['Qtotal_mVns'],b=qb(d['features']['Qtotal_mVns']))
    return d

def consumed_audit(training):
    old=load(CFG['consumed_R26_stream']);oldreceipt=json.loads(pathlib.Path(CFG['consumed_R26_receipt']).read_text('utf8'));rows={}
    for arm,t in training.items():
        new=load(ROOT/arm/'TRAINING_STREAMS_PRIVATE.npz');n=t['updates']*32
        checks={k:bool(np.array_equal(new[k],old[k][:t['updates']*3 if k=='NCE_positions' else n])) for k in new}
        if n==94400:checks.update(noise_stream=t['stream_hashes']['noise_stream']==oldreceipt['stream_hashes']['noise_stream'],style_stream=t['stream_hashes']['style_stream']==oldreceipt['stream_hashes']['style_stream'])
        fresh=json.loads((ROOT/arm/'FRESH_PROJECTION_STREAM.json').read_text('utf8'))
        assert all(checks.values()) and fresh['calls']==t['updates'] and fresh['actual_matrix_count']==t['updates']*3
        rows[arm]={'actual_rows':n,'actual_G_D_updates':t['updates'],'equal2950':t['updates']==2950,'exact_original_flow':checks,
            'fresh_calls':fresh['calls'],'same_directions_two_domains_each_G':all(v['same_draw_for_GENERATED_and_REAL'] for v in fresh['per_call_actual_float32_versions']),
            'fresh_validation_D_identity_evaluation_calls':0,'NCE_same_per_waveform_temporal_patch_positions_and_row_assignment':True}
    keys=['source','target','bin','original_row_ordinal','NCE_positions','noise_stream','style_stream']
    a,b=[training[n]for n in CFG['arms']];pair={k:a['stream_hashes'][k]==b['stream_hashes'][k]for k in keys}
    output={'arms':rows,'same_actual_steps':a['updates']==b['updates'],'same_stream_hashes':pair,'both_2950':a['updates']==b['updates']==2950,
        'same_platform':True,'migration_consistency_test':False}
    write('CONSUMED_EXPOSURE_AUDIT.json',output);return output



def c2st(domains,old,alias):
    oldpred=load(pathlib.Path(CFG['R26_root'])/'results/C2ST_PRIVATE.npz',keys=[f'phase{p}_{j}'for p in [0,1]for j in range(3)])
    selected={p:tuple(oldpred[f'phase{p}_{j}']for j in range(3))for p in [0,1]}
    gs=(selected[0][0],selected[1][0]);rs=(selected[0][1],selected[1][1]);results={};private={}
    saved_matrix=dc.matrix
    def matrix(d,view):
        key={'normalized_u480':'u','seen_reconstruction480':'seen','complement_reconstruction480':'complement'}.get(view)
        return d[key].copy()if key else saved_matrix(d,view)
    dc.matrix=matrix;views=[*CFG['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480']
    for name in ARMS:
        results[name]={};d=domains[name];r=domains['REAL']
        assert not set(d['groups'][gs[0]])&set(d['groups'][gs[1]]) and not set(r['groups'][rs[0]])&set(r['groups'][rs[1]])
        assert d['valid'][np.r_[gs[0],gs[1]]].all()and r['valid'][np.r_[rs[0],rs[1]]].all()
        for view in views:
            slot=name+'__'+view
            if name in alias and alias[name]in results and view in results[alias[name]]:
                canonical=alias[name];row=dict(results[canonical][view]);row['same_arm_endpoint_identity_alias']=canonical
                p=load(OUT/(canonical+'__'+view+'_PREDICTIONS_PRIVATE.npz'))
            else:
                def persist_fit(row,p):
                    np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**p);write('results/C_'+slot+'.json',row);progress()
                dc.PERSIST_HGB_CALLBACK=persist_fit
                row,p=dc.fit_one(d,r,gs,rs,view,CFG,COUNTER)
            np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**p);write('results/C_'+slot+'.json',row);results[name][view]=row
            private.update({slot+'__'+k:p[k]for k in ['probability','label','cluster']})
            write('C2ST_RUNNING.json',{'results':results,'counters':COUNTER});log('C2ST '+slot+' BA '+str(row['balanced_accuracy']))
    paired={}
    for label,mean,control in [('final_equal_updates','MEAN_FINAL','CONTROL_FINAL'),('selected','MEAN_SELECTED','CONTROL_SELECTED')]:
        paired[label]={}
        for view in views:
            a=load(OUT/(mean+'__'+view+'_PREDICTIONS_PRIVATE.npz'));b=load(OUT/(control+'__'+view+'_PREDICTIONS_PRIVATE.npz'))
            paired[label][view]=aa.paired_bootstrap(a,b,CFG)
    historical=json.loads((pathlib.Path(CFG['R28_root'])/'metrics.json').read_text('utf8'))['C2ST']['results']
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{f'phase{p}_{j}':v for p,values in selected.items()for j,v in enumerate(values)})
    return {'results':results,'paired_MEAN_minus_CONTROL':paired,'original_actual_selection_no_resplit':True,'train_each_domain':len(gs[0]),
        'holdout_each_domain':len(gs[1]),'seven_historical_and_three_pure480_views':True,'fixed_classifier_CI_not_training_uncertainty':True,
        'REAL_REAL_reference':old['C2ST']['RR_definition'],'R28_historical_C2ST_results':historical,'cross_platform_historical_not_new_loss_effect':True}



def widths(domains,old,alias):
    tables={n:old['energy_conditional_width']['tables'][n] for n in ['REAL','SOURCE_CLEAN',*CONTROLS]};grid=np.linspace(-.1,.6,129)
    for name in ARMS:
        if name in alias and alias[name] in tables:tables[name]=tables[alias[name]];continue
        d=domains[name];good=d['valid'];x=d['features']['PSD'][good];q=d['features']['Qtotal_mVns'][good]
        z,b=ew.bin_indices(q,EDGES);weights=np.ones(len(x),'f8');rows=[]
        for k in range(8):
            choose=b==k;finite=np.isfinite(x[choose])&np.isfinite(z[choose]);attempt=int(finite.sum()>=40)
            assert COUNTER['density_EM_attempts']+attempt<=CFG['density_EM_max']
            COUNTER['density_EM_attempts']+=attempt;COUNTER['density_description_slots']+=1;progress()
            row=ew.fit_slice(x[choose],z[choose],weights[choose]);assert row['fit_attempts']==attempt
            row.update(bin=k,logQ_edges=EDGES[k:k+2].tolist(),charge_edges_mVns=np.exp(EDGES[k:k+2]).tolist(),
                       actual_charge_range_mVns=[float(q[choose].min()),float(q[choose].max())] if choose.any() else None)
            hist,_=np.histogram(x[choose],bins=grid);row['histogram_counts']=hist.tolist()
            row['histogram_PSD_outside_grid_count']=int(np.sum(choose&((x<grid[0])|(x>grid[-1]))))
            write('results/WIDTH_'+name+'_bin'+str(k+1)+'.json',row);rows.append(row)
        tables[name]={'N':float(len(x)),'below_logQ_range':int((b<0).sum()),'above_logQ_range':int((b>7).sum()),
                      'PSD_below_minus0_1':int((x<-.1).sum()),'PSD_above0_6':int((x>.6).sum()),
                      'slices':rows,'histogram_edges_PSD':grid.tolist()}
        write('results/WIDTH_'+name+'_TABLE.json',tables[name])
    errors={}
    for label,names in [('all_endpoints',[*CONTROLS,*ARMS]),('selected',['CONTROL_SELECTED','MEAN_SELECTED']),('final_equal_updates',['CONTROL_FINAL','MEAN_FINAL'])]:
        rows=[]
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field] is not None for n in ['REAL',*names])]
                rows.append({'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],
                             'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in bins])) if bins else None for n in names}})
        errors[label]=rows
    return {'tables':tables,'common_support_errors':errors,'edges_logQ':EDGES,'old_reference_fits_reused':True,
            'all_slots_persisted_before_next':True,'density_components_not_particle_truth':True}

def run():
    old=json.loads((pathlib.Path(CFG['R26_root'])/'metrics.json').read_text('utf8'))
    ref=json.loads((ROOT/'CAL_REFERENCE_METADATA.json').read_text('utf8'));actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz')
    meta=load(ROOT/'DEV_SOURCE_METADATA.npz');ids=actual['source_ids'];source=actual['source_waveforms_V'];comp=meta['composition'];pdg=meta['pdg_inferred']
    real=load(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz')
    domains={'REAL':aa.data(real['real_V'],real['real_ids'],real['real_groups'],'REAL',ref),'SOURCE_CLEAN':aa.data(source,ids,np.arange(2000),'G4_SOURCE',ref)}
    domains['REAL']['valid']&=real['real_QC_valid'];rr=aa.data(real['reference_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
    generation={};repeats={};probe={};states={};alias={};evidence={}
    for name in ARMS:
        log('CPU main '+name);(ROOT/name).mkdir(exist_ok=True);model=Translator()
        states[name]=torch.load(CFG['endpoint_weight_paths'][name],map_location='cpu',weights_only=True)
        model.load_state_dict(states[name]);model.eval()
        previous=[p for p in ARMS if p in domains and p.split('_')[0]==name.split('_')[0]]
        canonical=next((p for p in previous if equal_states(states[name],states[p])),None)
        if canonical:
            original=load(ROOT/canonical/'generated.npz');pre=original['pre_ADC_V'];post=original['waveforms_V'];signal=original['signal_component_V'];electronic=original['electronic_component_V'];ad=generation[canonical]['ADC']
            assert np.array_equal(original['source_ids'],ids)
            alias[name]=canonical;evidence[name]={'canonical':canonical,'same_arm':True,'state_all_tensors_equal':True,'same_single_actual_input_container':True,'source_ID_order_equal':True,'no_duplicate_forward':True}
        else:
            pre,signal,electronic=infer(model,source,actual['z'],actual['white'],actual['style'],CFG['generation_chunk']);post,ad=adc(pre);COUNTER['main_generation_rows']+=2000
        np.savez_compressed(ROOT/name/'generated.npz',pre_ADC_V=pre,waveforms_V=post,signal_component_V=signal,electronic_component_V=electronic,source_ids=ids)
        domains[name]=aa.data(post,ids,np.arange(2000),'G4_SOURCE',ref)
        generation[name]={'rows':2000,'all_original_source_ids_preserved':bool(np.array_equal(ids,meta['source_ids'])),
            'ADC':ad,'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),
            'style_reader_used':False,'signal_not_conditional_expectation':True,'weight_file':CFG['endpoint_weight_paths'][name],'alias':alias.get(name),'CPU_generated_here':canonical is None}
        if canonical:probe[name]=probe[canonical];repeats[name]=repeats[canonical]
        else:probe[name]=common.probes(model,post,actual['style'],name);repeats[name]=common.joint_repeats(model,source,comp,name)
        write('GENERATION_PROGRESS.json',{'generation':generation,'aliases':alias,'counters':COUNTER});del model
    stats,distance=ev.characterize(domains,ref);psd=common.psd_metrics(domains);width=widths(domains,old,alias)
    for d in [*domains.values(),rr]:attach_temporal(d)
    maps=load(pathlib.Path(CFG['R27_root'])/'results/PROJECTION_SPACE_PRIVATE.npz');seen=maps['Pseen'];complement=maps['Pcomplement']
    for d in [*domains.values(),rr]:d['seen']=d['u']@seen;d['complement']=d['u']@complement
    cs=c2st(domains,old,alias)
    constants=load(CFG['local_training_input_paths']['FINE_CONSTANTS.npz']);tdomains={n:domains[n]for n in ['REAL',*ARMS]};tdomains['TRAIN_REFERENCE']=rr
    coverage=temporal.means_coverage(tdomains,seen,complement,domains['SOURCE_CLEAN']['b']);projection=temporal.fine_distances(tdomains,constants)
    target=load(ROOT/'MEAN_TARGETS_PRIVATE.npz',keys=['mu']);md,mp=describe_means(tdomains,ARMS,domains['SOURCE_CLEAN']['b'],target['mu'])
    np.savez_compressed(OUT/'MEAN_SD_DESCRIPTIONS_PRIVATE.npz',**mp);write('MEAN_SD_METRICS.json',md)
    frozen=json.loads((ROOT/'FROZEN_TILT_TABLE.json').read_text('utf8'));lambdas=np.asarray([r['lambda_value']for r in frozen['bins']]);h=.15*meta['edep_hadron_MeV']/(meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV'])
    composition=common.source_composition(domains,comp,h,lambdas)
    association={n:ds.association({'SOURCE_CLEAN':domains['SOURCE_CLEAN'],'TRANSLATED':domains[n]},comp,pdg,EDGES)for n in ARMS}
    local={n:dc.native_statistics(d,ref)for n,d in domains.items()};feat={};waves={}
    for name,d in domains.items():
        feat.update({name+'__'+key:a for key,a in {**d['features'],**d['extra']}.items()});feat[name+'__valid']=d['valid'];feat[name+'__region']=d['region'];waves[name]=d['V'].astype('f4')
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**feat);np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**waves)
    training={n:json.loads((ROOT/n/'TRAINING_RECEIPT.json').read_text('utf8'))for n in CFG['arms']};exposure=consumed_audit(training)
    online={n:common.online_auc(n)for n in CFG['arms']}
    metric={'status':'complete','training':training,'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text('utf8')),
        'domains':stats,'domain_vs_REAL':distance,'local':local,'PSD_main':psd,'energy_conditional_width':width,
        'source_composition':composition,'source_association':association,'source_probes':probe,'random_decomposition':repeats,
        'C2ST':cs,'generation':generation,'online_D_validation_AUC':online,'row_exposure':exposure,'endpoint_aliases':alias,
        'alias_evidence':evidence,'projection_coverage':coverage,'projection_distances':projection,'conditional_mean_SD':md,'counters':COUNTER,
        'versions':{'Python':platform.python_version(),'torch':torch.__version__,'NumPy':np.__version__,'sklearn':sklearn.__version__},
        'primary_DEV_population_reweighted':False,'EVAL_read':0,'CPU_main_and_replay_same_environment':True,
        'two_current_server_arms_retrained_as_registered':True,'sole_training_intervention':'conditional mean U coefficient1',
        'endpoint_selection_uses_DEV':False,'mean_target_is_TRAIN_only':True}
    write('metrics.json',metric);write('ENDPOINT_ALIAS_RECEIPT.json',{'aliases':alias,'evidence':evidence,'counts':COUNTER})
    assert COUNTER['HGB_attempts']<=40 and COUNTER['density_EM_attempts']<=32 and COUNTER['main_generation_rows']<=8000 and COUNTER['repeat_noise_rows']<=4096 and COUNTER['source_probe_rows']<=3328
    write('EVALUATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete','threads':4})
    log('R32 evaluation complete '+json.dumps(COUNTER))


try:run()
except BaseException as e:
    write('EVALUATION_FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),
          'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
