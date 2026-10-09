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
import endpoint_common as common
import temporal_common as temporal
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);START=time.monotonic()
CFG=json.loads((ROOT/'config.json').read_text('utf8'));ARMS=CFG['endpoints'];CONTROLS=['HOMO_SELECTED','HOMO_FINAL']
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
    new=load(ROOT/'FRESH_FINE/TRAINING_STREAMS_PRIVATE.npz');old=load(CFG['consumed_R26_stream']);n=training['updates']*32
    checks={k:bool(np.array_equal(new[k],old[k][:training['updates']*3 if k=='NCE_positions' else n])) for k in new}
    oldreceipt=json.loads(pathlib.Path(CFG['consumed_R26_receipt']).read_text('utf8'))
    if n==94400:
        checks.update(noise_stream=training['stream_hashes']['noise_stream']==oldreceipt['stream_hashes']['noise_stream'],
                      style_stream=training['stream_hashes']['style_stream']==oldreceipt['stream_hashes']['style_stream'])
    assert all(checks.values()),checks
    fresh=json.loads((ROOT/'FRESH_FINE/FRESH_PROJECTION_STREAM.json').read_text('utf8'))
    assert fresh['calls']==training['updates'];assert fresh['actual_matrix_count']==training['updates']*3
    rec={'actual_rows':n,'actual_G_D_updates':training['updates'],'equal2950_updates':training['updates']==2950,
         'exact_R26_actual_row_order_random_bindings':checks,'fresh_G_calls':fresh['calls'],
         'fresh_matrix_count':fresh['actual_matrix_count'],'same_directions_two_domains_each_call':all(v['same_draw_for_GENERATED_and_REAL'] for v in fresh['per_call_actual_float32_versions']),
         'fresh_stream_consumers_validation_D_identity_evaluation':0,'same_NCE_positions_and_same_assigned_original_rows':True}
    write('CONSUMED_EXPOSURE_AUDIT.json',rec);return rec

def c2st(domains,old,alias):
    oldpred=load(pathlib.Path(CFG['R26_root'])/'results/C2ST_PRIVATE.npz')
    selected={p:tuple(oldpred[f'phase{p}_{j}'] for j in range(3)) for p in [0,1]}
    gs=(selected[0][0],selected[1][0]);rs=(selected[0][1],selected[1][1]);results={};paired={};private={}
    saved_matrix=dc.matrix
    def matrix(d,view):
        key={'normalized_u480':'u','seen_reconstruction480':'seen','complement_reconstruction480':'complement'}.get(view)
        return d[key].copy() if key else saved_matrix(d,view)
    dc.matrix=matrix
    oldpure=json.loads((pathlib.Path(CFG['R27_root'])/'C_CLASSIFIER_METRICS.json').read_text('utf8'))['results']
    views=[*CFG['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480']
    for name in ARMS:
        control='HOMO_SELECTED' if name=='FRESH_SELECTED' else 'HOMO_FINAL';results[name]={};paired[name]={}
        d=domains[name];r=domains['REAL']
        assert not set(d['groups'][gs[0]])&set(d['groups'][gs[1]])
        assert not set(r['groups'][rs[0]])&set(r['groups'][rs[1]])
        assert d['valid'][np.r_[gs[0],gs[1]]].all() and r['valid'][np.r_[rs[0],rs[1]]].all()
        for view in views:
            slot=name+'__'+view
            if name in alias and alias[name] in results and view in results[alias[name]]:
                canonical=alias[name];row=results[canonical][view];p=load(OUT/(canonical+'__'+view+'_PREDICTIONS_PRIVATE.npz'))
            else:
                def persist_fit(row,p):
                    np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**p)
                    write('results/C_'+slot+'.json',row);progress()
                dc.PERSIST_HGB_CALLBACK=persist_fit
                row,p=dc.fit_one(d,r,gs,rs,view,CFG,COUNTER)
            np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**p)
            write('results/C_'+slot+'.json',row);results[name][view]=row
            if view in CFG['C2ST_views']:
                cp={k:oldpred[control+'__'+view+'__'+k] for k in ['probability','label','cluster']}
                control_result=old['C2ST']['results'][control][view]
            else:
                cp=load(pathlib.Path(CFG['R27_root'])/'results'/(control+'__'+view+'_PREDICTIONS_PRIVATE.npz'))
                control_result=oldpure[control][view]
            matched=np.array_equal(p['label'],cp['label']) and np.array_equal(p['cluster'],cp['cluster'])
            paired[name][view]=aa.paired_bootstrap(p,cp,CFG) if matched else {'defined':False,'reason':'labels/groups do not match exactly'}
            paired[name][view]['old_control']=control;paired[name][view]['old_control_result']=control_result
            private.update({slot+'__'+k:p[k] for k in ['probability','label','cluster']})
            write('C2ST_RUNNING.json',{'results':results,'paired_vs_R26':paired,'counters':COUNTER})
            log('C2ST '+slot+' BA '+str(row['balanced_accuracy']))
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{f'phase{p}_{j}':v for p,values in selected.items() for j,v in enumerate(values)})
    return {'results':results,'paired_vs_corresponding_R26':paired,'reference_results_reused':True,
            'original_actual_R26_selection_no_resplit':True,'train_each_domain':len(gs[0]),'holdout_each_domain':len(gs[1]),
            'seven_historical_and_three_pure480_views':True,'fixed_classifier_CI_not_training_uncertainty':True,
            'REAL_REAL_reference':old['C2ST']['RR_definition']}

def widths(domains,old,alias):
    tables={n:old['energy_conditional_width']['tables'][n] for n in ['REAL','SOURCE_CLEAN',*CONTROLS]};grid=np.linspace(-.1,.6,129)
    for name in ARMS:
        if name in alias and alias[name] in tables:tables[name]=tables[alias[name]];continue
        d=domains[name];good=d['valid'];x=d['features']['PSD'][good];q=d['features']['Qtotal_mVns'][good]
        z,b=ew.bin_indices(q,EDGES);weights=np.ones(len(x),'f8');rows=[]
        for k in range(8):
            choose=b==k;finite=np.isfinite(x[choose])&np.isfinite(z[choose]);attempt=int(finite.sum()>=40)
            assert COUNTER['density_EM_attempts']+attempt<=16
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
    for label,names in [('all_endpoints',[*CONTROLS,*ARMS]),('selected',['HOMO_SELECTED','FRESH_SELECTED']),('final_equal_updates',['HOMO_FINAL','FRESH_FINAL'])]:
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
    real=load(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz');ow=load(pathlib.Path(CFG['R26_root'])/'results/PLOT_WAVEFORMS_PRIVATE.npz')
    domains={'REAL':aa.data(real['real_V'],real['real_ids'],real['real_groups'],'REAL',ref)}
    domains['REAL']['valid']&=real['real_QC_valid']
    for n in ['SOURCE_CLEAN',*CONTROLS]:domains[n]=aa.data(ow[n],ids,np.arange(2000),'G4_SOURCE',ref)
    rr=aa.data(real['reference_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
    generation={};repeats={n:old['random_decomposition'][n] for n in CONTROLS};probe={n:old['source_probes'][n] for n in CONTROLS};states={};alias={};evidence={}
    for name in ARMS:
        log('CPU main '+name);(ROOT/name).mkdir(exist_ok=True);model=Translator()
        states[name]=torch.load(CFG['endpoint_weight_paths'][name],map_location='cpu',weights_only=True);model.load_state_dict(states[name]);model.eval()
        pre,signal,electronic=infer(model,source,actual['z'],actual['white'],actual['style'],CFG['generation_chunk']);post,ad=adc(pre);COUNTER['main_generation_rows']+=2000
        np.savez_compressed(ROOT/name/'generated.npz',pre_ADC_V=pre,waveforms_V=post,signal_component_V=signal,electronic_component_V=electronic,source_ids=ids)
        domains[name]=aa.data(post,ids,np.arange(2000),'G4_SOURCE',ref)
        for p in [x for x in ARMS if x!=name and x in domains]:
            if equal_states(states[name],states[p]) and np.array_equal(post,domains[p]['V']):
                alias[name]=p;evidence[name]={'canonical':p,'state_all_tensors_equal':True,'actual2000_output_bitwise_equal':True};break
        generation[name]={'rows':2000,'all_original_source_ids_preserved':bool(np.array_equal(ids,meta['source_ids'])),
                          'ADC':ad,'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),
                          'style_reader_used':False,'signal_not_conditional_expectation':True,'weight_file':CFG['endpoint_weight_paths'][name],'alias':alias.get(name)}
        if name in alias:probe[name]=probe[alias[name]];repeats[name]=repeats[alias[name]]
        else:probe[name]=common.probes(model,post,actual['style'],name);repeats[name]=common.joint_repeats(model,source,comp,name)
        write('GENERATION_PROGRESS.json',{'generation':generation,'aliases':alias,'counters':COUNTER});del model
    stats,distance=ev.characterize(domains,ref);psd=common.psd_metrics(domains);width=widths(domains,old,alias)
    for d in [*domains.values(),rr]:attach_temporal(d)
    maps=load(pathlib.Path(CFG['R27_root'])/'results/PROJECTION_SPACE_PRIVATE.npz')
    seen=maps['Pseen'];complement=maps['Pcomplement']
    for d in [*domains.values(),rr]:d['seen']=d['u']@seen;d['complement']=d['u']@complement
    cs=c2st(domains,old,alias)
    constants=load(CFG['training_input_paths']['FINE_CONSTANTS.npz']);tdomains={n:domains[n] for n in ['REAL',*CONTROLS,*ARMS]};tdomains['TRAIN_REFERENCE']=rr
    coverage=temporal.means_coverage(tdomains,seen,complement,domains['SOURCE_CLEAN']['b'])
    projection=temporal.fine_distances(tdomains,constants)
    frozen=json.loads((ROOT/'FROZEN_TILT_TABLE.json').read_text('utf8'));lambdas=np.asarray([r['lambda_value'] for r in frozen['bins']]);h=.15*meta['edep_hadron_MeV']/(meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV'])
    composition=common.source_composition(domains,comp,h,lambdas)
    association={n:ds.association({'SOURCE_CLEAN':domains['SOURCE_CLEAN'],'TRANSLATED':domains[n]},comp,pdg,EDGES) for n in [*CONTROLS,*ARMS]}
    local={n:dc.native_statistics(d,ref) for n,d in domains.items()};feat={};waves={}
    for name,d in domains.items():
        feat.update({name+'__'+key:a for key,a in {**d['features'],**d['extra']}.items()});feat[name+'__valid']=d['valid'];feat[name+'__region']=d['region'];waves[name]=d['V'].astype('f4')
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**feat);np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**waves)
    training=json.loads((ROOT/'FRESH_FINE/TRAINING_RECEIPT.json').read_text('utf8'));exposure=consumed_audit(training)
    online={'FRESH_FINE':common.online_auc('FRESH_FINE'),'HOMOGENEOUS_REORDER':old['online_D_validation_AUC']['HOMOGENEOUS_REORDER']}
    metric={'status':'complete','training':{'FRESH_FINE':training},'control_training_reused':old['training']['HOMOGENEOUS_REORDER'],
            'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text('utf8')),'domains':stats,'domain_vs_REAL':distance,'local':local,'PSD_main':psd,
            'energy_conditional_width':width,'source_composition':composition,'source_association':association,'source_probes':probe,
            'random_decomposition':repeats,'C2ST':cs,'generation':generation,'online_D_validation_AUC':online,'row_exposure':exposure,
            'endpoint_aliases':alias,'alias_evidence':evidence,'projection_coverage':coverage,'projection_distances':projection,'counters':COUNTER,
            'versions':{'Python':platform.python_version(),'torch':torch.__version__,'NumPy':np.__version__,'sklearn':sklearn.__version__},
            'primary_DEV_population_reweighted':False,'EVAL_read':0,'CPU_main_and_replay_same_environment':True,'old_control_retrained':False,
            'sole_training_intervention':'fresh independent fine projection directions once per G update','endpoint_selection_uses_DEV':False}
    write('metrics.json',metric);write('ENDPOINT_ALIAS_RECEIPT.json',{'aliases':alias,'evidence':evidence,'counts':COUNTER})
    assert COUNTER['HGB_attempts']<=20 and COUNTER['density_EM_attempts']<=16 and COUNTER['main_generation_rows']<=4000 and COUNTER['repeat_noise_rows']<=2048 and COUNTER['source_probe_rows']<=1664
    write('EVALUATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                          'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete','threads':4})
    log('R28 evaluation complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:
    write('EVALUATION_FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),
          'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
