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
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);START=time.monotonic()
CFG=json.loads((ROOT/'config.json').read_text('utf8'));ARMS=CFG['endpoints']
VALLEY=json.loads((ROOT/'VALLEY_DEFINITION.json').read_text('utf8'));EDGES=np.asarray(json.loads(pathlib.Path(CFG['local_input_paths']['width_definition']).read_text('utf8'))['edges_logQ'])
PRIOR_FAILURE=json.loads((ROOT/'EVALUATION_FAILURE_210037_216109.json').read_text('utf8'));COUNTER=dict(PRIOR_FAILURE['counters'])
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
def log(x):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),x,flush=True)
def write(n,v):(ROOT/n).write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
def load(path,keys=None):
    with np.load(path,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def qb(q):return np.searchsorted(EDGES,np.log(np.maximum(q,1e-30)),side='right')-1
for n in ['ROOT','OUT','CFG','VALLEY','EDGES','COUNTER','ev','dc','aa','ew','ds','infer','adc','summarize','load','qb','write','log','roc_auc_score','torch','np','pathlib','json','hashlib','wasserstein_distance','ks_2samp']:
    setattr(common,n,globals()[n])
def equal_states(a,b):return list(a)==list(b)and all(torch.equal(a[k],b[k])for k in a)

def c2st(domains,rr,selection,old,alias):
    original={k:(np.asarray(v['train'],dtype='i4'),np.asarray(v['holdout'],dtype='i4'))for k,v in selection['indices'].items()};chosen={};counts={}
    for phase in [0,1]:
        g=original['GEN'][phase];g=g[np.logical_and.reduce([domains[n]['valid'][g]for n in ['SOURCE_CLEAN','STANDARDIZED',*ARMS]])]
        r=original['REAL'][phase];r=r[domains['REAL']['valid'][r]];t=original['TRAIN_REFERENCE'][phase];t=t[rr['valid'][t]];n=min(len(g),len(r),len(t));chosen[phase]=(g[:n],r[:n],t[:n]);counts['train'if phase==0 else'holdout']={'N_each':n,'source_common_QC':len(g),'REAL':len(r),'TRAIN_REFERENCE':len(t)}
    oldp=load(pathlib.Path(CFG['R23_root'])/'results/C2ST_PRIVATE.npz');same_old=all(np.array_equal(oldp[f'phase{p}_{j}'],chosen[p][j])for p in [0,1]for j in [0,1,2])
    result={'STANDARDIZED':old['C2ST']['results']['STANDARDIZED'],'SOURCE_CLEAN':old['C2ST']['reference_results_reused']['SOURCE_CLEAN']};pred={};private={};pair={};finalpair={}
    supported=counts['train']['N_each']>=150 and counts['holdout']['N_each']>=75
    if supported:
        gs=(chosen[0][0],chosen[1][0]);rs=(chosen[0][1],chosen[1][1])
        for name in ARMS:
            if name in alias:
                a=alias[name];result[name]=result[a];pred[name]={v:({k:oldp['STANDARDIZED__'+v+'__'+k]for k in ['probability','label','cluster']}if a=='STANDARDIZED'else pred[a][v])for v in CFG['C2ST_views']}
                continue
            result[name]={};pred[name]={}
            assert not set(domains[name]['groups'][gs[0]])&set(domains[name]['groups'][gs[1]])
            assert not set(domains['REAL']['groups'][rs[0]])&set(domains['REAL']['groups'][rs[1]])
            for view in CFG['C2ST_views']:
                result[name][view],pred[name][view]=dc.fit_one(domains[name],domains['REAL'],gs,rs,view,CFG,COUNTER)
                log('C2ST '+name+' '+view+' '+str(result[name][view]['balanced_accuracy']))
        for name in ARMS:
            pair[name]={}
            for view in CFG['C2ST_views']:
                p=pred[name][view];control={k:oldp['STANDARDIZED__'+view+'__'+k]for k in ['probability','label','cluster']}
                if same_old and np.array_equal(p['label'],control['label'])and np.array_equal(p['cluster'],control['cluster']):pair[name][view]=aa.paired_bootstrap(p,control,CFG)
                else:pair[name][view]={'defined':False,'reason':'fixed selection/group mismatch'}
                private.update({name+'__'+view+'__'+k:v for k,v in p.items()})
        for view in CFG['C2ST_views']:
            finalpair[view]=aa.paired_bootstrap(pred['HOMO_FINAL'][view],pred['R23_D1_FINAL2950'][view],CFG)
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{'phase'+str(p)+'_'+str(j):a for p,v in chosen.items()for j,a in enumerate(v)})
    return{'supported':supported,'counts':counts,'results':result,'endpoint_minus_R23_D1_SELECTED_paired_CI':pair,'HOMO_FINAL_minus_R23_D1_FINAL_paired_CI':finalpair,'same_old_selection':same_old,'reference_results_reused':True,'RR_definition':old['C2ST']['RR_definition'],'aliases_reused_only_after_state_and_output_evidence':alias,'fixed_classifier_CI_not_training_uncertainty':True}

def widths(domains,old,alias):
    tables={'REAL':old['energy_conditional_width']['tables']['REAL'],'SOURCE_CLEAN':old['energy_conditional_width']['tables']['SOURCE_CLEAN'],'STANDARDIZED':old['energy_conditional_width']['tables']['STANDARDIZED']}
    for name in ARMS:
        if name in alias:tables[name]=tables[alias[name]];continue
        d=domains[name];good=d['valid'];COUNTER['density_description_slots']+=8;tables[name]=ew.section(d['features']['PSD'][good],d['features']['Qtotal_mVns'][good],EDGES);COUNTER['density_EM_attempts']+=sum(r['fit_attempts']for r in tables[name]['slices'])
        assert COUNTER['density_EM_attempts']<=24
    common_errors={}
    for label,names in [('all_endpoints',['STANDARDIZED',*ARMS]),('new_vs_old_final',['HOMO_FINAL','R23_D1_FINAL2950']),('new_selected_vs_old_selected',['STANDARDIZED','HOMO_SELECTED'])]:
        rows=[]
        for k in[0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8)if all(tables[n]['slices'][i]['valid']and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field]is not None for n in ['REAL',*names])]
                rows.append({'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field])for i in bins]))if bins else None for n in names}})
        common_errors[label]=rows
    return{'tables':tables,'common_support_errors':common_errors,'edges_logQ':EDGES,'old_reference_fits_reused':True,'aliases':alias,'density_components_not_particle_truth':True}

def row_exposure_audit(training):
    actual=load(ROOT/'HOMOGENEOUS_REORDER/TRAINING_STREAMS_PRIVATE.npz');old=load(CFG['actual_training_stream']);perm=actual['original_row_ordinal'];n=len(perm);order=np.argsort(perm)
    checks={k:bool(np.array_equal(actual[k],old[k][perm]))for k in old if k!='NCE_positions'}
    checks['NCE_original_update_positions']=bool(np.array_equal(actual['NCE_positions'],old['NCE_positions'][:training['updates']*3]))
    allrows=n==94400 and np.array_equal(np.sort(perm),np.arange(94400));perbin=[]
    for b in range(8):
        use=actual['bin']==b;expected=old['bin']==b
        perbin.append({'bin_id_1based':b+1,'new_rows':int(use.sum()),'old_rows':int(expected.sum()),'new_unique_source':len(np.unique(actual['source'][use])),'old_unique_source':len(np.unique(old['source'][expected])),'new_unique_REAL':len(np.unique(actual['target'][use])),'old_unique_REAL':len(np.unique(old['target'][expected]))})
    result={'actual_rows':n,'full_94400_row_multiset_trained':allrows,'row_binding_checks':checks,'actual_updates_equal_old2950':training['updates']==2950,'per_bin_exposure':perbin,'no_per_update_pairing_claim':True,'ordinal_sha256':hashlib.sha256(perm.tobytes()).hexdigest()}
    assert all(checks.values());write('CONSUMED_EXPOSURE_AUDIT.json',result);return result

def saved_probes(name,main):
    a=load(ROOT/'PROBE_INPUTS_CORRECTED.npz');p=load(OUT/(name+'_SOURCE_PROBES_PRIVATE.npz'));post=p['waveforms_V'];f=ev.features(post);sf=ev.features(a['source_waveforms_V']);tags=a['probe_kind'];baseline=main[a['reference_parent_index']];bf=ev.features(baseline);rows={}
    assert np.array_equal(p['source_parent_index'],a['source_parent_index'])and np.array_equal(p['noise_origin_parent_index'],a['noise_origin_parent_index'])and np.array_equal(p['reference_parent_index'],a['reference_parent_index'])
    for tag in np.unique(tags):
        use=tags==tag;rows[str(tag)]={'N':int(use.sum()),'QC':int(f['valid'][use].sum()),'waveform_RMS_vs_correct_reference_V':ev.summary(np.sqrt(np.mean((post[use]-baseline[use])**2,1))),'PSD_vs_reference':ev.summary(f['features']['PSD'][use]-bf['features']['PSD'][use]),'source_PSD':ev.summary(sf['features']['PSD'][use]),'output_PSD':ev.summary(f['features']['PSD'][use]),'output_Q_mVns':ev.summary(f['features']['Qtotal_mVns'][use])}
    same=tags=='same_input_same_noise';fixed=tags=='fixed_source_changed_noise'
    return{'rows':rows,'new_generated_rows':len(post),'same_input_same_noise_bitwise':bool(np.array_equal(post[same],main[a['source_parent_index'][same]])),'fixed_source_parent_index':int(a['source_parent_index'][fixed][0]),'fixed_source_unique_actual_parents':int(len(np.unique(a['source_parent_index'][fixed]))),'corrected_R21_parent_semantics_reused':True,'nonzero_sensitivity_not_correct_physics':True,'summary_from_saved_attempt1_only':True}
def saved_joint(name,source,comp):
    p=load(OUT/(name+'_joint8_REPEAT_PRIVATE.npz'));parent=p['source_parent_index'];assert len(parent)==1024 and np.all(parent.reshape(128,8)==parent.reshape(128,8)[:,:1])
    summary,f=summarize(p['waveforms_V'],p['signal_component_V'],p['electronic_component_V'],parent,8,comp,EDGES,source,VALLEY)
    return{'groups':{'joint8':summary},'actual_rows':1024,'source_parent_union':128,'new_independent_source_parents':0,'actual_R21_joint8_input_reused':True,'style_only_and_electronic_only_new_outputs':0,'summary_from_saved_attempt1_only':True}
def unavailable_widths(old,alias):
    tables={'REAL':old['energy_conditional_width']['tables']['REAL'],'SOURCE_CLEAN':old['energy_conditional_width']['tables']['SOURCE_CLEAN'],'STANDARDIZED':old['energy_conditional_width']['tables']['STANDARDIZED']}
    for n in ARMS:
        if n in alias:tables[n]=tables[alias[n]]
        else:tables[n]={'slices':[{'bin_id_1based':i+1,'valid':False,'components':[],'fit_attempts':None,'reason':'engineering_delivery_unavailable: completed fit summary lost before persistence; not a scientific fit failure'}for i in range(8)],'scientific_fit_success_or_failure':'unknown because unsaved summary','no_repeat_fit':True}
    return{'tables':tables,'common_support_errors':{},'edges_logQ':EDGES,'old_reference_fits_reused':True,'aliases':alias,'density_components_not_particle_truth':True,'delivery_unavailable':True,'completed_EM_attempts_preserved':PRIOR_FAILURE['counters']['density_EM_attempts'],'no_new_EM_in_continuation':True}

def run():
    old=json.loads((pathlib.Path(CFG['R23_root'])/'metrics.json').read_text('utf8'));ref=json.loads((ROOT/'CAL_REFERENCE_METADATA.json').read_text('utf8'));actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');meta=load(ROOT/'DEV_SOURCE_METADATA.npz');ids=actual['source_ids'];source=actual['source_waveforms_V'];comp=meta['composition'];pdg=meta['pdg_inferred'];real=load(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz');ow=load(pathlib.Path(CFG['R23_root'])/'results/PLOT_WAVEFORMS_PRIVATE.npz')
    domains={'REAL':aa.data(real['real_V'],real['real_ids'],real['real_groups'],'REAL',ref)}
    for n in ['SOURCE_CLEAN','STANDARDIZED']:domains[n]=aa.data(ow[n],ids,np.arange(2000),'G4_SOURCE',ref)
    domains['REAL']['valid']&=real['real_QC_valid'];rr=aa.data(real['reference_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
    generation={};repeats={'STANDARDIZED':old['random_decomposition']['STANDARDIZED']};probe={'STANDARDIZED':old['source_probes']['STANDARDIZED']};states={'STANDARDIZED':torch.load(pathlib.Path(CFG['R23_root'])/'STANDARDIZED/TRANSLATOR.pt',map_location='cpu',weights_only=True)};alias={};aliases_evidence={}
    for name in ARMS:
        log('reuse saved main/probes/joint '+name);saved=load(ROOT/name/'generated.npz');pre=saved['pre_ADC_V'];post=saved['waveforms_V'];signal=saved['signal_component_V'];electronic=saved['electronic_component_V'];assert np.array_equal(saved['source_ids'],ids)
        states[name]=torch.load(CFG['endpoint_weight_paths'][name],map_location='cpu',weights_only=True);domains[name]=aa.data(post,ids,np.arange(2000),'G4_SOURCE',ref)
        previous=['STANDARDIZED']+[x for x in ARMS if x!=name and x in domains]
        for p in previous:
            if equal_states(states[name],states[p])and np.array_equal(post,domains[p]['V']):alias[name]=alias.get(p,p);aliases_evidence[name]={'canonical_endpoint':alias[name],'state_all_tensors_equal':True,'actual2000_postADC_bitwise_equal':True,'rows_generated_before_alias':2000};break
        step=(1.-(-1.))/2**14;code=np.floor((pre.astype('f8')+.5)/step);ad={'clip_low_samples':int((code<0).sum()),'clip_high_samples':int((code>=2**14).sum()),'from_saved_pre_ADC_no_new_ADC_generation':True}
        generation[name]={'rows':2000,'all_original_source_ids_preserved':bool(np.array_equal(ids,meta['source_ids'])),'ADC':ad,'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),'style_reader_used':False,'signal_not_conditional_expectation':True,'weight_file':CFG['endpoint_weight_paths'][name],'alias':alias.get(name),'post_ADC_bitwise_vs_frozen_D1_selected':bool(np.array_equal(post,ow['STANDARDIZED'])),'maxabs_vs_frozen_D1_selected_V':float(np.max(np.abs(post.astype('f8')-ow['STANDARDIZED'].astype('f8')))),'generation_from_attempt1_saved_payload':True}
        if name in alias:probe[name]=probe[alias[name]];repeats[name]=repeats[alias[name]]
        else:probe[name]=saved_probes(name,post);repeats[name]=saved_joint(name,source,comp)
    ns,nd=ev.characterize(domains,ref);stats=ns;distance=nd;psd=common.psd_metrics(domains);width=unavailable_widths(old,alias);selection=json.loads((ROOT/'EVALUATOR_SELECTIONS_PRIVATE.json').read_text('utf8'));cs=c2st(domains,rr,selection,old,alias)
    frozen=json.loads((ROOT/'FROZEN_TILT_TABLE.json').read_text('utf8'));lambdas=np.asarray([r['lambda_value']for r in frozen['bins']]);h=.15*meta['edep_hadron_MeV']/(meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV']);composition=common.source_composition(domains,comp,h,lambdas)
    association={n:ds.association({'SOURCE_CLEAN':domains['SOURCE_CLEAN'],'TRANSLATED':domains[n]},comp,pdg,EDGES)for n in ['STANDARDIZED',*ARMS]};local={n:dc.native_statistics(d,ref)for n,d in domains.items()};feat={};waves={}
    for name,d in domains.items():
        feat.update({name+'__'+key:a for key,a in{**d['features'],**d['extra']}.items()});feat[name+'__valid']=d['valid'];feat[name+'__region']=d['region'];waves[name]=d['V'].astype('f4')
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**feat);np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**waves)
    training=json.loads((ROOT/'HOMOGENEOUS_REORDER/TRAINING_RECEIPT.json').read_text('utf8'));exposure=row_exposure_audit(training);online={'HOMOGENEOUS_REORDER':common.online_auc('HOMOGENEOUS_REORDER'),'STANDARDIZED':old['online_D_validation_AUC']['STANDARDIZED']}
    metric={'status':'complete','training':{'HOMOGENEOUS_REORDER':training},'control_training_reused':old['training']['STANDARDIZED'],'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text('utf8')),'domains':stats,'domain_vs_REAL':distance,'local':local,'PSD_main':psd,'energy_conditional_width':width,'source_composition':composition,'source_association':association,'source_probes':probe,'random_decomposition':repeats,'C2ST':cs,'generation':generation,'online_D_validation_AUC':online,'row_exposure':exposure,'endpoint_aliases':alias,'alias_evidence':aliases_evidence,'reorder_construction':json.loads((ROOT/'REORDER_CONSTRUCTION_RECEIPT.json').read_text('utf8')),'counters':COUNTER,'versions':{'Python':platform.python_version(),'torch':torch.__version__,'NumPy':np.__version__,'sklearn':sklearn.__version__},'critic_center_scale_combo_not_isolated':True,'primary_DEV_population_reweighted':False,'EVAL_read':0,'CPU_main_and_replay_same_environment':True,'old_control_retrained':False,'engineering_failure_preserved':PRIOR_FAILURE,'continuation_new_waveform_generation':0,'density_results_delivery_missing':True,'sole_training_intervention':'complete bound-row permutation/batch reorganization','endpoint_selection_uses_DEV':False}
    write('metrics.json',metric);write('ENDPOINT_ALIAS_RECEIPT.json',{'aliases':alias,'evidence':aliases_evidence,'reused_downstream_evaluations':True,'main_generation_always_actual_for_each_endpoint':True,'counts':COUNTER})
    assert COUNTER['HGB_attempts']<=21 and COUNTER['density_EM_attempts']<=24 and COUNTER['main_generation_rows']==6000 and COUNTER['repeat_noise_rows']<=3072 and COUNTER['source_probe_rows']<=2496
    write('EVALUATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete','threads':4});log('R26 evaluation complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:
    write('EVALUATION_FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
