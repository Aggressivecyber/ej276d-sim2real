from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,ast,hashlib,platform,shutil
import numpy as np,torch,sklearn
from scipy.stats import wasserstein_distance,ks_2samp
from sklearn.metrics import roc_auc_score
import evaluation as ev,diagnostic_core as dc,assessment as aa,data_support as ds
import endpoint_common as common,temporal_common as temporal
from random_diagnostics import summarize
from mean_description import describe as describe_means
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results';START=time.monotonic();CFG=json.loads((ROOT/'config.json').read_text('utf8'));ARMS=CFG['endpoints'];NAMES=ARMS
VALLEY=json.loads((ROOT/'VALLEY_DEFINITION.json').read_text('utf8'));EDGES=np.asarray(json.loads(pathlib.Path(CFG['local_input_paths']['width_definition']).read_text('utf8'))['edges_logQ'])
COUNTER=json.loads((ROOT/'EVALUATION_FAILURE_012533_554982.json').read_text('utf8'))['counters'].copy();torch.set_num_threads(4);torch.set_num_interop_threads(4)
def write(n,v):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def qb(q):
    l=np.log(np.maximum(q,1e-30));b=np.searchsorted(EDGES,l,side='right')-1;b[l==EDGES[-1]]=7;return b
def attach(d):
    d['s']=d['V']-d['V'][:,:100].mean(1,keepdims=True);d['p']=np.maximum(d['s'].max(1,keepdims=True),.0025);d['u']=d['s']/d['p'];d['q']=d['features']['Qtotal_mVns'];d['b']=qb(d['q']);return d
for n in ['ROOT','OUT','CFG','VALLEY','EDGES','COUNTER','NAMES','ev','dc','aa','ds','summarize','load','qb','write','roc_auc_score','torch','np','pathlib','json','hashlib','wasserstein_distance','ks_2samp']:setattr(common,n,globals()[n])
# Reuse the verified same-row audit; importing the original main would execute it.
tree=ast.parse((ROOT/'evaluate_endpoints.py').read_text('utf8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='consumed_audit');exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'saved_consumed_audit','exec'),globals())
ref=json.loads((ROOT/'CAL_REFERENCE_METADATA.json').read_text('utf8'));actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');meta=load(ROOT/'DEV_SOURCE_METADATA.npz');real=load(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz');old=json.loads((pathlib.Path(CFG['R26_root'])/'metrics.json').read_text('utf8'))
ids=actual['source_ids'];comp=meta['composition'];pdg=meta['pdg_inferred'];assert np.array_equal(ids,meta['source_ids'])
assert {str(k):int((comp==k).sum())for k in np.unique(comp)}=={'mixed':65,'pure_EE':417,'pure_had':1518}
saved_source=load(pathlib.Path(CFG['R26_root'])/'results/PLOT_WAVEFORMS_PRIVATE.npz',keys=['SOURCE_CLEAN'])['SOURCE_CLEAN']
domains={'REAL':aa.data(real['real_V'],real['real_ids'],real['real_groups'],'REAL',ref),'SOURCE_CLEAN':aa.data(saved_source,ids,np.arange(2000),'G4_SOURCE',ref)};domains['REAL']['valid']&=real['real_QC_valid']
source_input=aa.data(actual['source_waveforms_V'],ids,np.arange(2000),'ACTUAL_NETWORK_SOURCE_INPUT',ref)
expected=[int(((comp=='pure_had')&(qb(domains['SOURCE_CLEAN']['features']['Qtotal_mVns'])==i)).sum())for i in [2,3]]
assert expected==[207,182],('restored_frozen_source_condition_counts',expected)
source_binding={'all_original_ID_order_equal':True,'composition_counts':{'pure_EE':417,'pure_had':1518,'mixed':65},'primary_source_bin3_4_pure_had_N':expected,'source_condition_reference':str(pathlib.Path(CFG['R26_root'])/'results/PLOT_WAVEFORMS_PRIVATE.npz')+'::SOURCE_CLEAN','actual_network_input':str(ROOT/'ACTUAL_GENERATION_INPUTS.npz')+'::source_waveforms_V','same_reference_as_original_R28_entry':True,'reference_not_claimed_equal_to_actual_network_input':True,'first_attempt_used_wrong_SOURCE_CLEAN_domain_preserved':True,'no_training_or_main_generation_input_changed':True,'no_fit_or_generation_repeated':True}
write('SOURCE_REFERENCE_CORRECTION.json',source_binding)
generation={};probes={};repeats={};private={};waves={};association={}
for name in ARMS:
    a=load(ROOT/name/'generated.npz');assert np.array_equal(a['source_ids'],ids);domains[name]=aa.data(a['waveforms_V'],ids,np.arange(2000),'G4_SOURCE',ref)
    generation[name]={'rows':2000,'all_original_source_ids_preserved':True,'ADC':{'clip_count':int(((a['pre_ADC_V']<-.5)|(a['pre_ADC_V']>.5)).sum()),'pre_ADC_finite':bool(np.isfinite(a['pre_ADC_V']).all()),'bits':14,'Vrange':[-.5,.5]},'pre_ADC_sha256':hashlib.sha256(a['pre_ADC_V'].tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(a['waveforms_V'].tobytes()).hexdigest(),'weight_file':CFG['endpoint_weight_paths'][name],'CPU_generated_here':True,'style_reader_used':False,'signal_not_conditional_expectation':True}
    p=load(OUT/(name+'_SOURCE_PROBES_PRIVATE.npz'));pf=ev.features(p['waveforms_V']);bf=ev.features(a['waveforms_V'][p['reference_parent_index']]);tags=p['probe_kind'];rows={}
    for tag in np.unique(tags):
        u=tags==tag;rows[str(tag)]={'N':int(u.sum()),'QC':int(pf['valid'][u].sum()),'waveform_RMS_vs_correct_reference_V':ev.summary(np.sqrt(np.mean((p['waveforms_V'][u]-a['waveforms_V'][p['reference_parent_index'][u]])**2,1))),'PSD_vs_reference':ev.summary(p['PSD'][u]-bf['features']['PSD'][u]),'output_PSD':ev.summary(p['PSD'][u]),'output_Q_mVns':ev.summary(p['Q_mVns'][u])}
    fixed=tags=='fixed_source_changed_noise';same=tags=='same_input_same_noise'
    probes[name]={'rows':rows,'new_generated_rows':len(tags),'same_input_same_noise_bitwise':bool(np.array_equal(p['waveforms_V'][same],a['waveforms_V'][p['source_parent_index'][same]])),'fixed_source_parent_index':int(p['source_parent_index'][fixed][0]),'fixed_source_unique_actual_parents':int(len(np.unique(p['source_parent_index'][fixed]))),'corrected_R21_parent_semantics_reused':True,'nonzero_sensitivity_not_correct_physics':True,'summarized_saved_outputs_only':True}
    j=load(OUT/(name+'_joint8_REPEAT_PRIVATE.npz'));s,jf=summarize(j['waveforms_V'],j['signal_component_V'],j['electronic_component_V'],j['source_parent_index'],8,comp,EDGES,actual['source_waveforms_V'],VALLEY)
    repeats[name]={'groups':{'joint8':s},'actual_rows':1024,'source_parent_union':128,'new_independent_source_parents':0,'actual_R21_joint8_input_reused':True,'style_only_and_electronic_only_new_outputs':0,'summarized_saved_outputs_only':True}
    association[name]=ds.association({'SOURCE_CLEAN':domains['SOURCE_CLEAN'],'TRANSLATED':domains[name]},comp,pdg,EDGES)
stats,distance=ev.characterize(domains,ref);psd=common.psd_metrics(domains);lambdas=np.asarray([x['lambda_value']for x in json.loads((ROOT/'FROZEN_TILT_TABLE.json').read_text('utf8'))['bins']]);h=.15*meta['edep_hadron_MeV']/(meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV']);composition=common.source_composition(domains,comp,h,lambdas)
rr=aa.data(real['reference_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
for d in [*domains.values(),rr]:attach(d)
td={n:domains[n]for n in ['REAL',*ARMS]};td['TRAIN_REFERENCE']=rr;target=load(ROOT/'MEAN_TARGETS_PRIVATE.npz',keys=['mu'])
first_md=json.loads((ROOT/'MEAN_SD_METRICS.json').read_text('utf8'))
for n in ['MEAN_SD_METRICS.json','A_COVERAGE_METRICS.json','results/MEAN_SD_DESCRIPTIONS_PRIVATE.npz','results/TEMPORAL_DIAGNOSTICS_PRIVATE.npz']:
    p=ROOT/n;shutil.copy2(p,p.with_name(p.name+'.attempt001'))
md,mp=describe_means(td,ARMS,domains['SOURCE_CLEAN']['b'],target['mu']);assert all(md['domains'][n]['observed_Q']==first_md['domains'][n]['observed_Q']for n in md['domains'])
write('MEAN_SD_METRICS.json',md);np.savez_compressed(OUT/'MEAN_SD_DESCRIPTIONS_PRIVATE.npz',**mp)
coverage=json.loads((ROOT/'A_COVERAGE_METRICS.json').read_text('utf8'));oldtp=load(OUT/'TEMPORAL_DIAGNOSTICS_PRIVATE.npz')
maps=load(pathlib.Path(CFG['R27_root'])/'results/PROJECTION_SPACE_PRIVATE.npz');seen=maps['Pseen'];complement=maps['Pcomplement'];maxrec=0.
# Only source-Q descriptions depend on the corrected frozen source bin.
for name in ARMS:
    d=domains[name];b=domains['SOURCE_CLEAN']['b'];rows=[]
    for i in [None,*range(8)]:
        use=d['valid']if i is None else d['valid']&(b==i);raw=np.ones(len(b),bool)if i is None else b==i;r=domains['REAL'];rmask=r['valid']if i is None else r['valid']&(r['b']==i);n=int(use.sum());nr=int(rmask.sum());label='overall'if i is None else 'bin'+str(i+1)
        row={'bin_id_1based':None if i is None else i+1,'source_count':int(raw.sum()),'QC_n':n,'invalid_n':int((raw&~d['valid']).sum()),'REAL_observed_Q_QC_n':nr,'qualified':n>=40 and nr>=40,'reason':None if n>=40 and nr>=40 else'a_domain_below40','conditioning':'candidate source Q vs REAL observed Q; not identical conditioning'}
        if n:
            mean=d['u'][use].mean(0);med=np.median(d['u'][use],axis=0);oldtp[name+'__source_Q__'+label+'__mean_u']=mean;oldtp[name+'__source_Q__'+label+'__median_u']=med
            if nr:
                delta=mean-r['u'][rmask].mean(0);ss=delta@seen;cc=delta@complement;total=float(delta@delta);st=float(ss@ss);ct=float(cc@cc);maxrec=max(maxrec,float(np.max(np.abs(ss+cc-delta))))
                row.update(mean_difference_sqnorm=total,seen_sqnorm=st,complement_sqnorm=ct,seen_fraction=st/total if total else None,complement_fraction=ct/total if total else None,energy_closure_error=st+ct-total)
                for suffix,val in [('delta_u',delta),('delta_seen',ss),('delta_complement',cc)]:oldtp[name+'__source_Q__'+label+'__'+suffix]=val
        rows.append(row)
    coverage[name]['source_Q']=rows
assert maxrec<1e-10;write('A_COVERAGE_METRICS.json',coverage);np.savez_compressed(OUT/'TEMPORAL_DIAGNOSTICS_PRIVATE.npz',**oldtp)
tables={n:old['energy_conditional_width']['tables'][n]for n in ['REAL','SOURCE_CLEAN']}
tables.update({n:json.loads((OUT/('WIDTH_'+n+'_TABLE.json')).read_text('utf8'))for n in ARMS});errors={}
for label,names in [('all_endpoints',ARMS),('selected',['CONTROL_SELECTED','MEAN_SELECTED']),('final_equal_updates',['CONTROL_FINAL','MEAN_FINAL'])]:
    rows=[]
    for k in [0,1]:
        for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
            bins=[i for i in range(8)if all(tables[n]['slices'][i]['valid']and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field]is not None for n in ['REAL',*names])]
            rows.append({'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field])for i in bins]))if bins else None for n in names}})
    errors[label]=rows
width={'tables':tables,'common_support_errors':errors,'edges_logQ':EDGES,'old_reference_fits_reused':True,'all_slots_persisted_before_next':True,'density_components_not_particle_truth':True,'assembled_existing_slots_no_refit':True}
cs=json.loads((ROOT/'C2ST_RUNNING.json').read_text('utf8'));paired={}
for label,m,c in [('final_equal_updates','MEAN_FINAL','CONTROL_FINAL'),('selected','MEAN_SELECTED','CONTROL_SELECTED')]:
    paired[label]={v:aa.paired_bootstrap(load(OUT/(m+'__'+v+'_PREDICTIONS_PRIVATE.npz')),load(OUT/(c+'__'+v+'_PREDICTIONS_PRIVATE.npz')),CFG)for v in cs['results'][m]}
cs.update(paired_MEAN_minus_CONTROL=paired,original_actual_selection_no_resplit=True,train_each_domain=600,holdout_each_domain=250,seven_historical_and_three_pure480_views=True,fixed_classifier_CI_not_training_uncertainty=True,REAL_REAL_reference=old['C2ST']['RR_definition'],R28_historical_C2ST_results=json.loads((pathlib.Path(CFG['R28_root'])/'metrics.json').read_text('utf8'))['C2ST']['results'],cross_platform_historical_not_new_loss_effect=True,no_HGB_refit=True)
local={n:dc.native_statistics(d,ref)for n,d in domains.items()}
for name,d in domains.items():
    private.update({name+'__'+key:a for key,a in{**d['features'],**d['extra']}.items()});private[name+'__valid']=d['valid'];private[name+'__region']=d['region'];waves[name]=d['V'].astype('f4')
np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**private);np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**waves)
training={n:json.loads((ROOT/n/'TRAINING_RECEIPT.json').read_text('utf8'))for n in CFG['arms']};exposure=consumed_audit(training);online={n:common.online_auc(n)for n in CFG['arms']}
metric={'status':'complete_after_saved_output_summary_recovery','training':training,'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text('utf8')),'domains':stats,'domain_vs_REAL':distance,'local':local,'PSD_main':psd,'energy_conditional_width':width,'source_composition':composition,'source_association':association,'source_probes':probes,'random_decomposition':repeats,'C2ST':cs,'generation':generation,'online_D_validation_AUC':online,'row_exposure':exposure,'endpoint_aliases':{},'alias_evidence':{},'projection_coverage':coverage,'projection_distances':json.loads((ROOT/'B_PROJECTION_METRICS.json').read_text('utf8')),'conditional_mean_SD':md,'counters':COUNTER,'versions':{'Python':platform.python_version(),'torch':torch.__version__,'NumPy':np.__version__,'sklearn':sklearn.__version__},'primary_DEV_population_reweighted':False,'EVAL_read':0,'CPU_main_and_replay_same_environment':True,'two_current_server_arms_retrained_as_registered':True,'sole_training_intervention':'conditional mean U coefficient1','endpoint_selection_uses_DEV':False,'mean_target_is_TRAIN_only':True,'source_reference_correction':source_binding,'actual_network_source_input_feature_summary':{k:ev.summary(v)for k,v in source_input['features'].items()},'saved_output_recovery_new_fits_or_generations':0}
write('metrics.json',metric);write('EVALUATION_RECOVERY_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'new_HGB_EM_response_inference':0,'preserved_attempt001_files':True,'restored_original_SOURCE_CLEAN_reference':True,'observation_condition_descriptions_exactly_unchanged':True,'original_cumulative_counters':COUNTER,'source_population_assertions_pass':True})
print(json.dumps({'summary_complete':True,'PID':os.getpid(),'counters':COUNTER,'source_bins3_4':expected,'common_mean_bins':md['common_observed_Q_bin_ids_1based'],'common_mean_MSE':md['common_bin_equal_mean_full480_MSE']}),flush=True)
