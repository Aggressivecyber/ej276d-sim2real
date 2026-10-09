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
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);START=time.monotonic()
CFG=json.loads((ROOT/'config.json').read_text('utf8'));ARMS=['D5']
VALLEY=json.loads((ROOT/'VALLEY_DEFINITION.json').read_text('utf8'));EDGES=np.asarray(json.loads(pathlib.Path(CFG['local_input_paths']['width_definition']).read_text('utf8'))['edges_logQ'])
COUNTER={'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'repeat_noise_rows':0,'source_probe_rows':0,'EVAL_read':0,'CAL_waveform_decode':0,'response_updates':0}
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)

def log(x):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),x,flush=True)
def write(name,value):(ROOT/name).write_text(json.dumps(ev.safe(value),ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
def load(path,keys=None):
    with np.load(path,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def qb(q):return np.searchsorted(EDGES,np.log(np.maximum(q,1e-30)),side='right')-1

def psd_metrics(domains):
    tables={};values={};private={}
    for name,d in domains.items():
        p=d['features']['PSD'];q=d['features']['Qtotal_mVns'];valid=d['valid'];b=qb(q);rows=[];values[name]=[]
        b[np.log(np.maximum(q,1e-30))==EDGES[-1]]=7
        for i in range(8):
            x=np.sort(p[valid&(b==i)]);values[name].append(x);de=VALLEY['bins'][i]
            rows.append({'bin_id_1based':i+1,'N':len(x),'fraction_of_QC':len(x)/int(valid.sum())if valid.any()else None,'valley_fraction':float(np.mean((x>=de['low'])&(x<=de['high'])))if len(x)and de['valid']else None,'valley_low':de['low'],'valley_high':de['high']})
            private[name+'__bin'+str(i+1)+'__PSD']=x
        tables[name]={'source_n':len(p),'QC_n':int(valid.sum()),'outside_8bins':int(np.sum(valid&((b<0)|(b>7)))),'bins':rows}
    comparisons={}
    for name in domains:
        if name=='REAL':continue
        rows=[]
        for i in range(8):
            a=values[name][i];r=values['REAL'][i];ok=len(a)>0 and len(r)>0
            ca=tables[name]['bins'][i]['valley_fraction'];ra=tables['REAL']['bins'][i]['valley_fraction']
            rows.append({'bin_id_1based':i+1,'candidate_N':len(a),'REAL_N':len(r),'defined':ok,'W1':float(wasserstein_distance(a,r))if ok else None,'KS':float(ks_2samp(a,r).statistic)if ok else None,'valley_fraction_difference':ca-ra if ca is not None and ra is not None else None})
        comparisons[name]={'bins':rows,'all_8_supported':all(r['defined']for r in rows),'mean8_W1':float(np.mean([r['W1']for r in rows]))if all(r['defined']for r in rows)else None}
    np.savez_compressed(OUT/'PSD_SECTIONS_PRIVATE.npz',**private)
    return{'tables':tables,'vs_REAL':comparisons,'edges_logQ':EDGES,'population_weights_changed':False,'old_frozen_valley_reused':True}

def widths(domains,old):
    tables=dict(old['energy_conditional_width']['tables'])
    for name in ARMS:
        d=domains[name];good=d['valid'];COUNTER['density_description_slots']+=8
        tables[name]=ew.section(d['features']['PSD'][good],d['features']['Qtotal_mVns'][good],EDGES)
        COUNTER['density_EM_attempts']+=sum(r['fit_attempts']for r in tables[name]['slices'])
        assert COUNTER['density_EM_attempts']<=8
    common_errors={}
    for setname,names in [('D5_and_frozen_D1',['STANDARDIZED','D5']),('D5_D1_and_R21',['R21_GLOBAL_STYLE','STANDARDIZED','D5'])]:
        rows=[]
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8)if all(tables[n]['slices'][i]['valid']and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field]is not None for n in ['REAL',*names])]
                rows.append({'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field])for i in bins]))if bins else None for n in names}})
        common_errors[setname]=rows
    return{'tables':tables,'common_support_errors':common_errors,'edges_logQ':EDGES,'old_reference_fits_reused':True,'new_fitted_domain_only':'D5','density_components_not_particle_truth':True}

def c2st(domains,rr,selection,old):
    original={k:(np.asarray(v['train'],dtype='i4'),np.asarray(v['holdout'],dtype='i4'))for k,v in selection['indices'].items()};chosen={};counts={}
    for phase in [0,1]:
        g=original['GEN'][phase];g=g[np.logical_and.reduce([domains[n]['valid'][g]for n in ['SOURCE_CLEAN','STANDARDIZED','D5']])]
        r=original['REAL'][phase];r=r[domains['REAL']['valid'][r]]
        t=original['TRAIN_REFERENCE'][phase];t=t[rr['valid'][t]];n=min(len(g),len(r),len(t));chosen[phase]=(g[:n],r[:n],t[:n]);counts['train'if phase==0 else'holdout']={'N_each':n,'source_common_QC':len(g),'REAL':len(r),'TRAIN_REFERENCE':len(t)}
    oldp=load(pathlib.Path(CFG['R23_root'])/'results/C2ST_PRIVATE.npz')
    same_old=all(np.array_equal(oldp[f'phase{p}_{j}'],chosen[p][j])for p in [0,1]for j in [0,1,2])
    supported=counts['train']['N_each']>=150 and counts['holdout']['N_each']>=75;result=dict(old['C2ST']['results']);prediction={};private={};paired={}
    if supported:
        gs=(chosen[0][0],chosen[1][0]);rs=(chosen[0][1],chosen[1][1]);name='D5';result[name]={}
        assert not set(domains[name]['groups'][gs[0]])&set(domains[name]['groups'][gs[1]])
        assert not set(domains['REAL']['groups'][rs[0]])&set(domains['REAL']['groups'][rs[1]])
        for view in CFG['C2ST_views']:
            result[name][view],prediction[view]=dc.fit_one(domains[name],domains['REAL'],gs,rs,view,CFG,COUNTER)
            private.update({name+'__'+view+'__'+k:v for k,v in prediction[view].items()});log('C2ST D5 '+view+' '+str(result[name][view]['balanced_accuracy']))
            control={k:oldp['STANDARDIZED__'+view+'__'+k]for k in ['probability','label','cluster']}
            if same_old and np.array_equal(control['label'],prediction[view]['label'])and np.array_equal(control['cluster'],prediction[view]['cluster']):paired[view]=aa.paired_bootstrap(prediction[view],control,CFG)
            else:paired[view]={'defined':False,'reason':'fixed selection or label/group mismatch; no paired CI fabricated'}
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{'phase'+str(p)+'_'+str(j):a for p,v in chosen.items()for j,a in enumerate(v)})
    return{'supported':supported,'counts':counts,'results':result,'D5_minus_STANDARDIZED_paired_CI':paired,'same_old_selection':same_old,'reference_results_reused':old['C2ST']['reference_results_reused'],'RR_definition':old['C2ST']['RR_definition'],'new_fits_only_7':True,'old_R23_classifiers_predictions_reused':True,'fixed_classifier_CI_not_training_uncertainty':True}

def source_composition(domains,comp,hdev,lambdas):
    source=domains['SOURCE_CLEAN'];sb=qb(source['features']['Qtotal_mVns']);rows={};private={};weights=np.ones(2000);within=(sb>=0)&(sb<8);weights[within]=np.exp(lambdas[sb[within]]*hdev[within]);auxiliary={}
    for name,d in domains.items():
        if name=='REAL':continue
        ob=qb(d['features']['Qtotal_mVns']);rows[name]={}
        for category in ['pure_EE','pure_had','mixed']:
            table={}
            for mode,b in [('source_condition',sb),('output_condition',ob)]:
                tab=[]
                for i in range(8):
                    use=d['valid']&(comp==category)&(b==i);p=d['features']['PSD'][use];de=VALLEY['bins'][i]
                    population=(comp==category)&(b==i)
                    tab.append({'bin_id_1based':i+1,'source_count':int(population.sum()),'N':int(use.sum()),'invalid_count':int((population&~d['valid']).sum()),'output_PSD':ev.summary(p),'valley_fraction':float(np.mean((p>=de['low'])&(p<=de['high'])))if len(p)and de['valid']else None});private[name+'__'+category+'__'+mode+'__bin'+str(i+1)]=p
                table[mode]=tab
            rows[name][category]=table
        aux=[]
        for i in range(8):
            use=d['valid']&(sb==i);real=domains['REAL'];ruse=real['valid']&(qb(real['features']['Qtotal_mVns'])==i);p=d['features']['PSD'][use];r=real['features']['PSD'][ruse];w=weights[use];de=VALLEY['bins'][i]
            aux.append({'source_bin_id_1based':i+1,'candidate_N':len(p),'REAL_observed_bin_N':len(r),'Kish_ESS':float(w.sum()**2/np.dot(w,w))if len(w)else None,'weighted_PSD_mean':float(np.average(p,weights=w))if len(p)else None,'weighted_valley_fraction':float(np.average((p>=de['low'])&(p<=de['high']),weights=w))if len(p)and de['valid']else None,'weighted_W1_vs_REAL':float(wasserstein_distance(p,r,u_weights=w))if len(p)and len(r)else None})
        auxiliary[name]=aux
    primary={n:{'bin'+str(i+1):rows[n]['pure_had']['source_condition'][i]for i in[2,3]}for n in rows}
    for n in rows:assert [primary[n]['bin3']['source_count'],primary[n]['bin4']['source_count']]==[207,182],(n,'purehad source population changed')
    np.savez_compressed(OUT/'SOURCE_COMPOSITION_PRIVATE.npz',**private,source_Qbin=sb,auxiliary_dev_weight=weights)
    return{'all_bins_and_compositions':rows,'primary_pure_had_source_bins3_4':primary,'auxiliary_TRAIN_frozen_weight':auxiliary,'auxiliary_not_primary_population':True,'soft_h_to_high_PSD_is_assumption_not_truth':True}

def probes(model,main,mainstyle,name):
    a=load(ROOT/'PROBE_INPUTS_CORRECTED.npz');x=a['source_waveforms_V'];style=mainstyle[a['noise_origin_parent_index']];pre,signal,electronic=infer(model,x,a['z'],a['white'],style,CFG['generation_chunk']);post,ad=adc(pre);COUNTER['source_probe_rows']+=len(post)
    f=ev.features(post);sf=ev.features(x);tags=a['probe_kind'];baseline=main[a['reference_parent_index']];bf=ev.features(baseline);rows={}
    for tag in np.unique(tags):
        use=tags==tag;rows[str(tag)]={'N':int(use.sum()),'QC':int(f['valid'][use].sum()),'waveform_RMS_vs_correct_reference_V':ev.summary(np.sqrt(np.mean((post[use]-baseline[use])**2,1))),'PSD_vs_reference':ev.summary(f['features']['PSD'][use]-bf['features']['PSD'][use]),'source_PSD':ev.summary(sf['features']['PSD'][use]),'output_PSD':ev.summary(f['features']['PSD'][use]),'output_Q_mVns':ev.summary(f['features']['Qtotal_mVns'][use])}
    payload={k:a[k]for k in ['source_parent_index','source_parent_ids','noise_origin_parent_index','reference_parent_index','source_waveform_modified_counterfactual','probe_kind']}
    np.savez_compressed(OUT/(name+'_SOURCE_PROBES_PRIVATE.npz'),**payload,style=style,style_origin_main_parent_index=a['noise_origin_parent_index'],signal_component_V=signal,electronic_component_V=electronic,waveforms_V=post,PSD=f['features']['PSD'],Q_mVns=f['features']['Qtotal_mVns'])
    same=tags=='same_input_same_noise';fixed=tags=='fixed_source_changed_noise'
    return{'rows':rows,'new_generated_rows':len(post),'same_input_same_noise_bitwise':bool(np.array_equal(post[same],main[a['source_parent_index'][same]])),'fixed_source_parent_index':int(a['source_parent_index'][fixed][0]),'fixed_source_unique_actual_parents':int(len(np.unique(a['source_parent_index'][fixed]))),'corrected_R21_parent_semantics_reused':True,'nonzero_sensitivity_not_correct_physics':True}

def joint_repeats(model,source,comp,name):
    a=load(ROOT/'JOINT8_INPUTS_PRIVATE.npz');parent=a['source_parent_index'];assert len(parent)==1024 and np.all(parent.reshape(128,8)==parent.reshape(128,8)[:,:1])
    pre,signal,electronic=infer(model,a['source_waveforms_V'],a['z'],a['white'],a['style'],CFG['generation_chunk']);post,ad=adc(pre);COUNTER['repeat_noise_rows']+=len(post)
    summary,f=summarize(post,signal,electronic,parent,8,comp,EDGES,source,VALLEY);summary['ADC']=ad
    np.savez_compressed(OUT/(name+'_joint8_REPEAT_PRIVATE.npz'),source_parent_index=parent,waveforms_V=post,signal_component_V=signal,electronic_component_V=electronic,PSD=f['features']['PSD'],Q_mVns=f['features']['Qtotal_mVns'],valid=f['valid'],parent_composition=comp[parent.reshape(128,8)[:,0]])
    return{'groups':{'joint8':summary},'actual_rows':1024,'source_parent_union':128,'new_independent_source_parents':0,'actual_R21_joint8_input_reused':True,'style_only_and_electronic_only_new_outputs':0}

def online_auc(name):
    a=load(ROOT/name/'ONLINE_D_SCORES_PRIVATE.npz');trace=json.loads((ROOT/name/'TRAINING_TRACE.json').read_text('utf8'));result=[]
    for r in trace:
        step=r['validation']['step'];gp=a['step'+str(step)+'_gen'];rp=a['step'+str(step)+'_REAL'];y=np.r_[np.zeros(len(gp)),np.ones(len(rp))];score=np.r_[gp,rp];w=np.r_[a['source_weight'],a['REAL_weight']];use=w>0
        result.append({'step':step,'REAL_label_is1':True,'weighted_AUC':float(roc_auc_score(y[use],score[use],sample_weight=w[use])),'unweighted_AUC':float(roc_auc_score(y,score)),'scores_used_in_selection':False})
    return result

def prefix_audit(training,old):
    nt=training['D5']['updates'];no=old['training']['STANDARDIZED']['updates'];n=min(nt,no)
    streams={'D5':load(ROOT/'D5/TRAINING_STREAMS_PRIVATE.npz'),'STANDARDIZED':load(pathlib.Path(CFG['R23_root'])/'STANDARDIZED/TRAINING_STREAMS_PRIVATE.npz')};checks={}
    for key in ['source','target','bin','source_uniform','target_uniform','NCE_positions']:
        rows=n*3 if key=='NCE_positions'else n*CFG['batch'];a=streams['D5'][key][:rows];b=streams['STANDARDIZED'][key][:rows]
        checks[key]={'common_prefix_identical':bool(np.array_equal(a,b)),'rows':rows,'common_sha256':hashlib.sha256(a.tobytes()).hexdigest()}
    noise_checks={};common_hashes={}
    for name,base,count,record in [('D5',ROOT,nt,training['D5']),('STANDARDIZED',pathlib.Path(CFG['R23_root']),no,old['training']['STANDARDIZED'])]:
        rs=json.loads((base/name/'TRAINING_RANDOM_STATES.json').read_text('utf8'));rn=np.random.default_rng();rn.bit_generator.state=rs['noise_initial'];rt=np.random.default_rng();rt.bit_generator.state=rs['style_initial'];hn=hashlib.sha256();hs=hashlib.sha256();cn=None;cs=None
        for i in range(count):
            z=rn.normal(size=(32,8)).astype('f4');w=rn.normal(size=(32,480)).astype('f4');style=rt.normal(size=(32,8)).astype('f4');hn.update(z.tobytes());hn.update(w.tobytes());hs.update(style.tobytes())
            if i+1==n:cn=hn.hexdigest();cs=hs.hexdigest()
        noise_checks[name]={'consumed_noise_hash_matches':hn.hexdigest()==record['stream_hashes']['noise_stream'],'consumed_style_hash_matches':hs.hexdigest()==record['stream_hashes']['style_stream'],'final_noise_state_matches':rn.bit_generator.state==rs['noise_final'],'final_style_state_matches':rt.bit_generator.state==rs['style_final'],'no_waveform_generation':True}
        common_hashes[name]={'old_z_white':cn,'style':cs}
    good=all(v['common_prefix_identical']for v in checks.values())and common_hashes['D5']==common_hashes['STANDARDIZED']and all(all(v[k]for k in ['consumed_noise_hash_matches','consumed_style_hash_matches','final_noise_state_matches','final_style_state_matches'])for v in noise_checks.values())
    result={'common_prefix_updates':n,'D5_actual_updates':nt,'old_D1_actual_updates':no,'actual_consumed_stream_checks':checks,'noise_style_state_and_hash_checks':noise_checks,'common_prefix_noise_style_hashes':common_hashes,'all_common_prefix_checks_pass':good,'full_completed_lengths_equal':nt==no,'extra_D_does_not_consume_original_G_rng':True,'equal_compute_or_D_exposure':False}
    write('PAIRING_PREFIX_AUDIT.json',result);assert good,'original G stream prefix mismatch';return result

def run():
    old=json.loads((pathlib.Path(CFG['R23_root'])/'metrics.json').read_text('utf8'));ref=json.loads((ROOT/'CAL_REFERENCE_METADATA.json').read_text('utf8'));actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');meta=load(ROOT/'DEV_SOURCE_METADATA.npz');ids=actual['source_ids'];source=actual['source_waveforms_V'];comp=meta['composition'];pdg=meta['pdg_inferred'];real=load(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz')
    ow=load(pathlib.Path(CFG['R23_root'])/'results/PLOT_WAVEFORMS_PRIVATE.npz')
    domains={'REAL':aa.data(real['real_V'],real['real_ids'],real['real_groups'],'REAL',ref)}
    for name in ['SOURCE_CLEAN','R21_GLOBAL_STYLE','RAW','STANDARDIZED']:domains[name]=aa.data(ow[name],ids,np.arange(2000),'G4_SOURCE',ref)
    domains['REAL']['valid'] &= real['real_QC_valid'];rr=aa.data(real['reference_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
    generation={};repeats={};probe={}
    for name in ARMS:
        log('CPU D5 main and registered probes/repeats');model=Translator();model.load_state_dict(torch.load(ROOT/name/'TRANSLATOR.pt',map_location='cpu',weights_only=True));model.eval()
        pre,signal,electronic=infer(model,source,actual['z'],actual['white'],actual['style'],CFG['generation_chunk']);post,ad=adc(pre);COUNTER['main_generation_rows']+=2000
        np.savez_compressed(ROOT/name/'generated.npz',pre_ADC_V=pre,waveforms_V=post,signal_component_V=signal,electronic_component_V=electronic,source_ids=ids)
        domains[name]=aa.data(post,ids,np.arange(2000),'G4_SOURCE',ref)
        generation[name]={'rows':2000,'all_original_source_ids_preserved':bool(np.array_equal(ids,meta['source_ids'])),'ADC':ad,'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),'style_reader_used':False,'signal_not_conditional_expectation':True,'post_ADC_bitwise_vs_frozen_R23_D1':bool(np.array_equal(post,ow['STANDARDIZED'])),'new_vs_frozen_D1_maxabs_V':float(np.max(np.abs(post.astype('f8')-ow['STANDARDIZED'].astype('f8'))))}
        probe[name]=probes(model,post,actual['style'],name);repeats[name]=joint_repeats(model,source,comp,name);del model
    ns,nd=ev.characterize({'REAL':domains['REAL'],'D5':domains['D5']},ref);stats={**old['domains'],'D5':ns['D5']};distance={**old['domain_vs_REAL'],'D5':nd['D5']}
    npd=psd_metrics({'REAL':domains['REAL'],'D5':domains['D5']});psd={**old['PSD_main'],'tables':{**old['PSD_main']['tables'],'D5':npd['tables']['D5']},'vs_REAL':{**old['PSD_main']['vs_REAL'],'D5':npd['vs_REAL']['D5']},'old_reference_results_reused':True}
    width=widths(domains,old);selection=json.loads((ROOT/'EVALUATOR_SELECTIONS_PRIVATE.json').read_text('utf8'));cs=c2st(domains,rr,selection,old)
    frozen=json.loads((ROOT/'FROZEN_TILT_TABLE.json').read_text('utf8'));lambdas=np.asarray([r['lambda_value']for r in frozen['bins']]);h=.15*meta['edep_hadron_MeV']/(meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV'])
    ncomp=source_composition({n:domains[n]for n in ['REAL','SOURCE_CLEAN','D5']},comp,h,lambdas);composition={**old['source_composition']}
    for key in ['all_bins_and_compositions','primary_pure_had_source_bins3_4','auxiliary_TRAIN_frozen_weight']:composition[key]={**old['source_composition'][key],'D5':ncomp[key]['D5']}
    association={**old['source_association'],'D5':ds.association({'SOURCE_CLEAN':domains['SOURCE_CLEAN'],'TRANSLATED':domains['D5']},comp,pdg,EDGES)}
    local={**old['local'],'D5':dc.native_statistics(domains['D5'],ref)};feat={};waves={}
    for name,d in domains.items():
        feat.update({name+'__'+key:a for key,a in {**d['features'],**d['extra']}.items()});feat[name+'__valid']=d['valid'];feat[name+'__region']=d['region'];waves[name]=d['V'].astype('f4')
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**feat);np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**waves)
    training={'D5':json.loads((ROOT/'D5/TRAINING_RECEIPT.json').read_text('utf8'))};online={**old['online_D_validation_AUC'],'D5':online_auc('D5')};pairprefix=prefix_audit(training,old)
    metric={'status':'complete','training':training,'control_training_reused':old['training'],'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text('utf8')),'domains':stats,'domain_vs_REAL':distance,'local':local,'PSD_main':psd,'energy_conditional_width':width,'source_composition':composition,'source_association':association,'source_probes':{**old['source_probes'],**probe},'random_decomposition':{**old['random_decomposition'],**repeats},'C2ST':cs,'generation':generation,'online_D_validation_AUC':online,'pairing':json.loads((ROOT/'PAIRING_RECEIPT.json').read_text('utf8')),'paired_common_prefix':pairprefix,'counters':COUNTER,'versions':{'Python':platform.python_version(),'torch':torch.__version__,'NumPy':np.__version__,'sklearn':sklearn.__version__},'critic_center_scale_combo_not_isolated':True,'primary_DEV_population_reweighted':False,'EVAL_read':0,'CPU_main_and_replay_same_environment':True,'old_control_retrained':False,'sole_training_intervention':'four extra independent D steps per original G step','equal_compute_or_D_exposure':False}
    write('metrics.json',metric)
    assert COUNTER['HGB_attempts']<=7 and COUNTER['density_EM_attempts']<=8 and COUNTER['main_generation_rows']==2000 and COUNTER['repeat_noise_rows']==1024 and COUNTER['source_probe_rows']==832
    write('EVALUATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete','threads':4});log('R24 evaluation complete '+json.dumps(COUNTER))

try:run()
except BaseException as e:
    write('EVALUATION_FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
