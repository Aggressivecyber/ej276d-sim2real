from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pathlib,json,io,time,datetime,traceback,signal,hashlib,platform
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();CFG=json.loads((ROOT/'config.json').read_text());OUT=ROOT/'results';OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
def alarm(a,b):raise TimeoutError('R18 cumulative scientific/stage wall limit')
signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,min(1740,(datetime.datetime.fromisoformat(CFG['deadline_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-90)))
import numpy as np,torch,sklearn
from scipy.stats import spearmanr
import evaluation as ev,diagnostic_core as dc,assessment as aa,energy_width as ew
from common_math import distances
from model import Translator,infer,adc
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
COUNTER={'response_fit_attempts':0,'G_updates':0,'D_updates':0,'training_s':0.,'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'repeat_noise_rows':0,'source_probe_rows':0,'EVAL_read':0,'CAL_waveform_decode':0,'new_G4':0,'raw_long_waveform_decode':0};CACHE={}
def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
def write(n,v):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def read(k):
    if k not in CACHE:
        p=pathlib.Path(CFG['remote_input_root'])/k;n=CFG['input_files'][k]['bytes']
        with p.open('rb')as f:b=f.read(n+1)
        if len(b)!=n:raise RuntimeError('existing input size changed '+k)
        CACHE[k]=b
    return CACHE[k]
def zload(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in (z.files if keys is None else keys)}
def clean_expectation(ee,had):
    ee=np.asarray(ee,dtype='f8');had=np.asarray(had,dtype='f8');u=np.maximum(np.arange(512)-128.,0)[None,:];out=np.zeros((len(ee),512))
    for energy,taus,frac in [(ee,[13,35,270],[.7,.15,.15]),(.15*had,[13,59,460],[.5,.1,.4])]:
        for tau,weight in zip(taus,frac):out+=energy[:,None]*weight*(np.exp(-u/tau)-np.exp(-u/5))/(tau-5)
    energy=ee+.15*had;q=(energy+.03918)/.000132506/1000;out*=q[:,None]/out.sum(1,keepdims=True)
    peak=out.argmax(1);threshold=out.max(1)*.5;t=np.arange(512)[None,:];below=(t<peak[:,None])&(out<threshold[:,None]);tp=np.where(below,t,-1).max(1)+1;ix=(tp[:,None]-120+np.arange(480)[None,:]).clip(0,511)
    return np.take_along_axis(out,ix,axis=1).astype('f4'),q.astype('f4')
def composition(ee,had):
    return np.where((ee>0)&(had==0),'pure_EE',np.where((had>0)&(ee==0),'pure_had',np.where((ee>0)&(had>0),'mixed','zero')))
def inputs():
    v=np.frombuffer(read('REAL_V'),dtype='<f4',count=5200*480).reshape(5200,480).copy();qc=np.frombuffer(read('REAL_QC'),dtype='<u2',count=5200);g=np.frombuffer(read('REAL_groups'),dtype='<u2',count=5200);idx=np.frombuffer(read('REAL_original_index'),dtype='<u4',count=5200);roles=np.frombuffer(read('REAL_roles'),dtype='u1',count=5200)
    assert np.all(roles[:2000]==0)and np.all(roles[2000:]==1)and np.array_equal(g,idx//50)
    vg=json.loads(read('TRAIN_roles'))['REAL_internal_validation_artificial_groups'];fit=(qc[2000:]==0)&~np.isin(g[2000:],vg);val=(qc[2000:]==0)&np.isin(g[2000:],vg);assert fit.sum()==2592 and val.sum()==599
    ref=json.loads(read('CAL_reference'));edges=np.array(json.loads(read('width_definition'))['edges_logQ']);sel=json.loads(read('R7_selections'))
    with np.load(io.BytesIO(read('SIM_identity')),allow_pickle=False)as z:rids=z['real_event_ids'].copy();sids=z['sim_event_ids'].copy();labels=z['source_label'].copy()
    idsfit=np.array(['RAW_ORIGINAL_INDEX|'+str(i)for i in idx[2000:][fit]]);pick=ev.order(idsfit,'R7-RR-TRAIN-2000')[:2000];assert np.array_equal(idx[2000:][fit][pick],sel['TRAIN_reference_original_indices'])
    real=aa.data(v[:2000],rids,g[:2000],'REAL',ref,qc[:2000]);rr=aa.data(v[2000:][fit][pick],idsfit[pick],g[2000:][fit][pick],'TRAIN_REFERENCE',ref)
    add=zload(pathlib.Path(CFG['remote_R14_root'])/'TRAIN_ADDITION.npz');good=add['QC_reason_u16']==0;assert good.sum()==18455
    target=np.concatenate([v[2000:][fit],add['waveforms_V'][good]]).astype('f4');fitidx=np.concatenate([idx[2000:][fit],add['original_index'][good]]);assert len(target)==21047 and not np.isin(fitidx//50,vg).any()
    assert np.array_equal(fitidx,np.fromfile(pathlib.Path(CFG['remote_R14_root'])/'expanded_fit_original_index.bin',dtype='<u4'))
    gc=zload(pathlib.Path(CFG['remote_input_root'])/'G4_TRAIN_cache');mask=gc['internal_training'].astype(bool);assert mask.sum()==6593 and (~mask).sum()==1599
    source=gc['clean_expectation_V'][mask].astype('f4');allval=np.flatnonzero(~mask);vi=allval[ev.order(gc['event_ids'][allval],'R18-source-validation')[:512]];sv=gc['clean_expectation_V'][vi].astype('f4')
    gi=zload(pathlib.Path(CFG['remote_input_root'])/'generation_inputs');dev,q=clean_expectation(gi['edep_ee_MeV'],gi['edep_hadron_MeV']);assert dev.shape==(2000,480)and np.array_equal(sids,gi['sim_event_ids'])and np.allclose(q,gi['Qsource_Vns'],rtol=2e-6,atol=1e-7)
    comp=composition(gi['edep_ee_MeV'],gi['edep_hadron_MeV']);labelstr=np.array([x.decode()if isinstance(x,bytes)else str(x)for x in labels]);pdg=np.where(np.isin(labelstr,['neutron','2112']),2112,22).astype('i4')
    rolesrec={'G4_TRAIN_fit':6593,'G4_internal_val':1599,'G4_val_used_fixed':512,'REAL_TRAIN_fit':21047,'REAL_internal_val':599,'DEV_source_parents':2000,'REAL_DEV':2000,'TRAIN_PDG_counts':{str(x):int((gc['pdg'][mask]==x).sum())for x in np.unique(gc['pdg'][mask])},'TRAIN_composition':{str(x):int((composition(gc['edep_ee_MeV'][mask],gc['edep_hadron_MeV'][mask])==x).sum())for x in np.unique(composition(gc['edep_ee_MeV'][mask],gc['edep_hadron_MeV'][mask]))},'DEV_composition':{str(x):int((comp==x).sum())for x in np.unique(comp)},'DEV_PDG_counts_inferred':{str(x):int((pdg==x).sum())for x in np.unique(pdg)},'DEV_PDG_provenance':'inferred from historical exact2112/non2112 labels and R2 bound CSV onlyPDG22/2112; not explicitly stored per DEV event','all_source_parent_ids_preserved':True,'source_waveforms':'existing analytic response to G4 totalEE/had deposit, no microscopic timing','unpaired_domains':True,'EVAL_read':0,'CAL_numeric_decode':0,'raw_long_waveform_decode':0,'new_G4':0,'artificial50record_groups_one_acquisition':True}
    write('DATA_ROLES.json',rolesrec);np.savez_compressed(ROOT/'DEV_SOURCE_METADATA.npz',source_ids=sids,edep_ee_MeV=gi['edep_ee_MeV'],edep_hadron_MeV=gi['edep_hadron_MeV'],composition=comp,pdg_inferred=pdg,Qsource_Vns=q,source_label=labels)
    return source,target,sv,v[2000:][val].astype('f4'),dev,sids,comp,pdg,real,rr,ref,edges,sel,rolesrec
def c2st(domains,rr,sel):
    names=['SOURCE_CLEAN','TRANSLATED'];sp={k:(np.array(v['train'],dtype='i4'),np.array(v['holdout'],dtype='i4'))for k,v in sel['indices'].items()};chosen={};counts={}
    for phase in [0,1]:
        g=sp['GEN'][phase];g=g[np.logical_and.reduce([domains[n]['valid'][g]for n in names])];r=sp['REAL'][phase];r=r[domains['REAL']['valid'][r]];t=sp['TRAIN_REFERENCE'][phase];t=t[rr['valid'][t]];n=min(len(g),len(r),len(t));chosen[phase]=(g[:n],r[:n],t[:n]);counts['train'if phase==0 else'holdout']={'N_each':n,'source_common_QC':len(g),'REAL':len(r),'TRAIN_REFERENCE':len(t)}
    supported=counts['train']['N_each']>=150 and counts['holdout']['N_each']>=75;out={};pred={};private={}
    if supported:
        gs=(chosen[0][0],chosen[1][0]);rs=(chosen[0][1],chosen[1][1]);ts=(chosen[0][2],chosen[1][2])
        for name in names+['REAL_REAL']:
            d=rr if name=='REAL_REAL'else domains[name];selection=ts if name=='REAL_REAL'else gs;assert not set(d['groups'][selection[0]])&set(d['groups'][selection[1]]);out[name]={};pred[name]={}
            for view in CFG['C2ST_views']:
                out[name][view],pred[name][view]=dc.fit_one(d,domains['REAL'],selection,rs,view,CFG,COUNTER);private.update({name+'__'+view+'__'+k:v for k,v in pred[name][view].items()});log('C2ST '+name+' '+view+' '+str(out[name][view]['balanced_accuracy']))
    paired={v:aa.paired_bootstrap(pred['TRANSLATED'][v],pred['SOURCE_CLEAN'][v],CFG)for v in CFG['C2ST_views']}if supported else{}
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{'phase'+str(k)+'_'+str(j):v for k,a in chosen.items()for j,v in enumerate(a)})
    return {'supported':supported,'counts':counts,'results':out,'TRANSLATED_minus_SOURCE_paired_CI':paired,'RR_definition':'sameN TRAIN-reference vs DEV, same acquisition artificial groups; not physical equivalence','no_additional_fits':True}
def psd_metrics(domains,edges):
    from scipy.stats import wasserstein_distance,ks_2samp
    valley=json.loads((ROOT/'VALLEY_DEFINITION.json').read_text());tabs={};values={};private={}
    for name,d in domains.items():
        p=d['features']['PSD'];q=d['features']['Qtotal_mVns'];good=d['valid'];lq=np.log(np.maximum(q,1e-30));rows=[];values[name]=[]
        for i in range(8):
            ix=good&(lq>=edges[i])&((lq<edges[i+1])if i<7 else(lq<=edges[i+1]));x=np.sort(p[ix]);values[name].append(x);de=valley['bins'][i];rows.append({'bin_id_1based':i+1,'N':len(x),'fraction_of_QC':len(x)/int(good.sum())if good.any()else None,'valley_fraction':float(np.mean((x>=de['low'])&(x<=de['high'])))if len(x)and de['valid']else None,'valley_low':de['low'],'valley_high':de['high']});private[name+'__bin'+str(i+1)+'__PSD']=x
        tabs[name]={'source_n':len(p),'QC_n':int(good.sum()),'outside_8bins':int(np.sum(good&((lq<edges[0])|(lq>edges[-1])))),'bins':rows}
    comp={}
    for name in ['SOURCE_CLEAN','TRANSLATED']:
        rows=[]
        for i in range(8):
            a=values[name][i];b=values['REAL'][i];ok=len(a)>0 and len(b)>0;rows.append({'bin_id_1based':i+1,'candidate_N':len(a),'REAL_N':len(b),'defined':ok,'W1':float(wasserstein_distance(a,b))if ok else None,'KS':float(ks_2samp(a,b).statistic)if ok else None,'valley_fraction_difference':tabs[name]['bins'][i]['valley_fraction']-tabs['REAL']['bins'][i]['valley_fraction']if ok else None})
        comp[name]={'bins':rows,'all_8_supported':all(x['defined']for x in rows),'mean8_W1':float(np.mean([x['W1']for x in rows]))if all(x['defined']for x in rows)else None}
    np.savez_compressed(OUT/'PSD_SECTIONS_PRIVATE.npz',**private);return {'tables':tabs,'vs_REAL':comp,'edges_logQ':edges,'valley_definition_reused_not_loss_target':True}
def widths(domains,edges):
    prior=json.loads((pathlib.Path(CFG['remote_R17_root'])/'metrics.json').read_text());tabs={'REAL':prior['energy_conditional_width']['tables']['REAL']}
    for n in ['SOURCE_CLEAN','TRANSLATED']:
        d=domains[n];good=d['valid'];COUNTER['density_description_slots']+=8;tabs[n]=ew.section(d['features']['PSD'][good],d['features']['Qtotal_mVns'][good],edges);COUNTER['density_EM_attempts']+=sum(x['fit_attempts']for x in tabs[n]['slices'])
    errors=[]
    for k in [0,1]:
        for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
            ids=[i for i in range(8)if all(tabs[n]['slices'][i]['valid']and len(tabs[n]['slices'][i]['components'])==2 and tabs[n]['slices'][i]['components'][k][field]is not None for n in tabs)]
            errors.append({'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in ids],'MAE':{n:float(np.mean([abs(tabs[n]['slices'][i]['components'][k][field]-tabs['REAL']['slices'][i]['components'][k][field])for i in ids]))if ids else None for n in ['SOURCE_CLEAN','TRANSLATED']}})
    return {'tables':tabs,'edges_logQ':edges,'common_support_errors':errors,'REAL_fits_reused_no_new_fit':True,'density_components_not_particle_truth':True}
def association(domains,comp,pdg,edges):
    source=domains['SOURCE_CLEAN'];out=domains['TRANSLATED'];sq=source['features']['Qtotal_mVns'];sp=source['features']['PSD'];oq=out['features']['Qtotal_mVns'];op=out['features']['PSD'];good=source['valid']&out['valid'];sourcebin=np.searchsorted(edges,np.log(np.maximum(sq,1e-30)),side='right')-1;outbin=np.searchsorted(edges,np.log(np.maximum(oq,1e-30)),side='right')-1
    def corr(a,b,m):
        ok=m&np.isfinite(a)&np.isfinite(b);return {'n':int(ok.sum()),'Spearman':float(spearmanr(a[ok],b[ok]).statistic)if ok.sum()>2 and len(np.unique(a[ok]))>1 and len(np.unique(b[ok]))>1 else None}
    matrix=np.zeros((10,10),dtype='i4')
    for a,b in zip(np.clip(sourcebin+1,0,9),np.where(out['valid'],np.clip(outbin+1,0,9),9)):matrix[a,b]+=1
    rows={}
    for category,mask in [('all',np.ones(2000,dtype=bool))]+[(str(n),comp==n)for n in np.unique(comp)]+[('PDG_inferred_'+str(n),pdg==n)for n in np.unique(pdg)]:
        bins=[]
        for i in range(8):
            use=mask&good&(sourcebin==i);bins.append({'source_bin_1based':i+1,'N':int(use.sum()),'source_PSD':ev.summary(sp[use]),'output_PSD':ev.summary(op[use]),'output_minus_source_PSD':ev.summary((op-sp)[use]),'PSD_ranking':corr(sp,op,use)})
        rows[category]={'source_n':int(mask.sum()),'joint_QC_n':int((mask&good).sum()),'source_Q_to_output_Q':corr(sq,oq,mask&good),'source_PSD_to_output_PSD':corr(sp,op,mask&good),'logQ_change':ev.summary(np.log(np.maximum(oq[mask&good],1e-30)/sq[mask&good])),'bins':bins}
    return {'composition_and_PDG':rows,'migration_source_8bin_rows_to_output_8bin_columns_with_low_high_invalid':matrix,'migration_labels':['below',*['bin'+str(i+1)for i in range(8)],'above_or_invalid'],'no_microscopic_particle_truth_claim':True}
def probes(model,source,ids,comp,edges,z,w):
    pick=ev.order(ids,'R18-source-probe')[:128];base=source[pick];zz=z[pick];ww=w[pick];sources=[];noisez=[];noisew=[];tags=[];parent=[]
    def add(tag,x,iz,iw,pi):sources.append(x);noisez.append(iz);noisew.append(iw);tags.extend([tag]*len(x));parent.extend(pi.tolist())
    add('same_input_same_noise',base,zz,ww,pick);add('source_roll_no_fixed_points',np.roll(base,1,axis=0),zz,ww,pick);add('fixed_source_changed_noise',np.repeat(base[:1],128,axis=0),zz,ww,pick)
    for scale in [.5,1.5]:add('amplitude_x'+str(scale),base[:64]*scale,zz[:64],ww[:64],pick[:64])
    ee,_=clean_expectation(np.ones(64),np.zeros(64));ha,_=clean_expectation(np.zeros(64),np.ones(64));q=ev.features(base[:64])['features']['Qtotal_mVns']/1000;eq=ev.features(ee)['features']['Qtotal_mVns']/1000;hq=ev.features(ha)['features']['Qtotal_mVns']/1000;ee*=q[:,None]/eq[:,None];ha*=q[:,None]/hq[:,None]
    for fraction in [0,.25,.5,.75,1]:add('analytic_fast_slow_'+str(fraction),(1-fraction)*ee+fraction*ha,zz[:64],ww[:64],pick[:64])
    x=np.concatenate(sources).astype('f4');zz=np.concatenate(noisez);ww=np.concatenate(noisew);pre,mean,random=infer(model,x,zz,ww,CFG['generation_chunk']);post,ad=adc(pre);COUNTER['source_probe_rows']=len(x);f=ev.features(post);sf=ev.features(x);tags=np.array(tags);parent=np.array(parent);rows={}
    main=zload(ROOT/'generated.npz',['waveforms_V'])['waveforms_V'];baseline=main[parent];bf=ev.features(baseline)
    for tag in np.unique(tags):
        m=tags==tag;rows[tag]={'N':int(m.sum()),'QC':int(f['valid'][m].sum()),'waveform_RMS_vs_main_parent_V':ev.summary(np.sqrt(np.mean((post[m]-baseline[m])**2,1))),'PSD_vs_main_parent':ev.summary(f['features']['PSD'][m]-bf['features']['PSD'][m]),'Q_vs_main_parent_mVns':ev.summary(f['features']['Qtotal_mVns'][m]-bf['features']['Qtotal_mVns'][m]),'source_Q_mVns':ev.summary(sf['features']['Qtotal_mVns'][m]),'source_PSD':ev.summary(sf['features']['PSD'][m]),'output_Q_mVns':ev.summary(f['features']['Qtotal_mVns'][m]),'output_PSD':ev.summary(f['features']['PSD'][m])}
    np.savez_compressed(OUT/'SOURCE_PROBES_PRIVATE.npz',source_waveforms_V=x,z=zz,white=ww,parent_index=parent,probe_kind=tags,waveforms_V=post,PSD=f['features']['PSD'],Q_mVns=f['features']['Qtotal_mVns'])
    return {'rows':rows,'new_generated_rows':len(x),'source_roll_has_zero_fixed_points':True,'same_input_same_noise_bitwise':bool(np.array_equal(post[tags=='same_input_same_noise'],main[pick])),'analytic_interpolations_not_G4_events':True,'nonzero_sensitivity_not_physical_correctness':True},pick
def repeats(model,source,ids,comp,edges):
    chosen=[]
    for c in ['pure_EE','pure_had','mixed']:
        ix=np.flatnonzero(comp==c);chosen.extend(ix[ev.order(ids[ix],'R18-repeat-parent')[:42]].tolist())
    other=ev.order(ids,'R18-repeat-fill');chosen.extend([i for i in other if i not in chosen][:128-len(chosen)]);pick=np.array(chosen[:128]);parent=np.repeat(pick,8);rng=np.random.default_rng(CFG['repeat_seed']);z=rng.normal(size=(1024,8)).astype('f4');w=rng.normal(size=(1024,480)).astype('f4');pre,_,_=infer(model,source[parent],z,w,CFG['generation_chunk']);post,ad=adc(pre);COUNTER['repeat_noise_rows']=len(post);d=ev.features(post);p=d['features']['PSD'].reshape(128,8);q=d['features']['Qtotal_mVns'].reshape(128,8);good=d['valid'].reshape(128,8);valley=json.loads((ROOT/'VALLEY_DEFINITION.json').read_text());sf=ev.features(source[pick]);sb=np.searchsorted(edges,np.log(sf['features']['Qtotal_mVns']),side='right')-1;rows=[]
    for j in range(128):
        a=p[j][good[j]];i=int(sb[j]);de=valley['bins'][i]if 0<=i<8 else None;both=bool(de and de['valid']and len(a)and (a<de['low']).any()and(a>de['high']).any());rows.append({'source_parent_index':int(pick[j]),'composition':str(comp[pick[j]]),'source_Qbin_1based':i+1 if 0<=i<8 else None,'source_PSD':float(sf['features']['PSD'][j]),'valid_draws':len(a),'output_PSD':ev.summary(a),'both_sides_of_frozen_valley':both})
    by={str(c):{'parents':int(np.sum(comp[pick]==c)),'within_parent_PSD_SD':ev.summary(np.array([r['output_PSD']['sample_sd']for r in rows if r['composition']==c and r['output_PSD']['sample_sd']is not None])),'parents_straddling_valley':sum(r['both_sides_of_frozen_valley']for r in rows if r['composition']==c)}for c in np.unique(comp[pick])}
    np.savez_compressed(OUT/'REPEAT_NOISE_PRIVATE.npz',source_parent_index=parent,z=z,white=w,waveforms_V=post,PSD=p,Q_mVns=q,valid=good,parent_composition=comp[pick]);return {'parent_count':128,'draws_each':8,'rows':rows,'by_composition':by,'no_parent_GMM_fit':True,'same_parent_draws_not_independent_G4_events':True}
def run():
    from training import train
    log('R18 full-wave unpaired input load')
    xs,xt,vs,vt,source,ids,comp,pdg,real,rr,ref,edges,sel,roles=inputs();log('training source6593 REAL21047; all definitions frozen')
    model,training=train(xs,xt,vs,vt,edges,CFG,ROOT,COUNTER,log)
    rng=np.random.default_rng(CFG['generation_seed']);z=rng.normal(size=(2000,8)).astype('f4');w=rng.normal(size=(2000,480)).astype('f4');np.savez_compressed(ROOT/'ACTUAL_GENERATION_INPUTS.npz',source_waveforms_V=source,z=z,white=w,source_ids=ids)
    pre,mean,random=infer(model,source,z,w,CFG['generation_chunk']);post,ad=adc(pre);COUNTER['main_generation_rows']=2000;np.savez_compressed(ROOT/'generated.npz',waveforms_V=post,pre_ADC_V=pre,mean_V=mean,random_branch_V=random,source_ids=ids)
    source_adc,sourcead=adc(source);domains={'REAL':real,'SOURCE_CLEAN':aa.data(source_adc,ids,np.arange(2000),'G4_SOURCE',ref),'TRANSLATED':aa.data(post,ids,np.arange(2000),'G4_SOURCE',ref)}
    log('main2000 source records saved; probes and repeated noise')
    probe,probeidx=probes(model,source,ids,comp,edges,z,w);rep=repeats(model,source,ids,comp,edges)
    stats,dist=ev.characterize(domains,ref);cs=c2st(domains,rr,sel);psd=psd_metrics(domains,edges);width=widths(domains,edges);assoc=association(domains,comp,pdg,edges);local={k:dc.native_statistics(d,ref)for k,d in domains.items()}
    feat={};waves={}
    for n,d in domains.items():
        feat.update({n+'__'+k:v for k,v in {**d['features'],**d['extra']}.items()});feat[n+'__valid']=d['valid'];feat[n+'__region']=d['region'];waves[n]=d['V'].astype('f4')
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**feat);np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**waves)
    metrics={'status':'complete','training':training,'data_roles':roles,'domains':stats,'domain_vs_REAL':dist,'local':local,'PSD_main':psd,'energy_conditional_width':width,'source_association':assoc,'source_probes':probe,'same_parent_noise':rep,'C2ST':cs,'generation':{'source_ids_preserved':True,'rows':2000,'ADC':ad,'source_ADC':sourcead,'pre_ADC_payload_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest()},'counters':COUNTER,'versions':{'Python':platform.python_version(),'torch':torch.__version__,'NumPy':np.__version__,'sklearn':sklearn.__version__},'no_particle_truth_from_real_density':True,'EVAL_read':0};write('metrics.json',metrics)
    from plotting import plots
    plots(domains,metrics,source,post,comp,ROOT)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete','GPU_allocated_peak_bytes':training['GPU_allocated_peak_bytes'],'threads':4});log('R18 complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:write('FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
