from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pathlib,json,time,datetime,traceback,signal,hashlib,platform
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();CFG=json.loads((ROOT/'config.json').read_text());OUT=ROOT/'results';OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
def alarm(a,b):raise TimeoutError('R21 cumulative scientific/stage wall limit')
if hasattr(signal,'SIGALRM'):
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,min(1740,(datetime.datetime.fromisoformat(CFG['deadline_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-90)))
# On Windows the owned local supervisor enforces the full hard wall and RSS limits.
import numpy as np,torch,sklearn
from scipy.stats import wasserstein_distance,ks_2samp
import evaluation as ev,diagnostic_core as dc,assessment as aa,energy_width as ew,data_support as ds
from model import Translator,infer,adc,charge_psd
from fine_core import FineLoss
from random_diagnostics import repeats
from training import train
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
COUNTER={'response_fit_attempts':0,'G_updates':0,'D_updates':0,'training_s':0.,'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'repeat_noise_rows':0,'source_probe_rows':0,'EVAL_read':0,'CAL_waveform_decode':0,'new_G4':0,'raw_long_waveform_decode':0,'structural_check_rows':0,'style_reader_updates':0}
OLD=pathlib.Path(CFG['remote_R18_root']);BASE_OLD=pathlib.Path(CFG['remote_R19_root']);R20=pathlib.Path(CFG['remote_R20_root']);REFERENCE=json.loads((ROOT/'REFERENCE_METRICS.json').read_text());VALLEY=json.loads((ROOT/'VALLEY_DEFINITION.json').read_text())
def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
def write(n,v):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as a:return{k:a[k].copy()for k in(a.files if keys is None else keys)}
def qbins(q,edges):return np.searchsorted(edges,np.log(np.maximum(q,1e-30)),side='right')-1
def affinity(ee,had):return .15*had/(ee+.15*had)

def psd_metrics(domains,edges):
    tabs={};values={};private={}
    for name,d in domains.items():
        p=d['features']['PSD'];q=d['features']['Qtotal_mVns'];good=d['valid'];lq=np.log(np.maximum(q,1e-30));rows=[];values[name]=[]
        for b in range(8):
            use=good&(lq>=edges[b])&((lq<edges[b+1])if b<7 else(lq<=edges[b+1]));x=np.sort(p[use]);values[name].append(x);de=VALLEY['bins'][b]
            rows.append({'bin_id_1based':b+1,'N':len(x),'fraction_of_QC':len(x)/int(good.sum())if good.any()else None,'valley_fraction':float(np.mean((x>=de['low'])&(x<=de['high'])))if len(x)and de['valid']else None,'valley_low':de['low'],'valley_high':de['high']});private[name+'__bin'+str(b+1)+'__PSD']=x
        tabs[name]={'source_n':len(p),'QC_n':int(good.sum()),'outside_8bins':int(np.sum(good&((lq<edges[0])|(lq>edges[-1])))),'bins':rows}
    comp={}
    for name in ['SOURCE_CLEAN','R19_TILT','R20_FINE','GLOBAL_STYLE']:
        rows=[]
        for b in range(8):
            a=values[name][b];r=values['REAL'][b];ok=len(a)>0 and len(r)>0
            rows.append({'bin_id_1based':b+1,'candidate_N':len(a),'REAL_N':len(r),'defined':ok,'W1':float(wasserstein_distance(a,r))if ok else None,'KS':float(ks_2samp(a,r).statistic)if ok else None,'valley_fraction_difference':tabs[name]['bins'][b]['valley_fraction']-tabs['REAL']['bins'][b]['valley_fraction']if ok else None})
        comp[name]={'bins':rows,'all_8_supported':all(x['defined']for x in rows),'mean8_W1':float(np.mean([x['W1']for x in rows]))if all(x['defined']for x in rows)else None}
    np.savez_compressed(OUT/'PSD_SECTIONS_PRIVATE.npz',**private)
    return{'tables':tabs,'vs_REAL':comp,'edges_logQ':edges,'population_weights_changed':False,'old_frozen_valley_reused':True}

def widths(domains,edges):
    old=REFERENCE['energy_conditional_width']['tables'];tabs={'REAL':old['REAL'],'SOURCE_CLEAN':old['SOURCE_CLEAN'],'R19_TILT':old['R19_TILT'],'R20_FINE':old['FINE']}
    for name in ['GLOBAL_STYLE']:
        d=domains[name];good=d['valid'];COUNTER['density_description_slots']+=8;tabs[name]=ew.section(d['features']['PSD'][good],d['features']['Qtotal_mVns'][good],edges);COUNTER['density_EM_attempts']+=sum(x['fit_attempts']for x in tabs[name]['slices'])
    errors=[]
    for k in [0,1]:
        for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
            common=[b for b in range(8)if all(tabs[n]['slices'][b]['valid']and len(tabs[n]['slices'][b]['components'])==2 and tabs[n]['slices'][b]['components'][k][field]is not None for n in tabs)]
            errors.append({'component':k,'field':field,'common_bin_ids_1based':[b+1 for b in common],'MAE':{n:float(np.mean([abs(tabs[n]['slices'][b]['components'][k][field]-tabs['REAL']['slices'][b]['components'][k][field])for b in common]))if common else None for n in ['SOURCE_CLEAN','R19_TILT','R20_FINE','GLOBAL_STYLE']}})
    return{'tables':tabs,'common_support_errors':errors,'edges_logQ':edges,'old_REAL_SOURCE_R19_TILT_fits_reused':True,'density_components_not_particle_truth':True}

def c2st(domains,rr,sel):
    sp={k:(np.asarray(v['train'],dtype='i4'),np.asarray(v['holdout'],dtype='i4'))for k,v in sel['indices'].items()};chosen={};counts={}
    for phase in [0,1]:
        g=sp['GEN'][phase];g=g[np.logical_and.reduce([domains[n]['valid'][g]for n in ['SOURCE_CLEAN','GLOBAL_STYLE']])];r=sp['REAL'][phase];r=r[domains['REAL']['valid'][r]];t=sp['TRAIN_REFERENCE'][phase];t=t[rr['valid'][t]];n=min(len(g),len(r),len(t));chosen[phase]=(g[:n],r[:n],t[:n]);counts['train'if phase==0 else'holdout']={'N_each':n,'source_common_QC':len(g),'REAL':len(r),'TRAIN_REFERENCE':len(t)}
    supported=counts['train']['N_each']>=150 and counts['holdout']['N_each']>=75;out={};pred={};private={}
    oldpred=load(R20/'results/C2ST_PRIVATE.npz');sameold=all(np.array_equal(oldpred['phase'+str(p)+'_'+str(j)],chosen[p][j])for p in[0,1]for j in[0,1,2])
    if supported:
        gs=(chosen[0][0],chosen[1][0]);rs=(chosen[0][1],chosen[1][1])
        for name in ['GLOBAL_STYLE']:
            d=domains[name];assert not set(d['groups'][gs[0]])&set(d['groups'][gs[1]]);out[name]={};pred[name]={}
            for view in CFG['C2ST_views']:
                out[name][view],pred[name][view]=dc.fit_one(d,domains['REAL'],gs,rs,view,CFG,COUNTER)
                private.update({name+'__'+view+'__'+k:v for k,v in pred[name][view].items()});log('C2ST '+name+' '+view+' '+str(out[name][view]['balanced_accuracy']))
    oldpaired={}
    if supported and sameold:
        for view in CFG['C2ST_views']:
            prefix='FINE__'+view+'__';p={k[len(prefix):]:a for k,a in oldpred.items()if k.startswith(prefix)}
            oldpaired[view]=aa.paired_bootstrap(pred['GLOBAL_STYLE'][view],p,CFG)
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{'phase'+str(p)+'_'+str(j):a for p,v in chosen.items()for j,a in enumerate(v)})
    oldref=REFERENCE['C2ST']['reference_results_reused']
    return {'supported':supported,'counts':counts,'results':out,'GLOBAL_STYLE_minus_R20_FINE_paired_CI':oldpaired,
      'reference_results_reused':{'R20_FINE':REFERENCE['C2ST']['results']['FINE'],'R19_TILT':oldref['R19_TILT'],'SOURCE_CLEAN':oldref['SOURCE_CLEAN'],'REAL_REAL':oldref['REAL_REAL']},
      'same_old_selection':sameold,'RR_definition':REFERENCE['C2ST']['RR_definition'],'new_fits_only_7':True}


def source_composition(domains,comp,edges,hdev,lambdas):
    source=domains['SOURCE_CLEAN'];sb=qbins(source['features']['Qtotal_mVns'],edges);rows={};private={}
    for n in ['SOURCE_CLEAN','R19_TILT','R20_FINE','GLOBAL_STYLE']:
        d=domains[n];ob=qbins(d['features']['Qtotal_mVns'],edges);rows[n]={}
        for category in ['pure_EE','pure_had','mixed']:
            tables={}
            for mode,bb in [('source_condition',sb),('output_condition',ob)]:
                tab=[]
                for b in range(8):
                    use=d['valid']&(comp==category)&(bb==b);p=d['features']['PSD'][use];de=VALLEY['bins'][b]
                    tab.append({'bin_id_1based':b+1,'N':int(use.sum()),'output_PSD':ev.summary(p),'valley_fraction':float(np.mean((p>=de['low'])&(p<=de['high'])))if len(p)and de['valid']else None})
                    private[n+'__'+category+'__'+mode+'__bin'+str(b+1)]=p
                tables[mode]=tab
            rows[n][category]=tables
    auxiliary={};weights=np.ones(2000);inrange=(sb>=0)&(sb<8);weights[inrange]=np.exp(lambdas[sb[inrange]]*hdev[inrange])
    for n in ['SOURCE_CLEAN','R19_TILT','R20_FINE','GLOBAL_STYLE']:
        d=domains[n];out=[]
        for b in range(8):
            use=d['valid']&(sb==b);real=domains['REAL'];rb=qbins(real['features']['Qtotal_mVns'],edges);ruse=real['valid']&(rb==b);a=d['features']['PSD'][use];r=real['features']['PSD'][ruse];w=weights[use];de=VALLEY['bins'][b]
            out.append({'source_bin_id_1based':b+1,'candidate_N':len(a),'REAL_observed_bin_N':len(r),'Kish_ESS':float(w.sum()**2/np.dot(w,w))if len(w)else None,'weighted_PSD_mean':float(np.average(a,weights=w))if len(a)else None,'weighted_valley_fraction':float(np.average((a>=de['low'])&(a<=de['high']),weights=w))if len(a)and de['valid']else None,'weighted_W1_vs_REAL':float(wasserstein_distance(a,r,u_weights=w))if len(a)and len(r)else None})
        auxiliary[n]=out
    np.savez_compressed(OUT/'SOURCE_COMPOSITION_PRIVATE.npz',**private,auxiliary_dev_weight=weights,source_Qbin=sb)
    primary={n:{'bin'+str(b+1):rows[n]['pure_had']['source_condition'][b]for b in[2,3]}for n in rows}
    return{'all_bins_and_compositions':rows,'primary_pure_had_source_bins3_4':primary,'auxiliary_TRAIN_frozen_weight':auxiliary,'auxiliary_not_primary_population':True,'soft_h_to_high_PSD_is_assumption_not_truth':True}

def probes(model,main,mainstyle,arm):
    a=load(ROOT/'PROBE_INPUTS_CORRECTED.npz');x=a['source_waveforms_V'];z=a['z'];w=a['white'];style=mainstyle[a['noise_origin_parent_index']];pre,signal,electronic=infer(model,x,z,w,style,CFG['generation_chunk']);post,ad=adc(pre);COUNTER['source_probe_rows']+=len(post);f=ev.features(post);sf=ev.features(x);tags=a['probe_kind'];ref=a['reference_parent_index'];baseline=main[ref];bf=ev.features(baseline);rows={}
    for tag in np.unique(tags):
        use=tags==tag;rows[str(tag)]={'N':int(use.sum()),'QC':int(f['valid'][use].sum()),'waveform_RMS_vs_correct_reference_V':ev.summary(np.sqrt(np.mean((post[use]-baseline[use])**2,1))),'PSD_vs_reference':ev.summary(f['features']['PSD'][use]-bf['features']['PSD'][use]),'source_PSD':ev.summary(sf['features']['PSD'][use]),'output_PSD':ev.summary(f['features']['PSD'][use]),'output_Q_mVns':ev.summary(f['features']['Qtotal_mVns'][use])}
    # Preserve every actual source/noise/reference relation from the corrected R18 package.
    payload={k:a[k]for k in ['source_parent_index','source_parent_ids','noise_origin_parent_index','reference_parent_index','source_waveform_modified_counterfactual','probe_kind']}
    np.savez_compressed(OUT/(arm+'_SOURCE_PROBES_PRIVATE.npz'),**payload,style=style,style_origin_main_parent_index=a['noise_origin_parent_index'],signal_component_V=signal,electronic_component_V=electronic,waveforms_V=post,PSD=f['features']['PSD'],Q_mVns=f['features']['Qtotal_mVns'])
    same=tags=='same_input_same_noise';fixed=tags=='fixed_source_changed_noise'
    return{'rows':rows,'new_generated_rows':len(post),'same_input_same_noise_bitwise':bool(np.array_equal(post[same],main[a['source_parent_index'][same]])),'fixed_source_parent_index':int(a['source_parent_index'][fixed][0]),'fixed_source_unique_actual_parents':int(len(np.unique(a['source_parent_index'][fixed]))),'corrected_R18_parent_semantics_reused':True,'nonzero_sensitivity_not_correct_physics':True}


def run():
    log('R21 one GLOBAL_STYLE arm; old actual streams/constants reused, no old reference retraining')
    xs,xt,vs,vt,source,ids,comp,pdg,real,rr,ref,edges,sel,roles=ds.inputs()
    gc=ds.zload(pathlib.Path(CFG['remote_input_root'])/'G4_TRAIN_cache');mask=gc['internal_training'].astype(bool);allval=np.flatnonzero(~mask);vi=allval[ev.order(gc['event_ids'][allval],'R18-source-validation')[:512]]
    h=affinity(gc['edep_ee_MeV'][mask],gc['edep_hadron_MeV'][mask]);hv=affinity(gc['edep_ee_MeV'][vi],gc['edep_hadron_MeV'][vi]);cats=ds.composition(gc['edep_ee_MeV'][mask],gc['edep_hadron_MeV'][mask])
    sb=qbins(ev.features(xs)['features']['Qtotal_mVns'],edges);tb=qbins(ev.features(xt)['features']['Qtotal_mVns'],edges)
    frozen=json.loads((ROOT/'FROZEN_TILT_TABLE.json').read_text());tables=frozen['bins'];assert [r['bin_id_1based'] for r in tables]==list(range(1,9));lambdas=np.asarray([r['lambda_value'] for r in tables],dtype='f8');fine_constants=load(ROOT/'FINE_CONSTANTS.npz')
    write('FINE_CONSTANTS_REUSE_RECEIPT.json',{'role':'frozen REAL TRAIN21047 constants from R20 before its updates; no refit','copied_without_migration_test':True,'used_before_new_updates':True})
    np.savez_compressed(ROOT/'SOURCE_PRIOR_PRIVATE.npz',source_ids=gc['event_ids'][mask],h=h,source_bin=sb,lambda_by_bin=lambdas,TRAIN_REAL_bin=tb)
    actual=load(R20/'ACTUAL_GENERATION_INPUTS.npz');assert np.array_equal(ids,actual['source_ids'])
    # Reuse the saved source input itself; do not make CPU-versus-server recomputation an input-equivalence gate.
    source=actual['source_waveforms_V']
    mainstyle=np.random.default_rng(CFG['style_main_seed']).normal(size=(2000,8)).astype('f4');actual['style']=mainstyle;np.savez_compressed(ROOT/'ACTUAL_GENERATION_INPUTS.npz',**actual)
    source_adc,_=adc(source);old19=load(BASE_OLD/'TILT/generated.npz',['waveforms_V'])['waveforms_V'];old20=load(R20/'FINE/generated.npz',['waveforms_V'])['waveforms_V']
    domains={'REAL':real,'SOURCE_CLEAN':aa.data(source_adc,ids,np.arange(2000),'G4_SOURCE',ref),'R19_TILT':aa.data(old19,ids,np.arange(2000),'G4_SOURCE',ref),'R20_FINE':aa.data(old20,ids,np.arange(2000),'G4_SOURCE',ref)}
    log('starting GLOBAL_STYLE')
    model,tr,counts=train('GLOBAL_STYLE',xs,xt,vs,vt,h,hv,sb,tb,lambdas,edges,fine_constants,CFG,ROOT/'GLOBAL_STYLE',COUNTER,log)
    sampled_comp={str(c):{'draws':int(counts[cats==c].sum()),'distinct_source':int(((counts>0)&(cats==c)).sum())}for c in np.unique(cats)}
    pre,signal,electronic=infer(model,source,actual['z'],actual['white'],mainstyle,CFG['generation_chunk']);post,ad=adc(pre);COUNTER['main_generation_rows']+=2000
    np.savez_compressed(ROOT/'GLOBAL_STYLE/generated.npz',pre_ADC_V=pre,waveforms_V=post,signal_component_V=signal,electronic_component_V=electronic,source_ids=ids)
    domains['GLOBAL_STYLE']=aa.data(post,ids,np.arange(2000),'G4_SOURCE',ref)
    generation={'rows':2000,'all_original_source_ids_preserved':True,'ADC':ad,'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),'style_reader_used':False,'signal_not_conditional_expectation':True}
    probe=probes(model,post,mainstyle,'GLOBAL_STYLE');repeat=repeats(model,source,comp,edges,CFG,ROOT,OUT,VALLEY,COUNTER);del model
    pairing={'same4000updates':tr['updates']==REFERENCE['training']['FINE']['updates'],'old_R20_FINE_stream_hashes':tr['old_R20_FINE_full_stream_match'],'initial_J20_abs_difference':tr['initial_J20_abs_difference'],'initial_G':'R19 TILT selectedstep3000','initial_D':'R18 INITIAL_CRITIC','new_equal_Adam':True,'new_style_rng_separate':True,'changes':'global FiLM plus latent reconstruction package; not variance-only or a single factor'};write('PAIRING_RECEIPT.json',pairing)
    stats,dist=ev.characterize(domains,ref);psd=psd_metrics(domains,edges);width=widths(domains,edges);cs=c2st(domains,rr,sel)
    meta=load(ROOT/'DEV_SOURCE_METADATA.npz');hdev=affinity(meta['edep_ee_MeV'],meta['edep_hadron_MeV']);composition=source_composition(domains,comp,edges,hdev,lambdas)
    association=ds.association({'SOURCE_CLEAN':domains['SOURCE_CLEAN'],'TRANSLATED':domains['GLOBAL_STYLE']},comp,pdg,edges)
    local={k:dc.native_statistics(d,ref)for k,d in domains.items()};feat={};waves={}
    for name,d in domains.items():
        feat.update({name+'__'+k:a for k,a in {**d['features'],**d['extra']}.items()});feat[name+'__valid']=d['valid'];feat[name+'__region']=d['region'];waves[name]=d['V'].astype('f4')
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**feat);np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**waves)
    m={'status':'complete','training':{'GLOBAL_STYLE':tr},'reference_training_reused':{n:REFERENCE['training']['FINE'][n]for n in ['updates','selected_step','initial_validation','selected_validation','final_validation']},
      'tilt':{'tables':tables,'sampled_composition':sampled_comp,'source_inside_8bins':int(((sb>=0)&(sb<8)).sum()),'source_outside_8bins':int(((sb<0)|(sb>7)).sum()),'REAL_TRAIN_inside_8bins':int(((tb>=0)&(tb<8)).sum())},
      'data_roles':roles,'domains':stats,'domain_vs_REAL':dist,'local':local,'PSD_main':psd,'energy_conditional_width':width,'source_composition':composition,'source_association':association,
      'source_probes':probe,'random_decomposition':repeat,'C2ST':cs,'generation':generation,'pairing':pairing,'counters':COUNTER,
      'versions':{'Python':platform.python_version(),'torch':torch.__version__,'NumPy':np.__version__,'sklearn':sklearn.__version__},
      'fine_constants':json.loads((ROOT/'FINE_CONSTANTS_RECEIPT.json').read_text()),'fine_constants_role':'R20 frozen TRAIN-only, no refit','h_not_NN_input_or_particle_truth':True,'primary_DEV_population_reweighted':False,'EVAL_read':0}
    write('metrics.json',m)
    from plotting import plots
    plots(domains,m,ROOT)
    assert COUNTER['HGB_attempts']<=12 and COUNTER['density_EM_attempts']<=8 and COUNTER['G_updates']<=4000 and COUNTER['D_updates']<=4000 and COUNTER['repeat_noise_rows']<=4096 and COUNTER['structural_check_rows']<=128
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete','GPU_allocated_peak_bytes':tr['GPU_allocated_peak_bytes'],'threads':4});log('R21 complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:write('FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
