from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import pathlib,json,io,hashlib,time,datetime,traceback
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();cfg=json.loads((ROOT/'config.json').read_text())
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
import evaluation as ev
import diagnostic_core as dc
import energy_width as ew
from common_math import distances
OUT=ROOT/'results';OUT.mkdir(exist_ok=True);FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
counter={'HGB_attempts':0,'fixed_inference_rows':0,'response_updates':0,'EVAL_read':0,'SSH':0,'new_G4':0,'density_EM_attempts':0}

def write(n,x):
    (ROOT/n).parent.mkdir(parents=True,exist_ok=True);(ROOT/n).write_text(json.dumps(ev.safe(x),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
def check():
    if time.monotonic()-START>cfg['numeric_wall_limit_s']-30:raise RuntimeError('worker numeric budget')
    if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(cfg['deadline_UTC']):raise RuntimeError('stage deadline')
def read(k):
    p=cfg['input_files'][k];b=pathlib.Path(p['path']).read_bytes()
    if len(b)!=p['bytes'] or hashlib.sha256(b).hexdigest()!=p['sha256']:raise RuntimeError(k+' input binding drift')
    return b
def npz(k):return np.load(io.BytesIO(read(k)),allow_pickle=False)
def data(v,ids,groups,typ,ref,qc=None):
    d=ev.features(v);d['ids']=np.asarray(ids);d['groups']=np.asarray(groups)
    d['cluster']=np.array([typ+'|'+str(g) for g in groups]);d['region']=np.searchsorted(ref['CAL_charge_quantile_edges_mV_ns'],d['features']['Qtotal_mVns'],side='right')
    if qc is not None:d['valid'] &= np.asarray(qc)==0
    return dc.extra_features(d)

def engineering():
    import interface_core
    interface_core.np=np
    from interface_core import constant_fraction_tp0,digitize,_window_waveform
    rows=[];t=np.arange(512);pulse=np.maximum(0,np.exp(-np.maximum(t-128,0)/35)-np.exp(-np.maximum(t-128,0)/5)).astype('f4')
    for name,x in [('single',pulse),('precursor',pulse.copy()),('edge',np.r_[1.,np.zeros(511)].astype('f4')),('zero',np.zeros(512,dtype='f4'))]:
        if name=='precursor':x[20]=.3*float(x.max())
        n=constant_fraction_tp0(x,.5);a=torch.tensor(x);pi=int(a.argmax());below=torch.nonzero((torch.arange(512)<pi)&(a<.5*a.max())).flatten();q=int(below[-1])+1 if len(below) else pi
        assert n==q
        out=_window_waveform(x,n,480,120);indices=np.clip(np.arange(480)+n-120,0,511);assert np.array_equal(out,x[indices])
        rows.append({'case':name,'CFD_numpy':n,'CFD_torch':q,'edge_padding_same':True})
    v=np.linspace(-.49,.49,101);a=digitize(v);b=(((torch.tensor(v,dtype=torch.float32)+.5)*16383).round().clamp(0,16383)/16383-.5).numpy()
    rows.append({'case':'ADC_f64_vs_f32_away_from_half_codes','maxabs_V':float(np.max(np.abs(a-b))),'within_1e_7':bool(np.max(np.abs(a-b))<1e-7)});assert rows[-1]['within_1e_7']
    z=np.array([.03,.6,.8,.8,.3,.03]);y=np.array([0,0,1,0,1,1]);g=np.arange(len(y)).astype(str);r=dc.bootstrap(y,z,g,100,123)
    assert r['valid_replicates']>90
    return {'fixed_engineering_cases':rows,'regional_feature_invariance':dc.locality_checks(),'scientific_arrays_used_for_fixture':False,'CFD_implementations_identical_for_cases_not_physical_calibration':True}

def intervention(kind,pack,c,q,z,w):
    if kind=='signal_block_mean20':from model_block import SmoothResponse
    else:from model import SmoothResponse
    model=SmoothResponse(**pack['constructor'])
    state=pack['state_dict']
    if kind=='signal_block_mean20':
        result=model.load_state_dict({k:v for k,v in state.items() if k!='time'},strict=False);assert result.missing_keys==['time'] and not result.unexpected_keys
    else:model.load_state_dict(state)
    model.eval();bits=[]
    for name,p in model.named_parameters():
        assert torch.equal(p.detach(),state[name]);bits.append(hashlib.sha256(p.detach().numpy().tobytes()).hexdigest())
    outputs=[];metadata={};chunk=32 if kind=='signal_block_mean20' else 256
    with torch.no_grad():
        for first in range(0,2000,chunk):
            check();counter['fixed_inference_rows']+=min(chunk,2000-first)
            if counter['fixed_inference_rows']>cfg['generation_max_rows']:raise RuntimeError('inference budget')
            a,b=model(c[first:first+chunk],q[first:first+chunk],z[first:first+chunk],w[first:first+chunk],quantize=kind!='ADC_off')
            outputs.append(a.numpy());
            for key,value in b.items():metadata.setdefault(key,[]).append(value.numpy())
    return np.concatenate(outputs),{k:np.concatenate(v) for k,v in metadata.items()},{'kind':kind,'rows':2000,'same_all_trainable_parameters':True,'parameter_tensor_SHA256s':bits,'chunk':chunk,'only_derived_time_buffer_replaced':kind=='signal_block_mean20','response_updates':0}

def run():
    log('engineering and input bindings');write('ENGINEERING_CHECKS.json',engineering())
    ref=json.loads(read('CAL_reference'));roles=json.loads(read('TRAIN_roles'))
    with npz('SIM_identity') as p:ids=p['sim_event_ids'].copy();rids=p['real_event_ids'].copy()
    with npz('generated') as p:
        gen=p['waveforms_V'].copy();assert np.array_equal(ids,p['sim_event_ids'])
        geometry={k:p[k].copy() for k in ['main_padding','main_quantization_clip','main_CFD_full_index','main_window_left','main_event_phase_ns']}
    raw=read('REAL_V');real=np.frombuffer(raw,dtype='<f4',count=5200*480).reshape(5200,480).copy();del raw
    qc=np.frombuffer(read('REAL_QC'),dtype='<u2',count=5200).copy();groups=np.frombuffer(read('REAL_groups'),dtype='<u2',count=5200).copy();original=np.frombuffer(read('REAL_original_index'),dtype='<u4',count=5200).copy()
    rr=np.frombuffer(read('REAL_roles'),dtype='u1',count=5200);assert np.all(rr[:2000]==0) and np.all(rr[2000:]==1) and np.array_equal(groups,original//50)
    realgeom=np.frombuffer(read('REAL_geometry'),dtype='<i4',count=5200*9).reshape(5200,9).copy()
    normalization=np.frombuffer(read('REAL_normalization'),dtype='<f4',count=5200*2).reshape(5200,2).copy()
    fit=(qc[2000:]==0)&~np.isin(groups[2000:],roles['REAL_internal_validation_artificial_groups']);val=(qc[2000:]==0)&~fit
    assert fit.sum()==2592 and val.sum()==599
    fitids=np.array(['RAW_ORIGINAL_INDEX|'+str(x) for x in original[2000:][fit]])
    pick=ev.order(fitids,'R7-RR-TRAIN-2000')[:2000];train=real[2000:][fit][pick];trainids=fitids[pick];traingroups=groups[2000:][fit][pick]
    assert not set(traingroups)&set(groups[:2000])
    realdev=real[:2000];df=ev.features(realdev);peak=df['features']['peak_V']
    t=np.arange(480);tail=np.where(t>=120,np.exp(-(t-120)/80),0);rng=np.random.default_rng(cfg['positive_seed'])
    positive=realdev+peak[:,None]*(.02*tail[None,:]+.015*rng.normal(size=realdev.shape))
    domains={'GEN':data(gen,ids,np.arange(2000),'G',ref), 'REAL':data(realdev,rids,groups[:2000],'R',ref,qc[:2000]),
             'TRAIN_REFERENCE':data(train,trainids,traingroups,'R',ref),'POSITIVE':data(positive,rids,groups[:2000],'R',ref,qc[:2000])}
    splits,selection=dc.make_splits(domains,cfg);write('results/DIAGNOSTIC_SELECTIONS.json',{'summary':selection,'indices':{k:{'train':x[0],'holdout':x[1]} for k,x in splits.items()},'TRAIN_reference_original_indices':original[2000:][fit][pick]})
    source_stats={'REAL_TRAIN_fit':dc.native_statistics(data(real[2000:][fit],fitids,groups[2000:][fit],'R',ref),ref),
                  'REAL_TRAIN_validation':dc.native_statistics(data(real[2000:][val],np.array(['RAW_ORIGINAL_INDEX|'+str(x) for x in original[2000:][val]]),groups[2000:][val],'R',ref),ref)}
    native_stats={name:dc.native_statistics(d,ref) for name,d in domains.items()};source_stats.update(native_stats)
    processing={'REAL_saved_geometry':{'source_DEV':2000,'QC_n':int((qc[:2000]==0).sum()),'padding_rows':int(np.any(realgeom[:2000,5:7]>0,axis=1).sum()),'ADC_clipped_rows':int(np.any(realgeom[:2000,7:9]>0,axis=1).sum()),'CFD_numpy_saved':ev.summary(realgeom[:2000,2]),'baseline_full_V':ev.summary(normalization[:2000,0]),'peak_full_V':ev.summary(normalization[:2000,1])},
                'GEN_saved_geometry':{k:ev.summary(v.astype(float)) for k,v in geometry.items()},'differences':{'REAL':{'raw_to_mean':10000/500,'full_len':500,'CFD_baseline_n':100,'ADC_dtype_before_normalization':'float64->float32'},'GEN':{'raw_point_signal':'continuous analytic at1ns','full_len':512,'CFD_baseline_n':102,'ADC_arithmetic':'Torch float32','phase_formula':'.5*tanh(z5)'}},'software_differences_not_causal_proof':True}
    results={};saved={};transformed={};operations={};contrasts={'GEN_vs_REAL':'GEN','REAL_TRAIN_vs_DEV':'TRAIN_REFERENCE','positive_vs_REAL':'POSITIVE'}
    operators=['native',*['fractional_align','coarse_ADC4','smooth_sigma1','align_then_smooth']]
    for op in operators:
        check();log('operator '+op)
        if op=='native':current=domains
        else:
            current={};operations[op]={}
            for name,d in domains.items():
                v,a=dc.transform(d['V'],op);operations[op][name]=a;current[name]=data(v,d['ids'],d['groups'],'G' if name=='GEN' else 'R',ref)
                current[name]['valid']=d['valid'].copy()
        transformed[op]={'REAL':current['REAL'],'GEN':current['GEN']}
        for item in cfg['matrix']:
            if item['operator']!=op:continue
            name=contrasts[item['contrast']];key=op+'|'+item['view']+'|'+item['contrast']
            r,p=dc.fit_one(current[name],current['REAL'],splits[name],splits['REAL'],item['view'],cfg,counter);results[key]=r
            for k,v in p.items():saved[key+'__'+k]=v
            log(key+f" BA={r['balanced_accuracy']:.4f}")
    intervention_outputs={};intervention_info={}
    with npz('generation_inputs') as p:c=torch.tensor(p['G4_condition']);q=torch.tensor(p['Qsource_Vns']);assert np.array_equal(ids,p['sim_event_ids'])
    pack=torch.load(io.BytesIO(read('model')),map_location='cpu',weights_only=True);rngstate=torch.load(io.BytesIO(read('generation_rng')),map_location='cpu',weights_only=True)
    g=torch.Generator();g.set_state(rngstate['generator_before_draw']);z=torch.randn((2000,8),generator=g);w=torch.randn((2000,512),generator=g)
    for kind in ['ADC_off','signal_block_mean20']:
        log('fixed inference '+kind);v,meta,info=intervention(kind,pack,c,q,z,w);intervention_outputs[kind]=v;intervention_info[kind]={**info,'padding_rows':int(meta['padding'].sum()),'clipped_rows':int(meta['quantization_clip'].sum()),'maxabs_difference_from_saved_main_V':float(np.max(np.abs(v-gen))), 'window_left_difference_rows':int(np.count_nonzero(meta['window_left']!=geometry['main_window_left']))}
        d=data(v,ids,np.arange(2000),'G',ref);d['valid']=domains['GEN']['valid'].copy();transformed[kind]={'REAL':domains['REAL'],'GEN':d}
        for view in ['full_voltage','full_historical_shape']:
            key=kind+'|'+view+'|GEN_vs_REAL';r,p=dc.fit_one(d,domains['REAL'],splits['GEN'],splits['REAL'],view,cfg,counter);r['controls_reused']='native|'+view+'|REAL_TRAIN_vs_DEV and native|'+view+'|positive_vs_REAL';results[key]=r
            for k,v in p.items():saved[key+'__'+k]=v
            log(key+f" BA={r['balanced_accuracy']:.4f}")
    assert counter['HGB_attempts']==52 and counter['fixed_inference_rows']==4000
    np.savez_compressed(OUT/'FIXED_INTERVENTION_WAVEFORMS.npz',sim_event_ids=ids,ADC_off_V=intervention_outputs['ADC_off'],signal_block_mean20_V=intervention_outputs['signal_block_mean20'])
    np.savez_compressed(OUT/'C2ST_HOLDOUT_PREDICTIONS.npz',**saved)
    descriptor=json.loads(read('width_definition'));edges=np.array(descriptor['edges_logQ']) if 'edges_logQ' in descriptor else np.array(descriptor['logQ_edges'])
    width={};after={}
    for op,dd in transformed.items():
        width[op]={};after[op]={}
        for name,d in dd.items():
            check();valid_after=ev.features(d['V'])['valid'];after[op][name]={'original_QC_n':int(d['valid'].sum()),'QC_after_operator_n':int((d['valid']&valid_after).sum()),'features':{k:ev.summary(x[d['valid']]) for k,x in {**d['features'],**d['extra']}.items()}}
            if op not in ['ADC_off','signal_block_mean20']:
                b=d['valid']&valid_after;v=ew.section(d['features']['PSD'][b],d['features']['Qtotal_mVns'][b],edges,with_histogram=False);counter['density_EM_attempts']+=sum(s['fit_attempts'] for s in v['slices']);width[op][name]=v
    same=dc.same_N_conditional(domains)
    metrics={'status':'complete','question':'controlled development residual diagnosis','C2ST':results,'matrix':cfg['matrix'],'selection':selection,'source_statistics':source_stats,'processing_chain':processing,'common_operator_receipts':operations,'after_operator_statistics':after,'interventions':intervention_info,'same_N_conditional_reference':same,'PSD_density_widths':width,
             'CAL_reference_valid':[s['valid'] for s in ref['slots']],'q4_no_fallback':True,'counters':counter,'REAL_REAL_definition':'TRAIN fit2000 vs DEV2000; same acquisition disjoint role groups; not R4 half EVAL','negative_control_not_equivalence_tolerance':True,'DEV_historical_exposure':True,'no_response_fit_or_EVAL':True,'elapsed_before_plots_s':time.monotonic()-START,'worker_PID':os.getpid()}
    write('metrics.json',metrics);write('results/SAME_N_CONDITIONAL_REFERENCE.json',same);write('results/PROCESSING_CHAIN_DIAGNOSTICS.json',processing)
    from plotting import plots
    plots(domains,transformed,metrics,cfg,FIG)
    write('NUMERIC_RECEIPT.json',{'status':'complete','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':counter,'Torch':torch.__version__,'NumPy':np.__version__,'threads':4,'EVAL_read':0,'SSH':0,'response_updates':0})
    log('complete '+json.dumps(counter))

try:run()
except BaseException as e:
    write('FAILURE.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exception':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':counter,'no_scientific_recipe_change_on_repair':True});raise
