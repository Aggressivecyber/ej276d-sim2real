from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,traceback
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'));START=time.monotonic()
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:C['CPU_threads']])
import numpy as np
import scipy
from teacher_core import endpoint
from canonical import canonical
from coupling_core import integer_coupling,artificial_check
from weighted_metrics import desc,compare,correlation,boundary,stats,w1,wave_once,wave_zero,wave_add,wave_finish,CATEGORIES

COUNTER={'coupling_constructions':0,'teacher_edges':0,'new_response_updates':0,'HGB_fits':0,'density_fits':0,'ODE_generation':0}
def utc():return datetime.now(timezone.utc).isoformat()
def write(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def progress(phase):
    x={'UTC':utc(),'PID':os.getpid(),'phase':phase,'elapsed_s':time.monotonic()-START,'counter':dict(COUNTER)};write('PROGRESS.json',x);print(json.dumps(x),flush=True)
def load(n,keys):
    with np.load(C['input_paths'][n],allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def binrows(a):return {r['raw_bin0']:r for r in a}
def main():
    write('ARTIFICIAL_CONSTRUCTION_CHECK.json',artificial_check())
    binding=json.loads((R/'INPUT_BINDING.json').read_text('utf8'))['inputs']
    for k,p in C['input_paths'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==binding[k]['sha256'],k
    train=load('TRAIN_path',['source','target','source_ids','source_bin','target_bin','h','edges']);source=train['source'];real=train['target'];edges=load('constants_path',['edges_logQ'])['edges_logQ'];B=load('basis',['B'])['B']
    valley=json.loads(Path(C['input_paths']['valley_definition']).read_text('utf8'))
    assert valley['source']=='R3B results/WIDTH_PRETRAIN.json REAL_TRAIN_fit2592' and not valley['new_fit']
    assert source.shape==(6593,480) and real.shape==(21047,480) and np.array_equal(train['edges'],edges)
    sf=desc(source,edges);rf=desc(real,edges);rc=desc(canonical(real),edges)
    assert sf['valid'].all() and rf['valid'].all() and np.array_equal(sf['rawbin'],train['source_bin']) and np.array_equal(rf['rawbin'],train['target_bin'])
    si=[];ri=[];units=[];den=[];group=[];bins=[];construction=[]
    for b in CATEGORIES:
        ss=np.flatnonzero(sf['rawbin']==b);rr=np.flatnonzero(rf['rawbin']==b)
        aa,bb,mm,gg,rec=integer_coupling(sf['Q_Vns'][ss],sf['PSD'][ss],rf['Q_Vns'][rr],rf['PSD'][rr],C['charge_quantile_groups'])
        si.append(ss[aa]);ri.append(rr[bb]);units.append(mm);den.append(np.full(len(mm),len(rr)*len(source),dtype='i8'));group.append(gg);bins.append(np.full(len(mm),b,dtype='i8'))
        rec.update(raw_bin0=b,global_bin_mass=float(len(ss)/len(source)),REAL_original_bin_mass=float(len(rr)/len(real)),global_denominator=len(rr)*len(source))
        construction.append(rec)
    si=np.concatenate(si);ri=np.concatenate(ri);units=np.concatenate(units);den=np.concatenate(den);groups=np.concatenate(group);bin0=np.concatenate(bins);weight=units.astype('f8')/den
    assert len(np.unique(si))==6593 and len(np.unique(ri))==21047 and np.isclose(weight.sum(),1,rtol=0,atol=1e-14)
    sw=np.bincount(si,weights=weight,minlength=6593);rw=np.bincount(ri,weights=weight,minlength=21047)
    assert np.max(np.abs(sw-1/6593))<1e-16 and np.all(rw>0)
    np.savez_compressed(R/'COUPLING_PRIVATE.npz',source_index=si,REAL_index=ri,integer_mass=units,global_denominator=den,weight=weight,raw_bin0=bin0,charge_group=groups,source_ids=train['source_ids'],SOURCE_marginal=sw,REAL_marginal=rw)
    COUNTER['coupling_constructions']=1
    write('COUPLING_RECEIPT.json',{'UTC':utc(),'all_source_parents':6593,'all_REAL_parents':21047,'edges':len(si),'per_bin':construction,'source_float_marginal_error_max':float(np.max(np.abs(sw-1/6593))),
        'exact_integer_margins_all':all(r['integer_margins_exact'] for r in construction),'REAL_conditional_uniform':True,'REAL_global_uniform':False,'all_original_source_parents_uniform':True,'source_weights_for_evaluation_changed':False,'new_training_sampling_measure':'original TRAIN parent equal weights, not historical h-tilted exposure law'})
    progress('coupling_constructed')
    preparts={};postparts={};boundparts={};qerrors=[];perrors=[];delta_rms=[];wteacher=wave_zero();maxq=maxp=0.;baseline=True;examples={}
    for lo in range(0,len(si),512):
        sl=slice(lo,min(lo+512,len(si)));rr=real[ri[sl]];t=endpoint(rr,sf['Q_Vns'][si[sl]],B);pre=desc(t,edges);post=desc(canonical(t),edges);bo=boundary(t)
        qerror=np.abs(pre['Q_Vns']-sf['Q_Vns'][si[sl]])/sf['Q_Vns'][si[sl]];perror=np.abs(pre['PSD']-rf['PSD'][ri[sl]])
        maxq=max(maxq,float(qerror.max()));maxp=max(maxp,float(perror.max()));qerrors.append(qerror);perrors.append(perror);delta_rms.append(np.sqrt(np.mean((t-rr)**2,axis=1)))
        baseline &= np.array_equal(t[:,:100],rr[:,:100]) and np.array_equal(t[:,400:],rr[:,400:])
        for output,parts in [(pre,preparts),(post,postparts),(bo,boundparts)]:
            for k,v in output.items():parts.setdefault(k,[]).append(v)
        wave_add(wteacher,t,bin0[sl],weight[sl])
        for b in np.unique(bin0[sl]):
            key='bin_'+str(b)
            if key not in examples:
                k=int(np.flatnonzero(bin0[sl]==b)[0]);examples[key]={'edge':lo+k,'source_index':int(si[lo+k]),'REAL_index':int(ri[lo+k]),'source':source[si[lo+k]],'real':rr[k],'teacher':t[k]}
        COUNTER['teacher_edges']+=len(t)
    pf={k:np.concatenate(v) for k,v in preparts.items()};cf={k:np.concatenate(v) for k,v in postparts.items()};bd={k:np.concatenate(v) for k,v in boundparts.items()}
    original_boundary=boundary(real);qerrors=np.concatenate(qerrors);perrors=np.concatenate(perrors);delta_rms=np.concatenate(delta_rms)
    wr=np.full(len(real),1/len(real));ws=np.full(len(source),1/len(source));matched={k:v[ri] for k,v in rf.items()}
    pre_own=compare(pf,rf,weight,wr);pre_source=compare(pf,rf,weight,wr,bin0);post_own=compare(cf,rc,weight,wr);post_source=compare(cf,rc,weight,wr,bin0)
    bypre=binrows(pre_source);bypost=binrows(post_own)
    pm=[bypre[b]['features']['PSD']['W1'] for b in range(7)];cm=[bypost[b]['features']['PSD']['W1'] for b in range(7)]
    pmean=float(np.mean(pm)) if all(v is not None for v in pm) else None;cmean=float(np.mean(cm)) if all(v is not None for v in cm) else None
    valley_rows=[]
    for b in [2,3]:
        de=valley['bins'][b];ix=(bin0==b)&(train['h'][si]==1);parents=(sf['rawbin']==b)&(train['h']==1);mass=float(weight[ix].sum());assert mass>0 and de['valid']
        original=float(np.mean((sf['PSD'][parents]>=de['low'])&(sf['PSD'][parents]<=de['high'])))
        raw=float(np.dot(weight[ix],(pf['PSD'][ix]>=de['low'])&(pf['PSD'][ix]<=de['high']))/mass)
        post=float(np.dot(weight[ix],(cf['PSD'][ix]>=de['low'])&(cf['PSD'][ix]<=de['high']))/mass)
        valley_rows.append({'bin':b+1,'h_one_source_parents':int(parents.sum()),'original_source_fraction':original,'teacher_preADC_fraction':raw,'teacher_canonical_fraction':post,
            'preADC_delta':raw-original,'canonical_delta':post-original,'limit':C['source_valley_delta_max'],'preADC_pass':raw-original<=C['source_valley_delta_max'],'canonical_pass':post-original<=C['source_valley_delta_max'],'low':de['low'],'high':de['high']})
    cliprows=[]
    for b in CATEGORIES:
        ix=bin0==b;mass=float(weight[ix].sum());bad=ix&pf['range_exceeded'];parents=sf['rawbin']==b
        cliprows.append({'raw_bin0':b,'source_parents':int(parents.sum()),'source_input_range_exceeded_parents':int(sf['range_exceeded'][parents].sum()),'teacher_edges':int(ix.sum()),'teacher_range_exceeded_edges':int(bad.sum()),
            'source_parents_with_any_teacher_range_excess':int(len(np.unique(si[bad]))),'teacher_range_exceeded_mass':float(weight[bad].sum()),'teacher_range_exceeded_conditional_fraction':float(weight[bad].sum()/mass),'teacher_invalid_mass':float(weight[ix&~pf['valid']].sum()),'canonical_invalid_mass':float(weight[ix&~cf['valid']].sum())})
    associations={};bounds={}
    for b in [None,*CATEGORIES]:
        ix=np.ones(len(si),bool) if b is None else bin0==b
        associations[str(b)]={'mass':float(weight[ix].sum()),'Q_preADC':correlation(sf['Q_Vns'][si[ix]],pf['Q_Vns'][ix],weight[ix]),'PSD_preADC':correlation(sf['PSD'][si[ix]],pf['PSD'][ix],weight[ix]),'Q_canonical':correlation(sf['Q_Vns'][si[ix]],cf['Q_Vns'][ix],weight[ix]),'PSD_canonical':correlation(sf['PSD'][si[ix]],cf['PSD'][ix],weight[ix])}
        bounds[str(b)]={k:{'teacher':stats(x[ix],weight[ix]),'paired_correction':stats((x-original_boundary[k][ri])[ix],weight[ix])} for k,x in bd.items()}
    baseline_rows=[{'bin':b+1,'RMS_W1':bypre[b]['features']['baseline_RMS_V']['W1'],'ACF1_W1':bypre[b]['features']['baseline_ACF1']['W1']} for b in range(8)]
    gates={'exact_full_equal_source_marginal':bool(all(r['integer_margins_exact'] for r in construction)),
        'charge_PSD_baseline_structure':bool(maxq<=C['Q_relative_max'] and maxp<=C['PSD_preservation_max'] and baseline and pf['valid'].all()),
        'no_teacher_ADC_range_excess':bool(not pf['range_exceeded'].any()),
        'preADC_overall_PSD':bool(pre_own[0]['features']['PSD']['W1'] is not None and pre_own[0]['features']['PSD']['W1']<=C['PSD_overall_max']),
        'preADC_common_Q1_Q7_PSD':bool(pmean is not None and pmean<=C['PSD_conditional_mean_max']),
        'canonical_overall_PSD':bool(post_own[0]['features']['PSD']['W1'] is not None and post_own[0]['features']['PSD']['W1']<=C['PSD_overall_max']),
        'canonical_common_Q1_Q7_PSD':bool(cmean is not None and cmean<=C['PSD_conditional_mean_max']),
        'baseline_RMS_ACF1_all8':bool(all(r['RMS_W1'] is not None and r['ACF1_W1'] is not None and max(r['RMS_W1'],r['ACF1_W1'])<=C['baseline_W1_max'] for r in baseline_rows)),
        'source_h_one_Q3_Q4_valleys':bool(all(r['preADC_pass'] and r['canonical_pass'] for r in valley_rows))}
    passed=all(gates.values());decision='support_separate_neural_training_registration' if passed else 'close_fixed_equal_parent_16_group_teacher'
    waveT=wave_finish(wteacher);waveR=wave_once(real,rf['rawbin'],wr);waveS=wave_once(source,sf['rawbin'],ws)
    wavecmp=[{'raw_bin0':b,'mass':float(waveT['mass'][j]),**{k+'_difference_RMS':float(np.sqrt(np.mean((waveT[k][j]-waveR[k][j])**2))) for k in ['V_mean','V_SD','U_mean','U_SD']}} for j,b in enumerate([*CATEGORIES,None])]
    result={'UTC':utc(),'status':'complete','decision':decision,'all_teacher_conditions_pass':passed,'gates':gates,'failed_gates':[k for k,v in gates.items() if not v],'counter':COUNTER,
        'source_parents':6593,'REAL_parents':21047,'edges':len(si),'relative_Q_error_max':maxq,'PSD_preservation_error_max':maxp,'baseline_tail_exact':baseline,
        'PSD_summary':{'preADC_overall_W1':pre_own[0]['features']['PSD']['W1'],'preADC_common_Q1_Q7_mean_W1':pmean,'canonical_overall_W1':post_own[0]['features']['PSD']['W1'],'canonical_common_Q1_Q7_mean_W1':cmean},
        'preADC_vs_REAL_own_bins':pre_own,'preADC_vs_REAL_source_bins':pre_source,'canonical_vs_REAL_own_bins':post_own,'canonical_vs_REAL_source_bins':post_source,
        'teacher_vs_actual_matched_REAL':compare(pf,matched,weight,weight,bin0),'actual_matched_REAL_vs_full_REAL':compare(matched,rf,weight,wr),
        'valleys':valley_rows,'ADC_support':cliprows,'source_associations':associations,'boundaries':bounds,'baseline_by_bin':baseline_rows,'waveform_vs_REAL':wavecmp,'waveform_correction_RMS_V':stats(delta_rms,weight),
        'model_goal_complete':False,'original_A_F_unchanged':True,'evaluation_source_population_changed':False,'training_population_measure_changed':True,'new_response_training_started':False,'DEV_read':0,'EVAL_read':0,'CAL_decode':0}
    write('RESULT.json',result)
    payload={'source_index':si,'REAL_index':ri,'integer_mass':units,'global_denominator':den,'weight':weight,'source_bin0':bin0,'source_h':train['h'],'source_ids':train['source_ids'],'relative_Q_error':qerrors,'PSD_preservation_error':perrors,'waveform_correction_RMS_V':delta_rms}
    for name,f in [('SOURCE',sf),('REAL',rf),('REAL_CANONICAL',rc),('TEACHER_PRE',pf),('TEACHER_CANONICAL',cf)]:
        for k,v in f.items():payload[name+'__'+k]=v
    for name,w in [('SOURCE',waveS),('REAL',waveR),('TEACHER',waveT)]:
        for k,v in w.items():payload[name+'__wave_'+k]=v
    for k,v in bd.items():payload['boundary__'+k]=v
    np.savez_compressed(R/'FEATURES_PRIVATE.npz',**payload)
    ep={}
    for name,v in examples.items():
        for k,x in v.items():ep[name+'__'+k]=np.asarray(x)
    np.savez_compressed(R/'FIXED_EXAMPLES_PRIVATE.npz',**ep)
    from render_teacher import render
    render(R,result,construction,payload,examples)
    write('NUMERIC_RECEIPT.json',{'UTC':utc(),'status':'complete','PID':os.getpid(),'elapsed_s':time.monotonic()-START,'counter':COUNTER,'decision':decision,'model_goal_complete':False,'environment':{'python':os.sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'CPU_affinity':sorted(os.sched_getaffinity(0))}})
    progress('complete')
try:main()
except BaseException as exc:
    write('FAILURE.json',{'UTC':utc(),'PID':os.getpid(),'counter':COUNTER,'elapsed_s':time.monotonic()-START,'error':repr(exc),'traceback':traceback.format_exc()});raise
