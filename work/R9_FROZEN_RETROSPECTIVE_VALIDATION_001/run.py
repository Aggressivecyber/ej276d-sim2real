from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import pathlib,io,json,time,datetime,hashlib,traceback,sys
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic()
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
import numpy as np
import sklearn
import evaluation as ev
import diagnostic_core as dc
import assessment as aa
from common_math import distances,fom
import plotting as plot
CFG=json.loads((ROOT/'config.json').read_text());OUT=ROOT/'results';FIG=OUT/'figures'
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True);ev.OUT=OUT;ev.FIG=FIG;ev.CFG=CFG['evaluation']
CAND=CFG['candidates'];CACHE={};READS=[]
COUNTER={'HGB_attempts':0,'density_EM_attempts':0,'response_updates':0,'new_generation':0,'SSH':0,'new_G4':0,'EVAL_records_decoded':0}
def write(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(ev.safe(obj),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def read(key):
    if key not in CACHE:
        s=CFG['input_files'][key];p=pathlib.Path(s['path'])
        with p.open('rb') as f:b=f.read(s['bytes']+1)
        if len(b)!=s['bytes'] or hashlib.sha256(b).hexdigest()!=s['sha256']:raise RuntimeError(key+' input binding drift')
        CACHE[key]=b;READS.append({'key':key,**s})
    return CACHE[key]
def jread(key):return json.loads(read(key))
def npz(key):return np.load(io.BytesIO(read(key)),allow_pickle=False)
def full_order(ids,prefix):return np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256((prefix+'|'+str(ids[i])).encode()).digest()),dtype='i8')
def take(d,ix):
    n=len(d['V']);out={}
    for key,value in d.items():
        if isinstance(value,dict):out[key]={k:(v[ix] if isinstance(v,np.ndarray) and len(v)==n else v) for k,v in value.items()}
        elif isinstance(value,np.ndarray) and value.ndim and len(value)==n:out[key]=value[ix]
        else:out[key]=value
    return out
def joint(a,b,scale):
    def matrix(d):
        q=d['features']['Qtotal_mVns'][d['valid']];p=d['features']['PSD'][d['valid']]
        return (np.column_stack([np.log(q),p])-np.asarray(scale['mean']))/np.asarray(scale['population_sd'])
    x,y=matrix(a),matrix(b)
    if not len(x) or not len(y):return {'W1':None,'counts':[len(x),len(y)]}
    angles=np.arange(128)*np.pi/128;direction=np.column_stack([np.cos(angles),np.sin(angles)])
    u=x@direction.T;v=y@direction.T;values=[distances(u[:,i],v[:,i])['W1'] for i in range(128)]
    return {'W1':float(np.mean(values)),'direction_W1':values,'counts':[len(x),len(y)],'angles':128,'scale':'frozen R4 TRAIN-fit mean/population SD, never refit on EVAL'}
def compare(a,b):
    out={};fa={**a['features'],**a['extra']};fb={**b['features'],**b['extra']}
    for r in [None,0,1,2,3]:
        ma=ev.mask(a,r);mb=ev.mask(b,r);key='overall' if r is None else 'q'+str(r+1)
        out[key]={k:distances(v[ma&np.isfinite(v)],fb[k][mb&np.isfinite(fb[k])]) for k,v in fa.items()}
    return out
def rr_reference(allreal,ref,scale):
    with npz('RR_SELECTIONS') as z:selections={k:z[k].copy() for k in z.files}
    assert len(selections)==64
    rows=[];first=None
    for repeat in range(64):
        pair=selections['repeat'+str(repeat)];assert pair.shape==(2,2000)
        a,b=[take(allreal,ix) for ix in pair]
        assert a['valid'].all() and b['valid'].all() and not set(a['groups'])&set(b['groups'])
        assert not set(a['ids'])&set(b['ids'])
        fields=compare(a,b);foms={}
        for r in [None,0,1,2,3]:
            key='overall' if r is None else 'q'+str(r+1);slot=ref['slots'][0 if r is None else r+1]
            foms[key]=[fom(d['features']['PSD'][ev.mask(d,r)],slot['cut'] if slot['valid'] else None) for d in [a,b]]
        rows.append({'repeat':repeat,'each_N':2000,'groups':list(map(lambda d:len(np.unique(d['groups'])),[a,b])),'features':fields,'joint_sliced_W1':joint(a,b,scale),'FoM':foms})
        if first is None:first=(a,b)
    summary={}
    for region in ['overall','q1','q2','q3','q4']:
        summary[region]={}
        for field in rows[0]['features'][region]:
            summary[region][field]={}
            for metric in ['W1','KS']:
                values=np.array([r['features'][region][field][metric] for r in rows if r['features'][region][field][metric] is not None])
                summary[region][field][metric]={'defined_repeats':len(values),'median':float(np.median(values)) if len(values) else None,'range5_95':np.quantile(values,[.05,.95]).tolist() if len(values) else None}
    values=np.array([r['joint_sliced_W1']['W1'] for r in rows])
    result={'same_R4_saved64_selections':True,'same_acquisition_artificial_groups':True,'not_physical_equivalence_tolerance':True,'each_N':2000,
            'summary':summary,'joint_sliced_W1':{'median':float(np.median(values)),'range5_95':np.quantile(values,[.05,.95]).tolist()},'records':rows}
    write('results/REAL_REAL_REFERENCE.json',result);return result,first
def select(d,typ,mask=None):
    keys=d['ids'] if typ=='SIM' else d['groups'];flag=np.array([ev.hashint('R1-SPLIT|'+typ+'|'+str(x))%10<7 for x in keys])
    valid=d['valid'] if mask is None else mask
    train=np.flatnonzero(valid&flag);test=np.flatnonzero(valid&~flag)
    train=train[ev.order(d['ids'][train],'R1-C2ST-SAMPLE')];test=test[ev.order(d['ids'][test],'R1-C2ST-SAMPLE')]
    assert not set(d['groups'][train])&set(d['groups'][test])
    return train,test
def classifiers(domains,real,rrfirst,positive):
    common=np.logical_and.reduce([domains[n]['valid'] for n in CAND])
    base=select(domains['R3B'],'SIM',common);target=select(real,'REAL')
    control_a,control_b=[select(d,'REAL') for d in rrfirst];pos=select(positive,'REAL')
    allselections=[base,target,control_a,control_b,pos]
    nt=min(CFG['train_each_domain'],min(len(a) for a,b in allselections));nv=min(CFG['holdout_each_domain'],min(len(b) for a,b in allselections))
    if min(nt,nv)<40:raise RuntimeError('frozen classifier common support below40')
    cut=lambda s:(s[0][:nt],s[1][:nv])
    base,target,control_a,control_b,pos=map(cut,[base,target,control_a,control_b,pos])
    assert np.array_equal(pos[0],target[0]) and np.array_equal(pos[1],target[1]),'positive/native REAL sample drift'
    pairs=[(n,domains[n],real,base,target) for n in CAND]+[('REAL_vs_REAL',rrfirst[0],rrfirst[1],control_a,control_b),('PERTURBED_REAL',positive,real,pos,target)]
    saved={};selections={};out={}
    for name,a,b,asplit,bsplit in pairs:
        atr,ate=asplit;btr,bte=bsplit
        train_groups=set(a['cluster'][atr])|set(b['cluster'][btr]);test_groups=set(a['cluster'][ate])|set(b['cluster'][bte])
        assert not train_groups&test_groups,'whole parent/artificial-group leakage'
        selections[name]={'first_train_positions':atr,'first_test_positions':ate,'first_train_ids':a['ids'][atr],'first_test_ids':a['ids'][ate],
                          'second_train_positions':btr,'second_test_positions':bte,'second_train_ids':b['ids'][btr],'second_test_ids':b['ids'][bte],
                          'train_each_domain':nt,'holdout_each_domain':nv,'train_test_group_overlap':0}
        for view in CFG['C2ST_views']:
            row,pred=dc.fit_one(a,b,asplit,bsplit,view,CFG,COUNTER)
            out[name+'|'+view]=row
            for k,v in pred.items():saved[name+'|'+view+'__'+k]=v
            ev.log('R9 HGB '+name+' '+view+' BA='+str(row['balanced_accuracy']))
    paired={}
    for name in CAND[1:]:
        for view in CFG['C2ST_views']:
            key=name+'|'+view;old='R3B|'+view
            get=lambda n:{k:saved[n+'__'+k] for k in ['probability','label','cluster']}
            paired[key+'_minus_R3B']=aa.paired_bootstrap(get(key),get(old),CFG)
    write('results/C2ST_SPLITS.json',selections);np.savez_compressed(OUT/'C2ST_HOLDOUT_PREDICTIONS.npz',**saved)
    return out,paired,{'common_source_QC':int(common.sum()),'source_records':2000,'train_each_domain':nt,'holdout_each_domain':nv,'all_candidates_and_views_same_source_real_indices':True,'RR_definition':'Saved R4 repetition0 halves, original groups disjoint','bootstrap_repeats':500}
def widths(domains,allreal,edges):
    old=jread('R4_WIDTHS');r8=jread('R8_metrics')['energy_conditional_width'];r8c=jread('R8C_metrics')['energy_conditional_width']
    tables={n:old['main'][n] for n in ['REAL','SIM','GEN_W0','R3B']}
    tables.update({n:r8['tables'][n] for n in ['JOINT','JOINT_SIGNAL']});tables.update({n:r8c['tables'][n] for n in CAND[-2:]})
    checks=[]
    for name,d in {**domains,'REAL_all':allreal}.items():
        table=old['REAL_EVAL_all'] if name=='REAL_all' else tables[name]
        good=d['valid'];ps=d['features']['PSD'][good];q=d['features']['Qtotal_mVns'][good];z=np.log(q)
        bins=np.searchsorted(edges,z,side='right')-1;bins[z==edges[-1]]=7
        assert table['N']==len(ps),name+' old density population drift'
        grid=np.asarray(table['histogram_edges_PSD'])
        for i,row in enumerate(table['slices']):
            mask=bins==i;hist,_=np.histogram(ps[mask],bins=grid)
            assert row['N']==int(mask.sum()) and np.array_equal(hist,row['histogram_counts']),name+' density slice drift'
        checks.append({'domain':name,'N':len(ps),'eight_counts_and_histograms_match_saved_tables':True,'new_fit':0})
    common_out={}
    for family,names in CFG['width_common_support_sets'].items():
        rows=[]
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                common=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field] is not None for n in names)]
                error={n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in common])) if common else None for n in names if n!='REAL'}
                rows.append({'component':k,'field':field,'common_qualified_bin_ids_1based':[i+1 for i in common],'common_n':len(common),'mean_abs_error':error})
        common_out[family]=rows
    result={'edges_logQ':edges,'tables':tables,'REAL_all_supplement':old['REAL_EVAL_all'],'REAL_MAIN_group_bootstrap':old['REAL_MAIN_group_bootstrap'],
            'checks':checks,'common_support_paired_errors':common_out,'all_coverage_and_failure_reasons':{n:{str(s['bin']+1):{'valid':s['valid'],'N':s['N'],'failure_reasons':s['failure_reasons']} for s in t['slices']} for n,t in tables.items()},
            'density_fit_reused_exact_data':True,'new_density_fits':0,'not_particle_truth':True}
    write('results/ENERGY_WIDTH_METRICS.json',result);return result
def main():
    ready=json.loads((ROOT/'READINESS.json').read_text())
    for s in ready['execution_sources']:
        b=pathlib.Path(s['path']).read_bytes()
        assert hashlib.sha256(b).hexdigest()==s['sha256'],'execution source drift'
    assert hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest()==ready['config_sha256']
    ref=jread('CAL_reference');assert not ref['slots'][4]['valid'];edges=np.asarray(jread('width_definition')['edges_logQ'])
    scale=jread('STANDARDIZATION');assert scale['n']==2592 and np.all(np.asarray(scale['population_sd'])>0)
    identity=jread('EVAL_identity')
    with npz('EVAL_EXTRACT') as z:v=z['window_V_f32'].copy();qc=z['QC_reason_u16'].copy();orig=z['original_index'].copy()
    COUNTER['EVAL_records_decoded']=len(v);assert v.shape==(7600,480) and np.array_equal(orig,identity['original_index'])
    ids=np.asarray(identity['source_id']);groups=orig//50;allreal=aa.data(v,ids,groups,'REAL',ref,qc);assert int(allreal['valid'].sum())==7583
    with npz('EVAL_MAIN_SELECTION') as z:ix=z['selected_positions'].copy();savedorig=z['original_index'].copy();savedids=z['source_id'].copy()
    assert len(ix)==2000 and np.array_equal(orig[ix],savedorig) and np.array_equal(ids[ix],savedids)
    eligible=np.flatnonzero(allreal['valid']);assert np.array_equal(ix,eligible[full_order(ids[eligible],'R4-main-real-v1')[:2000]])
    real=take(allreal,ix);real['CAL_edges']=ref['CAL_charge_quantile_edges_mV_ns']
    with npz('SIM_identity') as z:
        simids=z['sim_event_ids'].copy();peak=z['normalization_peak_V'].copy();raw=z['raw'].copy()
        meta={k:z[k].copy() for k in ['source_label','sim_energy','edep_ee_MeV','edep_hadron_MeV','physical_r_h']}
    assert len(simids)==2000 and len(set(simids))==2000
    source={}
    with npz('generation_inputs') as z:
        for k in ['G4_condition','Qsource_Vns','sim_event_ids']:source[k]=z[k].copy()
    assert np.array_equal(source['sim_event_ids'],simids)
    domains={'SIM':aa.data(raw.astype('f8')*peak[:,None],simids,np.arange(2000),'SIM',ref),'REAL':real};views={}
    with npz('GEN_W0') as z:
        assert np.array_equal(z['sim_event_ids'],simids) and np.array_equal(z['before'],raw)
        domains['GEN_W0']=aa.data(z['after'].astype('f8')*peak[:,None],simids,np.arange(2000),'SIM',ref)
    for name in CAND:
        with npz(name) as z:
            assert np.array_equal(simids,z['sim_event_ids']),name+' parent order drift'
            mainkey='waveforms_V';waves=[z[k].copy() for k in [mainkey,'independent_view_2_V','independent_view_3_V']]
        views[name]=[ev.features(w) for w in waves];domains[name]=aa.data(waves[0],simids,np.arange(2000),'SIM',ref)
    stats,comparison=ev.characterize(domains,ref);allstats,_=ev.characterize({'REAL':allreal},ref)
    extra={n:dc.native_statistics(d,ref) for n,d in domains.items()}
    supplementary={n:compare(d,allreal) for n,d in domains.items() if n!='REAL'}
    joints={n:joint(d,real,scale) for n,d in domains.items() if n!='REAL'}
    alljoints={n:joint(d,allreal,scale) for n,d in domains.items() if n!='REAL'}
    rr,first=rr_reference(allreal,ref,scale);ev.log('R9 fixed64 same-N reference complete')
    rng=np.random.default_rng(202610081);t=np.arange(480);tail=np.where(t>=120,np.exp(-(t-120)/80),0.)
    perturb=real['V']+.02*real['features']['peak_V'][:,None]*tail[None,:]+.015*real['features']['peak_V'][:,None]*rng.normal(size=real['V'].shape)
    positive=aa.data(perturb,real['ids'],real['groups'],'REAL',ref)
    classification,paired,clsmeta=classifiers(domains,real,first,positive)
    widths_result=widths(domains,allreal,edges)
    nominal,association,nominal_region=aa.source_nominal(domains,source['Qsource_Vns'],source['G4_condition'])
    variation=aa.parent_variation(views)
    pvdata={};r8=jread('R8_metrics');r8c=jread('R8C_metrics')
    for n in ['JOINT','JOINT_SIGNAL']:pvdata[n]={'three_view':variation[n],'training_selection':r8['training'][n]}
    for n in CAND[-2:]:pvdata[n]={'three_view':variation[n],'training_selection':r8c['training'][n],'source_input_controls':r8c['source_input_controls'][n],'shuffle_self_permutation':jread('R8C_layout')['source_self_permutation'][n]}
    controls={'R8_replay':jread('R8_replay'),'R8C_replay':jread('R8C_replay'),'R3B_fixed_condition_pairs':jread('R3B_fixed_pairs'),
              'newer_arm_controlled_energy_composition_pairs':'not stored as a separate controlled pair; bundle shuffle/replacement does not isolate energy or composition',
              'R8_source_controls':'R8 paired mechanism on/off, TRAIN-only selected checkpoint; no new source shuffle in R8','source_controls_not_truth':True}
    per={'sim_event_ids':simids,'REAL_main_ids':real['ids'],'REAL_main_original_index':orig[ix],'source_nominal_region':nominal_region}
    for n,d in {**domains,'REAL_all':allreal}.items():
        for k,a in {**d['features'],**d['extra']}.items():per[n+'__'+k]=a
        for k in ['valid','region']:per[n+'__'+k]=d[k]
    np.savez_compressed(OUT/'PER_EVENT_FEATURES.npz',**per)
    write('results/DATA_ROLES.json',{'source_parents2000':simids,'EVAL_original_records':7600,'EVAL_QC':7583,'REAL_main_saved_positions':ix,'REAL_main_original_index':orig[ix],'REAL_main_source_id':real['ids'],
          'same_acquisition':True,'group_formula':'original_index//50 artificial record blocks','new_response_fit':0,'CAL_or_TRAIN_statistic_refit':0,'EVAL_feedback_to_response':False,
          'legacy_source_label_counts':{str(s):int(np.sum(meta['source_label']==s)) for s in np.unique(meta['source_label'])},'gamma_label_is_legacy_non2112_not_PDG22_truth':True})
    result={'status':'complete','question':'all5 frozen responses on previously viewed same-acquisition benchmark','frozen_SHA256':CFG['frozen_SHA256'],'frozen_UTC':CFG['frozen_UTC'],
            'domains':stats,'domain_vs_REAL':comparison,'regional_local_statistics':extra,'REAL_all_supplement':allstats['REAL'],'domain_vs_REAL_all':supplementary,
            'joint_sliced_W1':joints,'joint_sliced_W1_REAL_all':alljoints,'REAL_REAL_summary':rr['summary'],'REAL_REAL_joint':rr['joint_sliced_W1'],
            'C2ST':classification,'C2ST_common_selection':clsmeta,'paired_C2ST_vs_R3B':paired,'energy_conditional_width':widths_result,
            'source_nominal_condition':nominal,'source_association':association,'three_view_within_parent_variation':variation,'previous_control_and_replay_evidence':controls,
            'CAL_charge_edges_mVns':ref['CAL_charge_quantile_edges_mV_ns'],'CAL_reference_valid':[s['valid'] for s in ref['slots']],
            'q4_no_fallback':True,'unknown_all_preserved':True,'EVAL_original':7600,'EVAL_QC':7583,'REAL_main':2000,
            'retrospective_same_acquisition_not_blind_or_external':True,'response_updates':0,'EVAL_feedback':False,'SSH':0,'new_G4':0,'counters':COUNTER,
            'versions':{'Python':sys.version,'NumPy':np.__version__,'sklearn':sklearn.__version__},'worker_PID':os.getpid(),'numeric_before_plots_s':time.monotonic()-START}
    write('metrics.json',result);write('INPUTS.json',READS)
    plot.plots(domains,allreal,result,ref,FIG)
    result['numeric_elapsed_s']=time.monotonic()-START;write('metrics.json',result)
    write('NUMERIC_RECEIPT.json',{'status':'complete','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'versions':result['versions'],'response_updates':0,'SSH':0,'new_G4':0})
    ev.log('R9 complete '+str(result['numeric_elapsed_s'])+'s')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write('FAILURE.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'error':repr(e),'traceback':traceback.format_exc(),'counters':COUNTER,'scientific_definition_not_changed':True});raise
