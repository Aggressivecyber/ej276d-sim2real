"""Fixed DEV comparison; never feeds metrics back to the response training."""
from __future__ import annotations
import json, pathlib, time
import numpy as np
from scipy.stats import spearmanr
import evaluation as ev
import diagnostic_core as dc
import assessment as aa
import energy_width as ew


def write(path,obj):
    path.write_text(json.dumps(ev.safe(obj),ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def inputs(cfg,root):
    base=pathlib.Path(cfg['remote_input_root'])
    real=np.fromfile(base/'REAL_V',dtype='<f4',count=5200*480).reshape(5200,480)
    qc=np.fromfile(base/'REAL_QC',dtype='<u2',count=5200)
    groups=np.fromfile(base/'REAL_groups',dtype='<u2',count=5200)
    idx=np.fromfile(base/'REAL_original_index',dtype='<u4',count=5200)
    roles=np.fromfile(base/'REAL_roles',dtype='u1',count=5200)
    assert np.all(roles[:2000]==0) and np.all(roles[2000:]==1) and np.array_equal(groups,idx//50)
    vg=json.loads((base/'TRAIN_roles').read_text())['REAL_internal_validation_artificial_groups']
    fit=(qc[2000:]==0)&~np.isin(groups[2000:],vg)
    assert fit.sum()==2592 and not set(groups[:2000])&set(groups[2000:])
    ref=json.loads((base/'CAL_reference').read_text())
    edges=np.array(json.loads((base/'width_definition').read_text())['edges_logQ'])
    selection=json.loads((base/'R7_selections').read_text())
    with np.load(base/'SIM_identity',allow_pickle=False) as z:rids=z['real_event_ids'].copy()
    with np.load(root/'results/generated.npz',allow_pickle=False) as z:g={k:z[k] for k in z.files}
    fitids=np.array(['RAW_ORIGINAL_INDEX|'+str(v) for v in idx[2000:][fit]])
    pick=ev.order(fitids,'R7-RR-TRAIN-2000')[:2000]
    assert np.array_equal(idx[2000:][fit][pick],selection['TRAIN_reference_original_indices'])
    domains={'REAL':aa.data(real[:2000],rids,groups[:2000],'REAL',ref,qc[:2000])}
    for name in ['R3B','NSF']:
        domains[name]=aa.data(g[name+'_V'],g['ids'],np.arange(2000),'SIM',ref)
    rfit=aa.data(real[2000:][fit],fitids,groups[2000:][fit],'REAL',ref)
    rr=aa.data(real[2000:][fit][pick],fitids[pick],groups[2000:][fit][pick],'REAL',ref)
    return domains,rfit,rr,ref,edges,selection


def c2st(domains,rr,selection,cfg,counter,out):
    sp={k:(np.array(v['train'],dtype='i4'),np.array(v['holdout'],dtype='i4')) for k,v in selection['indices'].items()}
    chosen={};counts={};names=['R3B','NSF']
    for phase in range(2):
        g=sp['GEN'][phase];g=g[domains['R3B']['valid'][g]&domains['NSF']['valid'][g]]
        r=sp['REAL'][phase];r=r[domains['REAL']['valid'][r]]
        t=sp['TRAIN_REFERENCE'][phase];t=t[rr['valid'][t]]
        cap=cfg['train_each_domain'] if phase==0 else cfg['holdout_each_domain']
        n=min(cap,len(g),len(r),len(t));chosen[phase]=(g[:n],r[:n],t[:n])
        counts['train' if phase==0 else 'holdout']={'N_each':n,'available_GEN_common_QC':len(g),'REAL':len(r),'TRAIN_REFERENCE':len(t)}
    supported=counts['train']['N_each']>=cfg['C2ST_min_train'] and counts['holdout']['N_each']>=cfg['C2ST_min_holdout']
    results={};pred={};paired={}
    if supported:
        gs=(chosen[0][0],chosen[1][0]);rs=(chosen[0][1],chosen[1][1]);ts=(chosen[0][2],chosen[1][2])
        assert not set(domains['REAL']['groups'][rs[0]])&set(domains['REAL']['groups'][rs[1]])
        for name in names+['REAL_REAL']:
            d=rr if name=='REAL_REAL' else domains[name];sel=ts if name=='REAL_REAL' else gs
            assert not set(d['groups'][sel[0]])&set(d['groups'][sel[1]])
            results[name]={};pred[name]={}
            for view in cfg['C2ST_views']:
                results[name][view],pred[name][view]=dc.fit_one(d,domains['REAL'],sel,rs,view,cfg,counter)
                ev.log(f'C2ST {name} {view} BA={results[name][view]["balanced_accuracy"]:.4f}')
        paired={v:aa.paired_bootstrap(pred['NSF'][v],pred['R3B'][v],cfg) for v in cfg['C2ST_views']}
    saved={n+'__'+v+'__'+k:a for n,views in pred.items() for v,fields in views.items() for k,a in fields.items()}
    saved.update({'phase'+str(k)+'_'+str(j):v for k,a in chosen.items() for j,v in enumerate(a)})
    np.savez_compressed(out/'C2ST_PRIVATE.npz',**saved)
    return {'supported':supported,'counts':counts,'results':results,'paired_NSF_minus_R3B':paired,
            'original_R7_positions_only':True,'RR_same_acquisition_TRAIN_reference_vs_DEV':True,
            'common_QC_and_same_N_all_arms':True,'RR_not_equivalence_threshold':True}


def widths(domains,edges,counter):
    tables={}
    for name,d in domains.items():
        counter['density_description_slots']+=8
        if counter['density_description_slots']>24:raise RuntimeError('width slots cap')
        use=d['valid']
        tables[name]=ew.section(d['features']['PSD'][use],d['features']['Qtotal_mVns'][use],edges)
    errors=[]
    for component in [0,1]:
        for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
            bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][component][field] is not None for n in domains)]
            errors.append({'component':component,'field':field,'common_bin_ids_1based':[i+1 for i in bins],
                'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][component][field]-tables['REAL']['slices'][i]['components'][component][field]) for i in bins])) if bins else None for n in ['R3B','NSF']}})
    return {'tables':tables,'edges_logQ':edges,'common_support_errors':errors,
            'not_particle_truth_or_CAL_refit':True,'qualified_bins_1based':{n:[i+1 for i,s in enumerate(t['slices']) if s['valid']] for n,t in tables.items()}}


def nearest(domains,reference,out):
    result={'TRAIN_reference_N':len(reference['V']),'all2000_no_QC_resampling':True,'not_proof_of_no_memory':True};priv={}
    for view,key in [('V','V'),('own_shape','shape')]:
        r=np.asarray(reference[key],dtype='f8');keep=np.isfinite(r).all(axis=1);ri=np.flatnonzero(keep);r=r[keep];r2=(r*r).sum(axis=1)
        for name,d in domains.items():
            a=np.asarray(d[key],dtype='f8');ix=np.flatnonzero(np.isfinite(a).all(axis=1));ds=np.full(len(a),np.nan);nn=np.full(len(a),-1,dtype='i4');exact=0
            for start in range(0,len(ix),64):
                sl=ix[start:start+64];x=a[sl]
                sq=((x*x).sum(axis=1)[:,None]+r2[None,:]-2*x@r.T)/480
                j=np.argmin(sq,axis=1);ds[sl]=np.sqrt(np.maximum(sq[np.arange(len(sl)),j],0));nn[sl]=ri[j]
                exact+=sum(np.array_equal(row,r[k]) for row,k in zip(x,j))
            result.setdefault(view,{})[name]={'all_query_n':len(a),'defined_query_n':len(ix),'RMS_distance':ev.summary(ds),'exact_to_nearest_count':int(exact)}
            priv[name+'__'+view+'__distance']=ds;priv[name+'__'+view+'__index']=nn
    np.savez_compressed(out/'NN_PRIVATE.npz',**priv)
    return result


def sensitivity(domains,ref,root):
    with np.load(root/'results/SENSITIVITY_PRIVATE.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
    with np.load(root/'results/GENERATION_INPUTS_PRIVATE.npz',allow_pickle=False) as z:conditions=z['condition'].copy();qsource=z['qsource'].copy()
    ix=p['indices'];results={}
    for name in ['R3B','NSF']:
        baseline=domains[name];results[name]={}
        for kind in ['permuted','fresh']:
            v=p[name+'_'+kind];d=ev.features(v);good=d['valid']&baseline['valid'][ix]
            row={'parents':len(ix),'both_QC':int(good.sum()),'waveform_RMS_change_V':ev.summary(np.sqrt(np.mean((v-baseline['V'][ix])**2,axis=1))),
                 'relative_waveform_RMS_change':ev.summary(np.sqrt(np.mean((v-baseline['V'][ix])**2,axis=1))/np.maximum(np.sqrt(np.mean(baseline['x'][ix]**2,axis=1)),1e-12))}
            row['feature_absolute_changes']={k:ev.summary(np.abs(a[good]-baseline['features'][k][ix][good])) for k,a in d['features'].items()}
            results[name][kind]=row
        association={}
        for label,x in [('source_log_energy',conditions[:,0]),('source_hadron_fraction',conditions[:,1]),('source_Q_mVns',qsource*1000)]:
            association[label]={}
            for field in ['PSD','Qtotal_mVns','rise_10_90_ns']:
                y=baseline['features'][field];good=baseline['valid']&np.isfinite(x)&np.isfinite(y)
                association[label][field]={'n':int(good.sum()),'Spearman_rho':float(spearmanr(x[good],y[good]).statistic) if good.sum()>2 and np.ptp(x[good])>0 and np.ptp(y[good])>0 else None}
        results[name]['association']=association
    return {'not_source_truth_or_causal_identification':True,'same64_parent_positions':True,'permuted_source_holds_noise_fixed':True,'fresh_noise_holds_source_fixed':True,'results':results}


def evaluate(cfg,root,counter):
    start=time.monotonic();out=root/'results'
    if (root/'METRICS_COMPACT.json').exists():raise RuntimeError('metrics already complete; do not refit')
    assert (root/'MODEL_FROZEN.json').exists(),'freeze checkpoint before DEV'
    domains,rfit,rr,ref,edges,selection=inputs(cfg,root)
    stats,dist=ev.characterize(domains,ref)
    local={n:dc.native_statistics(d,ref) for n,d in domains.items()}
    cs=c2st(domains,rr,selection,cfg,counter,out)
    wd=widths(domains,edges,counter);nn=nearest(domains,rfit,out);sens=sensitivity(domains,ref,root)
    features={n+'__'+k:v for n,d in domains.items() for k,v in {**d['features'],**d['extra']}.items()}
    features.update({n+'__valid':d['valid'] for n,d in domains.items()})
    np.savez_compressed(out/'FEATURES_PRIVATE.npz',**features)
    full={'domains':stats,'domain_vs_REAL':dist,'local':local,'C2ST':cs,'width':wd,'nearest':nn,'sensitivity':sens,
          'counters':counter,'evaluation_s':time.monotonic()-start,'EVAL_read':0,'same_acquisition_DEV_not_blind':True,
          'continuous_density_not_exact_ADC_mass':True,'single_registered_recipe':True}
    write(out/'metrics.json',full)
    compact={k:v for k,v in full.items() if k not in ['local','width']}
    compact['width']={k:v for k,v in wd.items() if k!='tables'}
    compact['width']['tables']={n:{k:v for k,v in t.items() if k!='histogram_edges_PSD'} for n,t in wd['tables'].items()}
    for t in compact['width']['tables'].values():
        t['slices']=[{k:v for k,v in s.items() if k!='histogram_counts'} for s in t['slices']]
    write(root/'METRICS_COMPACT.json',compact)
    from plot_trial import plots
    plots(domains,full,root)
    ev.log(f'evaluation complete in {time.monotonic()-start:.1f}s')
