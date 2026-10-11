from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
import evaluation as ev
import diagnostic_core as dc
import energy_width as ew
from common_math import distances
from canonical import canonical

def data(v,ids,groups,typ,ref,qc=None,adc_domain=True):
    d=ev.features(canonical(v) if adc_domain else v);d['ids']=np.asarray(ids);d['groups']=np.asarray(groups);d['cluster']=np.array([typ+'|'+str(g) for g in groups])
    q=d['features']['Qtotal_mVns'];d['region']=np.searchsorted(ref['CAL_charge_quantile_edges_mV_ns'],q,side='right')
    d['region'][~np.isfinite(q)]=255
    if qc is not None:d['valid'] &= np.asarray(qc)==0
    return dc.extra_features(d)

def paired_bootstrap(a,b,cfg):
    assert np.array_equal(a['label'],b['label']) and np.array_equal(a['cluster'],b['cluster'])
    y=a['label'];p=a['probability'];q=b['probability'];_,inv=np.unique(a['cluster'],return_inverse=True);n=int(inv.max()+1)
    rng=np.random.default_rng(cfg['bootstrap_seed']);acc=[];aucs=[]
    _,tp=np.unique(p,return_inverse=True);_,tq=np.unique(q,return_inverse=True)
    def score(prob,tie,w):
        n0=w[y==0].sum();n1=w[y==1].sum()
        ba=.5*(w[(y==0)&(prob<.5)].sum()/n0+w[(y==1)&(prob>=.5)].sum()/n1)
        neg=np.bincount(tie[y==0],weights=w[y==0],minlength=int(tie.max()+1));pos=np.bincount(tie[y==1],weights=w[y==1],minlength=int(tie.max()+1))
        auc=float(np.dot(pos,np.cumsum(neg)-.5*neg)/(n0*n1));return ba,auc
    point_a=score(p,tp,np.ones(len(y)));point_b=score(q,tq,np.ones(len(y)))
    for _ in range(cfg['bootstrap_repeats']):
        w=np.bincount(rng.integers(n,size=n),minlength=n)[inv]
        if w[y==0].sum()==0 or w[y==1].sum()==0:continue
        aa=score(p,tp,w);bb=score(q,tq,w);acc.append(aa[0]-bb[0]);aucs.append(aa[1]-bb[1])
    return {'BA_delta_first_minus_second':point_a[0]-point_b[0],'BA_delta_95CI':np.quantile(acc,[.025,.975]),'AUC_delta_first_minus_second':point_a[1]-point_b[1],'AUC_delta_95CI':np.quantile(aucs,[.025,.975]),'valid_replicates':len(acc),'clusters':n,'paired_test_rows_and_groups':True,'conditional_on_two_fitted_classifiers':True,'response_training_uncertainty':False}

def same_N(domains,realtrain):
    names=[n for n in domains if n!='REAL'];full={**domains,'TRAIN_REFERENCE':realtrain};fields={**domains['REAL']['features'],**domains['REAL']['extra']};out={}
    for r in [None,0,1,2,3]:
        key='overall' if r is None else 'q'+str(r+1);inds={}
        for name,d in full.items():
            mask=d['valid'] if r is None else d['valid']&(d['region']==r)
            ix=np.flatnonzero(mask);inds[name]=ix[ev.order(d['ids'][ix],'R8-conditional-sameN')]
        n=min(map(len,inds.values()));dout={'N_per_domain':n,'source_counts':{k:len(v) for k,v in inds.items()},'comparison':{}}
        for name in [*names,'TRAIN_REFERENCE']:
            one={};a=full[name];b=full['REAL']
            for k in fields:
                va={**a['features'],**a['extra']}[k][inds[name][:n]];vb={**b['features'],**b['extra']}[k][inds['REAL'][:n]]
                one[k]={'candidate':ev.summary(va),'REAL':ev.summary(vb),'distance':distances(va[np.isfinite(va)],vb[np.isfinite(vb)])}
            dout['comparison'][name]=one
        out[key]=dout
    return out

def widths(domains,old,edges,viewfeatures,counter):
    tables={'REAL':old['PSD_density_widths']['native']['REAL'],'R3B':old['PSD_density_widths']['native']['GEN']}
    for name in ['JOINT','JOINT_SIGNAL']:
        d=domains[name];mask=d['valid'];tables[name]=ew.section(d['features']['PSD'][mask],d['features']['Qtotal_mVns'][mask],edges)
        counter['density_EM_attempts']+=sum(x['fit_attempts'] for x in tables[name]['slices'])
    errors=[]
    for k in [0,1]:
        for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
            common=[]
            for i in range(8):
                rows=[tables[n]['slices'][i] for n in ['REAL','R3B','JOINT','JOINT_SIGNAL']]
                if all(r['valid'] and len(r['components'])==2 and r['components'][k][field] is not None for r in rows):common.append(i)
            row={'component':k,'field':field,'common_qualified_bin_ids_1based':[i+1 for i in common],'common_n':len(common),'mean_abs_error':{}}
            for n in ['R3B','JOINT','JOINT_SIGNAL']:
                pairs=[abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in common]
                row['mean_abs_error'][n]=float(np.mean(pairs)) if pairs else None
            errors.append(row)
    return {'edges_logQ':edges,'edges_mVns':np.exp(edges),'tables':tables,'common_support_paired_errors':errors,'source_parent_three_view_width':{n:ew.parent_width(v,edges) for n,v in viewfeatures.items()},'old_REAL_R3B_fit_reused':True,'not_particle_truth_or_CAL_refit':True}

def source_nominal(domains,qsource,c):
    out={};association={};real=domains['REAL'];edges=real['CAL_edges'];nominal=np.searchsorted(edges,np.asarray(qsource)*1000,side='right')
    for name in [n for n in domains if n!='REAL']:
        d=domains[name];table=np.zeros((4,5),dtype=int);blocks={}
        for r in range(4):
            mask=nominal==r;good=mask&d['valid'];target=real['valid']&(real['region']==r)
            for j in range(4):table[r,j]=int(np.sum(mask&d['valid']&(d['region']==j)))
            table[r,4]=int(np.sum(mask&(~d['valid']|~np.isin(d['region'],[0,1,2,3]))))
            fields={**d['features'],**d['extra']}
            blocks['q'+str(r+1)]={'entire_source_parent_count':int(mask.sum()),'QC_n':int(good.sum()),'invalid_n':int((mask&~d['valid']).sum()),'REAL_own_q_QC_n':int(target.sum()),
                'features':{k:ev.summary(v[good]) for k,v in fields.items()},'vs_REAL_output_own_q':{k:distances(v[good&np.isfinite(v)],{**real['features'],**real['extra']}[k][target&np.isfinite({**real['features'],**real['extra']}[k])]) for k,v in fields.items()}}
        out[name]={'definition':'nominal source charge=Qsource_Vns*1000, frozen CAL edges; all parents retained; REAL reference uses its observed own charge','migration_rows_nominal_cols_output_q1_q2_q3_q4_invalid':table,'regions':blocks}
        assoc={}
        for cond,x in [('log_visible_energy_centered',c[:,0]),('visible_hadron_fraction',c[:,1]),('clean_PSD',c[:,2]),('clean_tail_charge_fraction',c[:,3])]:
            aa={}
            for key,y in {**d['features'],**d['extra']}.items():
                good=d['valid']&np.isfinite(x)&np.isfinite(y)
                aa[key]={'n':int(good.sum()),'Spearman_rho':float(spearmanr(x[good],y[good]).statistic) if good.sum()>2 and np.unique(x[good]).size>1 and np.unique(y[good]).size>1 else None}
            assoc[cond]=aa
        association[name]=assoc
    return out,association,nominal

def parent_variation(viewfeatures):
    out={}
    for n,features in viewfeatures.items():
        row={}
        for field in features[0]['features']:
            values=np.stack([f['features'][field] for f in features]);valid=np.isfinite(values).all(axis=0)&np.logical_and.reduce([f['valid'] for f in features])
            std=values[:,valid].std(axis=0,ddof=1)
            row[field]={'parent_count':int(valid.sum()),'undefined_parent_count':len(valid)-int(valid.sum()),'within_parent_SD':ev.summary(std)}
        out[n]={'random_views':3,'not_independent_real_events':True,'features':row}
    return out
