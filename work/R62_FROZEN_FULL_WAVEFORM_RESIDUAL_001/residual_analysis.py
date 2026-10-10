import numpy as np
from scipy.stats import spearmanr
from noise_core import q_np
from residual_math import *

def path_description(paths,reference,bins_s,bins_r):
    rf=features(reference);rb=boundaries(reference);rows=[];private={'REAL_reference_mean_V':reference.mean(0),'REAL_reference_SD_V':reference.std(0,ddof=1)}
    scopes=[('all',None)]+[('region'+str(b),b)for b in REGIONS]
    for domain,stages in paths.items():
        bins=bins_s if domain=='SOURCE'else bins_r
        for stage,v in stages.items():
            fs=features(v);bv=boundaries(v)
            private[domain+'_'+stage+'_mean_V']=v.mean(0);private[domain+'_'+stage+'_SD_V']=v.std(0,ddof=1)
            for scope,b in scopes:
                a=np.ones(len(v),bool)if b is None else bins==b;c=np.ones(len(reference),bool)if b is None else bins_r==b
                row={'domain':domain,'stage':stage,'scope':scope,'original_region':b,'N':int(a.sum()),'REAL_N':int(c.sum()),'feature_mean_W1':{},'voltage_segment_mean_W1':{}}
                for name,x,y in zip(FEATURES,fs,rf):
                    w=exact_w1_columns(x[a],y[c]);private['_'.join([domain,stage,scope,name,'W1'])]=w;row['feature_mean_W1'][name]=float(w.mean())
                    if name=='V480':row['voltage_segment_mean_W1']={str(lo)+':'+str(hi):float(w[lo:hi].mean())for lo,hi in [(0,100),(100,150),(150,400),(400,480)]}
                row['boundary_W1_V']=exact_w1_columns(bv[a],rb[c]).tolist();rows.append(row)
    find=lambda stage:next(r for r in rows if r['domain']=='SOURCE'and r['stage']==stage and r['scope']=='all')
    change=np.array(find('neural_raw')['boundary_W1_V'])-np.array(find('noise_initial')['boundary_W1_V'])
    selected=[1,5,6];flags=[bool(change[j]>1/16383)for j in selected]
    return {'rows':rows,'boundary_names':BOUNDARIES,'neural_boundary_change_V':dict(zip(BOUNDARIES,change.tolist())),
      'neural_boundary_increase':{'passed_components':int(sum(flags)),'required':2,'selected_names':[BOUNDARIES[j]for j in selected],'flags':flags,'activated':sum(flags)>=2},
      'stage_comparisons_are_diagnostics_not_new_acceptance_representations':True},private

def baseline_description(source,control,reference,bins_s,bins_r,rng,consume):
    sf,sb,sp=baseline_features(source);cf,cb,cp=baseline_features(control);rf,rb,rp=baseline_features(reference)
    rows=[];priv={'SOURCE_baseline_features':sf,'REAL_control_baseline_features':cf,'REAL_reference_baseline_features':rf}
    for scope,b in [('all',None)]+[('region'+str(b),b)for b in REGIONS]:
        a=np.arange(len(source))if b is None else np.flatnonzero(bins_s==b)
        c=np.arange(len(reference))if b is None else np.flatnonzero(bins_r==b)
        rc=covariance(rb[c]);sc=covariance(sb[a]);cc=covariance(cb[c]);null=[];ix=[]
        for k in range(32):
            chosen=c[rng.integers(0,len(c),size=len(a))].astype('<u2');consume('covariance_reference_rows',chosen);ix.append(chosen)
            value=covariance_error(covariance(rb[chosen]),rc);assert value is not None;null.append(value)
        src=covariance_error(sc,rc);ctl=covariance_error(cc,rc);assert src is not None and ctl is not None
        row={'scope':scope,'original_region':b,'SOURCE_N':len(a),'REAL_N':len(c),'SOURCE_covariance_relative_Frobenius':src,'REAL_control_covariance_relative_Frobenius':ctl,
          'empirical_reference_errors':null,'empirical_reference_quantiles':np.quantile(null,[0,.025,.5,.975,1]).tolist(),
          'SOURCE_exceeds_all_32_references_and_REAL_control':bool(src>max(null)and src>ctl),
          'SOURCE_baseline_feature_W1':exact_w1_columns(sf[a],rf[c]).tolist(),'REAL_control_baseline_feature_W1':exact_w1_columns(cf[c],rf[c]).tolist(),
          'zero_power_N':{'SOURCE':int(np.count_nonzero(sp[a]==0)),'REAL_control':int(np.count_nonzero(cp[c]==0)),'REAL_reference':int(np.count_nonzero(rp[c]==0))}}
        rows.append(row)
        for name,x in [('SOURCE',sc),('REAL_control',cc),('REAL_reference',rc)]:priv[scope+'_'+name+'_covariance_V2']=x
        for name,x in [('SOURCE',sb[a]),('REAL_control',cb[c]),('REAL_reference',rb[c])]:priv[scope+'_'+name+'_mean_baseline_V']=x.mean(0)
        priv[scope+'_reference_indices']=np.asarray(ix,dtype='u2')
    flagged=sum(r['SOURCE_exceeds_all_32_references_and_REAL_control']for r in rows if r['original_region']in range(8))
    return {'rows':rows,'feature_names':['RMS_V']+['ACF'+str(j)for j in range(1,21)]+['rFFT_band'+str(j)for j in range(1,11)],
      'frequency_cycles_per_sample':[[i/100,(i+4)/100]for i in range(1,51,5)],'baseline_joint_residual':{'regions':int(flagged),'required':6,'activated':flagged>=6},
      'reference_scope':'32 resamples from empirical REAL TRAIN, matched SOURCE N; not population confidence intervals or independent acquisitions'},priv

def loss_description(source,control,reference,bins_s,bins_r,stats,directions,streams,consume):
    s=normalized_feature_bank(source,bins_s,stats);c=normalized_feature_bank(control,bins_r,stats);r=normalized_feature_bank(reference,bins_r,stats)
    gs=[np.flatnonzero(bins_s==b)for b in REGIONS];gr=[np.flatnonzero(bins_r==b)for b in REGIONS]
    mass=np.asarray(stats['source_counts'])/np.sum(stats['source_counts'])
    full=np.stack([objective_table(s,r,gs,gr,directions),objective_table(r,r,gr,gr,directions),objective_table(c,r,gr,gr,directions)])
    batches=[];indices={k:[]for k in ['batch_SOURCE_rows','batch_REAL_A_rows','batch_REAL_B_rows','batch_REAL_control_rows']}
    for k in range(32):
        d={}
        for key,groups in [('batch_SOURCE_rows',gs),('batch_REAL_A_rows',gr),('batch_REAL_B_rows',gr),('batch_REAL_control_rows',gr)]:
            d[key]=np.stack([g[streams[key].integers(0,len(g),32)]for g in groups]).astype('<u2');consume(key,d[key]);indices[key].append(d[key])
        batches.append(np.stack([objective_table(s,r,d['batch_SOURCE_rows'],d['batch_REAL_A_rows'],directions),
          objective_table(r,r,d['batch_REAL_B_rows'],d['batch_REAL_A_rows'],directions),objective_table(c,r,d['batch_REAL_control_rows'],d['batch_REAL_A_rows'],directions)]))
    batches=np.asarray(batches);assert np.isfinite(full).all()and np.isfinite(batches).all()and np.max(full[1])==0
    total=(batches.mean((3,4))*mass[None,None,:]).sum(2)
    complete=(full.mean((2,3))*mass[None,:]).sum(1)
    per_feature=(batches.mean(4)*mass[None,None,:,None]).sum(2)
    names=['SOURCE_REAL','REAL_REAL','REAL_control_REAL']
    private={'full_tables':full,'batch_tables':batches,'batch_total_losses':total,'batch_feature_losses':per_feature,'source_original_region_mass':mass}
    private.update({k:np.asarray(v,dtype='u2')for k,v in indices.items()})
    rows={n:{'full_empirical_loss':float(complete[j]),'batch_loss_summary':summary(total[:,j]),'batch_losses':total[:,j].tolist(),
      'full_per_feature':(full[j].mean(2)*mass[:,None]).sum(0).tolist(),
      'full_region_contributions':(full[j].mean((1,2))*mass).tolist(),'batch_per_feature_mean':per_feature[:,j].mean(0).tolist(),
      'batch_per_feature_SD':per_feature[:,j].std(0,ddof=1).tolist()}for j,n in enumerate(names)}
    excess=total[:,0]-total[:,1]
    return {'comparisons':rows,'SOURCE_minus_REAL_REAL_paired_batch_excess':summary(excess),'fixed_batch_count':32,'rows_per_original_region':32,
      'source_original_region_mass':mass.tolist(),'fixed_directions_per_view':64,'new_neural_forward_rows':0,'new_fits':0,
      'scope':'Frozen endpoint resampling and fixed directions; not the training trajectory or independent-sample confidence bounds'},private

def psd_description(source,control,reference,raw_source,ids,h,bins_s,bins_r,stats):
    sp=psd(source);ip=psd(control);rp=psd(reference);original=psd(raw_source);sq=q_np(raw_source);oq=q_np(source)
    rows=[]
    for b in REGIONS:
        a=bins_s==b;c=bins_r==b
        rows.append({'region':b,'SOURCE_N':int(a.sum()),'REAL_N':int(c.sum()),'SOURCE_W1':float(exact_w1_columns(sp[a],rp[c])[0]),'REAL_control_W1':float(exact_w1_columns(ip[c],rp[c])[0])})
    valleys=[]
    for j,b in enumerate([2,3]):
        a=(bins_s==b)&(h==1);lo,hi=stats['valley_bounds'][j]
        entry={'region':b,'N':int(a.sum()),'bounds':[float(lo),float(hi)],'edge_half_width':.004}
        for name,v in [('original',original),('output',sp)]:
            entry[name+'_valley_mass']=float(np.mean((v[a]>=lo)&(v[a]<=hi)))
            entry[name+'_low_edge_mass']=float(np.mean(np.abs(v[a]-lo)<=.004));entry[name+'_high_edge_mass']=float(np.mean(np.abs(v[a]-hi)<=.004))
        valleys.append(entry)
    out={'overall_SOURCE_W1':float(exact_w1_columns(sp,rp)[0]),'common_Q1_Q7_mean_W1':float(np.mean([r['SOURCE_W1']for r in rows[1:8]])),
      'overall_REAL_control_W1':float(exact_w1_columns(ip,rp)[0]),'regions':rows,'valleys':valleys,'source_PSD_rank':float(spearmanr(original,sp).statistic),'source_Q_rank':float(spearmanr(sq,oq).statistic),'density_fits':0}
    private={'source_ids':ids,'source_h':h,'source_raw_region':bins_s,'REAL_raw_region':bins_r,'source_original_q':sq,'source_output_q':oq,'source_original_psd':original,'source_output_psd':sp,'REAL_reference_psd':rp,'REAL_output_psd':ip}
    return out,private
