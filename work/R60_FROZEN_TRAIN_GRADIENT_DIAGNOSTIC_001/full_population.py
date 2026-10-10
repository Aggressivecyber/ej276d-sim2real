import hashlib
import numpy as np
from scipy.stats import spearmanr
from scipy.special import expit
from noise_core import q_np,bins_for
from distribution_loss import CATEGORIES
from diagnostic_math import psd_np,noise_np,canonical_np,rounding_psd_interval,exact_w1_columns,full_feature_summary,order_population

def describe(source,real,source_ids,h,sraw,rraw,outS,outR,metaS,metaR,stats,edges,directions):
    ys=canonical_np(outS,True);yr=canonical_np(outR,True);reference=canonical_np(real,False)
    sp=psd_np(source);rp=psd_np(reference);yp=psd_np(ys);ip=psd_np(yr)
    sq=q_np(source);oq=q_np(ys);rq=q_np(real);iq=q_np(yr)
    assert all(np.isfinite(a).all()for a in [sp,rp,yp,ip,sq,oq,rq,iq])and np.all(oq>0)and np.all(iq>0)
    low,high=rounding_psd_interval(source)
    sn=noise_np(ys);rn=noise_np(reference);in_=noise_np(yr)
    rows=[]
    for b in CATEGORIES:
        a=sraw==b;c=rraw==b
        rows.append({'raw_region':b,'SOURCE_N':int(a.sum()),'REAL_N':int(c.sum()),
          'SOURCE_PSD_W1':float(exact_w1_columns(yp[a],rp[c])[0]),'SOURCE_raw_input_PSD_W1':float(exact_w1_columns(sp[a],rp[c])[0]),
          'same_treated_REAL_PSD_W1':float(exact_w1_columns(ip[c],rp[c])[0]),
          'SOURCE_projection_range_fraction':float(metaS['constraint_projection_range_active'][a].mean()),
          'SOURCE_projection_RMS_V_mean':float(metaS['constraint_projection_RMS_V'][a].mean()),
          'SOURCE_original_overrange_fraction':float(np.any(np.abs(source[a])>.5,axis=1).mean()),
          'SOURCE_output_PSD_quantiles':np.quantile(yp[a],[0,.05,.5,.95,1]).tolist(),
          'same_treated_REAL_PSD_quantiles':np.quantile(ip[c],[0,.05,.5,.95,1]).tolist()})
    valleys=[]
    for j,b in enumerate([2,3]):
        ix=(sraw==b)&(h==1);l,u=stats['valley_bounds'][j]
        valleys.append({'bin_1based':b+1,'N':int(ix.sum()),'bounds':[float(l),float(u)],
          'raw_SOURCE_valley_fraction':float(np.mean((sp[ix]>=l)&(sp[ix]<=u))),
          'output_valley_fraction':float(np.mean((yp[ix]>=l)&(yp[ix]<=u))),
          'output_soft_valley_fraction':float(np.mean(expit((yp[ix]-l)/.002)*expit((u-yp[ix])/.002)))})
    feature=full_feature_summary(ys,reference,sraw,rraw,stats,directions)
    order=order_population(sp,yp,low,high,h)
    _,ob,_=bins_for(ys,edges);_,ib,_=bins_for(yr,edges)
    noise={}
    for j,name in enumerate(['baseline_RMS_V','ACF1']):
        noise[name]={'SOURCE_W1':float(exact_w1_columns(sn[:,j],rn[:,j])[0]),'same_treated_REAL_W1':float(exact_w1_columns(in_[:,j],rn[:,j])[0]),
          'frozen_TRAIN_scale':float(stats['noise_scale'][j])}
    projection={}
    for label,meta,n in [('SOURCE',metaS,len(source)),('same_treated_REAL',metaR,len(real))]:
        projection[label]={'N':n,'range_active_events':int(meta['constraint_projection_range_active'].sum()),
          'range_active_fraction':float(meta['constraint_projection_range_active'].mean()),
          'RMS_correction_V_mean':float(meta['constraint_projection_RMS_V'].mean()),
          'RMS_correction_V_max':float(meta['constraint_projection_RMS_V'].max()),
          'rail_sample_count':int(meta['constraint_projection_rail_sample_count'].sum()),
          'preADC_relative_Q_error_max':float(meta['preADC_charge_relative_error_f32'].max())}
    result={'scope':'Original equal-parent TRAIN, one fixed noise realization per parent; no original A-F acceptance claim.',
      'SOURCE_N':len(source),'REAL_N':len(real),'overall_SOURCE_PSD_W1':float(exact_w1_columns(yp,rp)[0]),
      'common_Q1_Q7_mean_SOURCE_PSD_W1':float(np.mean([x['SOURCE_PSD_W1']for x in rows[1:8]])),
      'overall_same_treated_REAL_PSD_W1':float(exact_w1_columns(ip,rp)[0]),'per_original_region':rows,
      'same_treated_REAL_mV_RMS_error':float(np.sqrt(np.mean((yr-reference)**2))*1000),
      'source_PSD_rank_rho':float(spearmanr(sp,yp).statistic),'source_Q_rank_rho':float(spearmanr(sq,oq).statistic),
      'source_valleys':valleys,'noise':noise,'projection':projection,'full_window':feature,'source_order':order,
      'float32_PSD_interval_width_quantiles':np.quantile(high-low,[0,.05,.5,.95,1]).tolist(),
      'output_charge_bin_migrations':{'SOURCE':int(np.count_nonzero(ob!=sraw)),'same_treated_REAL':int(np.count_nonzero(ib!=rraw))},
      'canonical_output_SHA256':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()for k,v in [('SOURCE',ys),('same_treated_REAL',yr)]},
      'complete_480_mean_SD':{name:{'mean_V':v.mean(0).tolist(),'SD_V':v.std(0,ddof=1).tolist()}for name,v in [('SOURCE_raw_input',source.astype('f8')),('SOURCE_output',ys),('REAL_reference',reference),('same_treated_REAL',yr)]},
      'source_h_only_for_association_not_network_input':True,'REAL_particle_truth_assigned':False}
    private={'source_ids':source_ids,'source_h':h,'source_raw_region':sraw,'REAL_raw_region':rraw,
      'source_original_q':sq,'source_original_psd':sp,'source_PSD_interval_low':low,'source_PSD_interval_high':high,
      'source_output_q':oq,'source_output_psd':yp,'source_output_noise':sn,'source_output_region':ob,
      'REAL_original_q':rq,'REAL_reference_psd':rp,'REAL_reference_noise':rn,'REAL_output_q':iq,'REAL_output_psd':ip,'REAL_output_noise':in_,'REAL_output_region':ib}
    for label,meta in [('source',metaS),('REAL',metaR)]:
        for key in ['constraint_projection_range_active','constraint_projection_RMS_V','constraint_projection_maxabs_V','constraint_projection_rail_sample_count','preADC_charge_relative_error_f32']:
            private[label+'_'+key]=meta[key]
    return result,private
