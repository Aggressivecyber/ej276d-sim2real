"""Prespecified LONG64 minus SHORT64 paired comparison, no historical32 substitute."""
import numpy as np
import assessment as aa
def describe(metrics,cfg,root):
 short,long=cfg['arms'];paired={}
 for view in ['full_historical_shape','normalized_u480']:
  def pred(arm):
   with np.load(root/'results'/(arm+'__'+view+'_PREDICTIONS_PRIVATE.npz'),allow_pickle=False)as z:return{k:z[k].copy()for k in ['probability','label','cluster']}
  a,b=pred(long),pred(short);assert np.array_equal(a['label'],b['label'])and np.array_equal(a['cluster'],b['cluster'])
  paired[view]={**aa.paired_bootstrap(a,b,cfg),'first':long,'second':short,'defined':True,'new_classifier_fits':0}
 def difference(a,b,**kw):
  return {'long':a,'short':b,'absolute_delta_long_minus_short':a-b if a is not None and b is not None else None,
   'relative_change_long_minus_short_over_short':(a-b)/b if a is not None and b is not None and b>0 else None,
   'not_more_than10pct_worse':bool(a<=1.1*b)if a is not None and b is not None else False,**kw}
 d={}
 for key in ['PSD','baseline_RMS_V','ACF1']:
  d[key+'_overall_W1']=difference(metrics['domain_vs_REAL'][long]['overall'][key]['W1'],metrics['domain_vs_REAL'][short]['overall'][key]['W1'])
 pn=metrics['PSD_main'];bins=list(range(7));qmissing={n:[i+1 for i in bins if pn['tables'][n]['bins'][i]['N']<40]for n in ['REAL',short,long]}
 def mean(name):return float(np.mean([pn['vs_REAL'][name]['bins'][i]['W1']for i in bins]))if not qmissing['REAL']and not qmissing[name]else None
 d['PSD_common_Q_equal_mean_W1']=difference(mean(long),mean(short),common_bins_1based=[i+1 for i in bins],fixed_bins_not_candidate_intersection=True,unsupported_required_bins_by_domain=qmissing)
 wn=metrics['energy_conditional_width']['tables']
 for k in [0,1]:
  for field,label in [('mean','peak_position_MAE'),('sigma_PSD','raw_width_MAE')]:
   good=list(range(1,7));missing={n:[i+1 for i in good if not(wn[n]['slices'][i]['valid']and len(wn[n]['slices'][i]['components'])==2 and wn[n]['slices'][i]['components'][k][field]is not None)]for n in ['REAL',short,long]}
   def error(name):return float(np.mean([abs(wn[name]['slices'][i]['components'][k][field]-wn['REAL']['slices'][i]['components'][k][field])for i in good]))if not missing['REAL']and not missing[name]else None
   d['component'+str(k)+'_'+label]=difference(error(long),error(short),common_bins_1based=[i+1 for i in good],fixed_bins_not_candidate_intersection=True,failed_required_bins_by_domain=missing)
 assert len(d)==8
 ci_ok=all(paired[v]['BA_delta_95CI'][1]<0 for v in paired)
 distances_ok=all(r['not_more_than10pct_worse']for r in d.values())
 return {'paired_BA_long_minus_short':paired,'eight_prespecified_distance_differences':d,
  'both_paired_BA_CI_upper_below0':ci_ok,'all_eight_no_more_than10pct_worse':distances_ok,
  'complete_response_improvement_description_supported':ci_ok and distances_ok,
  'same64_steps_CPU12_chunk64':True,'equal_inference_unequal_historical_training_budget':True,
  'LONG_only_primary_candidate_no_SHORT_fallback':True,'no_DEV_checkpoint_selection':True}
