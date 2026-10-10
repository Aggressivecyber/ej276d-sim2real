from pathlib import Path
import json
import numpy as np
import assessment as aa


def describe(metrics, cfg, root, arm):
    oldroot = Path(cfg['R43_root'])
    old = json.loads((oldroot/'metrics.json').read_text('utf8'))
    short = 'SCALAR_TIME_FLOW_FINAL'
    paired = {}
    for view in ['full_historical_shape', 'normalized_u480']:
        try:
            with np.load(root/'results'/(arm+'__'+view+'_PREDICTIONS_PRIVATE.npz'), allow_pickle=False) as z:
                a = {k:z[k].copy() for k in ['probability', 'label', 'cluster']}
            with np.load(oldroot/'results'/(short+'__'+view+'_PREDICTIONS_PRIVATE.npz'), allow_pickle=False) as z:
                b = {k:z[k].copy() for k in ['probability', 'label', 'cluster']}
            if not (np.array_equal(a['label'], b['label']) and np.array_equal(a['cluster'], b['cluster'])):
                paired[view] = {'defined':False, 'reason':'saved fixed holdout/group binding differs'}
            else:
                paired[view] = {**aa.paired_bootstrap(a,b,cfg), 'defined':True,
                                'first':arm, 'second':'R43 total2950', 'new_classifier_fits':0}
        except (OSError, KeyError) as e:
            paired[view] = {'defined':False, 'reason':'missing saved paired fields', 'error':repr(e)}
    def difference(a,b,**kw):
        return {'long':a, 'short':b, 'absolute_delta_long_minus_short':a-b if a is not None and b is not None else None, **kw}
    d = {}
    for key in ['PSD', 'baseline_RMS_V', 'ACF1']:
        d[key+'_overall_W1'] = difference(metrics['domain_vs_REAL'][arm]['overall'][key]['W1'],
                                         old['domain_vs_REAL'][short]['overall'][key]['W1'])
    pn, po = metrics['PSD_main'], old['PSD_main']
    bins = [i for i in range(8) if all(t['N']>=40 for t in [pn['tables']['REAL']['bins'][i],
            pn['tables'][arm]['bins'][i], po['tables'][short]['bins'][i]])]
    mean = lambda table,name:float(np.mean([table['vs_REAL'][name]['bins'][i]['W1'] for i in bins])) if bins else None
    d['PSD_common_Q_equal_mean_W1'] = difference(mean(pn,arm),mean(po,short),common_bins_1based=[i+1 for i in bins])
    wn, wo = metrics['energy_conditional_width']['tables'], old['energy_conditional_width']['tables']
    for k in [0,1]:
        for field,label in [('mean','peak_position_MAE'),('sigma_PSD','raw_width_MAE')]:
            good = [i for i in range(8) if all(t['valid'] and len(t['components'])==2 and t['components'][k][field] is not None
                    for t in [wn['REAL']['slices'][i],wn[arm]['slices'][i],wo[short]['slices'][i]])]
            def error(table,name):
                return float(np.mean([abs(table[name]['slices'][i]['components'][k][field]-wn['REAL']['slices'][i]['components'][k][field])
                                     for i in good])) if good else None
            d['component'+str(k)+'_'+label] = difference(error(wn,arm),error(wo,short),common_bins_1based=[i+1 for i in good])
    assert len(d)==8
    return {'paired_BA_vs_R43':paired, 'eight_prespecified_distance_differences':d,
            'R43_reference':str(oldroot), 'R43_response_updates_repeated':0,
            'R43_classifier_or_density_refits':0, 'R43_environment':old.get('versions'),
            'same_registered_600_250_rows_and_group_order_required':True,
            'history_platform_difference_disclosed':True, 'no_DEV_checkpoint_selection':True}
