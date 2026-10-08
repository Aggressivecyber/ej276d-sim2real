"""Small paired report summary of existing fits; no new fit or array access."""
import json
import pathlib
import datetime
import os

ROOT=pathlib.Path(__file__).resolve().parent
w=json.loads((ROOT/'results/ENERGY_WIDTH_METRICS.json').read_text(encoding='utf8'))
a=w['before']['R3_DEV']['slices'];b=w['R3B_DEV']['slices'];r=w['before']['REAL_DEV']['slices']
indices=[i for i in range(8) if a[i]['valid'] and b[i]['valid'] and r[i]['valid']]
rows=[]
for component in [0,1]:
    for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
        pairs=[]
        for i in indices:
            x=a[i]['components'][component][field];y=b[i]['components'][component][field];z=r[i]['components'][component][field]
            assert x is not None and y is not None and z is not None
            pairs.append({'bin_ID_1_based':i+1,'REAL':z,'R3':x,'R3B':y,
                          'R3_absolute_error':abs(x-z),'R3B_absolute_error':abs(y-z),
                          'absolute_error_delta_R3B_minus_R3':abs(y-z)-abs(x-z)})
        old=sum(x['R3_absolute_error'] for x in pairs)/len(pairs)
        new=sum(x['R3B_absolute_error'] for x in pairs)/len(pairs)
        rows.append({'component':component,'field':field,'common_bins_1_based':[i+1 for i in indices],
                     'R3_MAE':old,'R3B_MAE':new,'MAE_delta_R3B_minus_R3':new-old,'paired_values':pairs})
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),
     'input':'results/ENERGY_WIDTH_METRICS.json','common_bins_1_based':[i+1 for i in indices],
     'criterion':'All REAL/R3/R3B two-component fits qualified with frozen criteria; same bins for all fields.',
     'new_fits':0,'source_array_decodes':0,'not_a_frozen_external_test':True,
     'unpaired_coverages_retained_in_original_metrics':True,'summary':rows}
(ROOT/'results/PAIRED_WIDTH_ERROR_SUMMARY.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({**{k:v for k,v in out.items() if k!='summary'},'summary':[{k:v for k,v in z.items() if k!='paired_values'} for z in rows]}))
