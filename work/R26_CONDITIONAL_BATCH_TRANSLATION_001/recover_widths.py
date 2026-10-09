from __future__ import annotations
import os,pathlib,json,time,datetime,traceback,hashlib
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results/WIDTH_RECOVERY';OUT.mkdir(exist_ok=False)
import numpy as np,torch
import energy_width as ew,evaluation as ev
torch.set_num_threads(4);torch.set_num_interop_threads(4)
permit_path=ROOT.parent.parent/'reviews/R26_ENGINEERING_RECOVERY_001.json';permit_bytes=permit_path.read_bytes();permit=json.loads(permit_bytes);cfg=json.loads((ROOT/'config.json').read_text('utf8'));metric=json.loads((ROOT/'metrics.json').read_text('utf8'))
assert permit['stage']==ROOT.name and permit['original_attempts_already_spent']==22 and permit['additional_recovery_EM_attempts_max']==22 and permit['amended_density_EM_attempts_total_max']==44
assert hashlib.sha256(pathlib.Path(permit['failure_receipt']).read_bytes()).hexdigest()==permit['failure_sha256']
(ROOT/'R26_ENGINEERING_RECOVERY_BINDING.json').write_text(json.dumps({'path':str(permit_path),'bytes':len(permit_bytes),'sha256':hashlib.sha256(permit_bytes).hexdigest(),'content':permit},indent=2)+'\n','utf8')
new_EM=0;completed_slots=0;tables=metric['energy_conditional_width']['tables'];edges=np.asarray(metric['PSD_main']['edges_logQ']);grid=np.linspace(-.1,.6,129)
def write(path,value):path.write_text(json.dumps(ev.safe(value),indent=2,allow_nan=False)+'\n','utf8')
try:
    with np.load(ROOT/'results/FEATURES_PRIVATE.npz',allow_pickle=False)as payload:
        for name in cfg['endpoints']:
            mask=payload[name+'__valid'];x=np.asarray(payload[name+'__PSD'][mask]);q=np.asarray(payload[name+'__Qtotal_mVns'][mask]);z,bins=ew.bin_indices(q,edges);mult=np.ones(len(x),'f8');rows=[]
            for k in range(8):
                select=(bins==k)&(mult>0);finite=np.isfinite(x[select])&np.isfinite(z[select]);expected=int(finite.sum()>=40)
                assert new_EM+expected<=permit['additional_recovery_EM_attempts_max'];new_EM+=expected
                write(OUT/'RECOVERY_PROGRESS.json',{'PID':os.getpid(),'endpoint':name,'bin_id_1based':k+1,'additional_EM_reserved_attempts':new_EM,'completed_slots':completed_slots,'original_EM_attempts':22})
                r=ew.fit_slice(x[select],z[select],mult[select]);assert r['fit_attempts']==expected
                r.update(bin=k,logQ_edges=edges[k:k+2].tolist(),charge_edges_mVns=np.exp(edges[k:k+2]).tolist(),actual_charge_range_mVns=[float(q[select].min()),float(q[select].max())]if select.any()else None)
                hist,_=np.histogram(x[select],bins=grid,weights=mult[select]);r['histogram_counts']=hist.tolist();r['histogram_PSD_outside_grid_count']=float(mult[select&((x<grid[0])|(x>grid[-1]))].sum())
                write(OUT/(name+'_bin'+str(k+1)+'.json'),r);rows.append(r);completed_slots+=1
            tables[name]={'N':float(mult.sum()),'below_logQ_range':float(mult[bins<0].sum()),'above_logQ_range':float(mult[bins>7].sum()),'PSD_below_minus0_1':float(mult[x<-.1].sum()),'PSD_above0_6':float(mult[x>.6].sum()),'slices':rows,'histogram_edges_PSD':grid.tolist()};write(OUT/(name+'_TABLE.json'),tables[name])
    common_errors={}
    for label,names in [('all_endpoints',['STANDARDIZED',*cfg['endpoints']]),('new_vs_old_final',['HOMO_FINAL','R23_D1_FINAL2950']),('new_selected_vs_old_selected',['STANDARDIZED','HOMO_SELECTED'])]:
        rows=[]
        for k in[0,1]:
            for field in['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8)if all(tables[n]['slices'][i]['valid']and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field]is not None for n in['REAL',*names])]
                rows.append({'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field])for i in bins]))if bins else None for n in names}})
        common_errors[label]=rows
    width={'tables':tables,'common_support_errors':common_errors,'edges_logQ':edges,'old_reference_fits_reused':True,'aliases':metric['endpoint_aliases'],'density_components_not_particle_truth':True,'delivery_unavailable':False,'recovered_under_addendum':True,'original_EM_attempts':22,'additional_recovery_EM_attempts':new_EM,'total_EM_attempts':22+new_EM,'slot_persistence_immediate':True}
    old_metric=(ROOT/'metrics.json').read_bytes();(ROOT/'metrics.pre_density_recovery_PRIVATE.json').write_bytes(old_metric)
    metric['energy_conditional_width']=width;metric['density_results_delivery_missing']=False;metric['counters'].update(density_EM_attempts=22+new_EM,density_description_slots=24,density_recovery_slot_executions=completed_slots,density_slot_executions_total=24+completed_slots)
    replay=json.loads((ROOT/'REPLAY_RECEIPT.json').read_text('utf8'));assert replay['pass']
    metadata_corrections={}
    for n in cfg['endpoints']:
        metadata_corrections[n]={'previous_ADC':metric['generation'][n]['ADC'],'correct_ADC':replay['endpoints'][n]['ADC'],'reason':'Use exact existing adc implementation on bitwise-identical replay pre_ADC instead of hand-coded continuation clip-count metadata','scientific_outputs_unchanged':True};metric['generation'][n]['ADC']=replay['endpoints'][n]['ADC']
    metric['engineering_recovery']={'authorization_path':str(permit_path),'authorization_sha256':hashlib.sha256(permit_bytes).hexdigest(),'additional_EM_attempts':new_EM,'original_cost_not_rolled_back':True,'new_training_generation_HGB':0,'additional_density_slot_executions':completed_slots}
    metric['NCE_explanation']='Negatives are sampled time patches within each waveform via bmm, not cross-event. Reordering changes association of fixed per-step positions to rows and parameter-update history; directly changes rank/distribution/D contexts and update order.'
    write(ROOT/'METADATA_CORRECTION.json',metadata_corrections);write(ROOT/'metrics.json',metric)
    receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':'complete','original_EM_attempts':22,'additional_EM_attempts':new_EM,'total_EM_attempts':22+new_EM,'original_density_slots':24,'additional_slot_executions':completed_slots,'completed_slots':completed_slots,'new_generation_HGB_or_training':0,'formula_parameters':ew.EM,'one_fixed_attempt_per_supported_original_slot':True,'all_slot_results_written_before_next_slot':True,'permit_sha256':hashlib.sha256(permit_bytes).hexdigest()};write(ROOT/'WIDTH_RECOVERY_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
except BaseException as e:
    write(ROOT/'WIDTH_RECOVERY_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'additional_EM_attempts_reserved':new_EM,'completed_slots':completed_slots,'elapsed_s':time.monotonic()-START,'no_further_retry_authorization':True});raise
