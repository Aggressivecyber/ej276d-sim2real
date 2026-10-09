from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,datetime,time
import numpy as np
import evaluation as ev
from model import adc
ROOT=pathlib.Path(__file__).resolve().parent;start=time.monotonic();cfg=json.loads((ROOT/'config.json').read_text('utf8'));m=json.loads((ROOT/'metrics.json').read_text('utf8'));edges=np.asarray(m['PSD_main']['edges_logQ']);valley=json.loads((ROOT/'VALLEY_DEFINITION.json').read_text('utf8'))
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def write(n,v):(ROOT/n).write_text(json.dumps(ev.safe(v),indent=2,allow_nan=False)+'\n','utf8')
def qb(q):
    z=np.log(np.maximum(q,1e-30));b=np.searchsorted(edges,z,side='right')-1;b[z==edges[-1]]=7;return b
a=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');meta=load(ROOT/'DEV_SOURCE_METADATA.npz');x=a['source_waveforms_V'];f=ev.features(x);comp=meta['composition'];refs={};identity={}
for key in ['R26_root','R28_root']:
    refs[key]=load(pathlib.Path(cfg[key])/'results/PLOT_WAVEFORMS_PRIVATE.npz',keys=['SOURCE_CLEAN'])['SOURCE_CLEAN']
    savedids=load(pathlib.Path(cfg[key])/'ACTUAL_GENERATION_INPUTS.npz',keys=['source_ids'])['source_ids'];identity[key]=bool(np.array_equal(savedids,a['source_ids']))
assert all(identity.values())and np.array_equal(a['source_ids'],meta['source_ids'])
quant,ad=adc(x);r=refs['R26_root'];rf=ev.features(r);prebin=qb(f['features']['Qtotal_mVns']);postbin=qb(rf['features']['Qtotal_mVns']);clipped=np.any((x<-.5)|(x>.5),axis=1)
checks={k:{'same_ADC_values':bool(np.array_equal(v,quant)),'maxabs_ADC_vs_reference_V':float(np.max(np.abs(v.astype('f8')-quant.astype('f8'))))}for k,v in refs.items()};assert all(z['same_ADC_values']for z in checks.values())
payload={'actual_network_input':str(ROOT/'ACTUAL_GENERATION_INPUTS.npz')+'::source_waveforms_V','historical_reference_paths':{k:str(pathlib.Path(cfg[k])/'results/PLOT_WAVEFORMS_PRIVATE.npz')+'::SOURCE_CLEAN'for k in refs},'parent_ID_order_checks':identity,'source_metadata_ID_equal':True,'rows':2000,'composition_counts':{str(k):int((comp==k).sum())for k in np.unique(comp)},'reference_reconstruction_check':checks,'ADC':ad,'source_ADC_clipped_parent_count':int(clipped.sum()),'source_waveform_difference':{'maxabs_V':float(np.max(np.abs(r.astype('f8')-x.astype('f8')))),'row_RMS_V':ev.summary(np.sqrt(np.mean((r.astype('f8')-x.astype('f8'))**2,1)))},'charge_difference_post_ADC_minus_actual_input_mVns':ev.summary(rf['features']['Qtotal_mVns']-f['features']['Qtotal_mVns']),'charge_difference_gate_input_minus_metadata_full512_target_mVns':ev.summary(f['features']['Qtotal_mVns']-meta['Qsource_Vns']*1000),'PSD_difference_post_ADC_minus_actual_input':ev.summary(rf['features']['PSD']-f['features']['PSD']),'source_Q_bin_changed_parent_count':int((prebin!=postbin).sum()),'pure_had_actual_preADC_counts_8bins':[int(((comp=='pure_had')&(prebin==i)).sum())for i in range(8)],'pure_had_reference_postADC_counts_8bins':[int(((comp=='pure_had')&(postbin==i)).sum())for i in range(8)],'verified_processing_difference':'SOURCE_CLEAN is the existing 14bit ADC [-.5,.5] clipping/rounding of the actual clean network input, not its unquantized input. This exactly reconstructs both saved references for all 2000 rows.','construction_sources':{'clean':'R18/numeric_worker.py clean_expectation: G4 EE+.15had, fixed exponential components, linear full512 charge target, CFD cropped480; R32 inherits actual clean input unchanged.','network_and_reference':'R18/numeric_worker.py run: save source in ACTUAL_GENERATION_INPUTS, infer(source), then source_adc=adc(source) for SOURCE_CLEAN evaluation. R28 explicitly reuses the R26 saved SOURCE_CLEAN reference.','Qsource_Vns':'analytic calibrated full512 target; not identical to the saved baseline-subtracted 100:400 gate integral'},'no_new_micro_timing_information':True,'not_a_migration_test':True,'no_response_inference_fits_or_generation':0}
rows={}
for name in ['CONTROL_FINAL','MEAN_FINAL']:
    g=load(ROOT/name/'generated.npz',keys=['waveforms_V','source_ids']);assert np.array_equal(g['source_ids'],a['source_ids']);gf=ev.features(g['waveforms_V']);out=[]
    for i in range(8):
        population=(comp=='pure_had')&(prebin==i);use=population&gf['valid'];p=gf['features']['PSD'][use];de=valley['bins'][i]
        out.append({'bin_id_1based':i+1,'source_count':int(population.sum()),'QC_n':int(use.sum()),'invalid_n':int((population&~gf['valid']).sum()),'output_PSD':ev.summary(p),'empirical_width_is_sample_SD_not_density_fit':True,'valley_fraction':float(np.mean((p>=de['low'])&(p<=de['high'])))if len(p)and de['valid']else None})
    rows[name]={'conditioning':'actual unquantized NN-input source-Q, not ADC-reference source-Q','all_8bins':out,'primary_actual_source_bins3_4':out[2:4],'all_parent_ids_preserved':True}
payload['actual_NN_input_source_Q_endpoints']=rows
write('SOURCE_SEMANTICS.json',payload)
m['source_reference_correction'].update(reference_is_ADC_of_actual_input=True,reference_source_bins_are_post_ADC=True,actual_NN_source_Q_report='SOURCE_SEMANTICS.json',reference_label_for_report='frozen post-ADC source reference Q');m['source_semantics']=payload
c='CONTROL_FINAL';n='MEAN_FINAL';gate={};comparisons={};common=m['conditional_mean_SD']['common_observed_Q_bin_ids_1based']
def bound(k,base,new,maxratio=1.1):
    ok=None if base is None or new is None else new<=base*maxratio if base>0 else new<=base
    gate[k]={'CONTROL_FINAL':base,'MEAN_FINAL':new,'relative_change':new/base-1 if base and new is not None else None,'allowed_increase_fraction':maxratio-1,'pass':ok,'undefined_is_not_pass':True}
for view in ['full_historical_shape','normalized_u480']:
    p=m['C2ST']['paired_MEAN_minus_CONTROL']['final_equal_updates'][view];gate[view+'_paired_BA']={'delta':p['BA_delta_first_minus_second'],'CI95':p['BA_delta_95CI'],'pass':p['BA_delta_95CI'][1]<0}
cm=m['conditional_mean_SD']['common_bin_equal_mean_full480_MSE'];gate['conditional_mean_MSE']={'common_bin_ids_1based':common,'CONTROL_FINAL':cm[c],'MEAN_FINAL':cm[n],'relative_reduction':1-cm[n]/cm[c]if cm[c]and cm[n]is not None else None,'required_reduction':.2,'pass':bool(cm[c]and cm[n]is not None and cm[n]<=.8*cm[c])if common else None}
for field,label in [('PSD','overall_PSD_W1'),('baseline_RMS_V','baseline_RMS_W1'),('ACF1','ACF1_W1')]:bound(label,m['domain_vs_REAL'][c]['overall'][field]['W1'],m['domain_vs_REAL'][n]['overall'][field]['W1'])
cv={k:float(np.mean([m['PSD_main']['vs_REAL'][k]['bins'][i-1]['W1']for i in common]))if common else None for k in [c,n]};bound('common_bin_PSD_W1',cv[c],cv[n]);gate['common_bin_PSD_W1']['common_bin_ids_1based']=common
we=m['energy_conditional_width']['common_support_errors']['all_endpoints']
for k in [0,1]:
    row=next(r for r in we if r['component']==k and r['field']=='sigma_PSD');label='component'+str(k)+'_raw_width_MAE';bound(label,row['MAE'][c],row['MAE'][n]);gate[label]['common_bin_ids_1based']=row['common_bin_ids_1based']
for i in [3,4]:
    r=m['source_composition']['primary_pure_had_source_bins3_4'];a1=r[c]['bin'+str(i)];a2=r[n]['bin'+str(i)];b=a1['valley_fraction'];v=a2['valley_fraction'];delta=v-b if b is not None and v is not None else None;gate['postADC_reference_pure_had_bin'+str(i)+'_valley']={'CONTROL_FINAL':b,'MEAN_FINAL':v,'difference_fraction':delta,'max_increase_fraction':.02,'source_count':a1['source_count'],'pass':delta<=.02 if delta is not None else None,'condition_is_frozen_post_ADC_reference_not_actual_input_Q':True}
gate['equal2950_and_original_flow']={'pass':m['row_exposure']['both_2950']and all(m['row_exposure']['same_stream_hashes'].values())}
adrows={k:m['generation'][k]['ADC']for k in m['generation']};gate['new_output_ADC_and_QC']={'ADC':adrows,'QC':{k:m['domains'][k]['QC_valid']if 'QC_valid'in m['domains'][k]else m['domains'][k].get('QC_n')for k in m['generation']},'pass':all(z['pre_ADC_finite']and z['clip_count']==0 for z in adrows.values())and all(m['local'][k]['QC_n']==2000 for k in m['generation'])}
out={'fixed_recipe_retained':all(r['pass']is True for r in gate.values()),'gate':gate,'same_platform_final_comparison':True,'unknown_not_automatically_pass':True,'operational_thresholds_not_physical_equivalence':True,'close_fixed_coefficient_recipe_if_fail':True}
write('GATE_METRICS.json',out);m['stage_gate']=out;write('metrics.json',m)
write('SOURCE_SEMANTICS_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'source_semantics_no_response_fit_or_generation':True,'derived_statistics_only':True,'historical_source_ADCs_exact':checks,'additional_HGB_EM':0})
print(json.dumps({'gate':out,'source_ADC_clipped_parents':int(clipped.sum()),'ADC_clip_count':ad['clip_count'],'actual_source_primary':{k:v['primary_actual_source_bins3_4']for k,v in rows.items()}}),flush=True)
