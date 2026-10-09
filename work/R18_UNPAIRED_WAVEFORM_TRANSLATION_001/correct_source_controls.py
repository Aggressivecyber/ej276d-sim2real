from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,shutil
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parent;folder=ROOT/'PROBE_REPAIR_001'
folder.mkdir(exist_ok=False)
import numpy as np
import evaluation as ev
import matplotlib.pyplot as plt
for name in ['metrics.json','METRICS_COMPACT.json','REPORT.md','RESOURCE_RECEIPT.json']:
    if (ROOT/name).exists():shutil.copyfile(ROOT/name,folder/name)
shutil.copyfile(ROOT/'results/figures/source_controls.png',folder/'source_controls_original.png')
with np.load(ROOT/'results/SOURCE_PROBES_PRIVATE.npz',allow_pickle=False)as f:z={k:f[k].copy()for k in f.files}
with np.load(ROOT/'generated.npz',allow_pickle=False)as f:main=f['waveforms_V'].copy()
with np.load(ROOT/'ACTUAL_GENERATION_INPUTS.npz',allow_pickle=False)as f:actual_source=f['source_waveforms_V'].copy();actual_z=f['z'].copy();actual_white=f['white'].copy();ids=f['source_ids'].copy()
tags=z['probe_kind'];original_ref=z['parent_index'].copy();source_anchor=original_ref.copy();reference=original_ref.copy();noise_origin=original_ref.copy();pick=original_ref[tags=='same_input_same_noise'];fixed=tags=='fixed_source_changed_noise';rolled=tags=='source_roll_no_fixed_points'
source_anchor[fixed]=pick[0];reference[fixed]=pick[0];source_anchor[rolled]=np.roll(pick,1)
direct=np.isin(tags,['same_input_same_noise','source_roll_no_fixed_points','fixed_source_changed_noise']);counterfactual=~direct
assert np.array_equal(z['source_waveforms_V'][direct],actual_source[source_anchor[direct]])
assert np.array_equal(z['z'],actual_z[noise_origin])and np.array_equal(z['white'],actual_white[noise_origin])
base=main[reference];post=z['waveforms_V'];d=ev.features(post);b=ev.features(base);sf=ev.features(z['source_waveforms_V']);metrics=json.loads((ROOT/'metrics.json').read_text());old=metrics['source_probes'];rows={}
for tag in np.unique(tags):
    use=tags==tag;rows[tag]={**old['rows'][tag],'waveform_RMS_vs_reference_V':ev.summary(np.sqrt(np.mean((post[use]-base[use])**2,1))),'PSD_vs_reference':ev.summary(d['features']['PSD'][use]-b['features']['PSD'][use]),'Q_vs_reference_mVns':ev.summary(d['features']['Qtotal_mVns'][use]-b['features']['Qtotal_mVns'][use]),'reference_rule':'single fixed source/noise0 output broadcast'if tag=='fixed_source_changed_noise'else'original main output for noise-origin parent','original_ambiguous_vs_main_parent_statistics_retained_in_PROBE_REPAIR':True}
    for key in ['waveform_RMS_vs_main_parent_V','PSD_vs_main_parent','Q_vs_main_parent_mVns']:rows[tag].pop(key,None)
new={**old,'rows':rows,'metadata_corrected_without_new_generation':True,'actual_source_parent_index_file':'results/SOURCE_PROBES_CORRECTED.npz','source_roll_actual_parent':'roll(pick,1); comparison reference remainspick','fixed_source_actual_and_reference_parent':'pick0 for all128; noise origins remainpick','counterfactual_anchor_not_a_new_G4_event':True,'old_mixed_source_reference_preserved':'PROBE_REPAIR_001/metrics.json'}
z.update(source_parent_index=source_anchor,source_parent_ids=ids[source_anchor],noise_origin_parent_index=noise_origin,noise_origin_parent_ids=ids[noise_origin],reference_parent_index=reference,reference_parent_ids=ids[reference],source_waveform_modified_counterfactual=counterfactual)
z.pop('parent_index');np.savez_compressed(ROOT/'results/SOURCE_PROBES_CORRECTED.npz',**z)
metrics['source_probes']=new;(ROOT/'metrics.json').write_text(json.dumps(ev.safe(metrics),ensure_ascii=False,allow_nan=False,indent=2)+'\n')
fig,axs=plt.subplots(1,3,figsize=(17,5));plt.rcParams.update({'font.size':9})
for ax,field,label in zip(axs[:2],['PSD','Q_mVns'],['PSD','Qtotal [mV ns]']):
    xx=[0,.25,.5,.75,1];vals=[z[field][tags=='analytic_fast_slow_'+str(t)]for t in xx];med=[np.median(a)for a in vals];lo=[np.quantile(a,.1)for a in vals];hi=[np.quantile(a,.9)for a in vals];ax.errorbar(xx,med,yerr=[np.array(med)-lo,np.array(hi)-med],fmt='o-');ax.set_xlabel('analytic slow-shape fraction');ax.set_ylabel(label);ax.grid(alpha=.2)
names=['same_input_same_noise','source_roll_no_fixed_points','fixed_source_changed_noise','amplitude_x0.5','amplitude_x1.5'];label=['same source / same noise','source roll / fixed noise','one source / changed noise','source amplitude x0.5','source amplitude x1.5'];axs[2].barh(label,[rows[n]['waveform_RMS_vs_reference_V']['mean']*1000 for n in names]);axs[2].set_xlabel('mean output RMS vs specified reference [mV]');fig.suptitle('Source controls: fixed-source noise changes use its own main output reference');fig.tight_layout();fig.savefig(ROOT/'results/figures/source_controls.png',dpi=140,bbox_inches='tight');plt.close(fig)
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'fits':0,'response_updates':0,'new_generation':0,'source_source_id_check':True,'noise_origin_z_white_check':True,'changed_main2000_metrics':False,'fixed_source_noise_RMS_reference_corrected_V':rows['fixed_source_changed_noise']['waveform_RMS_vs_reference_V'],'fixed_source_first_row_reference_maxabs_V':float(np.max(np.abs(post[fixed][0]-main[pick[0]]))),'source_roll_reference_rule':'source=roll(pick),noise=pick,reference=pick','stage_arrays_read_for_existing_statistic_correction_only':True,'original_payload_preserved':True}
(ROOT/'SOURCE_CONTROL_CORRECTION_RECEIPT.json').write_text(json.dumps(ev.safe(receipt),indent=2)+'\n');print(json.dumps(ev.safe(receipt)),flush=True)
