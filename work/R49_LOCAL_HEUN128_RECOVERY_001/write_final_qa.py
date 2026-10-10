from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
root=Path(__file__).resolve().parent
notes={
 'boundary_differences.png':'Final layout reopened after display-only repair: separate legend and title; all nine stored values and limits remain visible.',
 'c2st.png':'All ten fixed views readable; legend outside data; saved RAW two-dimensional reference remains missing.',
 'charge_psd_full_range.png':'Four domains share full axes; all events retained and outside counts explicit.',
 'charge_psd_main_body.png':'Common 0 to 0.6 PSD zoom; event counts and displayed outside counts explicit; original full-range figure preserved.',
 'conditional_psd_CDF.png':'All eight frozen bins, shared axes, domain counts and mismatching CDFs visible.',
 'conditional_waveforms.png':'All 480 original voltage points and observed-Q counts visible; raw-voltage means do not claim distributional equivalence.',
 'energy_peak_width.png':'Centers and raw widths show undefined gaps, including required Q3; components explicitly are not particle truth.',
 'initial_and_random_noise.png':'Seed/learned/cast functional decomposition includes covariance; expected seed variance not presented as eventwise noise.',
 'noise_curvature.png':'Baseline RMS/ACF and fixed prompt/postpeak curvature CDFs readable; prompt may include peak and early decline.',
 'paired_noise_rule_C2ST.png':'SOURCE and same-treated REAL direct primary-minus-control differences with fixed 500 paired group intervals; all ten views shown.',
 'same_treated_REAL_controls.png':'Only the three registered relative gates shown on left; waveform identity RMS on right explicitly descriptive.',
 'source_preservation.png':'Actual unquantized source bins 3/4 counts 213/181, valley proxy, ranks and source sensitivity explicitly limited.',
 'TRAIN_initial_noise_rule.png':'Both panels use common 0 to 0.60 mV vertical range and common horizontal range; noise amplitudes are not realized SD.',
 'TRAIN_noise_operator.png':'Historical unique consumed TRAIN calibration and colored kernel/variance normalization labeled; no R49 recalibration.',
 'TRAIN_numerical_check.png':'Actual local 128/256 CDF, unchanged 0.25/1 LSB gates and 1e-5 charge bound readable.',
 'training_source_identity.png':'Historical R48 50000-step trajectories labeled; R49 new training explicitly zero.'}
groups={
 'R49_LOCAL_VISUAL_QA_001':('2026-10-10T14:18:31+00:00',['c2st.png','charge_psd_full_range.png','charge_psd_main_body.png','paired_noise_rule_C2ST.png']),
 'R49_LOCAL_VISUAL_QA_002':('2026-10-10T14:18:50+00:00',['conditional_waveforms.png','conditional_psd_CDF.png','energy_peak_width.png','same_treated_REAL_controls.png']),
 'R49_LOCAL_VISUAL_QA_003':('2026-10-10T14:19:33+00:00',['source_preservation.png','noise_curvature.png','initial_and_random_noise.png']),
 'R49_LOCAL_VISUAL_QA_004':('2026-10-10T14:22:56+00:00',['TRAIN_initial_noise_rule.png','TRAIN_noise_operator.png','TRAIN_numerical_check.png','training_source_identity.png','boundary_differences.png'])}
mapping={name:(batch,timestamp)for batch,(timestamp,names)in groups.items()for name in names}
assert set(notes)==set(mapping)=={p.name for p in(root/'results/figures').glob('*.png')}
figures=[]
for name in sorted(notes):
    batch,stamp=mapping[name]
    figures.append({'file':name,'actually_opened':True,'opening_mechanism':'functions.exec tools.view_image original','visual_batch':batch,'observed_batch_clock_UTC':stamp,'SHA256':hashlib.sha256((root/'results/figures'/name).read_bytes()).hexdigest(),'layout_pass':True,'notes':notes[name]})
qa={'PID':os.getpid(),'recorded_UTC':datetime.now(timezone.utc).isoformat(),'actual_final_images_opened':16,'initial_boundary_also_opened':True,'new_science':0,'figures':figures,'initial_boundary_issue':'legend overlapped title; original PNG/source preserved in display_initial; final corrected PNG reopened'}
(root/'FIGURE_QA.json').write_bytes((json.dumps(qa,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8'))
receipt=json.loads((root/'BOUNDARY_LAYOUT_CORRECTION_001.json').read_text('utf8'))
native=json.loads((root/'render_boundary_layout.py.NATIVE_RECEIPT.json').read_text('utf8'))
receipt.update(completed_UTC=datetime.now(timezone.utc).isoformat(),native_tool_reference='1ba9a6',outer_native_exit=0,native_receipt='render_boundary_layout.py.NATIVE_RECEIPT.json',actual_worker_PID=native['actual_worker_PID'],supervisor_PID=native['supervisor_PID'],actual_elapsed_s=native['elapsed_s'],actual_worker_RSS_peak_bytes=native['actual_worker_RSS_peak_bytes'],final_PNG='results/figures/boundary_differences.png',final_PNG_SHA256=hashlib.sha256((root/'results/figures/boundary_differences.png').read_bytes()).hexdigest(),final_PNG_reopened=True,final_visual_batch='R49_LOCAL_VISUAL_QA_004',scientific_values_reused_unchanged=True)
(root/'BOUNDARY_LAYOUT_CORRECTION_001.json').write_bytes((json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode('utf8'))
compact=json.loads((root/'METRICS_COMPACT.json').read_text('utf8'))
forbidden={'source_ids','reference_ids','source_rows','target_rows','selected_indices','real_indices','source_indices','predictions','predicted_labels','ordinal','matched_target','original_index','reference_groups'}
hits=[]
def check(value,path=''):
    if isinstance(value,dict):
        for key,child in value.items():
            current=path+'/'+key
            if key in forbidden and isinstance(child,list):hits.append(current)
            check(child,current)
    elif isinstance(value,list):
        for index,child in enumerate(value):
            if isinstance(child,(dict,list)):check(child,path+'/'+str(index))
check(compact)
assert not hits,hits
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'final_images_actually_opened':16,'compact_no_explicit_event_payload_keys':True,'boundary_final_SHA256':receipt['final_PNG_SHA256'],'boundary_native_elapsed_s':native['elapsed_s'],'new_science':0}),flush=True)
