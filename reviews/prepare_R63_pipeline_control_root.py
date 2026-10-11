from pathlib import Path
import ast,json
R=Path(__file__).resolve().parents[1];S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'
oldstage='R61_PRECISION_RESOLVED_SOURCE_ORDER_001';stage=S.name
for name in ['launch_R61_entry_root.py','watch_R61_pipeline_root.py','fetch_R61_root.py','deploy_R61_post_root.py']:
 t=(R/'reviews'/name).read_text('utf8').replace(oldstage,stage).replace('R61','R63').replace('R59','R61')
 if name.startswith('launch_'):
  t=t.replace("assert train['stream_hashes']==cfg['expected_R61_stream_hashes']","assert {k:train['stream_hashes'][k]for k in cfg['expected_R61_stream_hashes']}==cfg['expected_R61_stream_hashes']")
  t=t.replace("train['single_factor_integrity']","train['objective_extension_integrity']").replace("'final_random_states_equal_R61'","'original_six_final_random_states_equal_R61'")
  t=t.replace("'frozen_R61_distribution_statistics_reused'])","'frozen_R61_distribution_statistics_reused','seventh_direction_stream_recorded_independently'])\nassert train['temporal_direction_calls']==50000 and train['coverage_trace_rows']==50000")
 if name.startswith('deploy_'):
  t=t.replace("'precision_audit.py'","'coverage_audit.py'")
 ast.parse(t);dest=R/'reviews'/name.replace('R61','R63');assert not dest.exists();dest.write_bytes(t.encode('utf8'))
p=S/'render_R63_figures.py'
t=p.read_text('utf8').replace("title='Direct unpaired distribution training'","title='Original seven loss terms'")
p.write_bytes(t.encode('utf8'))
print(json.dumps({'controller_scripts_created':4,'science_resubmission_on_observation_timeout':False,'training_fields_modified':False}))
