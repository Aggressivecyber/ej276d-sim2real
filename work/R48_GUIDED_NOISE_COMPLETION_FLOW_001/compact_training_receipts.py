from pathlib import Path
from datetime import datetime,timezone
import json,os
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'));rows={}
for arm in C['arms']:
    p=R/arm/'TRAINING_RECEIPT.json'
    if not p.exists():continue
    a=json.loads(p.read_text('utf8'));trace=json.loads((R/arm/'TRAINING_TRACE.json').read_text('utf8'))
    keep=['PID','UTC','arm','updates','elapsed_s','counter','actual_unique_SOURCE_REAL','original_loaded_source_REAL',
          'initial_state_SHA','stream_hashes','paired_stream_equality','source_actual_bin_mismatch','identity_actual_bin_mismatch',
          'GPU_allocated_peak_bytes','GPU_reserved_peak_bytes','branch_gradient_descriptions','input_amplitudes',
          'zero_input_weights_count','zero_input_gradient_checks','zero_input_value_checks','t_nonzero_gradient_steps','bin_nonzero_gradient_steps']
    rows[arm]={k:a[k]for k in keep if k in a};rows[arm]['initial_trace']=trace[0];rows[arm]['final_trace']=trace[-1]
    rows[arm]['unique_parent_population_increased']=False
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'training':rows,
     'new_response_training_runs_actual_completed':len(rows),'plan_runs':2,
     'training_metadata_correction':{'config_json_inherited_new_response_training_runs':0,
                                    'meaning':'inactive inherited planning metadata, not actual run count',
                                    'config_cpu_explicit_R48_plan':2},
     'local_stage_file_bytes_now':sum(p.stat().st_size for p in R.rglob('*')if p.is_file()),'scientific_calculation':False}
(R/'TRAINING_COMPACT.json').write_bytes((json.dumps(rec,indent=2)+'\n').encode())
print(json.dumps(rec),flush=True)
