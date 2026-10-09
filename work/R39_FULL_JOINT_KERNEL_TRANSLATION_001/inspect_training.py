from pathlib import Path
from datetime import datetime,timezone
import json,os,hashlib
ROOT=Path(__file__).resolve().parent;A=ROOT/'JOINT_KERNEL_FINAL'
r=json.loads((A/'TRAINING_RECEIPT.json').read_text('utf8'));cal=json.loads((A/'INITIAL_GRADIENT_SCALE.json').read_text('utf8'))
compact={k:v for k,v in cal.items()if k!='rows'};rows=[];counts={}
for row in cal['rows']:
    cc={str(b):row['bin_ids_1based'].count(b)for b in sorted(set(row['bin_ids_1based']))}
    for k,v in cc.items():counts[k]=counts.get(k,0)+v
    rows.append({**{k:v for k,v in row.items()if k not in ['source_rows','target_rows','original_row_ordinal','bin_ids_1based']},'bin_counts':cc})
compact.update(rows=rows,aggregate_bin_counts=counts,original_private_scale_JSON_sha256=hashlib.sha256((A/'INITIAL_GRADIENT_SCALE.json').read_bytes()).hexdigest(),reference_direction_stream_sha256=hashlib.sha256((A/'FRESH_PROJECTION_STREAM.json').read_bytes()).hexdigest())
(ROOT/'INITIAL_GRADIENT_SCALE_COMPACT.json').write_text(json.dumps(compact,indent=2,allow_nan=False)+'\n','utf8')
m=json.loads((A/'MMD_TRAINING_RECEIPT.json').read_text('utf8'));match=json.loads((ROOT/'MATCHED_TRAINING_FLOW_RECEIPT.json').read_text('utf8'))
out={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'updates':r['updates'],'selected_step':r['selected_step'],'step0_J20':r['initial_validation']['J'],'selected_J20':r['selected_validation']['J'],'final_J20':r['final_validation']['J'],'gradient_coefficient':cal['coefficient'],'aggregate_gradient_old_new':[cal['sum_old_grad_y_squared'],cal['sum_new_grad_y_squared']],'calibration_bin_counts':counts,'MMD':m,'original_flow_match':match,'GPU_peak_allocated_bytes':r['GPU_peak_allocated_bytes'],'training_s':r['training_s'],'actually_unique_source':r['actually_sampled_unique_source'],'actually_unique_REAL':r['actually_sampled_unique_REAL'],'c_reevaluation_calls':0,'actual_D_update_compute_calls_including_recovery':r['D_updates']+1,'unique_original_D_trajectory_updates':r['D_updates'],'step0_J20_compute_calls':2}
(ROOT/'TRAINING_COMPACT.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n','utf8');print(json.dumps(out),flush=True)
