from pathlib import Path
import shutil,ast
R=Path(__file__).resolve().parents[1];H=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001';S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'
shutil.copy2(H/'artificial_check.py',S/'R61_artificial_check.py')
(S/'artificial_check.py').write_bytes(b'from coverage_artificial_check import check, integration_check\n')
text=(H/'train_final.py').read_text('utf8').replace('R59','R61')
def replace(old,new):
 global text
 assert text.count(old)==1,(text.count(old),old)
 text=text.replace(old,new)
replace('from distribution_loss import DistributionLoss,CATEGORIES,NAMES','from distribution_loss import DistributionLoss,CATEGORIES,NAMES,fit_temporal_statistics,TEMPORAL_NAMES')
replace("    interval_low,interval_high=rounding_psd_interval(source)","""    temporal=fit_temporal_statistics(real,rraw,C)
    assert np.array_equal(temporal['temporal_REAL_counts'],statistics['REAL_counts'])
    np.savez_compressed(R/'TRAIN_TEMPORAL_CONSTANTS.npz',**temporal)
    write('TRAIN_TEMPORAL_CONSTANTS.json',{'UTC':utc(),'fitted_REAL_TRAIN_rows':len(real),'region_categories':CATEGORIES,
        'feature_names':TEMPORAL_NAMES,'center':temporal['temporal_center'].tolist(),'scale':temporal['temporal_scale'].tolist(),
        'scale_floor':C['temporal_scale_floor'],'population_standard_deviation':True,'zero_power_counts':temporal['temporal_zero_power_counts'].tolist(),
        'power_floor':C['temporal_power_floor'],'fitted_before_first_update':True,'scientific_response_fit':False,'teacher_fit':False,
        'new_statistics_SHA256':hashlib.sha256((R/'TRAIN_TEMPORAL_CONSTANTS.npz').read_bytes()).hexdigest(),
        'old_statistics_SHA256':hashlib.sha256((R/'TRAIN_DISTRIBUTION_CONSTANTS.npz').read_bytes()).hexdigest()})
    statistics.update(temporal)
    interval_low,interval_high=rounding_psd_interval(source)""")
replace("'new_normalization_is_TRAIN_only':False,'frozen_R61_TRAIN_normalization_reused':True", "'new_temporal_normalization_is_TRAIN_only':True,'original_seven_loss_frozen_R61_TRAIN_normalization_reused':True")
replace("    counter['response_fit_attempts']=1", """    coverage_columns=['original_seven_total','coordinate_RMS','temporal_joint','temporal_marginal','temporal_sliced',*[name+'_coordinate_RMS'for name in NAMES]];coverage_dense=[]
    counter['response_fit_attempts']=1""")
replace("        directions=objective.draw_directions(streams['projection_directions'])", "        directions=objective.draw_directions(streams['projection_directions'])\n        temporal_directions=objective.draw_temporal_directions(streams['baseline_projection_directions'])")
replace("torch.as_tensor(interval_high[si],device=dev))", "torch.as_tensor(interval_high[si],device=dev),temporal_directions)")
replace("        sN=len(si);active=", "        hashes['baseline_projection_directions']=chain_update(hashes['baseline_projection_directions'],temporal_directions)\n        sN=len(si);active=")
replace("        assert abs(order_row[0]-order_row[1]-row[3])<=1e-10+1e-10*abs(order_row[0])", "        assert abs(order_row[0]-order_row[1]-row[columns.index('source_order')])<=1e-10+1e-10*abs(order_row[0])")
replace("        global_max_q64=max(global_max_q64,float(row[10]));global_max_q32=max(global_max_q32,float(row[11]))", """        global_max_q64=max(global_max_q64,float(row[columns.index('max_relative_q_f64')]));global_max_q32=max(global_max_q32,float(row[columns.index('max_relative_q_f32')]))
        coverage_row=torch.stack([diagnostic[k]for k in coverage_columns]).cpu().numpy().astype('f8');coverage_dense.append(coverage_row)
        assert np.isfinite(coverage_row).all() and np.all(coverage_row>=0)
        assert abs(coverage_row[:3].sum()-row[0])<=1e-10+1e-10*abs(row[0])""")
replace("            trace.append(record);write", "            record['coverage_last_up_to1000_mean']=dict(zip(coverage_columns,np.asarray(coverage_dense[-min(len(coverage_dense),1000):]).mean(0).tolist()))\n            trace.append(record);write")
replace("'loss_columns':columns,'dense_losses':np.asarray(dense)","'coverage_columns':coverage_columns,'coverage_values':np.asarray(coverage_dense),'loss_columns':columns,'dense_losses':np.asarray(dense)")
replace("objective.direction_calls==50000 and all", "objective.direction_calls==50000 and objective.temporal_direction_calls==50000 and all")
replace("assert {k:v.hex()for k,v in hashes.items()}==C['expected_R61_stream_hashes']", "assert {k:hashes[k].hex()for k in C['expected_R61_stream_hashes']}==C['expected_R61_stream_hashes']")
replace("assert final_states==json.loads(Path(C['input_paths']['R61_final_random_states']).read_text('utf8'))", "assert {k:final_states[k]for k in C['expected_R61_stream_hashes']}==json.loads(Path(C['input_paths']['R61_final_random_states']).read_text('utf8'))")
replace("    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},D/'FINAL_FLOW.pt')", "    np.savez_compressed(D/'COVERAGE_BY_UPDATE_PRIVATE.npz',columns=np.asarray(coverage_columns),values=np.asarray(coverage_dense,dtype='f8'))\n    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},D/'FINAL_FLOW.pt')")
replace("'directions':'standard_normal((d,64)) float64; unit columns normalized float64; castfloat32; d=480,480,479,478'", "'directions':'standard_normal((d,64)) float64; unit columns normalized float64; castfloat32; d=480,480,479,478','baseline_directions':'independent seventh stream; standard_normal((31,64)) float64; normalize columns float64; castfloat32'")
replace("'single_factor_integrity':", "'objective_extension_integrity':")
replace("'final_random_states_equal_R61':True", "'original_six_final_random_states_equal_R61':True,'seventh_direction_stream_recorded_independently':True")
replace("'precision_order_trace_rows':len(order_dense)","'precision_order_trace_rows':len(order_dense),'coverage_columns':coverage_columns,'coverage_trace_rows':len(coverage_dense),'temporal_direction_calls':objective.temporal_direction_calls,'new_temporal_statistics_saved':True")
ast.parse(text);(S/'train_final.py').write_bytes(text.encode('utf8'))
for file in ['deploy_R61_root.py','poll_R61_root.py']:
 t=(R/'reviews'/file).read_text('utf8').replace('R61_PRECISION_RESOLVED_SOURCE_ORDER_001','R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001').replace('PRECISION_RESOLVED_RESPONSE','COVERAGE_RESPONSE').replace('R61_','R63_')
 if file.startswith('poll_'):
  t=t.replace("'TRAIN_POPULATION_AND_COORDINATES.json'","'TRAIN_POPULATION_AND_COORDINATES.json','TRAIN_TEMPORAL_CONSTANTS.json'")
 ast.parse(t);(R/'reviews'/file.replace('R61','R63')).write_bytes(t.encode('utf8'))
print('R63 driver and control source prepared; no scientific execution')
