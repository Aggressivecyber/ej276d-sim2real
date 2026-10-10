from __future__ import annotations
import os,json,time,pathlib,datetime
import numpy as np
import evaluation as ev
R=pathlib.Path(__file__).resolve().parent;START=time.monotonic()
C=json.loads((R/'config_post.json').read_text('utf8'));M=json.loads((R/'metrics.json').read_text('utf8'))
L=C['arms'][0];A=[L];RC=['JOINT_PULSE_REAL_CONTROL']
num=json.loads((R/'TRAIN_SOLVER_CHECK.json').read_text('utf8'));replay=json.loads((R/'REPLAY_RECEIPT.json').read_text('utf8'))
training={a:json.loads((pathlib.Path(C['historical_training_root'])/a/'TRAINING_RECEIPT.json').read_text('utf8'))for a in A}
def write(n,x):(R/n).write_bytes((json.dumps(ev.safe(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def load(p,keys=None):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
feat=load(R/'results/FEATURES_PRIVATE.npz')
def score_aliases(row):
 out=dict(row);ci=row.get('heldout_CI',{})
 if isinstance(ci,dict):
  for target,original in [('BA_95CI','balanced_accuracy_95CI'),('AUC_95CI','AUC_95CI')]:
   if original in ci:out[target]=ci[original]
 return out
cs={n:{v:score_aliases(x)for v,x in rows.items()}for n,rows in M['C2ST']['results'].items()}
psd=M['PSD_main'];width=M['energy_conditional_width']
def measure(x,b):
 d=x/b-1 if x is not None and b is not None and b>0 else (0. if x==b==0 else None)
 return {'candidate':x,'BASE':b,'relative_change':d,'pass':d is not None and d<=.10}
charge={}
for n in C['endpoints']:
 d=load(R/n/'generated.npz',keys=['preADC_charge_relative_error_f64','preADC_charge_relative_error_f32','source_q_Vns','waveforms_V','neural_flow_applied_all_rows','inside_TRAIN_support'])
 w=d['waveforms_V'].astype('f8');pq=w[:,100:400].sum(1)-3*w[:,:100].sum(1)
 charge[n]={'rows':len(w),'f64_max_relative_error':float(d['preADC_charge_relative_error_f64'].max()),
  'actual_f32_preADC_max_relative_error':float(d['preADC_charge_relative_error_f32'].max()),
  'postADC_relative_charge_error':ev.summary((pq-d['source_q_Vns'])/d['source_q_Vns']),
  'postADC_not_claimed_exact':True,'all_rows_neural_flow':bool(d['neural_flow_applied_all_rows'].all()),
  'inside_TRAIN_support_rows':int(d['inside_TRAIN_support'].sum())}
write('CHARGE_CONSERVATION_SUMMARY.json',charge)
gaps={n:[{'bin_id_1based':s['bin']+1,'N':s['N'],'failure_reasons':s['failure_reasons']}for s in t['slices']if not s['valid']]for n,t in width['tables'].items()}
gates={};primary=M['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4'];bd=M['boundary_differences']['results']
for arm,ctrl in zip(A,RC):
 paired=M['C2ST']['paired'][arm];full=['full_historical_shape','normalized_u480'];limits={'full_historical_shape':.934,'normalized_u480':.928}
 g1={'paired_minus_BASE':{v:paired[v]for v in full},'fixed_point_limits':limits,
  'candidate_BA':{v:cs[arm][v]['balanced_accuracy']for v in full},
  'both_BASE_CI_upper_negative':all(paired[v]['BA_delta_95CI'][1]<0 for v in full),
  'both_fixed_R37_point_limits':all(cs[arm][v]['balanced_accuracy']<=limits[v]+1e-14 for v in full)}
 g1['pass']=g1['both_BASE_CI_upper_negative']and g1['both_fixed_R37_point_limits']
 common=list(range(7));missingQ={n:[i+1 for i in common if psd['tables'][n]['bins'][i]['N']<40]for n in ['REAL','BASE',arm]}
 pw={n:float(np.mean([psd['vs_REAL'][n]['bins'][i]['W1']for i in common]))if not missingQ['REAL']and not missingQ[n]else None for n in ['BASE',arm]}
 meas={k+'_overall_W1':measure(M['domain_vs_REAL'][arm]['overall'][k]['W1'],M['domain_vs_REAL']['BASE']['overall'][k]['W1'])for k in ['PSD','baseline_RMS_V','ACF1']}
 meas['PSD_common_Q_equal_mean_W1']={**measure(pw[arm],pw['BASE']),'common_bins_1based':[i+1 for i in common],'N_each_domain_ge':40,'fixed_required_bins':True,'unsupported_required_bins_by_domain':missingQ}
 for row in width['common_support_errors'][arm+'_vs_BASE']:
  if row['field']in ['mean','sigma_PSD']:
   key='peak_position_MAE'if row['field']=='mean'else'raw_width_MAE'
   meas['component'+str(row['component'])+'_'+key]={**measure(row['MAE'][arm],row['MAE']['BASE']),
    'common_bins_1based':row['common_bin_ids_1based'],'density_failure_not_hidden_by_intersection':True}
 required=list(range(1,7));assert all(width['tables']['BASE']['slices'][i]['valid']for i in required)
 newfail=[i+1 for i in required if not width['tables'][arm]['slices'][i]['valid']]
 zeros={n:int(np.sum(feat[n+'__baseline_RMS_V']==0))for n in ['BASE',arm]}
 g2={'measures':meas,'BASE_valid_required_bins_1based':[i+1 for i in required],'new_required_density_failures':newfail,
  'all_density_support_gaps':gaps,'zero_baseline_counts':zeros,'zero_baseline_not_increased':zeros[arm]<=zeros['BASE']}
 g2['pass']=len(meas)==8 and all(v['pass']for v in meas.values())and not newfail and zeros[arm]<=zeros['BASE']
 valleys=[];ranks=[]
 for b in ['bin3','bin4']:
  x=primary[arm][b]['valley_fraction'];base=primary['BASE'][b]['valley_fraction'];d=x-base if x is not None and base is not None else None
  valleys.append({'actual_source_bin':b,'N':primary[arm][b]['source_count'],'candidate':x,'BASE':base,'delta':d,'pass':d is not None and d<=.02})
 for f in ['source_Q_to_output_Q','source_PSD_to_output_PSD']:
  x=M['source_association_actual_preADC_input'][arm]['composition_and_PDG']['all'][f]['Spearman'];b=M['source_association_actual_preADC_input']['BASE']['composition_and_PDG']['all'][f]['Spearman'];d=b-x if x is not None and b is not None else None
  ranks.append({'measure':f,'candidate':x,'BASE':b,'BASE_minus_candidate':d,'pass':d is not None and d<=.05})
 gen=M['generation'][arm];sens=M['source_probes'][arm]['rows']['source_roll_no_fixed_points']['waveform_RMS_vs_correct_reference_V']['mean']
 g3={'actual_source_pure_had_valleys':valleys,'whole_population_source_ranks':ranks,
  'source_ID_order_composition_weights_preserved':gen['all_source_ID_order_weights_and_composition_preserved']and M['data_roles']['DEV_composition']=={'mixed':65,'pure_EE':417,'pure_had':1518}and not M['primary_DEV_population_reweighted'],
  'no_new_invalid_nonfinite_ADC_clip':M['domains'][arm]['invalid']==0 and gen['ADC']['pre_ADC_finite']and gen['ADC']['clip_count']==0,
  'outside_TRAIN_support_rows':gen['TRAIN_support_outside_N'],'all_outside_rows_use_new_flow':gen['BASE_fallback_N']==0 and gen['standalone_flow_applied_N']==2000,
  'source_roll_RMS_V':sens,'sensitivity_not_correct_physical_mapping':True}
 g3['pass']=all(x['pass']for x in valleys+ranks)and all(g3[k]for k in ['source_ID_order_composition_weights_preserved','no_new_invalid_nonfinite_ADC_clip','all_outside_rows_use_new_flow'])and sens is not None and sens>0
 bounds=[]
 for key in bd[arm]:
  x=bd[arm][key]['W1_V'];b=bd['BASE'][key]['W1_V'];o=bd['INDEPENDENT_CFM'][key]['W1_V'];lim=max(b,o)+1/16383 if b is not None and o is not None else None
  bounds.append({'location':key,'candidate_W1_V':x,'BASE_W1_V':b,'INDEPENDENT_W1_V':o,'limit_V':lim,'pass':x is not None and lim is not None and x<=lim})
 g4={'boundaries':bounds,'full_SOURCE_perturbation':M['guide_probe'][arm],'pass':len(bounds)==9 and all(x['pass']for x in bounds)and M['guide_probe'][arm]['pass']}
 cp=M['C2ST']['same_treated_REAL_control_paired'][ctrl];control_rows={}
 for v in ['full_historical_shape','normalized_u480','baseline_local']:
  r=cp[v];control_rows[v]={'BASE_G_REAL_BA':cs['BASE_REAL_CONTROL'][v]['balanced_accuracy'],'new_G_REAL_BA':cs[ctrl][v]['balanced_accuracy'],**r,
   'point_delta_le0_01':r['BA_delta_first_minus_second']<=.01+1e-14,'CI_upper_le0_10':r['BA_delta_95CI'][1]<=.10}
  control_rows[v]['pass']=control_rows[v]['point_delta_le0_01']and control_rows[v]['CI_upper_le0_10']
 g5={'pre_registered_same_treated_scope':True,'comparison':'new SOURCE flow on REAL minus frozen BASE G on REAL',
  'three_views':control_rows,'RAW_RR_absolute_effects_descriptive':{n:M['C2ST']['paired'][n]for n in ['BASE_REAL_CONTROL',ctrl]},
  'RAW_identity_not_required':True,'pass':all(r['pass']for r in control_rows.values())}
 g6={'fixed_final50000_models_loaded':True,'new_R53_updates_per_arm':50000,'new_training_updates_per_arm':50000,'old_TRAIN_coordinate_constants_refit':False,'noise_calibration_unique_matched_REAL':True,
  'TRAIN_fixed128_vs256_pass':num['models'][arm]['pass'],'both_models_pass_before_DEV':num['both_pass'],'TRAIN_check_DEV_read':num['DEV_read_or_generated'],
  'all_preADC_f32_charge_error_le1e_5':all(charge[n]['actual_f32_preADC_max_relative_error']<=1e-5 for n in [arm,ctrl]),
  'all_main_rows_processed':all(charge[n]['all_rows_neural_flow']for n in [arm,ctrl]),'charge_error_summaries':{n:charge[n]for n in [arm,ctrl]},
  'independent_GPU_full_chain_replay_rows':sum(replay['endpoints'][n]['rows']for n in [arm,ctrl]),
  'bitwise_pass':all(replay['endpoints'][n]['bitwise_pass']for n in [arm,ctrl]),'CPU_threads':12,'chunk':64,
  'functional_components_not_independent_physical_noise':True,'solver256_reference_not_exact_truth':True}
 g6['frozen_final_training_stream_pass']=training[arm]['updates']==50000 and all(training[arm]['same_R48_source_identity_and_random_streams'].values()) and training[arm]['same_R48_target_total_occurrence_counts'] and json.loads((R/'TEACHER_VALIDATION.json').read_text())['passed']
 g6['pass']=g6['frozen_final_training_stream_pass']and num['models'][arm]['pass']and num['both_pass']and num['DEV_read_or_generated']==0 and g6['all_preADC_f32_charge_error_le1e_5']and g6['all_main_rows_processed']and g6['bitwise_pass']and g6['independent_GPU_full_chain_replay_rows']==4000
 gates[arm]={'gate1_complete_response':g1,'gate2_PSD_width_centers_noise':g2,'gate3_source_preservation':g3,
  'gate4_boundary_source':g4,'gate5_same_treated_REAL_control':g5,'gate6_numerical_replay':g6,
  'retain':all(x['pass']for x in [g1,g2,g3,g4,g5,g6]),'role':'primary'if arm==L else'fixed control; cannot be chosen as fallback'}
gate={'recipe':'R53 fixed joint charge-PSD occurrence coupling and pulse-only teacher; full guide and REAL identity',
 'models':gates,'primary':L,'primary_retain':gates[L]['retain'],'control_not_fallback':True,
 'decision':'retain_primary_pending_A_C'if gates[L]['retain']else'close_fixed_R53_joint_pulse_teacher_recipe',
 'prespecified_rules_changed_after_data':False,'paired_R53_minus_historical_R49_completion':M['paired_R53_minus_historical_R49_completion']}
views=['full_historical_shape','normalized_u480','full_voltage','Q_PSD_2D','baseline_local']
acceptance={}
for arm,ctrl in zip(A,RC):
 aa={n:{v:{'BA':cs[n][v]['balanced_accuracy'],'BA_95CI':cs[n][v]['BA_95CI'],
   'pass':cs[n][v]['BA_95CI'][0]>=.4 and cs[n][v]['BA_95CI'][1]<=.6}for v in views}for n in [arm,ctrl]}
 g2=gates[arm]['gate2_PSD_width_centers_noise'];required=list(range(1,7));rel={}
 for k in [0,1]:
  failed=[i+1 for i in required if not(width['tables']['REAL']['slices'][i]['valid']and width['tables'][arm]['slices'][i]['valid'])]
  vals=[]
  if not failed:
   for i in required:
    x=width['tables'][arm]['slices'][i]['components'][k]['sigma_PSD'];b=width['tables']['REAL']['slices'][i]['components'][k]['sigma_PSD']
    if x is None or b is None or b<=0:failed.append(i+1)
    else:vals.append(abs(x-b)/b)
  value=float(np.mean(vals))if not failed and len(vals)==6 else None
  rel[str(k)]={'required_bins_1based':[2,3,4,5,6,7],'failed_required_bins':failed,'mean_absolute_relative_rawSD_error':value,'pass':value is not None and value<=.10}
 b1=M['domain_vs_REAL'][arm]['overall']['PSD']['W1']<=.01
 b2=g2['measures']['PSD_common_Q_equal_mean_W1']['candidate']
 centers={str(k):next(row['MAE'][arm]for row in width['common_support_errors'][arm+'_vs_BASE']if row['component']==k and row['field']=='mean')for k in [0,1]}
 bpass=b1 and b2 is not None and b2<=.01 and all(x is not None and x<=.01 for x in centers.values())and all(x['pass']for x in rel.values())and gates[arm]['gate2_PSD_width_centers_noise']['pass']
 acceptance[arm]={'A':{'views':aa,'pass':all(x['pass']for r in aa.values()for x in r.values())},
  'B':{'overall_PSD_W1':M['domain_vs_REAL'][arm]['overall']['PSD']['W1'],'common_Q1_7_PSD_W1':b2,
   'peak_MAE_required_Q2_7':centers,'raw_width_relative_errors':rel,'original_noise_side_effects_required':True,'pass':bool(bpass)},
  'C':{'all_original_six_stage_gates':gates[arm]['retain'],'pass':gates[arm]['retain']},
  'D_E':'not_evaluated; requires separate registration after primary A-C pass','role':'primary'if arm==L else'fixed control, not replacement winner'}
gate['primary_A_C_pass']=all(acceptance[L][k]['pass']for k in ['A','B','C'])
gate['decision']='primary_pending_separately_registered_D_E'if gate['primary_A_C_pass']else'close_fixed_R53_joint_pulse_teacher_recipe'
write('GATE_METRICS.json',gate);write('ACCEPTANCE_A_C.json',acceptance)
def compact_scores(x):
 return {k:x[k]for k in ['balanced_accuracy','AUC','train_balanced_accuracy','train_AUC','BA_95CI','AUC_95CI','train_N','holdout_N','feature_dimension','training_balanced_accuracy','training_AUC','train_each_domain','holdout_each_domain','features','heldout_CI']if k in x}
tc={a:{k:r[k]for k in ['updates','elapsed_s','parameter_count','initial_state_SHA','stream_hashes','paired_stream_equality','GPU_allocated_peak_bytes','GPU_reserved_peak_bytes','source_loss_final','identity_loss_final','same_R48_source_identity_and_random_streams','same_R48_target_total_occurrence_counts']if k in r}for a,r in training.items()}
compact={'status':M['status'],'gate':gate,'acceptance_A_C':acceptance,'counters':M['counters'],'R53_training':tc,'new_training_runs':1,
 'C2ST':{'results':{n:{v:compact_scores(x)for v,x in rows.items()}for n,rows in cs.items()},
  'paired':M['C2ST']['paired'],'same_treated_REAL_control_paired':M['C2ST']['same_treated_REAL_control_paired']},
 'overall_domain_vs_REAL':{n:v['overall']for n,v in M['domain_vs_REAL'].items()},'PSD_main_vs_REAL':psd['vs_REAL'],
 'width_common_support_errors':width['common_support_errors'],'density_support_gaps':gaps,
 'source_primary_actual_Q':primary,'source_historical_ADC_reference_Q':M['source_composition']['historical_postADC_reference_Q_bins3_4'],
 'source_ranks_actual_Q':{n:r['composition_and_PDG']['all']for n,r in M['source_association_actual_preADC_input'].items()},
 'boundary_differences':M['boundary_differences'],'source_guide_summary':M['guide_probe'],
 'paired_R53_minus_historical_R49_completion':M['paired_R53_minus_historical_R49_completion'],
 'initial_noise':json.loads((R/'INITIAL_NOISE_DESCRIPTIONS.json').read_text('utf8')),
 'REAL_identity_error':json.loads((R/'REAL_IDENTITY_DESCRIPTIONS.json').read_text('utf8')),
 'replay':{'rows':replay['rows'],'pass':replay['pass']},'EVAL_read':0,'CAL_waveform_decode':0,
 'privacy':'aggregate only; complete identities, splits, predictions and waveforms remain private locally'}
write('METRICS_COMPACT.json',compact)
import render_R53_figures
render_R53_figures.run(R,C,M,gate,acceptance)
write('SUMMARY_PLOT_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'elapsed_s':time.monotonic()-START,'new_HGB':0,'new_EM':0,'new_neural_inference':0,'aggregate_and_plot_only':True})
print(json.dumps({'summary_complete':True,'PID':os.getpid(),'primary_A_C_pass':gate['primary_A_C_pass'],'elapsed_s':time.monotonic()-START}),flush=True)
