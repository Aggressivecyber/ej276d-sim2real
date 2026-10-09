from pathlib import Path
import ast,json,shutil,hashlib,datetime,os
ROOT=Path(__file__).resolve().parent;P=ROOT.parent.parent
R23=P/'work/R23_ONLINE_STANDARDIZED_CRITIC_001';R24=P/'work/R24_CRITIC_UPDATE_RATIO_001'
for n in ['model.py','legacy_model.py','fine_core.py','prior.py','evaluation.py','assessment.py','diagnostic_core.py','energy_width.py','data_support.py','random_diagnostics.py','common_math.py','supervise.py','owned_worker.py']:
    shutil.copyfile(R24/n,ROOT/n)
for n in ['ACTUAL_GENERATION_INPUTS.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json','VALLEY_DEFINITION.json','EVALUATOR_SELECTIONS_PRIVATE.json','FROZEN_TILT_TABLE.json']:
    shutil.copyfile(R24/n,ROOT/n)
cfg=json.loads((R24/'config.json').read_text('utf8'))
for k in list(cfg):
    if k.startswith('extra_D_'):del cfg[k]
cfg.update(stage=ROOT.name,registered_UTC='2026-10-09T12:40:38.792322+00:00',deadline_UTC='2026-10-09T14:00:38.792322+00:00',absolute_deadline_UTC='2026-10-09T14:10:38.792322+00:00',first_actual_command_UTC='2026-10-09T12:41:34.302904+00:00',first_actual_PID=56420,first_actual_tool='8a92fd',arms=['HOMOGENEOUS_REORDER'],G_D_update_ratio='1:1',D_updates_limit=2950,HGB_max_fits=21,HGB_plan=21,HGB_cap=21,width_slots=24,repeated_rows_planned=3072,repeated_rows_cap=3072,source_probe_rows=2496,main_generation_rows=6000,replay_rows=6000,reorder_seed=20261009261,critic_intervention='same STANDARDIZED critic; only row binding order and resulting batch composition changed',scientific_package='conditional batch/order intervention; rank/NCE context, online game, gradients and order all may change',old_control='R23 STANDARDIZED selected and final2950; no old response training',source_rng_seed_and_pairing='reconstruct old exact original PCG64 states/call order, attach z/white/style to original ordinal, permute rows; NCE remains per original update step',endpoints=['HOMO_SELECTED','HOMO_FINAL','R23_D1_FINAL2950'])
cfg['training_input_paths']={n:str(R23/n)for n in ['TRAIN_PRIVATE.npz','VALIDATION_INPUTS.npz','FINE_CONSTANTS.npz','CRITIC_SCALER.npz']}
cfg['actual_training_stream']=str(R23/'STANDARDIZED/TRAINING_STREAMS_PRIVATE.npz')
cfg['actual_training_random_states']=str(R23/'STANDARDIZED/TRAINING_RANDOM_STATES.json')
cfg['actual_training_receipt']=str(R23/'STANDARDIZED/TRAINING_RECEIPT.json')
cfg['old_final_G']=str(R23/'STANDARDIZED/FINAL_TRANSLATOR.pt')
cfg['endpoint_weight_paths']={'HOMO_SELECTED':str(ROOT/'HOMOGENEOUS_REORDER/TRANSLATOR.pt'),'HOMO_FINAL':str(ROOT/'HOMOGENEOUS_REORDER/FINAL_TRANSLATOR.pt'),'R23_D1_FINAL2950':cfg['old_final_G']}
(ROOT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n','utf8')
t=(R23/'online_training.py').read_text('utf8')
def replace(a,b):
    global t
    assert a in t,a;t=t.replace(a,b)
replace("if self.arm=='STANDARDIZED':","if self.arm in ['STANDARDIZED','HOMOGENEOUS_REORDER']:")
replace("pathlib.Path(cfg['remote_R18_root'])/'INITIAL_CRITIC.pt'","cfg['D_initial']")
replace("initial_rng={'noise':copy.deepcopy(rngnoise.bit_generator.state),'style':copy.deepcopy(rngstyle.bit_generator.state)}","initial_rng=copy.deepcopy(actual['original_rng_initial']);original_noise=actual.pop('original_noise');original_style=actual.pop('original_style');permutation=actual['original_row_ordinal'];grad_rows=[]")
replace("z=rngnoise.normal(size=(cfg['batch'],8)).astype('f4');white=rngnoise.normal(size=(cfg['batch'],480)).astype('f4');sty=rngstyle.normal(size=(cfg['batch'],8)).astype('f4');", "ordinals=permutation[sl];z=original_noise['z'][ordinals];white=original_noise['white'][ordinals];sty=original_style[ordinals];")
replace("ld.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),cfg['gradient_clip']);od.step()", "ld.backward();d_norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),cfg['gradient_clip']);od.step()")
replace("loss.backward();torch.nn.utils.clip_grad_norm_([*model.parameters(),*reader.parameters()],cfg['gradient_clip']);og.step()", "loss.backward();g_norm=torch.nn.utils.clip_grad_norm_([*model.parameters(),*reader.parameters()],cfg['gradient_clip']);grad_rows.append([float(g_norm),float(d_norm),min(1.,cfg['gradient_clip']/(float(g_norm)+1e-6)),min(1.,cfg['gradient_clip']/(float(d_norm)+1e-6))]);og.step()")
replace("'G_loss':float(loss),'D_loss':float(ld)","'G_loss':float(loss),'D_loss':float(ld),'G_reader_global_grad_norm':float(g_norm),'D_global_grad_norm':float(d_norm),'G_reader_clipped':float(g_norm)>cfg['gradient_clip'],'D_clipped':float(d_norm)>cfg['gradient_clip']")
replace("prefix={k:v[:count]for k,v in actual.items()if k!='NCE_positions'}", "prefix={k:v[:count]for k,v in actual.items()if isinstance(v,np.ndarray)and k!='NCE_positions'}")
replace("'noise_final':rngnoise.bit_generator.state,'style_initial':initial_rng['style'],'style_final':rngstyle.bit_generator.state", "'noise_final':actual['original_rng_final']['noise'],'style_initial':initial_rng['style'],'style_final':actual['original_rng_final']['style'],'reconstruction_call_order':'old perstep z32x8 then white32x480; independent style32x8; new training indexes precomputed original rows'")
replace("elapsed=time.monotonic()-clock;counter['training_s']+=elapsed;record=", "np.savez_compressed(root/'GRADIENT_NORMS_PRIVATE.npz',values=np.asarray(grad_rows,dtype='f8'));elapsed=time.monotonic()-clock;counter['training_s']+=elapsed;record=")
replace("'actually_sampled_unique_source':len(np.unique(prefix['source']))", "'actually_sampled_unique_source':len(np.unique(prefix['source'])),'actually_sampled_unique_REAL':len(np.unique(prefix['target'])),'actually_consumed_rows':count,'completed_all_original_rows':count==94400,'reorder_seed':cfg['reorder_seed'],'training_stop_reason':'updates_limit'if updates==cfg['updates']else'wallclock_limit','global_gradient_norms':{'columns':['G_reader_preclip','D_preclip','G_reader_clip_scale','D_clip_scale'],'quantiles':np.quantile(np.asarray(grad_rows),[0,.25,.5,.75,.95,1],axis=0).tolist()if grad_rows else None,'G_reader_clip_fraction':float(np.mean(np.asarray(grad_rows)[:,0]>cfg['gradient_clip']))if grad_rows else None,'D_clip_fraction':float(np.mean(np.asarray(grad_rows)[:,1]>cfg['gradient_clip']))if grad_rows else None}")
replace("'elapsed_s':time.monotonic()-clock},indent=2)","'elapsed_s':time.monotonic()-clock,'completed_original_row_prefix':updates*cfg['batch'],'no_retrain_after_scientific_undertraining':True},indent=2)")
(ROOT/'online_training.py').write_text(t,'utf8');ast.parse(t)
sources={n:{'bytes':(ROOT/n).stat().st_size,'sha256':hashlib.sha256((ROOT/n).read_bytes()).hexdigest()}for n in ['model.py','legacy_model.py','fine_core.py','prior.py','online_training.py','supervise.py','owned_worker.py']}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_science_started':True,'new_recipe_changes':'only row permutation/batch contexts; added recording of existing clip norms','sources':sources,'historical_inputs_readonly':True,'migration_tests':0},indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ready':str(ROOT),'source_count':len(sources)}))
