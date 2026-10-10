from pathlib import Path
import json,os,shutil,ast,datetime,hashlib
R=Path(__file__).resolve().parent;P=R.parents[1];OLD=R.parent/'R42_SOURCE_INITIALIZED_NEURAL_FLOW_001'
assert not (R/'PREPARATION_RECEIPT.json').exists()
reg=json.loads((P/'reviews/R43_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
cfg=json.loads((OLD/'config.json').read_text('utf8'));remote=json.loads((OLD/'remote_config.json').read_text('utf8'))
A='SCALAR_TIME_FLOW_FINAL';C='SCALAR_TIME_FLOW_REAL_CONTROL'
newremote='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R43_REMOTE_20261009_233131_UTC'
common=dict(stage=R.name,registered_UTC=reg['registered_UTC'],deadline_UTC=reg['executor_seal_deadline_UTC'],absolute_deadline_UTC=reg['stage_deadline_UTC'],
    remote_root=newremote,arms=[A],endpoints=[A,C],time_features='[t,zero16]; original frequency buffer unused',zero_time_input_weights=2048,
    parameter_initialization='new same-seed construction; never load previous response weights as initialization',
    selection='sole final2950; original fixed32/64 TRAIN prerequisite',TRAIN_constants_refit=0,PSD_coupling_refit_or_resample=0,
    first_actual_command={'tool':'ef875e','PID':53840,'native_exit':0,'UTC':'2026-10-09T23:32:02.1274872Z'})
cfg.update(common);remote.update(common)
cfg.update(constants_path=str(R/'UNIT_CHARGE_CONSTANTS.npz'),endpoint_weight_paths={A:str(R/A/'FINAL_FLOW.pt'),C:str(R/A/'FINAL_FLOW.pt')},
    frozen_R42_constants=str(OLD/'UNIT_CHARGE_CONSTANTS.npz'),frozen_R42_constants_json=str(OLD/'UNIT_CHARGE_CONSTANTS.json'),
    frozen_R42_coupling=str(OLD/'SOURCE_NEURAL_FLOW_FINAL/ACTUAL_CFM_STREAM_PRIVATE.npz'),
    frozen_R42_train_receipt=str(OLD/'SOURCE_NEURAL_FLOW_FINAL/TRAINING_RECEIPT.json'),
    historical_R42_initial_state=str(OLD/'SOURCE_NEURAL_FLOW_FINAL/INITIAL_FLOW.pt'),R42_root=str(OLD))
oldremote=json.loads((OLD/'remote_config.json').read_text('utf8'))['remote_root']
remote.update(constants_path=newremote+'/UNIT_CHARGE_CONSTANTS.npz',endpoint_weight_paths={A:newremote+'/'+A+'/FINAL_FLOW.pt',C:newremote+'/'+A+'/FINAL_FLOW.pt'},
    frozen_R42_constants=oldremote+'/UNIT_CHARGE_CONSTANTS.npz',frozen_R42_constants_json=oldremote+'/UNIT_CHARGE_CONSTANTS.json',
    frozen_R42_coupling=oldremote+'/SOURCE_NEURAL_FLOW_FINAL/ACTUAL_CFM_STREAM_PRIVATE.npz',
    frozen_R42_train_receipt=oldremote+'/SOURCE_NEURAL_FLOW_FINAL/TRAINING_RECEIPT.json',
    historical_R42_initial_state=oldremote+'/SOURCE_NEURAL_FLOW_FINAL/INITIAL_FLOW.pt')
sources=['evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py','endpoint_common.py','temporal_common.py',
    'common_math.py','mean_description.py','random_diagnostics.py','flow_diagnostics.py','flow_core.py','model.py','legacy_model.py',
    'owned_worker.py','supervise.py','linux_owned_worker.py','linux_supervise.py','launch_remote.py','resource_check.py','covariance_description.py',
    'generate_all.py','evaluate_saved.py','replay_all.py','reorder_core.py','retrieve.py','psd_order_core.py','deploy_training.py',
    'source_flow_core.py','flow_runtime.py','network.py','train_server.py','summarize_and_plot.py','plot_train_failure.py']
for n in sources:
    s=(OLD/n).read_text('utf8').replace('SOURCE_NEURAL_FLOW_FINAL',A).replace('SOURCE_NEURAL_FLOW_REAL_CONTROL',C)
    s=s.replace('R42 saved evaluation complete','R43 saved evaluation complete').replace('Source neural flow','Scalar-time flow')
    # Accurate support alias; actual NN execution remains all rows, including outside support.
    s=s.replace("'flow_applied'","'inside_TRAIN_support'")
    (R/n).write_text(s,'utf8')
payload=['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','DEV_SOURCE_METADATA.npz',
    'REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json','VALLEY_DEFINITION.json','AFFINE_CONSTANTS.npz',
    'UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json']
for n in payload:shutil.copy2(OLD/n,R/n)
shutil.copy2(OLD/'TRAIN_SOLVER_CHECK_PRIVATE.npz',R/'R42_TRAIN_SOLVER_CHECK_PRIVATE.npz')
shutil.copy2(OLD/'TRAIN_SOLVER_CHECK.json',R/'R42_TRAIN_SOLVER_CHECK.json')
# Network constructor and state shapes unchanged; only the actual time-input expression differs.
p=R/'network.py';s=p.read_text('utf8')
s=s.replace('time=t.to(y).reshape(-1,1);angle=2*math.pi*time*self.frequency[None,:].to(y)',
    'time=t.to(y).reshape(-1,1);time17=torch.cat((time,torch.zeros((len(y),16),dtype=y.dtype,device=y.device)),1)')
s=s.replace('torch.cat((time,angle.sin(),angle.cos(),onehot),1)','torch.cat((time17,onehot),1)')
p.write_text(s,'utf8')
# Reuse fixed constants and exact saved statistical coupling, rather than re-estimation.
p=R/'train_server.py';s=p.read_text('utf8').replace('import os,time,pathlib,json,datetime,traceback,hashlib,sys,copy','import os,time,pathlib,json,datetime,traceback,hashlib,sys,copy,shutil')
start=s.index("    with np.load(CFG['affine_path']")
end=s.index('    actual=reconstruct_and_reorder',start)
s=s[:start]+'''    shutil.copy2(CFG['frozen_R42_constants'],ROOT/'UNIT_CHARGE_CONSTANTS.npz')
    shutil.copy2(CFG['frozen_R42_constants_json'],ROOT/'UNIT_CHARGE_CONSTANTS.json')
    with np.load(ROOT/'UNIT_CHARGE_CONSTANTS.npz',allow_pickle=False)as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
    sq=q_np(source);rq=q_np(real)
    if not np.isfinite(sq).all()or not np.isfinite(rq).all()or not(sq>0).all()or not(rq>0).all():raise ValueError('TRAIN raw source/REAL nonpositive/nonfinite q; no clipping/deletion')
    def bins(q):
        v=np.log(1000.*q);b=np.searchsorted(edges,v,side='right')-1;b[v==edges[-1]]=7;return b
    sb=bins(sq);rb=bins(rq)
    with np.load(CFG['frozen_R42_coupling'],allow_pickle=False)as z:coupled={k:z[k].copy()for k in z.files}
    write('CONSTANTS_COUPLING_REUSE.json',{'constants_refits':0,'coupling_resorts_resamples':0,'constants':CFG['frozen_R42_constants'],'coupling':CFG['frozen_R42_coupling'],'target_statistical_not_true_pairs':True})
''' +s[end:]
s=s.replace("    reference=json.loads(pathlib.Path(CFG['original_training_receipt_path']).read_text('utf8'))['records']['INDEPENDENT_CFM']", "    reference=json.loads(pathlib.Path(CFG['frozen_R42_train_receipt']).read_text('utf8'))\n    assert all(np.array_equal(actual[k],coupled[k])for k in ['source','target','bin','original_row_ordinal'])")
s=s.replace("tt_all=np.concatenate([rt.uniform(0,1,size=32).astype('f4')for _ in range(2950)])", "tt_all=coupled['t'].copy();expected_t=np.concatenate([rt.uniform(0,1,size=32).astype('f4')for _ in range(2950)]);assert np.array_equal(tt_all,expected_t)")
s=s.replace("    torch.save(state(model),dest/'INITIAL_FLOW.pt')", "    initial_state=state(model);torch.save(initial_state,dest/'INITIAL_FLOW.pt')")
s=s.replace("    zero=model(torch.zeros((4,480)", "    time_inputs=[]\n    hook=model.embedding[0].register_forward_pre_hook(lambda module,args:time_inputs.append(args[0].detach().cpu().clone()))\n    zero=model(torch.zeros((4,480)")
s=s.replace("torch.tensor([0.,.2,.6,1.],device=DEVICE)","torch.tensor([0.,.5,1.,1.],device=DEVICE)")
s=s.replace("    assert torch.equal(zero,torch.zeros_like(zero))",'''    assert torch.equal(zero,torch.zeros_like(zero));hook.remove()
    vec=time_inputs[0];assert vec.shape==(4,25) and torch.count_nonzero(vec[:,1:17])==0
    np.savez_compressed(dest/'TIME_FEATURES_ARTIFICIAL_PRIVATE.npz',actual_embedding_input25=vec.numpy())
    initial_zero_columns=model.embedding[0].weight[:,1:17].detach().clone()
    zero_grad_checks=0;zero_value_checks=0;t_nonzero_steps=0;bin_nonzero_steps=0;other_nonzero_steps=0
    oldpath=pathlib.Path(CFG['historical_R42_initial_state']);initial_equal='unknown'
    if oldpath.is_file():
        oldinit=torch.load(oldpath,map_location='cpu',weights_only=True)
        initial_equal={'same_keys':set(oldinit)==set(initial_state),'each_tensor_bitwise_equal':{k:bool(torch.equal(v,oldinit[k]))for k,v in initial_state.items()}}
        assert initial_equal['same_keys'] and all(initial_equal['each_tensor_bitwise_equal'].values())
    write('INITIALIZATION_RECEIPT.json',{'actual_initial_state_saved':True,'historical_R42_state':str(oldpath),'historical_state_exists':oldpath.is_file(),'R42_actual_initial_state_comparison':initial_equal,'same_construction_seed':CFG['model_seed'],'R42_weights_loaded_as_initialization':False,'new_model_initialization_only':True,'time_embedding_input25':vec.tolist(),'zero_time_input_column_weight_count':2048,'original_frequency_buffer_unused':True})''')
s=s.replace("tick=time.monotonic();p,cost,groups,keys=ordered_coupling(source[sidx],real[tidx],b,source[sidx],real[tidx]);sorting_s+=time.monotonic()-tick", "p=coupled['within_batch_target_position'][sl].astype('i8');assert np.array_equal(tidx[p],coupled['matched_target'][sl])")
s=s.replace("        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();",'''        gg=model.embedding[0].weight.grad
        assert gg is not None and torch.count_nonzero(gg[:,1:17])==0
        zero_grad_checks+=1;t_nonzero_steps+=int(torch.count_nonzero(gg[:,0])>0);bin_nonzero_steps+=int(torch.count_nonzero(gg[:,17:])>0)
        other_nonzero_steps+=int(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for n,p in model.named_parameters()if n!='embedding.0.weight'))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        assert torch.equal(model.embedding[0].weight[:,1:17],initial_zero_columns);zero_value_checks+=1
        ''')
s=s.replace("src_key[sl]=keys[2];tgt_key[sl]=keys[5]", "src_key[sl]=coupled['source_raw_PSD'][sl];tgt_key[sl]=coupled['target_original_PSD'][sl]")
s=s.replace("'PSD_stable_coupling_statistical_not_truth':True", "'PSD_stable_coupling_statistical_not_truth':True,'exact_saved_R42_coupling_reused':True,'constants_refits':0,'time_features':'scalar t plus16zeros then8bins'")
s=s.replace("    write(A+'/TRAINING_RECEIPT.json',rec);",'''    parameter_checks={'zero_input_columns':list(range(1,17)),'zero_input_column_weight_count':2048,'zero_gradient_update_checks':zero_grad_checks,'unchanged_value_update_checks':zero_value_checks,'initial_final_zero_columns_bitwise_equal':bool(torch.equal(model.embedding[0].weight[:,1:17],initial_zero_columns)),'scalar_t_nonzero_gradient_steps':t_nonzero_steps,'bin_inputs_nonzero_gradient_steps':bin_nonzero_steps,'other_parameters_nonzero_gradient_steps':other_nonzero_steps,'total_parameters':rec['parameter_count'],'maximum_data_gradient_participating_parameter_count':rec['parameter_count']-2048,'all_parameters_finite':all(torch.isfinite(p).all()for p in model.parameters()),'final_zero_columns_gradient_maxabs':float(model.embedding[0].weight.grad[:,1:17].abs().max()),'actual_time17_and_embedding25_saved':True}
    assert zero_grad_checks==zero_value_checks==2950 and parameter_checks['initial_final_zero_columns_bitwise_equal'] and t_nonzero_steps>0 and bin_nonzero_steps>0
    write(A+'/SCALAR_TIME_PARTICIPATION.json',parameter_checks)
    write(A+'/TRAINING_RECEIPT.json',rec);''')
assert 'ordered_coupling(source'not in s
p.write_text(s,'utf8')
# Same128 numerical inputs, explicit copied fields. No TRAIN selection or fresh white reconstruction.
check=(OLD/'check_train_solver.py').read_text('utf8').replace('SOURCE_NEURAL_FLOW_FINAL',A)
start=check.index("with np.load(CFG['training_input_paths']")
end=check.index("x=np.concatenate",start)
check=check[:start]+'''with np.load(ROOT/'R42_TRAIN_SOLVER_CHECK_PRIVATE.npz',allow_pickle=False)as z:
    x=z['input_V'].copy();white=z['white'].copy();si=z['source_TRAIN_index'].copy();ri=z['REAL_TRAIN_index'].copy();wanted=z['white_original_row_ordinal'].copy();old_rms=z['RMS_difference_V'].copy()
with np.load(ROOT/'UNIT_CHARGE_CONSTANTS.npz',allow_pickle=False)as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
assert x.shape==white.shape==(128,480) and len(si)==len(ri)==64 and len(wanted)==128
''' +check[end:]
check=check.replace("x=np.concatenate([src[si],real[ri]]);net=", "net=")
check=check.replace("RMS_difference_V=rms)","RMS_difference_V=rms,R42_RMS_difference_V=old_rms,paired_RMS_difference_V=rms-old_rms)")
check=check.replace("'no_prefix_or_64step_candidate':True", "'no_prefix_or_64step_candidate':True,'exact_R42_inputs_white_indices_order_reused':True,'R42_reference_RMS_median_LSB':float(np.median(old_rms)/lsb),'R42_reference_RMS_p95_LSB':float(np.quantile(old_rms,.95)/lsb),'paired_RMS_difference_LSB_mean':float(np.mean((rms-old_rms)/lsb))")
(R/'check_train_solver.py').write_text(check,'utf8')
# Both successful and early-failure reporting will be completed from saved results only.
for name in ['plot_train_failure.py','summarize_and_plot.py']:
    p=R/name;s=p.read_text('utf8').replace('R42:','R43:').replace('source_initialized_neural_flow_recipe','scalar_time_neural_flow_recipe').replace('Standalone source-initialized','Standalone scalar-time source-initialized')
    p.write_text(s,'utf8')
for n,c in [('config.json',cfg),('remote_config.json',remote)]:(R/n).write_text(json.dumps(c,indent=2,allow_nan=False)+'\n','utf8')
(R/'START_RECEIPT.json').write_text(json.dumps(cfg['first_actual_command'],indent=2)+'\n','utf8')
(R/'PROTOCOL.md').write_text('''# R43 scalar-time-only source neural flow

Tasks R43 + inherited R42 are authoritative. Only time features change: [t,zero16]+bin8, same25input/1,371,841parameter architecture and constructor seed20261010420. Frequency buffer remains compatibility only, never read in forward. New model/newAdam .0002(.9,.999),clip5; no response checkpoint initialization.2048 embedding first-layer weights at columns1:17 must have zero data gradients and unchanged values at all2950steps. Original t/bin gradients and other finite parameters recorded. Actual initial model saved; historical R42 initial state compared only if actual saved original exists.

Exact R42 TRAIN mu/scale/edges and saved coupling/target/t reused, zero refits/resorts. Original source/target/bin/ordinal/white94400 and seven audit streams match. t PCG64(20261010331),white actually used,z/style/NCE audit unused. Same full480 unit-charge coordinates/tangent projection/1mV initial noise/only velocity MSE; float64 arithmetic/float32 NN and preADC, no final renormalization. Sole endpoint2950.

Before DEV, exact R42 saved128TRAIN input/white/indices/ordinal, fixed32/64Heun/chunk64; RMS median<=.25LSB,p95<=1LSB/preADCrelativeq<=1e-5.64reference never candidate. On failure only saved TRAIN report/graphs/seal, noDEV/HGB/EM/replay or third time/solver recipe. On pass automatic original complete6gates +main2000/newREAL2000/joint1024/probe832/extraSOURCE128,20HGB/16slots,4000independentCPUreplay. All2000NN including417outside; nearest-edge conditioning only,noBASEfallback. inside_TRAIN_support is original supportmask; neural_flow_applied_all_rows is execution flag. Functional components not physical independent signal/noise. Old references all0refits,RAW RR2Dpaired unknown.

Same phase/group600/250/500bootstrap and original five joint scientific gates, plus scalar-gradient/init/flow/numerical/replay evidence. Science2700/train900s,4threads/RSS+GPU4GiB/output512MiB; one job at a time. Seal<=2026-10-10T01:01:31.265601Z. No EVAL/CAL/newREAL/G4/install/agents/Git/migration tests. Old stages immutable.
''','utf8')
bindings={}
for p in R.glob('*.py'):
    b=p.read_bytes();ast.parse(b.decode('utf8'));bindings[p.name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
rec={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_AST':bindings,'dependency_payload_bytes':sum((R/n).stat().st_size for n in payload),'science_started':False,'old_response_weights_used':False,'old_stage_writes':0,'migration_content_tests':0,'TRAIN_constants_coupling_refits':0,'R42_initial_actual_file_exists':(OLD/'SOURCE_NEURAL_FLOW_FINAL/INITIAL_FLOW.pt').is_file()}
(R/'PREPARATION_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':rec['UTC'],'AST_count':len(bindings),'dependency_bytes':rec['dependency_payload_bytes']}),flush=True)
