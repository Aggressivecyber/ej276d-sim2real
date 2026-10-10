from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json
R=Path(__file__).resolve().parents[1]
S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
H=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'
C=json.loads((S/'config.json').read_text('utf8'))

def edit(name,changes):
    old=(H/name).read_text('utf8');value=old
    for before,after in changes:
        assert value.count(before)==1,(name,before,value.count(before))
        value=value.replace(before,after)
    (S/name).write_bytes(value.encode('utf8'))

edit('distribution_loss.py',[
    ('import torch\n','import torch\nfrom precision_order import precision_order_loss\n'),
    ('def __call__(self,output,raw,reference,identity_input,identity_bins,source_psd,source_h,directions):',
     'def __call__(self,output,raw,reference,identity_input,identity_bins,source_psd,source_h,directions,source_interval_low,source_interval_high):'),
    ("        ds=source_psd[:,None]-source_psd[None,:];dy=yp[:,None]-yp[None,:]\n        weight=(self.sw[:,None]*self.sw[None,:])*(ds!=0)\n        pieces['source_order']=(weight*torch.relu(-torch.sign(ds)*dy)).sum()/weight.sum().clamp_min(1e-30)/self.cfg['PSD_loss_scale']",
     "        pieces['source_order'],order_diagnostic=precision_order_loss(yp,source_psd,source_interval_low,source_interval_high,self.sw,self.cfg['PSD_loss_scale'])"),
    ("            'PSD_by_region':bybin.detach()}","            'PSD_by_region':bybin.detach(),**order_diagnostic}")])

edit('artificial_check.py',[
    ('from distribution_loss import weighted_w1','from distribution_loss import weighted_w1\nfrom precision_order import rounding_psd_interval,artificial_precision_check'),
    ("    return {'artificial_only':True,'scientific_data_decoded':False,'response_training_updates':0,'extra_LP_solves':0,",
     "    precision_record=artificial_precision_check()\n    return {'precision_order':precision_record,'artificial_only':True,'scientific_data_decoded':False,'response_training_updates':0,'extra_LP_solves':0,"),
    ("    original_psd=psd(torch.as_tensor(source,dtype=torch.float64)).numpy()",
     "    original_psd=psd(torch.as_tensor(source,dtype=torch.float64)).numpy()\n    interval_low,interval_high=rounding_psd_interval(source)"),
    ("    loss,pieces,diagnostics=objective(output,raw,torch.as_tensor(real[ri],dtype=torch.float64,device=device),torch.as_tensor(real[ii],dtype=torch.float64,device=device),torch.as_tensor(rb[ii],device=device),\n        torch.as_tensor(original_psd[si],device=device),torch.as_tensor(h[si],device=device),objective.draw_directions(rng))",
     "    directions=objective.draw_directions(rng)\n    args=(output,raw,torch.as_tensor(real[ri],dtype=torch.float64,device=device),torch.as_tensor(real[ii],dtype=torch.float64,device=device),torch.as_tensor(rb[ii],device=device),\n        torch.as_tensor(original_psd[si],device=device),torch.as_tensor(h[si],device=device),directions)\n    loss,pieces,diagnostics=objective(*args,torch.as_tensor(interval_low[si],device=device),torch.as_tensor(interval_high[si],device=device))\n    from R59_distribution_loss import DistributionLoss as R59Loss\n    old_loss,old_pieces,_=R59Loss(statistics,cfg,device)(*args)\n    others=[k for k in pieces if k!='source_order']\n    assert all(torch.equal(pieces[k],old_pieces[k])for k in others)\n    new_gradient=torch.autograd.grad(sum(pieces[k]for k in others),raw,retain_graph=True)[0]\n    old_gradient=torch.autograd.grad(sum(old_pieces[k]for k in others),raw,retain_graph=True)[0]\n    assert torch.equal(new_gradient,old_gradient)\n    assert torch.allclose(loss,old_loss-old_pieces['source_order']+pieces['source_order'],rtol=1e-12,atol=1e-12)\n    assert torch.allclose(old_pieces['source_order'],diagnostics['original_order_loss'],rtol=1e-12,atol=1e-12)\n    precision_record=artificial_precision_check(device)"),
    ("    return {'artificial_only':True,'device':str(device),'scientific_data_decoded':False,'optimizer_updates':0,'ten_region_all_seven_losses_finite':True,",
     "    return {'precision_order':precision_record,'six_other_loss_values_and_gradients_exactly_match_R59':True,'artificial_only':True,'device':str(device),'scientific_data_decoded':False,'optimizer_updates':0,'ten_region_all_seven_losses_finite':True,")])

edit('train_final.py',[
    ('import os,sys,json,time,hashlib,traceback,copy,math','import os,sys,json,time,hashlib,traceback,copy,math,shutil'),
    ('from distribution_loss import DistributionLoss,fit_statistics,CATEGORIES,NAMES',
     'from distribution_loss import DistributionLoss,CATEGORIES,NAMES\nfrom precision_order import rounding_psd_interval'),
    ("    if not torch.cuda.is_available():raise RuntimeError('registered CUDA unavailable')",
     "    reviewed=json.loads((R/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))\n    for record in reviewed['files']:assert hashlib.sha256((R/record['name']).read_bytes()).hexdigest()==record['sha256'],record['name']\n    historical=json.loads(Path(C['input_paths']['R59_training_receipt']).read_text('utf8'))\n    assert historical['initial_state_SHA']==C['expected_R59_initial_state_SHA']and historical['stream_hashes']==C['expected_R59_stream_hashes']\n    if not torch.cuda.is_available():raise RuntimeError('registered CUDA unavailable')"),
    ("    statistics=fit_statistics(real,sp,source_h,sraw,rraw,valley,C)\n    np.savez_compressed(R/'TRAIN_DISTRIBUTION_CONSTANTS.npz',**statistics)",
     "    with np.load(C['input_paths']['distribution_statistics'],allow_pickle=False)as z:statistics={k:z[k].copy()for k in z.files}\n    assert np.array_equal(statistics['source_counts'],[len(g)for g in groupsS])and np.array_equal(statistics['REAL_counts'],[len(g)for g in groupsR])\n    shutil.copy2(C['input_paths']['distribution_statistics'],R/'TRAIN_DISTRIBUTION_CONSTANTS.npz')\n    interval_low,interval_high=rounding_psd_interval(source)\n    assert np.all(interval_low<=sp)and np.all(sp<=interval_high)\n    np.savez_compressed(R/'SOURCE_PSD_PRECISION_PRIVATE.npz',source_ids=source_ids,source_original_psd=sp,PSD_interval_low=interval_low,PSD_interval_high=interval_high)\n    write('SOURCE_ORDER_PRECISION_DEFINITION.json',{'interval_rule':C['source_order_rule'],'source_parents':len(source),'new_threshold_fit':False,'all_original_denominator_pairs_retained':True,'interval_width_quantiles':np.quantile(interval_high-interval_low,[0,.05,.5,.95,1]).tolist(),'not_physical_measurement_uncertainty':True})"),
    ("'new_normalization_is_TRAIN_only':True,'normalization_not_response_or_teacher_fit':True",
     "'new_normalization_is_TRAIN_only':False,'frozen_R59_TRAIN_normalization_reused':True,'normalization_not_response_or_teacher_fit':True"),
    ("    initial_hash=state_hash(model.state_dict());parameter_count=sum(p.numel() for p in model.parameters())",
     "    initial_hash=state_hash(model.state_dict());parameter_count=sum(p.numel() for p in model.parameters())\n    assert initial_hash==C['expected_R59_initial_state_SHA'],('initialization_changed',initial_hash)"),
    ("    dense=[];trace=[];global_max_q64=global_max_q32=0.;allfinite=True",
     "    dense=[];trace=[];global_max_q64=global_max_q32=0.;allfinite=True\n    order_columns=['original_order_loss','omitted_order_loss','original_nonzero_pair_mass','resolved_pair_fraction'];order_dense=[]"),
    ("            torch.as_tensor(sp[si],device=dev),torch.as_tensor(source_h[si],device=dev),directions)",
     "            torch.as_tensor(sp[si],device=dev),torch.as_tensor(source_h[si],device=dev),directions,\n            torch.as_tensor(interval_low[si],device=dev),torch.as_tensor(interval_high[si],device=dev))"),
    ("        row=torch.stack(values).cpu().numpy().astype('f8');dense.append(row)",
     "        row=torch.stack(values).cpu().numpy().astype('f8');dense.append(row)\n        order_row=torch.stack([diagnostic[k]for k in order_columns]).cpu().numpy().astype('f8');order_dense.append(order_row)\n        assert np.isfinite(order_row).all()and np.all(order_row>=0)\n        assert abs(order_row[0]-order_row[1]-row[3])<=1e-10+1e-10*abs(order_row[0])"),
    ("            trace.append(record);write(ARM+'/TRAINING_TRACE.json',trace);write('TRAIN_PROGRESS.json',record);print(json.dumps(record),flush=True)",
     "            record['precision_order_last_up_to1000_mean']=dict(zip(order_columns,np.asarray(order_dense[-min(len(order_dense),1000):]).mean(0).tolist()))\n            trace.append(record);write(ARM+'/TRAINING_TRACE.json',trace);write('TRAIN_PROGRESS.json',record);print(json.dumps(record),flush=True)"),
    ("'loss_columns':columns,'dense_losses':np.asarray(dense),'initial_state_SHA':initial_hash",
     "'loss_columns':columns,'dense_losses':np.asarray(dense),'precision_order_columns':order_columns,'precision_order_values':np.asarray(order_dense),'initial_state_SHA':initial_hash"),
    ("    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},D/'FINAL_FLOW.pt')",
     "    assert {k:v.hex()for k,v in hashes.items()}==C['expected_R59_stream_hashes'],('R59_stream_mismatch',hashes)\n    final_states={k:copy.deepcopy(v.bit_generator.state)for k,v in streams.items()}\n    assert final_states==json.loads(Path(C['input_paths']['R59_final_random_states']).read_text('utf8'))\n    with np.load(C['input_paths']['R59_parent_exposures'],allow_pickle=False)as z:\n        for key,value in [('source_ids',source_ids),('source_counts',countsS),('REAL_reference_counts',countsR),('REAL_identity_counts',countsI),('source_bin0',sraw),('REAL_bin0',rraw)]:assert np.array_equal(z[key],value),key\n    np.savez_compressed(D/'PRECISION_ORDER_BY_UPDATE_PRIVATE.npz',columns=np.asarray(order_columns),values=np.asarray(order_dense,dtype='f8'))\n    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},D/'FINAL_FLOW.pt')"),
    ("'CPU_threads':C['CPU_threads'],'source_val_target_val_decoded':False,'DEV_read':0,'EVAL_read':0,'CAL_decode':0,'compute_budget_enforced':False,'model_goal_complete':False}",
     "'CPU_threads':C['CPU_threads'],'source_val_target_val_decoded':False,'DEV_read':0,'EVAL_read':0,'CAL_decode':0,'compute_budget_enforced':False,'model_goal_complete':False,\n        'single_factor_integrity':{'same_R59_initial_state':True,'all_six_consumed_stream_hashes_equal_R59':True,'all_parent_exposures_equal_R59':True,'final_random_states_equal_R59':True,'frozen_R59_distribution_statistics_reused':True,'sole_change':C['sole_scientific_change']},\n        'source_order_rule':C['source_order_rule'],'precision_order_columns':order_columns,'precision_order_trace_rows':len(order_dense)}")])

files=list(S.glob('*.py'))
for path in files:ast.parse(path.read_text('utf8'),filename=str(path))
source=(R/'work/R60_FROZEN_TRAIN_GRADIENT_DIAGNOSTIC_001/diagnostic_math.py').read_text('utf8')
new=(S/'precision_order.py').read_text('utf8')
def tree(text,name):
    return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)
assert tree(source,'rounding_psd_interval')==tree(new,'rounding_psd_interval')
assert tree(source,'psd_np')==tree(new,'psd_np')
for name in ['network.py','noise_core.py','source_flow_core.py']:
    assert (H/name).read_bytes()==(S/name).read_bytes()
record={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','training_implementation_prepared':True,
        'frozen_R59_network_noise_endpoint_exact_bytes':True,'R60_interval_algorithm_same_AST':True,'all_AST_pass':True,
        'scientific_execution_started':False,'scientific_data_decoded':False,
        'changes':['only source_order numerator is masked by disjoint stored-PSD intervals','load identical frozen R59 statistics instead of recomputing them',
                   'require matching initial state and consumed random streams/exposures/states','record omitted order penalty and resolved pair mass']}
(S/'TRAIN_IMPLEMENTATION_PREPARATION.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps(record,indent=2))
