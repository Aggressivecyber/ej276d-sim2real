from __future__ import annotations
import ast,datetime,json,os,pathlib,shutil
root=pathlib.Path(__file__).resolve().parent
failed=root/'failed_002';failed.mkdir(exist_ok=False)
for name in ['numeric_worker.py.stdout.log','numeric_worker.py.stderr.log','LOCAL_SCIENCE_TOKEN.json','ACTIVE_JOB.json','RUN_HISTORY.json','LOCAL_NATIVE_RECEIPT.json','READINESS.json','numeric_worker.py','training.py','local_supervise.py']:
    shutil.copyfile(root/name,failed/name)
for p in root.glob('FAILURE_*.json'):shutil.copyfile(p,failed/p.name)
history=json.loads((failed/'RUN_HISTORY.json').read_text('utf8'))
assert history[-1]['exit_code']==1 and history[-1]['worker_PID']==14520
last=max((json.loads(p.read_text('utf8'))for p in root.glob('FAILURE_*.json')),key=lambda r:r['PID']==14520)
assert last['PID']==14520 and last['counters']['D_updates']==1 and last['counters']['G_updates']==0
partial=root/'GLOBAL_STYLE'
assert partial.resolve().is_relative_to(root.resolve()) and (failed/'GLOBAL_STYLE').resolve().is_relative_to(root.resolve())
partial.rename(failed/'GLOBAL_STYLE')
eq=json.loads((failed/'GLOBAL_STYLE/STRUCTURAL_EQUIVALENCE.json').read_text('utf8'))
assert eq['pass'] and eq['before_parameter_updates'] and eq['total_forward_rows']==96
training=(root/'training.py').read_text('utf8')
old='self.records.append(x.copy());return x'
assert training.count(old)==1
training=training.replace(old,'self.records.append(x.copy());return x.astype(np.int64,copy=False)')
begin=training.index('    # This fixed TRAIN batch checks structure only:')
end=training.index('    torch.save(state(model)',begin)
replacement="""    # The same unchanged local model/weights already passed 96 structural rows
    # before failed_002 reached the NCE indexing interface. Reuse that evidence.
    eqpath=pathlib.Path(__file__).resolve().parent/'failed_002/GLOBAL_STYLE/STRUCTURAL_EQUIVALENCE.json'
    equivalence=json.loads(eqpath.read_text('utf8'))
    assert equivalence['pass'] and equivalence['all_inherited_parameters_equal']
    assert all(torch.equal(model.state_dict()[k].cpu(),v)for k,v in oldG.items())
    assert bool((model.style_film[-1].weight==0).all()) and bool((model.style_film[-1].bias==0).all())
    equivalence={**equivalence,'reused_after_NCE_dtype_only_repair':True,'new_structural_forward_rows':0,'evidence_path':str(eqpath)}
    (root/'STRUCTURAL_EQUIVALENCE.json').write_text(json.dumps(equivalence,indent=2)+'\\n')
    model.train()
"""
training=training[:begin]+replacement+training[end:]
training=training.replace("min(cfg['updates'],nold)+1", "min(cfg['updates'],nold)-1+1")
training=training.replace("cfg['training_wall_s']-5", "cfg['training_wall_s']-5-10.956509799987543")
assert 'return x.astype(np.int64,copy=False)' in training
ast.parse(training);(root/'training.py').write_text(training,'utf8')
supervisor=(root/'local_supervise.py').read_text('utf8').replace("ROOT/'failed_001/RUN_HISTORY.json'","ROOT/'failed_002/RUN_HISTORY.json'")
ast.parse(supervisor);(root/'local_supervise.py').write_text(supervisor,'utf8')
(root/'LOCAL_SCIENCE_TOKEN.json').unlink()
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'failed_run_counters':last['counters'],'failed_native_wall_s':history[-1]['elapsed_s'],
 'scientific_cumulative_before_retry_s':sum(h['elapsed_s']for h in history),
 'NCE_actual_positions_changed':False,'saved_NCE_dtype':'uint16 unchanged','index_interface_dtype':'int64',
 'structural_evidence_reused':True,'total_structural_rows_consumed':96,
 'remaining_completed_G_D_limit':3999,'failed_D_update_consumed':1,
 'failed_training_upper_wall_s':history[-1]['elapsed_s'],'training_wall_deducted':True,
 'scientific_recipe_changed':False}
(root/'ENGINEERING_REPAIR_002.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt))
