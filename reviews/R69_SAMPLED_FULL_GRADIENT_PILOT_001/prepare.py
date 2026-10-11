"""Prepare one new estimator pilot from the preserved R67 harness; no science."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, shutil, sys

P=Path(__file__).resolve().parent; R=P.parents[1]
OLD=R/'work/R67_EQUAL_COST_REPAIR_001'; S=R/'work/R69_SAMPLED_FULL_GRADIENT_PILOT_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P)
assert not S.exists()
S.mkdir()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def put(name,s): compile(s,name,'exec'); (S/name).write_text(s,encoding='utf8')
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b,1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

cfg=json.loads((OLD/'config.json').read_text('utf8'))
for k in ['full_updates','full_arm_cap_s','aggregate_budget_s','prior_attempt_native_wall_s','prior_invalid_attempt','remaining_reserve_after_repair_max_s']:
    cfg.pop(k,None)
now=datetime.now(timezone.utc)
cfg.update(stage=S.name,registered_UTC=None,prepared_UTC=now.isoformat(),
    remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R69_ROOT_'+now.strftime('%Y%m%d_%H%M%S_UTC'),
    science_budget=600,worker_cap_s=600,candidate_updates=100,candidate_arm_cap_s=200,
    sampled_block_count=4,block_sampler_seed=202610116901,relative_gradient_MSE_max=1.,
    probe_seeds=list(range(202610116911,202610116919)),REAL_control_seed=202610116999,bootstrap_seed=202610116980,
    arms={'candidate':'current_full_D_sampled4_of104_parameter_VJPs_plus_original_C','mini':'original_R63_nine_term_microbatch'},
    no_optimizer_update_in_gradient_admission=True,mathematical_gradient_unbiased_before_clip_only=True,
    original_Q_population_and_acceptance_unchanged=True,
    scientific_entry_requires_registration_and_root_review=True)
write(S/'config.json',cfg)
shutil.copy2(R/('tasks/'+S.name+'.md'),S/'PROTOCOL.md')
shutil.copy2(OLD/'mini_step.py',S/'mini_step.py')
shutil.copy2(OLD/'full_step.py',S/'reference_full_step.py')
shutil.copy2(OLD/'linux_owned_worker.py',S/'linux_owned_worker.py')
sup=(OLD/'linux_supervise.py').read_text('utf8').replace('R67','R69').replace('395','600')
put('linux_supervise.py',sup)

full=(OLD/'full_step.py').read_text('utf8')
ref=full[:full.index("inflight = 'single_final_cast_clip_Adam'")]
ref=once(ref,"for group in optimizer.param_groups:\n    group['lr'] = rate\n","")
assert 'optimizer.step(' not in ref and 'install_cpu_f64_clip_step(' not in ref
put('reference_gradient.py',ref)
step=once(full,"versions = tuple((p._version for p in params))",
    "sampled_indices, ht_scale = sample_blocks(block_rng, len(schedule), PC['sampled_block_count'])\n"
    "sampled_schedule = [schedule[int(i)] for i in sampled_indices]\n"
    "sampled_parent_rows = sum(hi-lo for lo,hi in sampled_schedule)\n"
    "sampled_block_history.append([int(i) for i in sampled_indices])\n"
    "versions = tuple((p._version for p in params))")
# The full first pass stays byte-for-byte. Only the graph-bearing replay loop changes.
step=once(step,"for lo, hi in schedule:\n    inflight = {'replay_forward_and_VJP_chunk': [lo, hi]}",
    "for lo, hi in sampled_schedule:\n    inflight = {'replay_forward_and_VJP_chunk': [lo, hi]}")
step=once(step,"assert torch.isfinite(D_gradient).all()\nD_gradient_norm",
    "D_gradient *= ht_scale\nassert torch.isfinite(D_gradient).all()\nD_gradient_norm")
step=once(step,"replay_row = np.array([6593, 6593, len(schedule), matched_y, matched_codes, matched_free, 1, 1, len(schedule), 1, 1, 1], dtype='i8')\nassert np.array_equal(replay_row, C['expected_replay_row'])",
    "replay_row = np.array([6593, sampled_parent_rows, len(sampled_schedule), matched_y, matched_codes, matched_free, 1, 1, len(sampled_schedule), 1, 1, 1], dtype='i8')\n"
    "assert matched_y == matched_codes == matched_free == PC['sampled_block_count']\n"
    "for lo,hi in sampled_schedule: sampled_backward_counts[lo:hi] += 1")
step=once(step,"== 13570", "== 6593 + sampled_parent_rows + 384")
step=once(step,"== 209", "== len(schedule) + len(sampled_schedule) + 1")
step=once(step,"== 104", "== len(sampled_schedule)")
old_counts="for key, expected in C['expected_completed_update_counts'].items():\n    assert expected % C['updates'] == 0\n    counter[key] += expected // C['updates']"
new_counts="""actual_update_counts = dict(new_response_updates=1,SOURCE_training_rows=320,REAL_reference_rows=320,REAL_identity_rows=64,
    full_D_SOURCE_law_rows=6593,full_D_REAL_law_rows=21047,
    scientific_neural_forward_rows=6593+sampled_parent_rows+384,
    scientific_neural_forward_calls=len(schedule)+len(sampled_schedule)+1,
    D_parameter_VJP_calls=len(sampled_schedule),C_parameter_reverse_calls=1,
    full_D_endpoint_reverse_calls=1,full_D_objective_calls=1,replay_chunks_checked=len(sampled_schedule))
assert set(actual_update_counts)==set(C['expected_completed_update_counts'])
for key,amount in actual_update_counts.items():counter[key]+=amount"""
step=once(step,old_counts,new_counts)
put('sampled_step.py',step)

s=(OLD/'pilot.py').read_text('utf8')
s=s.replace('R67 requires','R69 requires').replace("P/'full_step.py'","P/'sampled_step.py'")
s=s.replace("name=='full'","name=='candidate'").replace("make_arm('full')","make_arm('candidate')").replace("train(full,'full')","train(full,'candidate')").replace("evaluate(full,'full')","evaluate(full,'candidate')")
s=s.replace("PC['full_updates']","PC['candidate_updates']").replace("PC['full_arm_cap_s']","PC['candidate_arm_cap_s']")
s=s.replace('full20 arm exceeded120 seconds','sampled100 arm exceeded200 seconds').replace('fixed395-second remaining budget','fixed600-second worker budget')
s=s.replace('full_vs_mini_relative_improvement','candidate_vs_mini_relative_improvement').replace('full_vs_baseline_relative_improvement','candidate_vs_baseline_relative_improvement').replace('full_beats_mini_probes','candidate_beats_mini_probes').replace('full_mean=','candidate_mean=').replace('full_training=','candidate_training=')
s=s.replace('descriptive_paired_noise_MC_full_minus_mini_95_interval','descriptive_paired_noise_MC_candidate_minus_mini_95_interval')
s=s.replace("'Do not resume costly R65; no equal-cost benefit support at this starting state'", "'Close this fixed sampled4 estimator; no equal-cost joint benefit; R65 stays retired'")
s=once(s,'    import numpy as np\n',"    from sampled_blocks import original_schedule,sample_blocks,conditional_variance_of_sum\n"
    "    from check_sampled_blocks import run_checks\n"
    "    write('ARTIFICIAL_CHECK.json',run_checks())\n"
    "    import numpy as np\n")
s=once(s,"    mini_code=compile((P/'mini_step.py').read_text(),'bound_R63_update_body','exec')",
    "    mini_code=compile((P/'mini_step.py').read_text(),'bound_R63_update_body','exec')\n"
    "    reference_code=compile((P/'reference_gradient.py').read_text(),'original_full_no_update_gradient','exec')")
s=once(s,"schedule=[(lo,min(lo+64,len(source)))for lo in range(0,len(source),64)],",
    "schedule=original_schedule(len(source),64),\n"
    "            block_rng=np.random.Generator(np.random.PCG64(PC['block_sampler_seed'])),sampled_block_history=[],sampled_backward_counts=np.zeros(len(source),dtype='i8'),")
# Every stage receipt reports its own actual work, not R65's original complete-replay estimates.
s=once(s,"neural_forward_rows=n*(13570 if name=='candidate'else 384),",
    "neural_forward_rows=(H['physical']['neural_forward_rows'] if name=='candidate' else n*384),")
s=once(s,"        if name=='mini':\n            receipt.update(cost_match_absolute_s=abs(elapsed-threshold)",
    "        receipt['counter_delta']={k:int(H['counter'][k]-recovery['counter'][k])for k in H['counter']}\n"
    "        if name=='candidate':\n"
    "            receipt.update(block_count=104,selected_blocks_per_update=4,HT_scale=26.,\n"
    "                actual_D_backward_parent_occurrences=int(H['sampled_backward_counts'].sum()),\n"
    "                sampled_backward_unique_parents=int(np.count_nonzero(H['sampled_backward_counts'])),\n"
    "                D_parameter_VJP_calls=H['physical']['D_parameter_VJP_calls'],\n"
    "                all_parent_forward_and_distribution_exposure_preserved=True,all_parent_backward_exposure_each_step=False,\n"
    "                sampler_final_state=copy.deepcopy(H['block_rng'].bit_generator.state))\n"
    "            assert receipt['D_parameter_VJP_calls']==4*n\n"
    "            assert len(H['sampled_block_history'])==n\n"
    "            (P/name).mkdir(exist_ok=True)\n"
    "            np.savez_compressed(P/name/'BLOCK_SAMPLING_PRIVATE.npz',selected_blocks=np.asarray(H['sampled_block_history'],dtype='i8'),backward_parent_counts=H['sampled_backward_counts'])\n"
    "        if name=='mini':\n            receipt.update(cost_match_absolute_s=abs(elapsed-threshold)")
s=once(s,"torch_CPU_RNG=torch.get_rng_state(),torch_CUDA_RNG=torch.cuda.get_rng_state(0),receipt=receipt)",
    "torch_CPU_RNG=torch.get_rng_state(),torch_CUDA_RNG=torch.cuda.get_rng_state(0),receipt=receipt,\n"
    "            block_sampler_state=copy.deepcopy(H['block_rng'].bit_generator.state)if name=='candidate'else None)")
admission='''    def gradient_admission():
        progress('no_update_full_gradient_admission')
        H=make_arm('gradient_reference');before_model=state_hash(H['model'].state_dict());before_adam=digest(H['optimizer'].state_dict())
        squared=[]
        original_vjp=parameter_vjp_cpu_f64
        def counted_vjp(*args,**kwargs):
            g=original_vjp(*args,**kwargs);squared.append(float(torch.dot(g,g)));return g
        H['parameter_vjp_cpu_f64']=counted_vjp;H['step']=501
        t=time.monotonic();exec(reference_code,H);torch.cuda.synchronize()
        assert len(squared)==104
        d2=float(torch.dot(H['D_gradient'],H['D_gradient']))
        c2=float(torch.dot(H['C_gradient'],H['C_gradient']))
        dc2=float(torch.dot(H['combined'],H['combined']))
        sum2=math.fsum(squared)
        variance=conditional_variance_of_sum(104,PC['sampled_block_count'],sum2,d2)
        unchanged=before_model==state_hash(H['model'].state_dict())and before_adam==digest(H['optimizer'].state_dict())
        assert unchanged and math.isfinite(variance)and variance>=0
        ok=variance<=PC['relative_gradient_MSE_max']*d2 and variance<=PC['relative_gradient_MSE_max']*dc2
        out=dict(UTC=utc(),fixed_original_update=500,no_optimizer_update=True,model_and_Adam_unchanged=unchanged,
            source_parents=6593,REAL_parents=21047,original_block_count=104,sampled_block_count=4,HT_scale=26.,
            block_squared_norms=squared,sum_block_squared_norm=sum2,D_squared_norm=d2,C_squared_norm=c2,
            D_plus_C_squared_norm=dc2,conditional_MSE_trace=variance,
            relative_MSE_to_D=variance/d2 if d2>0 else None,
            relative_MSE_to_D_plus_C=variance/dc2 if dc2>0 else None,
            prospective_relative_MSE_limit=PC['relative_gradient_MSE_max'],pass_gate=bool(ok),
            one_frozen_theta_noise_directions_and_C_draw_only=True,clip_and_Adam_unbiased_claim=False,
            model_SHA256=before_model,Adam_SHA256=before_adam,elapsed_s=time.monotonic()-t,
            physical_counts=H['physical'],new_response_updates=0,
            numeric=dict(D_q64=H['dq64'],D_q32=H['dq32'],C_q64=H['cq64'],C_q32=H['cq32']))
        write('GRADIENT_ADMISSION.json',out)
        del H
        progress('gradient_admission_complete',pass_gate=bool(ok),relative_MSE_to_D=out['relative_MSE_to_D'],relative_MSE_to_D_plus_C=out['relative_MSE_to_D_plus_C'])
        return out

    admission=gradient_admission()
    RESULT['gradient_admission']=admission
    if not admission['pass_gate']:
        RESULT.update(status='NEGATIVE_GRADIENT_ADMISSION',new_response_updates=0,new_response_probes=0,
            decision='Close this fixed k4 estimator before training: conditional RMS gradient error exceeds a preregistered reference-gradient magnitude. No k or seed scan, no original-A-F impossibility claim.')
    else:
'''
start=s.index("    progress('loaded_common_start'")
end=s.index("    RESULT.update(UTC=utc()",start)
existing=s[start:end]
s=s[:start]+admission+''.join('    '+line if line.strip() else line for line in existing.splitlines(True))+s[end:]
s=once(s,"original_R65_50000_complete=False,automatic_long_resume=False)",
    "original_R65_50000_complete=False,automatic_long_resume=False,estimator='uniform4_of104_full_objective_parameter_VJP',\n"
    "    original_acceptance_unchanged=True,block_sampler_seed=PC['block_sampler_seed'])")
s=once(s,"progress('training_start',arm=name,full_updates=", "progress('training_start',arm=name,candidate_updates=")
s=once(s,"        if name=='candidate':\n            arm_deadline=time.monotonic()+PC['candidate_arm_cap_s']\n            def phase_check(_):\n                check_budget()\n                if time.monotonic()>=arm_deadline:raise TimeoutError('sampled100 arm exceeded200 seconds')\n            H['phase_write']=phase_check",
    "        arm_deadline=time.monotonic()+PC['candidate_arm_cap_s']\n        def phase_check(_):\n            check_budget()\n            if time.monotonic()>=arm_deadline:raise TimeoutError(name+' arm exceeded200 seconds')\n        H['phase_write']=phase_check")
s=once(s,"            check_budget();H['step']=501+n;exec(code,H);n+=1", "            phase_check('before_update');H['step']=501+n;exec(code,H);n+=1\n            if name=='mini':\n                H['counter']['scientific_neural_forward_rows']+=384\n                H['counter']['scientific_neural_forward_calls']+=1")
s=once(s,"            if name=='candidate':\n                if elapsed>PC['candidate_arm_cap_s']:raise TimeoutError('sampled100 arm exceeded200 seconds')", "            if elapsed>PC['candidate_arm_cap_s']:raise TimeoutError(name+' arm exceeded200 seconds')\n            if name=='candidate':")
s=once(s,"        receipt['counter_delta']={k:int(H['counter'][k]-recovery['counter'][k])for k in H['counter']}","        receipt['counter_delta']={k:int(H['counter'][k]-recovery['counter'][k])for k in H['counter']}\n        receipt['joint_microbatch_reverse_calls']=n if name=='mini'else 0")
put('pilot.py',s)
write(P/'PREPARATION_RECEIPT.json',dict(UTC=now.isoformat(),scientific_execution=False,source_R67_file_hashes={f.name:sha(f)for f in OLD.iterdir()if f.name in ['pilot.py','full_step.py','mini_step.py','linux_supervise.py','linux_owned_worker.py','config.json']},
    prepared_python_AST_parses=True,conditional_candidate_updates=100,worker_cap_s=600,
    resource=dict(guard,**memory())))
print(json.dumps(dict(status='PREPARED_NOT_REGISTERED',stage=S.name,remote_root=cfg['remote_root'],files=len(list(S.iterdir())))))
