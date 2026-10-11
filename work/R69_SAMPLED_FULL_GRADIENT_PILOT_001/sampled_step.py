current_update = step
inflight = 'draw_original_C_and_fresh_full_D_noise_and_directions'
phase_write('draws')
si = np.concatenate([g[streams['source_rows'].integers(0, len(g), size=C['batch_per_region'])] for g in groupsS]).astype('i8')
ri = np.concatenate([g[streams['REAL_reference_rows'].integers(0, len(g), size=C['batch_per_region'])] for g in groupsR]).astype('i8')
ii = streams['REAL_identity_rows'].integers(0, len(real), size=C['identity_batch']).astype('i8')
assert np.array_equal(sraw[si], np.repeat(CATEGORIES, 32)) and np.array_equal(rraw[ri], np.repeat(CATEGORIES, 32))
epsilonS = streams['source_epsilon'].standard_normal((len(si), 480)).astype('<f4')
epsilonI = streams['identity_epsilon'].standard_normal((len(ii), 480)).astype('<f4')
physical['C_SOURCE_noise_rows'] += len(si)
physical['C_identity_noise_rows'] += len(ii)
nS, _ = innovation(source[si], epsilonS, mu, edges, noise['p480'], noise['kappa2'], float(noise['tau2']), True)
nI, _ = innovation(real[ii], epsilonI, mu, edges, noise['p480'], noise['kappa2'], float(noise['tau2']), True)
directions = objective.draw_directions(streams['projection_directions'])
temporal_directions = objective.draw_temporal_directions(streams['baseline_projection_directions'])
physical['full_window_direction_draw_calls'] += 1
physical['temporal_direction_draw_calls'] += 1
epsilonD = streams['full_SOURCE_epsilon'].standard_normal((6593, 480)).astype('<f4')
full_SOURCE_epsilon_calls += 1
physical['D_source_noise_rows'] += 6593
nD, _ = innovation(source, epsilonD, mu, edges, noise['p480'], noise['kappa2'], float(noise['tau2']), True)
nd = torch.as_tensor(nD, device=dev)
rate = learning_rate(step)
for group in optimizer.param_groups:
    group['lr'] = rate
optimizer.zero_grad(set_to_none=True)
sampled_indices, ht_scale = sample_blocks(block_rng, len(schedule), PC['sampled_block_count'])
sampled_schedule = [schedule[int(i)] for i in sampled_indices]
sampled_parent_rows = sum(hi-lo for lo,hi in sampled_schedule)
sampled_block_history.append([int(i) for i in sampled_indices])
versions = tuple((p._version for p in params))
before_physical = dict(physical)
projected, raw_collected = ([], [])
phase_write('full_D_firstpass')
with torch.no_grad():
    for lo, hi in schedule:
        inflight = {'firstpass_chunk': [lo, hi]}
        output, raw = endpoint(model, xs[lo:hi], nd[lo:hi], sqt[lo:hi], sbt[lo:hi], sht[lo:hi], mt, st)
        physical['neural_forward_calls'] += 1
        physical['neural_forward_rows'] += hi - lo
        projected.append(output)
        raw_collected.append(raw)
    cache = torch.cat(projected)
    full_raw = torch.cat(raw_collected)
    dq64, dq32 = charge_range(cache, sqt)
    D_projection_fraction = float(((full_raw < -0.5) | (full_raw > 0.5)).any(1).double().mean())
    D_projection_RMS = float((cache - full_raw).square().mean().sqrt())
del projected, raw_collected, full_raw, output, raw
inflight = 'one_joint_D_objective_and_one_endpoint_reverse'
phase_write('full_D_objective_and_GY')
full_objective = FrozenDistributionObjective(statistics, C, directions, temporal_directions, sraw, rraw, feature_chunk=64, device=dev, uniform_fastpath=True)
distribution = joint_total_D(full_objective, cache, xr, layout)
physical['full_D_objective_calls'] += 1
physical['full_D_endpoint_reverse_calls'] += 1
physical['full_D_SOURCE_law_rows'] += 6593
physical['full_D_REAL_law_rows'] += 21047
gy = distribution['GY']
assert gy.dtype == torch.float64 and gy.shape == cache.shape and torch.isfinite(gy).all()
assert torch.isfinite(distribution['total']) and all((torch.isfinite(v) for v in distribution['pieces'].values()))
D_gradient = torch.zeros(parameter_count, dtype=torch.float64, device='cpu')
matched_y = matched_codes = matched_free = 0
phase_write('full_D_parameter_replay')
for lo, hi in sampled_schedule:
    inflight = {'replay_forward_and_VJP_chunk': [lo, hi]}
    output, raw = endpoint(model, xs[lo:hi], nd[lo:hi], sqt[lo:hi], sbt[lo:hi], sht[lo:hi], mt, st)
    physical['neural_forward_calls'] += 1
    physical['neural_forward_rows'] += hi - lo
    expected = cache[lo:hi]
    same_y = torch.equal(output.detach(), expected)
    same_codes = torch.equal(canonical_adc_torch(output.detach(), False), canonical_adc_torch(expected, False))
    same_free = torch.equal((output.detach() > -0.5) & (output.detach() < 0.5), (expected > -0.5) & (expected < 0.5))
    if not (same_y and same_codes and same_free):
        write('REPLAY_MISMATCH.json', {'step': step, 'chunk': [lo, hi], 'endpoint_bitwise': same_y, 'canonical_codes': same_codes, 'Box_free_mask': same_free})
        raise FloatingPointError('Original-schedule endpoint replay mismatch')
    matched_y += 1
    matched_codes += 1
    matched_free += 1
    physical['endpoint_replay_checked_chunks'] += 1
    D_gradient += parameter_vjp_cpu_f64(output, params, gy[lo:hi])
    physical['D_parameter_VJP_calls'] += 1
    del output, raw
assert tuple((p._version for p in params)) == versions
assert all((torch.equal(value, frozen_buffers[name]) for name, value in model.named_buffers()))
D_gradient *= ht_scale
assert torch.isfinite(D_gradient).all()
D_gradient_norm = float(torch.linalg.vector_norm(D_gradient))
del full_objective, gy, cache, nd, nD
inflight = 'original384_row_C_forward_original_nine_objective_four_C_reverse'
phase_write('original_C_and_combined_update')
bx = torch.as_tensor(np.r_[sb[si], rb[ii]], device=dev, dtype=torch.int64)
q = torch.as_tensor(np.r_[sq[si], rq[ii]], device=dev)
hc = torch.as_tensor(np.r_[sh[si], rh[ii]], device=dev)
x = torch.cat((xs[si], xr[ii]))
n = torch.as_tensor(np.concatenate((nS, nI)), device=dev)
output, raw = endpoint(model, x, n, q, bx, hc, mt, st)
physical['neural_forward_calls'] += 1
physical['neural_forward_rows'] += len(x)
constraints = original_C_microbatch(objective, output, raw, xr[ri], xr[ii], torch.as_tensor(rraw[ii], device=dev), torch.as_tensor(sp[si], device=dev), torch.as_tensor(source_h[si], device=dev), directions, torch.as_tensor(interval_low[si], device=dev), torch.as_tensor(interval_high[si], device=dev), temporal_directions)
physical['C_objective_calls'] += 1
physical['C_SOURCE_objective_rows'] += len(si)
physical['C_REAL_reference_objective_rows'] += len(ri)
physical['C_REAL_identity_objective_rows'] += len(ii)
assert torch.isfinite(constraints['total'])
cq64, cq32 = charge_range(output.detach(), q)
C_gradient = scalar_parameter_gradient_cpu_f64(constraints['total'], params)
physical['C_parameter_reverse_calls'] += 1
assert torch.isfinite(C_gradient).all()
C_gradient_norm = float(torch.linalg.vector_norm(C_gradient))
assert tuple((p._version for p in params)) == versions
assert all((torch.equal(value, frozen_buffers[name]) for name, value in model.named_buffers()))
combined = D_gradient + C_gradient
assert torch.isfinite(combined).all()
inflight = 'single_final_cast_clip_Adam'
update_meta = install_cpu_f64_clip_step(params, combined, optimizer, clip=C['gradient_clip'], capture=False)
physical['parameter_cast_calls'] += 1
physical['gradient_clip_calls'] += 1
physical['optimizer_step_calls'] += 1
assert update_meta['cast_calls'] == update_meta['clip_calls'] == update_meta['optimizer_step_calls'] == 1
assert all((torch.isfinite(p).all() for p in params))
norm = float(update_meta['pre_clip_norm'])
pieces = dict(distribution['pieces'], **{key: value.detach() for key, value in constraints['pieces'].items()})
total = float(distribution['total']) + float(constraints['total'].detach())
diagnostic = constraints['diagnostics']
with torch.no_grad():
    active = ((raw < -0.5) | (raw > 0.5)).any(1)
    correction = (output - raw).square().mean(1)
    C_S_projection_fraction = float((active[:320].double() * objective.sw).sum())
    C_R_projection_fraction = float(active[320:].double().mean())
    C_S_projection_RMS = float((correction[:320] * objective.sw).sum().sqrt())
    C_R_projection_RMS = float(correction[320:].mean().sqrt())
row_values = dict(pieces)
row_values.update(total_loss=total, learning_rate=rate, gradient_norm=norm, max_relative_q_f64=max(dq64, cq64), max_relative_q_f32=max(dq32, cq32), SOURCE_global_range_projection_fraction=C_S_projection_fraction, REAL_range_projection_fraction=C_R_projection_fraction, SOURCE_global_projection_RMS_V=C_S_projection_RMS, REAL_projection_RMS_V=C_R_projection_RMS, overall_PSD_W1=float(distribution['diagnostics']['overall_PSD_W1']), Q1_Q7_mean_PSD_W1=float(distribution['diagnostics']['Q1_Q7_mean_PSD_W1']), Q3_h_one_soft_valley=float(diagnostic['Q3_h_one_soft_valley']), Q4_h_one_soft_valley=float(diagnostic['Q4_h_one_soft_valley']), D_total=float(distribution['total']), C_total=float(constraints['total'].detach()), full_D_SOURCE_range_projection_fraction=D_projection_fraction, full_D_SOURCE_projection_RMS_V=D_projection_RMS, full_D_max_relative_q_f64=dq64, full_D_max_relative_q_f32=dq32, C_max_relative_q_f64=cq64, C_max_relative_q_f32=cq32, D_parameter_gradient_L2=D_gradient_norm, C_parameter_gradient_L2=C_gradient_norm)
row = np.array([float(row_values[key]) for key in columns], dtype='f8')
order_row = np.array([float(diagnostic[key]) for key in order_columns], dtype='f8')
discarded_values = {key: float(v) for key, v in constraints['discarded_D_values'].items()}
discarded_row = np.array([discarded_values[key] for key in D_PIECE_ORDER] + [float(constraints['original_nine_total_value']), sum(discarded_values.values())], dtype='f8')
assert np.isfinite(row).all() and np.isfinite(order_row).all() and np.isfinite(discarded_row).all()
assert np.all(order_row >= 0)
assert abs(order_row[0] - order_row[1] - float(pieces['source_order'])) <= 1e-10 + 1e-10 * abs(order_row[0])
assert abs(sum((float(pieces[key]) for key in C['full_D_scalar_order'])) - float(distribution['total'])) <= 1e-10 + 1e-10 * abs(total)
assert abs(sum((float(pieces[key]) for key in C['C_scalar_order'])) - float(constraints['total'].detach())) <= 1e-10 + 1e-10 * abs(total)
replay_row = np.array([6593, sampled_parent_rows, len(sampled_schedule), matched_y, matched_codes, matched_free, 1, 1, len(sampled_schedule), 1, 1, 1], dtype='i8')
assert matched_y == matched_codes == matched_free == PC['sampled_block_count']
for lo,hi in sampled_schedule: sampled_backward_counts[lo:hi] += 1
assert physical['neural_forward_rows'] - before_physical['neural_forward_rows'] == 6593 + sampled_parent_rows + 384
assert physical['neural_forward_calls'] - before_physical['neural_forward_calls'] == len(schedule) + len(sampled_schedule) + 1
assert physical['D_parameter_VJP_calls'] - before_physical['D_parameter_VJP_calls'] == len(sampled_schedule)
np.add.at(countsS, si, 1)
np.add.at(countsR, ri, 1)
np.add.at(countsI, ii, 1)
countsDS += 1
countsDR += 1
for key, values in (('source_rows', si.astype('<u2')), ('REAL_reference_rows', ri.astype('<u2')), ('REAL_identity_rows', ii.astype('<u2')), ('source_epsilon', epsilonS), ('identity_epsilon', epsilonI), ('full_SOURCE_epsilon', epsilonD)):
    hashes[key] = chain_update(hashes[key], values)
for direction in directions:
    hashes['projection_directions'] = chain_update(hashes['projection_directions'], direction)
hashes['baseline_projection_directions'] = chain_update(hashes['baseline_projection_directions'], temporal_directions)
actual_update_counts = dict(new_response_updates=1,SOURCE_training_rows=320,REAL_reference_rows=320,REAL_identity_rows=64,
    full_D_SOURCE_law_rows=6593,full_D_REAL_law_rows=21047,
    scientific_neural_forward_rows=6593+sampled_parent_rows+384,
    scientific_neural_forward_calls=len(schedule)+len(sampled_schedule)+1,
    D_parameter_VJP_calls=len(sampled_schedule),C_parameter_reverse_calls=1,
    full_D_endpoint_reverse_calls=1,full_D_objective_calls=1,replay_chunks_checked=len(sampled_schedule))
assert set(actual_update_counts)==set(C['expected_completed_update_counts'])
for key,amount in actual_update_counts.items():counter[key]+=amount
gradient_execution_counts['combined_gradient_cast_events'] += update_meta['cast_calls']
gradient_execution_counts['combined_clip_calls'] += update_meta['clip_calls']
gradient_execution_counts['optimizer_step_calls'] += update_meta['optimizer_step_calls']
dense.append(row)
order_dense.append(order_row)
replay_dense.append(replay_row)
discarded_dense.append(discarded_row)
updates = step
global_max_D_q64, global_max_D_q32 = (max(global_max_D_q64, dq64), max(global_max_D_q32, dq32))
global_max_C_q64, global_max_C_q32 = (max(global_max_C_q64, cq64), max(global_max_C_q32, cq32))
global_max_q64, global_max_q32 = (max(global_max_q64, dq64, cq64), max(global_max_q32, dq32, cq32))
inflight = None
if torch.cuda.max_memory_allocated() > C['GPU_limit_bytes'] or torch.cuda.max_memory_reserved() > C['GPU_limit_bytes']:
    raise MemoryError('Registered GPU allocated/reserved safety limit')
