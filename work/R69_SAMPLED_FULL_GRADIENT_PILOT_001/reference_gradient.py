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
optimizer.zero_grad(set_to_none=True)
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
for lo, hi in schedule:
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
