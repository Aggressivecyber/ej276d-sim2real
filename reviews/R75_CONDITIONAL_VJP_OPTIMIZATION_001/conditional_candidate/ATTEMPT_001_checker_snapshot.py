"""Artificial R74-vs-candidate endpoint, full NN VJP, Adam and boundary checks."""
from __future__ import annotations

import copy
import torch

from artificial_checks import fixture
from combined_loss import FullWaveformLoss as R74FullWaveformLoss
from conditional_vjp_loss import FullWaveformLoss
from exact_uniform_w1 import UniformW1Cache
from physical_operators import canonical_REAL_torch
from waveform_response import forward_prepared, slice_prepared


def _implementation(original, cls):
    result = copy.copy(original)
    result.__class__ = cls
    result.w1 = UniformW1Cache(original.device)
    return result


def _bits_equal(a, b):
    return a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _comparison(a, b):
    error = a - b
    return {"maxabs": float(error.abs().max()),
            "relative_L2": float(torch.linalg.vector_norm(error) /
                                 torch.linalg.vector_norm(a).clamp_min(1e-30)),
            "byte_equal": _bits_equal(a, b)}


def run_checks(device="cpu", seed=750001) -> dict:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    rng_devices = [] if device.type == "cpu" else [device.index]
    with torch.random.fork_rng(devices=rng_devices):
        return _run(device, seed)


def _run(device, seed):
    model, prepared, epsilon, nS, original, directions, temporal = fixture(device, seed)
    n, chunk = len(epsilon), 13
    params = list(model.parameters())
    initial = copy.deepcopy(model.state_dict())

    def forward():
        found = [[], [], []]
        for start in range(0, n, chunk):
            pre, adc, raw, _ = forward_prepared(model, slice_prepared(prepared, start, min(start+chunk, n)),
                                                epsilon[start:start+chunk])
            for part, value in zip(found, (adc, pre, raw)):
                part.append(value)
        return [torch.cat(part) for part in found]

    with torch.no_grad():
        saved = forward()
    base = [v[sl].detach() for v in saved for sl in (slice(0, nS), slice(nS, None))]
    reports = {}

    def compare_endpoint(label, endpoint, parameter_check=False):
        results = []
        for cls in (R74FullWaveformLoss, FullWaveformLoss):
            objective = _implementation(original, cls)
            leaves = [v.detach().clone().requires_grad_(True) for v in endpoint]
            acc = [torch.zeros_like(v, dtype=torch.float64) for v in leaves]
            values, value_tensors, group_grads = {}, {}, []
            first_diagnostics = None
            groups = 0
            for names, total, terms, diagnostics in objective.iter_gradient_groups(*leaves, directions, temporal):
                if groups == 0:
                    first_diagnostics = diagnostics
                    assert names == ("full_window", "coordinate_RMS")
                    assert total.requires_grad
                    if cls is FullWaveformLoss:
                        assert all(not v.requires_grad for v in terms.values())
                gradient = torch.autograd.grad(total, leaves, allow_unused=True)
                dense = []
                for target, local in zip(acc, gradient):
                    tensor = torch.zeros_like(target) if local is None else local.detach().to(torch.float64)
                    dense.append(tensor.clone())
                    target.add_(tensor)
                group_grads.append(dense)
                values.update({k: float(v.detach()) for k, v in terms.items()})
                value_tensors.update({k: v.detach().clone() for k, v in terms.items()})
                groups += 1
            assert groups == 8 and len(values) == 9
            parameter_gradients = None
            if parameter_check:
                all_gradient = [torch.cat(acc[2*j:2*j+2]) for j in range(3)]
                out = forward()
                assert all(_bits_equal(actual.detach(), expected) for actual, expected in zip(out, saved))
                parameter_gradients = torch.autograd.grad(out, params,
                    grad_outputs=[g.to(v.dtype) for g, v in zip(all_gradient, out)])
                parameter_gradients = [v.detach().clone() for v in parameter_gradients]
            results.append((values, value_tensors, acc, group_grads, parameter_gradients, first_diagnostics))
        old, new = results
        assert old[5] == new[5]
        value_errors = {k: abs(old[0][k]-new[0][k]) for k in old[0]}
        assert all(e <= 1e-11*max(1., abs(old[0][k])) for k, e in value_errors.items()), value_errors
        term_bits = {k: _bits_equal(old[1][k], new[1][k]) for k in old[1]}
        six = []
        for before, after in zip(old[2], new[2]):
            torch.testing.assert_close(before, after, rtol=1e-10, atol=1e-10)
            six.append(_comparison(before, after))
        first_group = []
        for before, after in zip(old[3][0], new[3][0]):
            torch.testing.assert_close(before, after, rtol=1e-10, atol=1e-10)
            first_group.append(_comparison(before, after))
        # Seven unmodified terms also retain their separate per-group gradients.
        unchanged = all(_bits_equal(g, h) for a, b in zip(old[3][1:], new[3][1:]) for g, h in zip(a, b))
        assert unchanged
        record = {"nine_term_value_errors": value_errors, "term_value_byte_equal": term_bits,
                  "six_leaf_gradient_comparisons": six, "first_pair_six_leaf_comparisons": first_group,
                  "other_seven_group_gradient_byte_equal": unchanged, "diagnostics_equal": True}
        if parameter_check:
            flat = lambda data: torch.cat([v.detach().reshape(-1).to(torch.float64) for v in data])
            pg = _comparison(flat(old[4]), flat(new[4]))
            assert pg["relative_L2"] <= 1e-6, pg
            after = []
            for entry in results:
                model.load_state_dict(initial)
                opt = torch.optim.Adam(params, lr=2e-4, betas=(.9, .999))
                for p, g in zip(params, entry[4]):
                    p.grad = g.clone()
                torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
                opt.step()
                after.append(flat(params).clone())
            adam = _comparison(after[0], after[1])
            assert adam["maxabs"] <= 1e-6, adam
            model.load_state_dict(initial)
            record.update(parameter_gradient=pg, Adam_parameter=adam,
                          complete_original_chunk_NN_VJP=True, nondivisible_tail=True)
        reports[label] = record

    compare_endpoint("original_50row_fixture", base, parameter_check=True)
    duplicated = [v.clone() for v in base]
    for i in range(2):
        bins = original.bins(duplicated[i])
        for slot in range(-1, 9):
            use = torch.nonzero(bins == slot).flatten()
            if len(use) > 1:
                for j in (i, i+2, i+4):
                    duplicated[j][use] = duplicated[j][use[0]].clone()
    compare_endpoint("stable_duplicate_and_feature_tie_endpoints", duplicated)
    zero_real = [v.clone() for v in base]
    zero_real[1] = original.reference.clone()
    compare_endpoint("REAL_zero_marginal_coordinate_norm", zero_real)

    migrated = [v.clone() for v in base]
    migration_counts = []
    for i in range(2):
        old_bins = original.bins(migrated[i])
        low = torch.nonzero(old_bins == -1).flatten()[0]
        row = low + (0 if i == 0 else nS)
        q = migrated[i][low, 100:400].sum() - 3*migrated[i][low, :100].sum()
        target = torch.exp((original.edges[0]+original.edges[1])*.5)/1000
        moved = migrated[i][low] + (target-q)*prepared["mu_condition"][row]
        migrated[i][low] = canonical_REAL_torch(moved)
        new_bins = original.bins(migrated[i])
        changed = int((old_bins != new_bins).sum())
        assert changed == 1 and int(new_bins[low]) == 0
        migration_counts.append(changed)
    compare_endpoint("current_output_bin_migration", migrated)

    arbitrary = []
    for upstream in (-3.25, 0., .125, 1e3):
        observed = []
        for cls in (R74FullWaveformLoss, FullWaveformLoss):
            leaves = [v.detach().clone().requires_grad_(True) for v in base]
            obj = _implementation(original, cls)
            _, total, _, _ = next(obj.iter_gradient_groups(*leaves, directions, temporal))
            observed.append(torch.autograd.grad(total, leaves[:2], grad_outputs=total.new_tensor(upstream)))
        metrics = []
        for before, after in zip(*observed):
            torch.testing.assert_close(before, after, atol=1e-10*max(1,abs(upstream)), rtol=1e-10)
            if upstream == 0:
                assert bool((after == 0).all())
            metrics.append(_comparison(before, after))
        arbitrary.append({"upstream": upstream, "canonical_gradients": metrics})

    requirement_flags = []
    for need_source, need_real in ((True, False), (False, True), (False, False)):
        values = []
        for cls in (R74FullWaveformLoss, FullWaveformLoss):
            leaves = [v.detach().clone() for v in base]
            leaves[0].requires_grad_(need_source)
            leaves[1].requires_grad_(need_real)
            _, total, _, _ = next(_implementation(original, cls).iter_gradient_groups(*leaves, directions, temporal))
            if need_source or need_real:
                use = [v for v in leaves[:2] if v.requires_grad]
                values.append(torch.autograd.grad(total, use))
            else:
                assert not total.requires_grad
                values.append((total.detach(),))
        assert all(torch.allclose(a,b,atol=1e-10,rtol=1e-10) for a,b in zip(*values))
        requirement_flags.append([need_source, need_real])

    failures = {}
    empty = [v.clone() for v in base]
    bins = original.bins(empty[0])
    empty[0][bins == 0] = empty[0][torch.nonzero(bins == 1).flatten()[0]].clone()
    nonfinite = [v.clone() for v in base]
    nonfinite[0][0, 0] = float("nan")
    for label, endpoints, missing_reference in (("empty_required_current_bin", empty, False),
                                                ("nonfinite_endpoint", nonfinite, False),
                                                ("occupied_bin_empty_reference", base, True)):
        errors = []
        for cls in (R74FullWaveformLoss, FullWaveformLoss):
            obj = _implementation(original, cls)
            if missing_reference:
                obj._reference_counts = list(obj._reference_counts)
                obj._reference_counts[0] = 0
            try:
                next(obj.iter_gradient_groups(*endpoints, directions, temporal))
            except (ValueError, FloatingPointError) as error:
                errors.append({"type": type(error).__name__, "message": str(error)})
        assert len(errors) == 2 and errors[0] == errors[1], errors
        failures[label] = errors[0]

    return {"pass_all": True, "device": str(device), "seed": seed,
            "evidence_level": "artificial_first_order_R74_equivalence_only", "SOURCE_N": nS, "REAL_N": n-nS,
            "rows": n, "NN_chunk": chunk, "cases": reports, "migration_counts": migration_counts,
            "arbitrary_upstream_comparisons": arbitrary, "requires_grad_flags_checked": requirement_flags,
            "preserved_failures": failures, "scientific_arrays_weights_read": False,
            "science_training_probe_run": False, "real_speedup_claimed": False}
