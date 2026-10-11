"""Finite artificial checks only; import does not run tests or write files.

run_checks() returns a JSON-serializable dict and never writes a file. The native
guard launcher and the root pilot own their respective receipt persistence.
"""

from fractions import Fraction
from itertools import combinations
import math
import sys


def run_checks():
    import numpy as np

    passed = []

    def require(condition, name):
        if not condition:
            raise AssertionError(name)
        passed.append(name)

    def rejects(call, name):
        try:
            call()
        except (TypeError, ValueError):
            passed.append(name)
        else:
            raise AssertionError(name)

    def legacy_state_equal(first, second):
        return (first[0] == second[0] and np.array_equal(first[1], second[1])
                and first[2:] == second[2:])

    legacy_before_import = np.random.get_state()
    untouched = np.random.Generator(np.random.PCG64(1701))
    untouched_before = untouched.bit_generator.state
    helper_previously_loaded = "sampled_blocks" in sys.modules
    from sampled_blocks import (ROUNDING_ULPS, conditional_variance_of_sum,
                                original_schedule, sample_blocks)
    require(legacy_state_equal(legacy_before_import, np.random.get_state()),
            "helper_import_preserves_legacy_numpy_rng")
    require(untouched_before == untouched.bit_generator.state,
            "helper_import_preserves_unrelated_pcg64_rng")

    # Five deliberately UNEQUAL row blocks, including a one-row final block.
    # All numbers and all sample means/variances below are exact Fractions.
    block_lengths = (2, 3, 4, 5, 1)
    rows = [(Fraction((-1) ** i * (i + 1), i + 2),
             Fraction(2 * i + 1, 3 * (i + 1)))
            for i in range(sum(block_lengths))]

    def vector_sum(vectors):
        return tuple(sum((v[d] for v in vectors), Fraction(0)) for d in range(2))

    def norm_squared(vector):
        return sum((v * v for v in vector), Fraction(0))

    block_gradients = []
    lo = 0
    for length in block_lengths:
        block_gradients.append(vector_sum(rows[lo:lo + length]))
        lo += length
    M, k = 5, 2
    total = vector_sum(block_gradients)
    sum_norm2 = sum((norm_squared(g) for g in block_gradients), Fraction(0))
    total_norm2 = norm_squared(total)
    subsets = list(combinations(range(M), k))
    estimates = []
    included_counts = [0] * M
    for subset in subsets:
        for block in subset:
            included_counts[block] += 1
        subtotal = vector_sum([block_gradients[b] for b in subset])
        estimates.append(tuple(Fraction(M, k) * v for v in subtotal))
    mean = tuple(sum((v[d] for v in estimates), Fraction(0)) / len(estimates)
                 for d in range(2))
    exact_error_variance = sum((norm_squared(tuple(v[d] - total[d] for d in range(2)))
                                for v in estimates), Fraction(0)) / len(estimates)
    exact_formula_variance = (Fraction(M * (M - k), k * (M - 1))
                              * (sum_norm2 - total_norm2 / M))
    require(mean == total, "fraction_exhaustive_ht_mean_equals_full_gradient")
    require(exact_error_variance == exact_formula_variance,
            "fraction_exhaustive_error_variance_equals_ht_formula")
    require(included_counts == [4] * M,
            "unequal_blocks_and_singleton_have_identical_inclusion_probability")
    helper_variance = conditional_variance_of_sum(M, k, sum_norm2, total_norm2)
    require(math.isclose(helper_variance, float(exact_formula_variance), rel_tol=1e-12, abs_tol=0.0),
            "float_helper_matches_exact_fraction_variance")
    require(conditional_variance_of_sum(M, M, sum_norm2, total_norm2) == 0.0,
            "full_sample_variance_is_zero_for_unequal_blocks")
    require(conditional_variance_of_sum(1, 1, 5.0, 5.0) == 0.0,
            "singleton_population_variance_is_zero")
    require(conditional_variance_of_sum(5, 2, 5.0, 25.0) == 0.0,
            "identical_block_gradients_have_zero_sampling_variance")

    schedule = original_schedule(6593, 64)
    require(len(schedule) == 104 and schedule[-1] == (6592, 6593),
            "original_schedule_has_104_blocks_and_final_one_parent")
    require(schedule[0] == (0, 64) and all(hi - start == 64 for start, hi in schedule[:-1]),
            "original_first_103_blocks_have_64_parents")
    require(all(schedule[i][1] == schedule[i + 1][0] for i in range(len(schedule) - 1))
            and sum(hi - start for start, hi in schedule) == 6593,
            "original_schedule_covers_every_parent_once_without_gaps")
    require(original_schedule(1, 64) == [(0, 1)], "schedule_retains_small_population")

    seed = 202610116901
    sampler_rng = np.random.Generator(np.random.PCG64(seed))
    direct_rng = np.random.Generator(np.random.PCG64(seed))
    legacy_before_sample = np.random.get_state()
    indices, scale = sample_blocks(sampler_rng, 104, 4)
    direct_indices = sorted(int(i) for i in direct_rng.choice(104, 4, replace=False))
    require(indices == direct_indices and sampler_rng.bit_generator.state == direct_rng.bit_generator.state,
            "sampler_is_exactly_one_private_pcg64_choice_without_replacement")
    require(len(indices) == len(set(indices)) == 4 and indices == sorted(indices)
            and 0 <= indices[0] <= indices[-1] < 104 and scale == 26.0,
            "sample_indices_are_original_sorted_unique_indices_with_scale_26")
    repeat_rng = np.random.Generator(np.random.PCG64(seed))
    require(sample_blocks(repeat_rng, 104, 4) == (indices, scale),
            "private_seed_reproduces_sample")
    require(legacy_state_equal(legacy_before_sample, np.random.get_state())
            and untouched_before == untouched.bit_generator.state,
            "sampling_advances_only_supplied_rng")
    all_rng = np.random.Generator(np.random.PCG64(710))
    require(sample_blocks(all_rng, 104, 104) == (list(range(104)), 1.0),
            "full_block_sample_retains_all_original_indices")

    rejects(lambda: original_schedule(0, 64), "reject_zero_population")
    rejects(lambda: original_schedule(6593, 0), "reject_zero_chunk")
    rejects(lambda: original_schedule(True, 64), "reject_bool_population")
    rejects(lambda: original_schedule(6593.0, 64), "reject_noninteger_population")
    rejects(lambda: sample_blocks(sampler_rng, 104, 0), "reject_zero_sample_size")
    rejects(lambda: sample_blocks(sampler_rng, 104, 105), "reject_sample_larger_than_population")
    rejects(lambda: sample_blocks(sampler_rng, 104, True), "reject_bool_sample_size")
    rejects(lambda: sample_blocks(np.random.Generator(np.random.MT19937(2)), 5, 2),
            "reject_non_pcg64_generator")
    rejects(lambda: conditional_variance_of_sum(5, 0, 1.0, 1.0), "variance_rejects_zero_sample")
    rejects(lambda: conditional_variance_of_sum(5, 2, -1.0, 1.0), "variance_rejects_negative_norm")
    rejects(lambda: conditional_variance_of_sum(5, 2, float('nan'), 1.0), "variance_rejects_nan_norm")
    rejects(lambda: conditional_variance_of_sum(5, 2, 1.0, float('inf')), "variance_rejects_infinite_norm")
    rejects(lambda: conditional_variance_of_sum(2, 1, 1.0, 2.000001),
            "variance_rejects_materially_negative_centered_sum")
    rejects(lambda: conditional_variance_of_sum(2, 1, 1e-250, 2e-250 * (1.0 + 1e-6)),
            "variance_rounding_tolerance_has_no_absolute_floor")
    rejects(lambda: conditional_variance_of_sum(5, 5, 0.0, 1.0),
            "full_sample_still_rejects_inconsistent_norms")
    rejects(lambda: conditional_variance_of_sum(104, 4, sys.float_info.max, 0.0),
            "variance_rejects_overflow")
    require(conditional_variance_of_sum(2, 1, 1.0, math.nextafter(2.0, math.inf)) == 0.0,
            "only_scale_relative_negative_roundoff_is_clamped")

    return {
        "status": "PASS_ARTIFICIAL_CPU_ONLY",
        "numpy_version": np.__version__,
        "helper_previously_loaded": helper_previously_loaded,
        "checks_passed": len(passed),
        "check_names": passed,
        "exact_fraction_fixture": {
            "M": M, "k": k, "block_lengths": list(block_lengths),
            "subsets_enumerated": len(subsets), "inclusion_counts": included_counts,
            "full_gradient": [str(v) for v in total],
            "mean_ht_gradient": [str(v) for v in mean],
            "sum_block_squared_norm": str(sum_norm2),
            "total_gradient_squared_norm": str(total_norm2),
            "mean_squared_estimation_error": str(exact_error_variance),
            "formula_variance": str(exact_formula_variance),
            "helper_variance_float": helper_variance,
        },
        "original_schedule": {"n": 6593, "chunk": 64, "M": len(schedule),
                              "last_block": list(schedule[-1]), "last_block_length": 1},
        "pcg64_fixture": {"seed": seed, "indices": indices, "scale": scale},
        "roundoff_policy": {"relative_float_epsilon_multiplier": ROUNDING_ULPS,
                            "absolute_floor": 0.0, "material_negative": "raise ValueError"},
        "scope": {"scientific_arrays_opened": 0, "model_runs": 0, "training_updates": 0,
                  "cuda_execution": 0, "network_calls": 0, "file_writes_by_run_checks": 0},
        "evidence_boundary": "Finite artificial expectation/variance and API checks; not scientific or CUDA validation.",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_checks(), ensure_ascii=False, indent=2, allow_nan=False))
