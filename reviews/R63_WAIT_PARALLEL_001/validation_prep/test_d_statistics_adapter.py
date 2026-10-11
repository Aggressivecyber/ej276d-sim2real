"""Artificial-array selftests; never reads scientific arrays or model artifacts.

Run with Python -B. The only non-source file read is this directory's immutable
static report.json, for its byte hash. Outputs stay beside this file on E or D.
NumPy is the only non-standard-library dependency.
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import time
import unittest

import numpy as np

from d_statistics_adapter import (ARMS, BOOTSTRAP_REPEATS, BOOTSTRAP_SEED, ESTIMATORS,
                                  _group_weighted_means, evaluate_d_statistics)


# Exact immutable report hash supplied by the completed static-preparation task.
EXPECTED_REPORT_SHA256 = "e1eafdc353f90b4c56eb0ec2d4cbd125cd26707035d92a446130279e02640c7b"


def fixture(candidate=(0.0, 8.0), real=(1.0, 9.0), sim=(2.0, 10.0), full=(1.0, 9.0),
            candidate_logq=None, real_logq=None, groups=None):
    """Manufacture predictions from literal errors around zero artificial truth."""
    count = len(candidate)
    groups = list(range(count)) if groups is None else list(groups)
    rows = [f"artificial-event-{index}" for index in range(count)]

    def packet(psd, logq):
        return {"row_ids": list(rows), "group_ids": list(groups),
                "psd": np.asarray(psd, dtype=np.float64), "logq": np.asarray(logq, dtype=np.float64)}

    truth = packet(np.zeros(count), np.zeros(count))
    cq = np.ones(count) if candidate_logq is None else candidate_logq
    rq = np.ones(count) if real_logq is None else real_logq
    return truth, {name: {"candidate": packet(candidate, cq), "real_only_1024": packet(real, rq),
                         "original_sim_1024": packet(sim, np.ones(count)),
                         "full_real": packet(full, np.ones(count))} for name in ESTIMATORS}


def evaluate(truth, estimates):
    return evaluate_d_statistics(truth=truth, estimates=estimates, frozen_input_design_valid=True)


def checks(report, estimator="Ridge"):
    return report["estimators"][estimator]["checks"]


class ArtificialDStatisticsTests(unittest.TestCase):
    def assertUndefined(self, report):
        self.assertEqual(report["D6"]["status"], "undefined")
        self.assertFalse(report["D6"]["pass"])
        json.dumps(report, allow_nan=False)

    def test_hand_whole_group_repetition_uses_event_weights(self):
        # Two events in group 0 with total error 4, one event in group 1 with error 10.
        # Drawing [0, 0, 1] yields total 18 across five repeated events: 18/5, not 14/3.
        actual = _group_weighted_means(np.array([4.0, 10.0]), np.array([2, 1]), np.array([2, 1]))
        self.assertEqual(float(actual), 18.0 / 5.0)
        self.assertNotEqual(float(actual), (2.0 + 2.0 + 10.0) / 3.0)
        rows = _group_weighted_means([[4.0, 10.0], [2.0, 8.0]], [2, 1], [2, 1])
        np.testing.assert_array_equal(rows, [18.0 / 5.0, 12.0 / 5.0])

    def test_all_1000_shared_draws_match_independent_expanded_events(self):
        truth, estimates = fixture(candidate=[0.5, 1.5, 8.0], real=[1.0, 3.0, 10.0],
                                   sim=[2.0, 4.0, 12.0], full=[1.0, 3.0, 10.0], groups=[2, 2, 10])
        truth["psd"] = np.array([0.1, -0.2, 0.3])
        for estimator in ESTIMATORS:
            for arm_index, arm in enumerate(ARMS):
                estimates[estimator][arm]["psd"] += truth["psd"]
                if estimator == "HGB":
                    offsets = (arm_index + 1) * np.array([-0.2, 0.4, -0.6])
                    estimates[estimator][arm]["psd"] += offsets
        result = evaluate(truth, estimates)
        self.assertEqual(result["bootstrap"]["group_order"], [2, 10])
        self.assertEqual(result["bootstrap"]["group_sizes"], [2, 1])
        unique = np.unique(truth["group_ids"])
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        reference = {(name, arm): [] for name in ESTIMATORS for arm in ARMS}
        draw_hash = hashlib.sha256()
        for _ in range(BOOTSTRAP_REPEATS):
            draw = rng.integers(0, len(unique), size=len(unique), dtype=np.int64)
            draw_hash.update(np.asarray(draw, dtype="<i8").tobytes())
            # Expand repeated group memberships to actual repeated event rows.
            indices = np.concatenate([np.flatnonzero(np.asarray(truth["group_ids"]) == unique[g])
                                      for g in draw])
            for estimator in ESTIMATORS:
                for arm in ARMS:
                    errors = np.abs(estimates[estimator][arm]["psd"][indices] - truth["psd"][indices])
                    reference[(estimator, arm)].append(float(np.mean(errors)))
        self.assertEqual(result["bootstrap"]["draws_sha256"], draw_hash.hexdigest())
        for estimator in ESTIMATORS:
            for arm in ARMS:
                np.testing.assert_allclose(result["estimators"][estimator]["bootstrap_psd_mae"][arm],
                                           reference[(estimator, arm)], rtol=1e-14, atol=1e-15)
            cr, rr, sr = (np.asarray(reference[(estimator, arm)]) for arm in ARMS[:3])
            np.testing.assert_allclose(checks(result, estimator)["D2"]["replicate_values"],
                                       (cr - rr) / rr, rtol=1e-14, atol=1e-15)
            np.testing.assert_allclose(checks(result, estimator)["D3"]["replicate_values"],
                                       cr - sr, rtol=1e-14, atol=1e-15)

    def test_ratio_ci_is_per_replicate_not_absolute_ci_divided_by_point_mae(self):
        truth, estimates = fixture()
        result = evaluate(truth, estimates)
        d = checks(result)
        # Candidate minus REAL is -1 for every event and every resample.
        # The baseline MAE varies from 1 to 9 across resamples.
        np.testing.assert_allclose(d["D2"]["confidence_interval"], [-1.0, -1.0 / 9.0])
        wrong_interval = np.array([-1.0, -1.0]) / 5.0
        self.assertFalse(np.allclose(d["D2"]["confidence_interval"], wrong_interval))
        self.assertAlmostEqual(d["D2"]["point_difference"], -0.2)
        self.assertEqual(d["D2"]["valid_replicates"], 1000)
        self.assertTrue(result["D6"]["pass"])
        for name in ESTIMATORS:
            self.assertTrue(all(item["pass"] for item in checks(result, name).values()))

    def test_equal_predictions_have_zero_paired_differences_and_fail_strict_ci(self):
        truth, estimates = fixture(candidate=[1, 9], sim=[1, 9])
        result = evaluate(truth, estimates)
        d = checks(result)
        for name in ("D2", "D3"):
            self.assertEqual(d[name]["confidence_interval"], [0.0, 0.0])
            self.assertEqual(d[name]["replicate_values"], [0.0] * 1000)
            self.assertFalse(d[name]["pass"])
            self.assertEqual(d[name]["status"], "failed")
        self.assertTrue(d["D4"]["pass"])
        self.assertTrue(d["D5"]["pass"])
        self.assertEqual(result["D6"]["status"], "failed")

    def test_any_strictly_negative_ci_passes_without_added_tolerance(self):
        almost_one = np.nextafter(1.0, 0.0)
        result = evaluate(*fixture(candidate=[almost_one], real=[1], sim=[1], full=[1]))
        for name in ("D2", "D3"):
            self.assertLess(checks(result)[name]["confidence_interval"][1], 0)
            self.assertTrue(checks(result)[name]["pass"])
        self.assertFalse(checks(result)["D1"]["pass"])

    def test_D1_inclusive_five_percent_gain_and_immediate_outside_fail(self):
        at_bound = evaluate(*fixture(candidate=[0.95], real=[1], sim=[2], full=[1]))
        outside = evaluate(*fixture(candidate=[np.nextafter(0.95, np.inf)], real=[1], sim=[2], full=[1]))
        self.assertTrue(checks(at_bound)["D1"]["pass"])
        self.assertFalse(checks(outside)["D1"]["pass"])
        self.assertEqual(checks(at_bound)["D1"]["comparison_lhs"],
                         checks(at_bound)["D1"]["comparison_rhs"])

    def test_D4_D5_inclusive_noninferiority_and_immediate_outside_fail(self):
        values = dict(candidate=[1.05], real=[2], sim=[2], full=[1], candidate_logq=[1.05], real_logq=[1])
        at_bound = evaluate(*fixture(**values))
        self.assertTrue(checks(at_bound)["D4"]["pass"])
        self.assertTrue(checks(at_bound)["D5"]["pass"])
        self.assertTrue(at_bound["D6"]["pass"])
        for field, check in (("candidate", "D4"), ("candidate_logq", "D5")):
            outside = dict(values)
            outside[field] = [np.nextafter(1.05, np.inf)]
            result = evaluate(*fixture(**outside))
            self.assertFalse(checks(result)[check]["pass"])
        self.assertEqual(checks(at_bound)["D4"]["comparison_rhs"], 1.05)
        self.assertEqual(checks(at_bound)["D5"]["comparison_rhs"], 1.05)

    def test_zero_REAL_PSD_reference_is_undefined_without_floor(self):
        result = evaluate(*fixture(candidate=[0, 0], real=[0, 0], sim=[1, 1], full=[1, 1]))
        self.assertUndefined(result)
        for name in ("D1", "D2"):
            self.assertEqual(checks(result)[name]["status"], "undefined")
            self.assertFalse(checks(result)[name]["pass"])
        self.assertEqual(checks(result)["D2"]["denominator_failures"], 1000)
        self.assertIsNone(checks(result)["D2"]["confidence_interval"])

    def test_any_zero_denominator_replicate_invalidates_whole_ci(self):
        result = evaluate(*fixture(candidate=[0, 1], real=[0, 2], sim=[1, 3], full=[1, 2]))
        d = checks(result)
        self.assertTrue(d["D1"]["pass"])
        self.assertEqual(d["D1"]["reference"], 1.0)
        self.assertGreater(d["D2"]["denominator_failures"], 0)
        self.assertGreater(d["D2"]["valid_replicates"], 0)
        self.assertEqual(d["D2"]["valid_replicates"] + d["D2"]["invalid_replicates"], 1000)
        self.assertEqual(sum(value is None for value in d["D2"]["replicate_values"]),
                         d["D2"]["invalid_replicates"])
        self.assertEqual(d["D2"]["status"], "undefined")
        self.assertIsNone(d["D2"]["confidence_interval"])
        self.assertUndefined(result)

    def test_zero_FULL_PSD_and_REAL_logq_references_are_undefined(self):
        result = evaluate(*fixture(candidate=[0.5], real=[1], sim=[2], full=[0],
                                   candidate_logq=[0], real_logq=[0]))
        for name in ("D4", "D5"):
            self.assertEqual(checks(result)[name]["status"], "undefined")
            self.assertFalse(checks(result)[name]["pass"])
        self.assertUndefined(result)

    def test_logq_uses_RMSE_not_MAE(self):
        # REAL errors [0, 2] have RMSE sqrt(2), whereas candidate [1.1, 1.1]
        # has RMSE 1.1: RMSE passes but a mistaken MAE comparison would fail.
        result = evaluate(*fixture(candidate_logq=[1.1, 1.1], real_logq=[0, 2]))
        d5 = checks(result)["D5"]
        self.assertAlmostEqual(d5["reference"], np.sqrt(2))
        self.assertAlmostEqual(d5["candidate"], 1.1)
        self.assertTrue(d5["pass"])
        self.assertGreater(1.1, 1.05 * np.mean([0, 2]))

    def test_row_order_mismatch_rejected_even_when_shapes_match(self):
        truth, estimates = fixture()
        estimates["HGB"]["original_sim_1024"]["row_ids"].reverse()
        result = evaluate(truth, estimates)
        self.assertUndefined(result)
        self.assertIn("row_ids", result["input_binding"]["reason"])

    def test_group_binding_mismatch_rejected(self):
        truth, estimates = fixture()
        estimates["Ridge"]["full_real"]["group_ids"] = [0, 0]
        result = evaluate(truth, estimates)
        self.assertUndefined(result)
        self.assertIn("group_ids", result["input_binding"]["reason"])

    def test_duplicate_mixed_or_malformed_identities_rejected(self):
        bad_values = [("row_ids", ["duplicate", "duplicate"]), ("row_ids", [1, "2"]),
                      ("group_ids", [True, False]), ("group_ids", [0.0, 1.0]),
                      ("row_ids", ["", "event"]), ("group_ids", [0])]
        for field, values in bad_values:
            with self.subTest(field=field, values=values):
                truth, estimates = fixture()
                truth[field] = values
                self.assertUndefined(evaluate(truth, estimates))

    def test_missing_packets_fields_and_nonfinite_values_are_undefined(self):
        modifications = [
            lambda t, e: t.pop("psd"),
            lambda t, e: e.pop("HGB"),
            lambda t, e: e["Ridge"].pop("original_sim_1024"),
            lambda t, e: e["HGB"]["candidate"].pop("group_ids"),
            lambda t, e: e["Ridge"]["full_real"].pop("logq"),
            lambda t, e: e["Ridge"]["candidate"].update(psd=np.array([np.nan, 8.])),
            lambda t, e: t.update(logq=np.array([0., np.inf])),
            lambda t, e: t.update(psd=np.ma.array([0., 0.], mask=[False, True])),
            lambda t, e: e.update(unexpected_estimator=copy.deepcopy(e["Ridge"])),
        ]
        for modification in modifications:
            truth, estimates = fixture()
            modification(truth, estimates)
            self.assertUndefined(evaluate(truth, estimates))

    def test_empty_or_wrong_shape_numeric_values_are_undefined(self):
        for values in ([], [[0, 8]], [0], [0, 8, 9], [True, False], [0j, 8j], ["0", "8"]):
            truth, estimates = fixture()
            estimates["Ridge"]["candidate"]["psd"] = values
            self.assertUndefined(evaluate(truth, estimates))
        truth, estimates = fixture(candidate=[], real=[], sim=[], full=[])
        self.assertUndefined(evaluate(truth, estimates))

    def test_design_declaration_is_required_and_never_defaulted_valid(self):
        self.assertIs(inspect.signature(evaluate_d_statistics).parameters[
            "frozen_input_design_valid"].default, inspect.Parameter.empty)
        truth, estimates = fixture()
        with self.assertRaises(TypeError):
            evaluate_d_statistics(truth=truth, estimates=estimates)
        for invalid in (False, None, 1, "true", np.bool_(True)):
            self.assertUndefined(evaluate_d_statistics(truth=truth, estimates=estimates,
                                                      frozen_input_design_valid=invalid))

    def test_one_successful_estimator_cannot_substitute_for_both(self):
        truth, estimates = fixture()
        estimates["HGB"]["candidate"]["psd"] = estimates["HGB"]["real_only_1024"]["psd"].copy()
        result = evaluate(truth, estimates)
        self.assertTrue(all(item["pass"] for item in checks(result, "Ridge").values()))
        self.assertFalse(checks(result, "HGB")["D1"]["pass"])
        self.assertEqual(result["D6"]["status"], "failed")
        self.assertFalse(result["D6"]["pass"])

    def test_finite_inputs_that_overflow_derived_errors_cannot_pass(self):
        truth, estimates = fixture()
        truth["psd"][0] = -1.0e308
        estimates["Ridge"]["candidate"]["psd"][0] = 1.0e308
        self.assertUndefined(evaluate(truth, estimates))

    def test_reproducible_json_safe_and_no_input_mutation(self):
        truth, estimates = fixture(groups=["artificial-b", "artificial-a"])
        before = copy.deepcopy((truth, estimates))
        first, second = evaluate(truth, estimates), evaluate(truth, estimates)
        self.assertEqual(json.dumps(first, sort_keys=True, allow_nan=False),
                         json.dumps(second, sort_keys=True, allow_nan=False))
        self.assertEqual(first["bootstrap"]["group_order"], ["artificial-a", "artificial-b"])
        for current, original in [(truth, before[0])] + [
                (estimates[name][arm], before[1][name][arm]) for name in ESTIMATORS for arm in ARMS]:
            for key in ("row_ids", "group_ids", "psd", "logq"):
                np.testing.assert_array_equal(current[key], original[key])


def set_single_cpu_affinity():
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                                  ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        current, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(current), ctypes.byref(system)):
            raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask failed")
        selected = current.value & -current.value
        if not kernel.SetProcessAffinityMask(handle, selected):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
        verified = ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(verified), ctypes.byref(system)):
            raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask verification failed")
        return {"before_mask": current.value, "selected_mask": selected,
                "observed_mask": verified.value, "cpu_count": verified.value.bit_count()}
    if hasattr(os, "sched_getaffinity"):
        previous = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(previous)})
        return {"before_cpu_count": len(previous), "selected": sorted(os.sched_getaffinity(0)),
                "cpu_count": len(os.sched_getaffinity(0))}
    raise RuntimeError("single-CPU affinity enforcement is unavailable")


def peak_process_memory():
    if os.name != "nt":
        return {"available": False}

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not kernel.K32GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return {"available": False, "error": ctypes.get_last_error()}
    return {"available": True, "peak_working_set_bytes": counters.PeakWorkingSetSize,
            "current_working_set_bytes": counters.WorkingSetSize,
            "peak_pagefile_bytes": counters.PeakPagefileUsage}


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def startTest(self, test):
        self._started = time.perf_counter()
        super().startTest(test)

    def addSuccess(self, test):
        self.records.append({"test": test.id(), "status": "passed",
                             "seconds": time.perf_counter() - self._started})
        super().addSuccess(test)

    def addFailure(self, test, err):
        self.records.append({"test": test.id(), "status": "failed", "detail": str(err[1])})
        super().addFailure(test, err)

    def addError(self, test, err):
        self.records.append({"test": test.id(), "status": "error", "detail": str(err[1])})
        super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.records.append({"test": subtest.id(), "status": "failed", "detail": str(err[1])})
        super().addSubTest(test, subtest, err)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def main():
    output = Path(__file__).resolve().parent
    if os.name == "nt" and output.drive.upper() not in ("E:", "D:"):
        raise RuntimeError("all outputs must be on E or D")
    if not sys.dont_write_bytecode:
        raise RuntimeError("invoke this artificial selftest with Python -B")
    started_utc = datetime.now(timezone.utc).isoformat()
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    report_before = sha256(output / "report.json")
    affinity = set_single_cpu_affinity()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ArtificialDStatisticsTests)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    report_after = sha256(output / "report.json")
    memory = peak_process_memory()
    resources_valid = affinity["cpu_count"] == 1 and memory.get("available", False) and (
        memory["peak_working_set_bytes"] < 512 * 1024 * 1024)
    immutable_valid = report_before == report_after == EXPECTED_REPORT_SHA256
    passed = result.wasSuccessful() and resources_valid and immutable_valid
    ratio_example = evaluate(*fixture())
    invalid_example = evaluate(*fixture(candidate=[0, 1], real=[0, 2], sim=[1, 3], full=[1, 2]))
    selftest = {
        "schema_version": "artificial-D-selftest-1", "status": "passed" if passed else "failed",
        "data_origin": "only manufactured literals and arrays in this selftest source",
        "scientific_arrays_read": False, "models_loaded_or_fit": False,
        "started_UTC": started_utc, "completed_UTC": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "tests": result.records, "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
                    "platform": platform.platform(), "pid": os.getpid(), "dont_write_bytecode": True,
                    "single_thread_environment": {name: os.environ[name] for name in (
                        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                    "affinity": affinity, "memory": memory,
                    "per_process_resource_check": resources_valid,
                    "resource_scope": "this artificial selftest process; no claim about other team processes"},
        "immutable_report": {"expected_sha256": EXPECTED_REPORT_SHA256, "before_sha256": report_before,
                             "after_sha256": report_after, "unchanged": immutable_valid},
        "decisive_artificial_examples": {
            "whole_group_weighted_mean": {"expected": 18 / 5, "incorrect_equal_group_mean": 14 / 3},
            "ratio_CI": {"observed": checks(ratio_example)["D2"]["confidence_interval"],
                         "incorrect_absolute_CI_divided_by_point_MAE": [-0.2, -0.2],
                         "artificial_D6_pass": ratio_example["D6"]["pass"],
                         "shared_draws_sha256": ratio_example["bootstrap"]["draws_sha256"]},
            "any_invalid_replicate": {key: checks(invalid_example)["D2"][key] for key in (
                "status", "pass", "requested_replicates", "valid_replicates", "invalid_replicates",
                "denominator_failures", "confidence_interval")},
        },
    }
    selftest_path = output / "ARTIFICIAL_SELFTEST.json"
    write_json(selftest_path, selftest)
    receipt = {
        "schema_version": "D-code-preparation-receipt-1",
        "status": "code_preparation_selftests_passed" if passed else "code_preparation_selftests_failed",
        "prepared_UTC": datetime.now(timezone.utc).isoformat(), "output_directory": str(output),
        "implementation": "pure NumPy statistics adapter; standard library only for binding hashes and selftest receipt",
        "checks_implemented": ["D1 PSD MAE point gain <= -5%", "D2 paired relative PSD MAE 95% CI upper < 0",
                               "D3 candidate minus REAL+original SIM absolute PSD MAE 95% CI upper < 0",
                               "D4 full REAL PSD MAE point noninferiority <= +5%",
                               "D5 REAL1024 logQ RMSE point noninferiority <= +5%",
                               "D6 every D1-D5 passes for Ridge and HGB and explicit frozen design validity"],
        "bootstrap": {"repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED, "quantile_method": "linear",
                      "resampling": "one shared whole-group draw per replicate across all 8 prediction packets",
                      "relative_CI": "ratio computed after each replicate's two weighted MAEs",
                      "invalid_replicates": "no discard; any invalid replicate makes that CI undefined and not passing"},
        "interface_contract": {"required_keyword_without_default": "frozen_input_design_valid",
                               "omitted_declaration": "signature raises TypeError; evaluation cannot run or pass",
                               "false_or_nonbool_declaration": "all checks undefined and not passing",
                               "packet_bindings": "truth and every arm separately supply same-order row_ids and group_ids",
                               "missing_zero_nonfinite": "undefined and not passing; no denominator floor",
                               "inclusive_point_bounds": "equivalent candidate <= (1 + threshold)*reference; no tolerance",
                               "strict_CI_bounds": "CI upper < 0; no tolerance"},
        "selftest": {"artifact": selftest_path.name, "tests_run": result.testsRun,
                     "all_passed": passed, "artificial_arrays_only": True},
        "execution": {"invocation": "Python313/python.exe -B test_d_statistics_adapter.py",
                      "affinity_cpus": affinity["cpu_count"], "numpy_threads_requested": 1,
                      "observed_peak_working_set_bytes": memory.get("peak_working_set_bytes"),
                      "per_process_below_512_MiB": resources_valid},
        "immutable_report": selftest["immutable_report"],
        "scientific_execution_or_acceptance": {
            "wired_to_actual_D": False, "actual_D_acceptance_tested": False,
            "actual_frozen_input_design_verified": False, "scientific_arrays_read": False,
            "models_or_identity_bins_read": False, "trained_generated_or_fitted": False,
            "D_or_E_registered_or_run": False, "frozen_R63_modified": False,
            "root_metadata_or_Git_modified": False,
            "limitation": "A caller-supplied true design flag is not proof of actual sample identities, training design, model freezing, or scientific acceptance."},
        "hashes_sha256": {path.name: sha256(path) for path in (
            output / "d_statistics_adapter.py", output / "test_d_statistics_adapter.py", selftest_path)},
    }
    write_json(output / "CODE_PREPARATION_RECEIPT.json", receipt)
    print(json.dumps({"status": receipt["status"], "tests_run": result.testsRun,
                      "report_unchanged": immutable_valid, "cpu_affinity_count": affinity["cpu_count"],
                      "peak_working_set_bytes": memory.get("peak_working_set_bytes"),
                      "output_directory": str(output)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
