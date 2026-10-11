"""Bound metadata and schema helpers. Importing never opens scientific files."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

import numpy as np


class AdmissionError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise AdmissionError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def integer(value, name):
    require(isinstance(value, int) and not isinstance(value, bool), name + " must be integer")
    return value


def finite_number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            name + " must be numeric")
    require(math.isfinite(value), name + " must be finite")
    return float(value)


def binding(row, name):
    require(isinstance(row, dict) and set(row) >= {"path", "sha256"}, name + " binding missing")
    require(isinstance(row["path"], str) and Path(row["path"]).is_absolute(),
            name + " requires explicit absolute path")
    require(isinstance(row["sha256"], str) and re.fullmatch("[0-9a-f]{64}", row["sha256"]),
            name + " requires exact sha256")
    return row


def read_json_bound(row, name):
    row = binding(row, name)
    require(sha_file(row["path"]) == row["sha256"], name + " file SHA mismatch")
    return json.loads(Path(row["path"]).read_text(encoding="utf-8"))


def verify_file(row, name):
    row = binding(row, name)
    require(sha_file(row["path"]) == row["sha256"], name + " file SHA mismatch")
    return Path(row["path"])


def validate_config(c):
    require(c.get("execution_enabled") is True,
            "static candidate disabled; root must bind a separate registered evaluation config")
    require(integer(c.get("schema_version"), "schema_version") == 1 and
            integer(c.get("acceptance_version"), "acceptance_version") == 5,
            "only the explicit v5 schema is implemented")
    require(c.get("purpose") == "registered_v5_full_response_generation_replay",
            "incorrect scientific role")
    require(isinstance(c.get("stage"), str) and c["stage"] and
            c.get("registered_stage") == c["stage"], "stage binding absent")
    require(c.get("generation_chunk") == 256, "candidate is fixed to original-order chunk256")
    require(c.get("straight_through") is False,
            "bind current R74 postcheck inference straight_through=False explicitly")
    require(c.get("inference_device") == "cuda:0", "no CPU fallback")
    require(c.get("raw_precision") == "loaded_raw_to_float64" and
            c.get("epsilon_precision") == "float32", "precision contract absent")
    require(c.get("CPU_threads") == 12 and c.get("exact_CPU_affinity") == list(range(12)),
            "original native CPU resource contract absent")
    require(c.get("RSS_limit_bytes") == 48 * 1024**3 and
            c.get("GPU_limit_bytes") == 10 * 1024**3, "native memory resource contract absent")
    require(Path(c.get("output_root", "")).is_absolute(), "output root must be explicit absolute")
    require(c.get("SOURCE_arm") and c.get("REAL_arm") and
            c["SOURCE_arm"] != c["REAL_arm"], "two explicit distinct output arms required")
    for key in ["SOURCE_arm", "REAL_arm"]:
        require(re.fullmatch("[A-Za-z0-9_]+", c[key]) is not None, "unsafe arm name")
    for key in ["config", "receipt", "numerical_reference", "postcheck_receipt",
                "TRAIN_replay", "DATA_receipt", "native_receipt"]:
        binding(c["training"][key], "training." + key)
    for key in ["shape_weights", "frozen_Q_weights", "original_Q_weights", "shape_statistics"]:
        binding(c["model"][key], "model." + key)
    for key in ["actual", "source_metadata", "REAL_packet", "probes", "joint8"]:
        binding(c["inputs"][key], "inputs." + key)
    require(Path(c["runtime_source_directory"]).is_absolute(), "explicit runtime bundle absent")
    for name in ["waveform_response.py", "public_charge_head.py",
                 "common_response.py", "physical_operators.py"]:
        row = binding(c["runtime_sources"][name], name)
        require(Path(row["path"]).resolve().parent == Path(c["runtime_source_directory"]).resolve()
                and Path(row["path"]).name == name, "runtime bundle path mismatch")
    require(c.get("expected_populations") == {
        "SOURCE": 2000, "REAL_control": 2000, "joint8": 1024,
        "probes": 832, "guide": 128, "original_support_outside": 417,
        "primary_valley_bin2": 213, "primary_valley_bin3": 181},
        "original held-out populations altered")
    output = Path(c["output_root"]).resolve()
    for row in list(c["model"].values()) + list(c["inputs"].values()) + list(c["training"].values()):
        require(output != Path(row["path"]).resolve().parent,
                "evaluation output must not overwrite original training/input directories")
    require(output != Path(c["runtime_source_directory"]).resolve(),
            "runtime bundle is read-only; output must be separate")
    return c


def validate_training_records(c, t, r, n, p, replay, data, native):
    """Pure metadata admission; call before any PT/NPZ or runtime module read."""
    u = integer(t.get("updates"), "training config updates")
    require(u > 0, "bound prospective training endpoint must be positive")
    require(integer(r.get("completed_updates"), "completed_updates") == u and
            integer(r.get("unique_selected_endpoint"), "unique_selected_endpoint") == u,
            "no engineering/partial/checkpoint endpoint may be evaluated")
    require(t.get("SOURCE_N") == 6593 and t.get("REAL_N") == 21047 and
            t.get("generation_chunk") == 256 and t.get("hidden") == 128 and
            t.get("shape_network") == [482, 128, 128, 480] and
            t.get("shape_dtype") == "float32" and t.get("physical_dtype") == "float64" and
            t.get("Q_head_frozen") is True, "bound training method mismatch")
    require(t.get("numeric_median_LSB_max") == .25 and
            t.get("numeric_p95_LSB_max") == 1. and
            t.get("preADC_relative_Qstar_error_max") == 1e-5 and
            t.get("numeric_CPU_reference_rows_per_domain") == 64,
            "original numerical lines changed")
    require(r.get("complete_parent_per_update") is True and
            r.get("heldout_arrays_read") is False, "training parent/held-out assertion absent")
    for key, expected in {
        "SOURCE_main_exposures": 6593*u, "REAL_main_exposures": 21047*u,
        "full_neural_firstpass_rows": 27640*u, "full_parameter_VJP_rows": 27640*u}.items():
        require(integer(r.get(key), key) == expected, key + " mismatch")
    require(r.get("final_weight_file_SHA256") == c["model"]["shape_weights"]["sha256"],
            "shape file not the completed receipt endpoint")
    require(r.get("Q_head_weight_SHA256") == c["model"]["frozen_Q_weights"]["sha256"],
            "frozen Q file must match actual training receipt")
    require(isinstance(r.get("final_state_SHA256"), str) and
            re.fullmatch("[0-9a-f]{64}", r["final_state_SHA256"]),
            "completed shape state digest absent")
    require(data.get("SOURCE_N") == 6593 and data.get("REAL_N") == 21047 and
            data.get("Q_head_weights_SHA256") == c["model"]["original_Q_weights"]["sha256"] and
            data.get("heldout_arrays_read") is False and
            data.get("shape_dtype") == "float32" and data.get("physical_dtype") == "float64",
            "original Q/data receipt binding absent")
    require(set(n.get("results", {})) == {"SOURCE", "REAL"}, "numeric domains missing")
    for domain, row in n["results"].items():
        require(integer(row.get("N"), domain + " numerical N") == 64,
                "full independent numerical sample changed")
        for key, limit in [("median_RMS_LSB", .25), ("p95_RMS_LSB", 1.),
                           ("independent_Qstar_relative_error_max", 1e-5)]:
            value = finite_number(row.get(key), domain + "." + key)
            require(0 <= value <= limit, "independent numerical reference failed")
        require(row.get("pass_gate") is True, "numeric receipt not PASS")
    require(p.get("numerical_and_TRAIN_replay_pass") is True and
            p.get("heldout_arrays_read") is False and p.get("TRAIN_replay_rows") == 4000 and
            p.get("independent_numpy_rows") == 128, "complete postcheck receipt absent")
    require(set(replay.get("results", {})) == {"SOURCE", "REAL"}, "TRAIN replay domains missing")
    for row in replay["results"].values():
        require(row.get("N") == 2000 and
                all(row.get(k) is True for k in ["preADC_bitwise", "canonical_ADC_bitwise",
                                                "Qstar_bitwise"]), "TRAIN serialization replay failed")
    require(native.get("entry") == "train_full.py" and
            integer(native.get("actual_worker_native_exit"), "actual worker exit") == 0 and
            native.get("termination_reason") is None and
            integer(native.get("actual_worker_PID"), "actual worker PID") > 0,
            "completed actual native worker exit0 absent")
    require(0 < finite_number(native.get("actual_worker_RSS_peak_bytes"), "worker RSS") <=
            c["RSS_limit_bytes"], "training resource receipt invalid")
    require(0 <= finite_number(native.get("GPU_process_memory_peak_bytes"), "GPU peak") <=
            c["GPU_limit_bytes"], "training GPU receipt invalid")
    return {"completed_updates": u, "final_state_SHA256": r["final_state_SHA256"],
            "native_worker_PID": native["actual_worker_PID"],
            "native_exit": 0, "numeric_gate_recomputed": True,
            "ADC_numpy_reference_flags_are_descriptions_not_new_gate": True}


def preflight_training(c):
    validate_config(c)
    names = ["config", "receipt", "numerical_reference", "postcheck_receipt",
             "TRAIN_replay", "DATA_receipt", "native_receipt"]
    rows = [read_json_bound(c["training"][name], "training." + name) for name in names]
    return validate_training_records(c, *rows)


def load_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text(encoding="utf-8"))
    validate_config(c)
    return c, {"path": str(path), "sha256": sha_file(path)}


def read_npz_bound(row, name, keys=None, required_names=()):
    """Future registered worker only. Never invoked by the artificial suite."""
    path = verify_file(row, name)
    with np.load(path, allow_pickle=False) as packet:
        require(set(required_names) <= set(packet.files), name + " required roles absent")
        select = packet.files if keys is None else keys
        require(set(select) <= set(packet.files), name + " schema fields absent")
        return {k: packet[k].copy() for k in select}


def q_window(x):
    a = np.asarray(x, dtype=np.float64)
    return a[..., 100:400].sum(-1) - 3. * a[..., :100].sum(-1)


def feature_Q_PSD(x):
    """Same original mean-centered SOURCE_INPUT Q/PSD definition, no ADC."""
    a = np.asarray(x, dtype=np.float64)
    y = a - a[..., :100].mean(-1, keepdims=True)
    qt = y[..., 100:400].sum(-1)
    qf = y[..., 100:150].sum(-1)
    qmV, qfmV = qt * 1000., qf * 1000.
    p = np.divide(qfmV, qmV, out=np.full_like(qmV, np.nan), where=qmV > 0)
    return qmV, 1. - p


def raw_bins_mVns(q_mVns, edges):
    q = np.asarray(q_mVns, dtype=np.float64)
    require(np.isfinite(q).all() and (q > 0).all(), "nonpositive/nonfinite Q; no deletion")
    edges = np.asarray(edges, dtype=np.float64)
    require(edges.shape == (9,) and np.isfinite(edges).all() and
            (np.diff(edges) > 0).all(), "frozen logQ edges invalid")
    logq = np.log(q)
    out = np.searchsorted(edges, logq, side="right") - 1
    out[logq == edges[-1]] = 7
    return out.astype(np.int64)


def raw_bins(q_Vns, edges):
    return raw_bins_mVns(1000. * np.asarray(q_Vns, dtype=np.float64), edges)


def adc_representations(canonical):
    a = np.asarray(canonical)
    require(a.dtype == np.float64 and a.ndim == 2 and a.shape[1] == 480,
            "actual canonical output must be full480 float64")
    require(np.isfinite(a).all() and (a >= -.5).all() and (a <= .5).all(),
            "actual ADC waveform invalid or outside range")
    code64 = np.rint((a + .5) * 16383.)
    require((code64 >= 0).all() and (code64 <= 16383).all(), "integer ADC code out of range")
    codes = code64.astype(np.int32)
    strict = -.5 + codes.astype(np.float64) / 16383.
    require(np.array_equal(np.rint((strict + .5) * 16383.).astype(np.int32), codes),
            "strict code representation mismatch")
    return codes, strict


def wave_rows(x, n, name):
    require(isinstance(x, np.ndarray) and x.shape == (n, 480) and
            x.dtype in [np.dtype("float32"), np.dtype("float64")] and np.isfinite(x).all(),
            name + " requires unchanged finite floating full480 rows")


def row_array(x, n, name):
    require(isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == n and
            x.dtype.kind != "O", name + " row schema mismatch")


def validate_main(actual, metadata, reference, edges):
    n = 2000
    wave_rows(actual["source_waveforms_V"], n, "SOURCE")
    wave_rows(reference["reference_V"], n, "REAL reference")
    for key in ["source_ids", "z", "white", "style"]:
        row_array(actual[key], n, "actual." + key)
    wave_rows(actual["white"], n, "actual white")
    require(len(np.unique(actual["source_ids"])) == n, "SOURCE parent IDs not unique")
    require(exact(actual["source_ids"], metadata["source_ids"]), "SOURCE metadata ID order mismatch")
    for key, value in metadata.items():
        row_array(value, n, "original source metadata." + key)
    require(set(metadata) >= {"composition", "Qsource_Vns", "source_label",
                              "pdg_inferred", "edep_ee_MeV", "edep_hadron_MeV"},
            "original source metadata fields absent")
    for key in ["reference_ids", "reference_groups"]:
        row_array(reference[key], n, key)
    require(len(np.unique(reference["reference_ids"])) == n, "REAL reference IDs not unique")
    comp = metadata["composition"]
    counts = {str(k): int((comp == k).sum()) for k in np.unique(comp)}
    require(counts == {"mixed": 65, "pure_EE": 417, "pure_had": 1518}, "source composition changed")
    b_support = raw_bins(q_window(actual["source_waveforms_V"]), edges)
    inside = (b_support >= 0) & (b_support < 8)
    require(int((~inside).sum()) == 417, "original support417 altered")
    fq, fp = feature_Q_PSD(actual["source_waveforms_V"])
    b_feature = raw_bins_mVns(fq, edges)
    for b, expected in [(2, 213), (3, 181)]:
        require(int(((comp == "pure_had") & (b_feature == b)).sum()) == expected,
                "original SOURCE_INPUT valley mother population changed")
    return {"support_bin": b_support, "inside_support": inside,
            "feature_Q_mVns": fq, "feature_PSD": fp, "feature_bin": b_feature,
            "guide_pick": np.flatnonzero(inside)[:128]}


def validate_aux(a, rows, main, kind):
    wave_rows(a["source_waveforms_V"], rows, kind)
    wave_rows(a["white"], rows, kind + " white")
    row_array(a["z"], rows, kind + " z")
    parent = a["source_parent_index"]
    require(parent.shape == (rows,) and parent.dtype.kind in "iu" and
            (parent >= 0).all() and (parent < 2000).all(), kind + " parent index invalid")
    if kind == "joint8":
        row_array(a["style"], rows, "joint8 style")
        require(rows == 1024, "joint8 size mismatch")
        groups = parent.reshape(128, 8)
        require(np.all(groups == groups[:, :1]) and len(np.unique(groups[:, 0])) == 128,
                "joint8 original128x8 parent layout changed")
        require(exact(a["source_waveforms_V"], main["source_waveforms_V"][parent]),
                "joint8 original parent waveforms changed")
    else:
        for key in ["source_parent_ids", "noise_origin_parent_index", "reference_parent_index",
                    "source_waveform_modified_counterfactual", "probe_kind"]:
            row_array(a[key], rows, "probe." + key)
        require(exact(a["source_parent_ids"], main["source_ids"][parent]),
                "probe parent IDs changed")
        for key in ["noise_origin_parent_index", "reference_parent_index"]:
            idx = a[key]
            require(idx.shape == (rows,) and idx.dtype.kind in "iu" and
                    (idx >= 0).all() and (idx < 2000).all(), "probe reference index invalid")
        require(a["source_waveform_modified_counterfactual"].dtype == np.bool_,
                "probe counterfactual flags must remain boolean")
        require("source_roll_no_fixed_points" in a["probe_kind"],
                "original source_roll probe absent")
    return parent


def main_inputs(c):
    actual = read_npz_bound(c["inputs"]["actual"], "actual",
                           ["source_waveforms_V", "source_ids", "z", "white", "style"])
    metadata = read_npz_bound(c["inputs"]["source_metadata"], "source_metadata")
    reference = read_npz_bound(c["inputs"]["REAL_packet"], "REAL_packet",
                              ["reference_V", "reference_ids", "reference_groups"],
                              ["real_V", "real_ids", "real_groups",
                               "reference_V", "reference_ids", "reference_groups"])
    return actual, metadata, reference


def identity_payload(domain, actual, metadata, reference, axes):
    n = 2000
    shared = {"original_row": np.arange(n, dtype=np.int64),
              "original_weight": np.full(n, 1/n, dtype=np.float64)}
    if domain == "SOURCE":
        result = {**metadata, **shared, "source_ids": actual["source_ids"],
                  "original_group": np.arange(n, dtype=np.int64),
                  "legacy_nominal_Qsource_Vns": metadata["Qsource_Vns"],
                  "original_source_bin_raw": axes["support_bin"],
                  "source_Qbin_actual_preADC": axes["support_bin"],
                  "inside_TRAIN_support": axes["inside_support"],
                  "source_q_Vns": q_window(actual["source_waveforms_V"]),
                  "original_SOURCE_INPUT_Q_mVns": axes["feature_Q_mVns"],
                  "original_SOURCE_INPUT_PSD": axes["feature_PSD"],
                  "original_primary_valley_bin_raw": axes["feature_bin"]}
    else:
        result = {**shared, "reference_ids": reference["reference_ids"],
                  "reference_groups": reference["reference_groups"],
                  "original_group": reference["reference_groups"]}
    return result


def merge_columns(*parts):
    out = {}
    for part in parts:
        for key, value in part.items():
            require(isinstance(value, np.ndarray) and value.dtype.kind != "O",
                    key + " must be a concrete nonobject array")
            if key in out:
                require(exact(out[key], value), "duplicate payload field mismatch: " + key)
            else:
                out[key] = value
    return out


def check_payload(data, n):
    for key in ["pre_ADC_V", "waveforms_V", "raw_preconstraint_V", "pre_constraint_projected_f64_V",
                "evaluation_canonical_V", "ADC_integer_codes", "projected_colored_innovation_V"]:
        require(key in data and data[key].shape == (n, 480), key + " missing full output rows")
    require(data["pre_ADC_V"].dtype == np.float32 and data["waveforms_V"].dtype == np.float64,
            "saved pipeline precision changed")
    require(data["ADC_integer_codes"].dtype == np.int32, "strict codes must be int32")
    for key in ["Q0", "Qstar", "QpreADC", "QADC", "predicted_target_Q_Vns",
                "original_bin_raw", "target_bin_raw", "preADC_bin_raw", "ADC_bin_raw",
                "Q_relative_error_f32", "Q_relative_error_f64", "response_applied_all_rows"]:
        require(key in data and data[key].shape == (n,), key + " missing row metadata")
    require(np.array_equal(data["Qstar"], data["predicted_target_Q_Vns"]),
            "target not independently saved before waveform")
    require(np.isfinite(data["Qstar"]).all() and (data["Qstar"] > 0).all(),
            "illegal Qstar")
    require(data["response_applied_all_rows"].dtype == np.bool_ and
            data["response_applied_all_rows"].all(), "rows silently skipped")
    return data


def summary(x):
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    require(np.isfinite(a).all(), "nonfinite descriptive summary; invalid rows must be explicit")
    if len(a) == 0:
        return {"N": 0, "mean": None, "median": None, "sample_sd": None}
    return {"N": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "sample_sd": float(a.std(ddof=1)) if len(a) > 1 else None,
            "min": float(a.min()), "max": float(a.max()), "p95": float(np.quantile(a, .95))}


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)


def save_output(path, payload):
    path = Path(path)
    require(not path.exists(), "do not overwrite a previous scientific generation")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    return {"path": str(path), "sha256": sha_file(path),
            "rows": len(payload["pre_ADC_V"]),
            "array_SHA256": {k: array_sha(v) for k, v in payload.items()}}


def owned_runtime(c, entry, module_directory):
    """Require the existing root-owned Linux RSS/GPU supervisor handshake."""
    require(sys.platform == "linux", "scientific entries require registered Linux supervisor")
    nonce = os.environ.get("OWNED_SCIENCE_NONCE")
    require(nonce, "owned worker handshake missing")
    ack = json.loads((Path(module_directory) / (entry + ".RSS_ACK.json")).read_text("utf-8"))
    require(ack.get("nonce") == nonce and ack.get("actual_PID") == os.getpid() and
            ack.get("actual_PID_memory_query_success") is True and
            ack.get("RSS_limit_bytes") == c["RSS_limit_bytes"], "owned actual PID/RSS binding missing")
    require(sorted(os.sched_getaffinity(0)) == c["exact_CPU_affinity"], "actual affinity mismatch")
    return {"actual_PID": os.getpid(), "nonce": nonce, "entry": entry,
            "RSS_guard": "existing registered external supervisor",
            "CPU_affinity": sorted(os.sched_getaffinity(0)), "handshake_verified": True}


def config_argument(entry, directory, argv=None):
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    # R74 owned_worker uses runpy and leaves its single entry token in argv.
    if os.environ.get("OWNED_SCIENCE_NONCE") and argv == [entry]:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(directory) / "config.json",
                        help="explicit root-bound config in the new evaluation directory")
    return parser.parse_args(argv).config
