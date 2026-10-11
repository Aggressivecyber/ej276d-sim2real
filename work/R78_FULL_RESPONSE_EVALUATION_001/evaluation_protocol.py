"""Frozen evaluation configuration and roles; import is read-only and has no I/O."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from evaluation_io import require

VIEWS = ["full_historical_shape", "normalized_u480", "full_voltage", "Q_PSD_2D", "baseline_local"]
HGB = {"max_iter": 100, "learning_rate": .06, "max_leaf_nodes": 15,
       "min_samples_leaf": 20, "l2_regularization": 1,
       "early_stopping": False, "random_state": 20261008}
CONTROLS = ["BASE", "INDEPENDENT_CFM"]
CONTROL_ALIASES = {"BASE": "BASE_SOURCE", "BASE_REAL_CONTROL": "BASE_REAL",
                   "RAW_REAL_REAL": "RAW_REFERENCE"}


def reference_paths(cfg):
    frozen = Path(cfg["frozen_eval_input_root"])
    return {
        "CAL_metadata": str(frozen/"CAL_REFERENCE_METADATA.json"),
        "valley": str(frozen/"VALLEY_DEFINITION.json"),
        "data_roles": str(frozen/"DATA_ROLES.json"),
        "width_definition": str(Path(cfg["local_input_paths"]["width_definition"])),
        "frozen_width_metrics": str(Path(cfg["R39_root"])/"metrics.json"),
        "BASE_output": str(Path(cfg["R32_root"])/"CONTROL_FINAL"/"generated.npz"),
        "INDEPENDENT_CFM_output": str(Path(cfg["R33_root"])/"INDEPENDENT_CFM"/"generated.npz"),
        "BASE_REAL_output": str(Path(cfg["old_BASE_REAL_generated"])),
        "fixed_selection": str(Path(cfg["R26_root"])/"results"/"C2ST_PRIVATE.npz"),
        "frozen_C2ST_metrics": str(Path(cfg["canonical_R51_root"])/"CANONICAL_C2ST.json"),
        "frozen_C2ST_predictions": str(Path(cfg["canonical_R51_root"])/"CANONICAL_PREDICTIONS_PRIVATE.npz"),
    }


def validate_post_config(cfg, adapter, admission):
    require(cfg.get("acceptance_version") == 5, "explicit acceptance_version5 required")
    require(cfg.get("SOURCE_arm") == adapter["SOURCE_arm"] and
            cfg.get("REAL_arm") == adapter["REAL_arm"], "evaluation SOURCE/REAL roles changed")
    require(cfg.get("updates") == admission["completed_updates"], "post endpoint differs from actual completed endpoint")
    require(cfg.get("C2ST") == HGB and cfg.get("C2ST_views") == VIEWS,
            "original classifier hyperparameters or view order changed")
    for key, value in {"train_each_domain":600, "holdout_each_domain":250,
                       "bootstrap_repeats":500, "bootstrap_seed":2026100871,
                       "HGB_max_fits":10}.items():
        require(cfg.get(key) == value, "original evaluation definition changed: "+key)
    require(isinstance(cfg.get("recipe_description"), str) and cfg["recipe_description"],
            "prospectively bound current method description required")
    require(isinstance(cfg.get("reference_bindings"), dict), "prebound reference SHA map required")
    for path in reference_paths(cfg).values():
        require(Path(path).is_absolute() and path in cfg["reference_bindings"],
                "explicit prebound old reference absent: "+path)
    return cfg


def validate_selection(saved):
    expected = {f"phase{p}_{j}" for p in [0,1] for j in range(3)}
    require(set(saved) == expected, "fixed R26 selection roles missing/changed")
    selected = {}
    for phase, n in [(0,600),(1,250)]:
        arrays = []
        for j in range(3):
            a = np.asarray(saved[f"phase{phase}_{j}"])
            require(a.ndim == 1 and a.dtype.kind in "iu" and len(a) == n and
                    len(np.unique(a)) == n and (a >= 0).all() and (a < 2000).all(),
                    "fixed600/250 selection changed")
            arrays.append(a)
        selected[phase] = tuple(arrays)
    for j in range(3):
        require(not np.intersect1d(selected[0][j],selected[1][j]).size,
                "fixed original train/held-out overlap")
    return selected


def generation_counters(generation, source_arm, real_arm):
    require(generation.get("status") == "complete5984_generation_only" and
            generation.get("all_generation_rows") == 5984 and
            set(generation.get("domains",{})) == {source_arm,real_arm},
            "incomplete/misassigned full generation")
    counter = generation["counter"]
    for key, expected in {"source_main_rows":2000, "REAL_control_rows":2000,
                          "repeat_noise_rows":1024, "source_probe_rows":832,
                          "extra_guide_probe_rows":128, "response_fit_or_update":0,
                          "HGB_attempts":0, "density_EM_attempts":0,
                          "final_EVAL_read":0}.items():
        require(counter.get(key) == expected, "generation row/fit accounting changed: "+key)
    return {"main_generation_rows":counter["source_main_rows"]+counter["REAL_control_rows"],
            "repeat_noise_rows":counter["repeat_noise_rows"],
            "source_probe_rows":counter["source_probe_rows"],
            "extra_guide_probe_rows":counter["extra_guide_probe_rows"]}
