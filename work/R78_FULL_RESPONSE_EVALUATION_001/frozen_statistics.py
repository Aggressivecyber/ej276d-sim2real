"""Original fixed HGB/EM/PSD descriptions; callable only by registered evaluator."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
import evaluation_primitives as ev
import diagnostic_primitives as dc
import assessment_primitives as aa
import width_primitives as ew
import source_diagnostics as sd
from evaluation_protocol import CONTROL_ALIASES, CONTROLS, validate_selection
from evaluation_io import require


def c2st(ctx, domains):
    cfg, arms, real_arm = ctx.cfg, ctx.arms, ctx.real_arm
    saved = ctx.reference_npz("fixed_selection",
        [f"phase{p}_{j}" for p in [0,1] for j in range(3)])
    selected = validate_selection(saved)
    hist = ctx.reference_json("frozen_C2ST_metrics")
    hp = ctx.reference_npz("frozen_C2ST_predictions")
    result, paired, private = {}, {}, {}
    def old_prediction(name, view):
        tag = CONTROL_ALIASES[name]+"__"+view+"__"
        return {k:hp[tag+k] for k in ["probability","label","cluster"]}
    for name, alias in CONTROL_ALIASES.items():
        result[name] = {}
        for view in cfg["C2ST_views"]:
            row = dict(hist["results"][alias+"__"+view])
            row.update(balanced_accuracy=row["BA"],
                       heldout_CI={"balanced_accuracy_95CI":row["BA_95CI"]},
                       reused_from=ctx.references["frozen_C2ST_metrics"]["path"], new_fit=False)
            result[name][view] = row
    paired["BASE_REAL_CONTROL"] = {
        v:dict(aa.paired_bootstrap(old_prediction("BASE_REAL_CONTROL",v),
                                  old_prediction("RAW_REAL_REAL",v),cfg),
               reference="RAW_REAL_REAL") for v in cfg["C2ST_views"]}
    previous, previous_callback = dc.matrix, getattr(dc,"PERSIST_HGB_CALLBACK",None)
    def matrix(d, view):
        return d["u"].copy() if view == "normalized_u480" else previous(d,view)
    dc.matrix = matrix
    try:
        for name in arms:
            d, r = dict(domains[name]), dict(domains["REAL"])
            j = 2 if name == real_arm else 0
            tr, te = selected[0][j], selected[1][j]
            rt, re = selected[0][1], selected[1][1]
            require(not set(d["groups"][tr]) & set(d["groups"][te]) and
                    not set(r["groups"][rt]) & set(r["groups"][re]),
                    "original fixed group leakage; no resplit allowed")
            r["cluster"] = np.array(["R|"+str(g) for g in r["groups"]])
            if name == real_arm:
                d["cluster"] = np.array(["R|"+str(g) for g in d["groups"]])
            result[name], paired[name] = {}, {}
            for view in cfg["C2ST_views"]:
                slot = name+"__"+view
                def persist(row,p):
                    np.savez_compressed(ctx.results/(slot+"_PREDICTIONS_PRIVATE.npz"),**p)
                    ctx.write("results/C_"+slot+".json",row)
                    ctx.progress()
                dc.PERSIST_HGB_CALLBACK = persist
                row,p = dc.fit_one(d,r,(tr,te),(rt,re),view,cfg,ctx.counter)
                ctx.counter["HGB_completed_slots"] += 1
                row.update(original_fixed_rows_retained=True,canonical_ADC_representation=True,
                    candidate_train_QC_n=int(d["valid"][tr].sum()),
                    candidate_holdout_QC_n=int(d["valid"][te].sum()))
                persist(row,p)
                result[name][view] = row
                private.update({slot+"__"+k:p[k] for k in ["probability","label","cluster"]})
                refname = "RAW_REAL_REAL" if name == real_arm else "BASE"
                paired[name][view] = dict(aa.paired_bootstrap(p,old_prediction(refname,view),cfg),
                                          reference=refname)
                ctx.write("C2ST_RUNNING.json",{"results":result,"paired":paired,
                                             "counters":ctx.counter})
                ctx.log("C2ST "+slot+" BA "+str(row["balanced_accuracy"]))
    finally:
        dc.matrix, dc.PERSIST_HGB_CALLBACK = previous, previous_callback
    same_treated = {}
    for view in cfg["C2ST_views"]:
        p = {k:private[real_arm+"__"+view+"__"+k] for k in ["probability","label","cluster"]}
        same_treated[view] = dict(aa.paired_bootstrap(p,old_prediction("BASE_REAL_CONTROL",view),cfg),
                                  reference="BASE_REAL_CONTROL")
    same_treated = {real_arm:same_treated}
    ctx.write("SAME_TREATED_REAL_CONTROL_PAIRED.json",same_treated)
    np.savez_compressed(ctx.results/"C2ST_PRIVATE.npz",**private,
        **{f"phase{p}_{j}":a for p,v in selected.items() for j,a in enumerate(v)})
    return {"results":result,"paired":paired,"same_treated_REAL_control_paired":same_treated,
        "original_fixed_selection_no_resplit":True,
        "fixed_classifier_CI_not_response_training_uncertainty":True,
        "historical_control_fits_reused_from_R51":True,
        "all_five_original_acceptance_views":True,"new_fits":ctx.counter["HGB_attempts"],
        "train_each_domain":600,"holdout_each_domain":250,
        "REAL_REAL_reference":"same-acquisition disjoint TRAIN reference/DEV artificial groups; not physical equivalence"}


def widths(ctx, domains, old):
    cfg, edges, arms = ctx.cfg, ctx.edges, ctx.arms
    tables = {n:old["energy_conditional_width"]["tables"][n]
              for n in ["REAL","SOURCE_CLEAN","BASE_REAL_CONTROL",*CONTROLS]}
    grid = np.linspace(-.1,.6,129)
    for name in arms:
        d = domains[name]
        good = d["valid"]
        x,q = d["features"]["PSD"][good], d["features"]["Qtotal_mVns"][good]
        z,b = ew.bin_indices(q,edges)
        weights,rows = np.ones(len(x),"f8"),[]
        for k in range(8):
            choose = b == k
            finite = np.isfinite(x[choose]) & np.isfinite(z[choose])
            attempt = int(finite.sum() >= 40)
            require(ctx.counter["density_EM_attempts"]+attempt <= 16 and
                    ctx.counter["density_description_slots"] < 16,
                    "original16 density slots/EM attempts exceeded")
            ctx.counter["density_EM_attempts"] += attempt
            ctx.counter["density_description_slots"] += 1
            ctx.progress()
            row = ew.fit_slice(x[choose],z[choose],weights[choose])
            require(row["fit_attempts"] == attempt,"original EM attempt semantics changed")
            ctx.counter["density_completed_slots"] += 1
            row.update(bin=k,logQ_edges=edges[k:k+2].tolist(),
                charge_edges_mVns=np.exp(edges[k:k+2]).tolist(),
                actual_charge_range_mVns=[float(q[choose].min()),float(q[choose].max())]
                                         if choose.any() else None)
            row["histogram_counts"] = np.histogram(x[choose],bins=grid)[0].tolist()
            row["histogram_PSD_outside_grid_count"] = int(np.sum(choose & ((x<grid[0])|(x>grid[-1]))))
            ctx.write("results/WIDTH_"+name+"_bin"+str(k+1)+".json",row)
            rows.append(row)
        tables[name] = {"N":float(len(x)),"below_logQ_range":int((b<0).sum()),
            "above_logQ_range":int((b>7).sum()),"PSD_below_minus0_1":int((x<-.1).sum()),
            "PSD_above0_6":int((x>.6).sum()),"slices":rows,"histogram_edges_PSD":grid.tolist()}
        ctx.write("results/WIDTH_"+name+"_TABLE.json",tables[name])
    errors = {}
    for label,names in [(ctx.source_arm+"_vs_BASE",["BASE",ctx.source_arm]),
                         ("SOURCE_and_BASE",["BASE",ctx.source_arm]),
                         ("all_new_and_BASE",["BASE",*arms])]:
        rows = []
        for k in [0,1]:
            for field in ["mean","sigma_PSD","detrended_sigma_PSD"]:
                bins = list(range(1,7))
                missing = {n:[i+1 for i in bins if not (
                    tables[n]["slices"][i]["valid"] and
                    len(tables[n]["slices"][i]["components"]) == 2 and
                    tables[n]["slices"][i]["components"][k][field] is not None)]
                    for n in ["REAL",*names]}
                rows.append({"component":k,"field":field,
                    "common_bin_ids_1based":[i+1 for i in bins],
                    "fixed_required_bins_not_candidate_intersection":True,
                    "failed_required_bins_by_domain":missing,
                    "MAE":{n:float(np.mean([abs(
                        tables[n]["slices"][i]["components"][k][field]-
                        tables["REAL"]["slices"][i]["components"][k][field]) for i in bins]))
                         if not missing["REAL"] and not missing[n] else None for n in names}})
        errors[label] = rows
    return {"tables":tables,"common_support_errors":errors,"edges_logQ":edges,
        "old_fits_reused":True,"all_slots_immediately_saved":True,
        "density_components_not_particle_truth":True}


def psd_metrics(ctx, domains):
    tables,values,private = {},{},{}
    for name,d in domains.items():
        p,q,valid = d["features"]["PSD"],d["features"]["Qtotal_mVns"],d["valid"]
        bins = sd.raw_bins(q,ctx.edges)
        rows,values[name] = [],[]
        for i in range(8):
            x = np.sort(p[valid & (bins==i)])
            values[name].append(x)
            definition = ctx.valley["bins"][i]
            rows.append({"bin_id_1based":i+1,"N":len(x),
                "fraction_of_QC":len(x)/int(valid.sum()) if valid.any() else None,
                "valley_fraction":float(np.mean((x>=definition["low"]) &
                    (x<=definition["high"]))) if len(x) and definition["valid"] else None,
                "valley_low":definition["low"],"valley_high":definition["high"]})
            private[name+"__bin"+str(i+1)+"__PSD"] = x
        tables[name] = {"source_n":len(p),"QC_n":int(valid.sum()),
            "outside_8bins":int(np.sum(valid & ((bins<0)|(bins>7)))),"bins":rows}
    comparisons = {}
    for name in domains:
        if name == "REAL": continue
        rows = []
        for i in range(8):
            a,r = values[name][i],values["REAL"][i]
            ok = len(a)>0 and len(r)>0
            ca,ra = tables[name]["bins"][i]["valley_fraction"],tables["REAL"]["bins"][i]["valley_fraction"]
            rows.append({"bin_id_1based":i+1,"candidate_N":len(a),"REAL_N":len(r),"defined":ok,
                "W1":float(wasserstein_distance(a,r)) if ok else None,
                "KS":float(ks_2samp(a,r).statistic) if ok else None,
                "valley_fraction_difference":ca-ra if ca is not None and ra is not None else None})
        comparisons[name] = {"bins":rows,"all_8_supported":all(r["defined"] for r in rows),
            "mean8_W1":float(np.mean([r["W1"] for r in rows]))
                        if all(r["defined"] for r in rows) else None}
    np.savez_compressed(ctx.results/"PSD_SECTIONS_PRIVATE.npz",**private)
    return {"tables":tables,"vs_REAL":comparisons,"edges_logQ":ctx.edges,
            "population_weights_changed":False,"old_frozen_valley_reused":True}


def waveform_moments(domains):
    descriptions,private = {},{}
    for name,d in domains.items():
        if name in ["SOURCE_INPUT","SOURCE_CLEAN"]: continue
        descriptions[name] = []
        for k in [None,*range(8)]:
            use = d["valid"] if k is None else d["valid"] & (d["b"]==k)
            real = domains["REAL"]
            target = real["valid"] if k is None else real["valid"] & (real["b"]==k)
            row = {"bin_id_1based":None if k is None else k+1,"N":int(use.sum()),
                "REAL_N":int(target.sum()),"qualified":int(use.sum())>=40 and int(target.sum())>=40}
            if use.any() and target.any():
                mv,rv = d["V"][use].mean(0),real["V"][target].mean(0)
                row["mean_full480_MSE_V2"] = float(np.mean((mv-rv)**2))
                private[name+"__"+str(k)+"__mean_V"] = mv
                if use.sum()>1 and target.sum()>1:
                    sv,rs = d["V"][use].std(0,ddof=1),real["V"][target].std(0,ddof=1)
                    row["SD_full480_RMS_difference_V"] = float(np.sqrt(np.mean((sv-rs)**2)))
                    private[name+"__"+str(k)+"__SD_V"] = sv
            descriptions[name].append(row)
    return descriptions,private
