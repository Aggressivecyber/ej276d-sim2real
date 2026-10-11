"""Saved v5 full-response evaluator. Import never reads arrays or performs a fit.

The root registers this file together with config_post.json and the sealed
generation/replay bundle. Actual execution requires its native owned supervisor.
No response model is imported or invoked by this evaluator.
"""
from __future__ import annotations
import os
import json
import time
import traceback
import platform
from pathlib import Path
import numpy as np
import sklearn
import scipy
import evaluation_primitives as ev
import diagnostic_primitives as dc
import assessment_primitives as aa
import source_diagnostics as sd
import frozen_statistics as fs
from evaluation_protocol import reference_paths, validate_post_config, generation_counters, CONTROLS
from evaluation_io import (
    load_config, preflight_training, owned_runtime, require, utc, sha_file, array_sha,
    exact, verify_file, read_json_bound, read_npz_bound, main_inputs, validate_main,
    identity_payload, check_payload, validate_aux, adc_representations, feature_Q_PSD, write_json,
)


class EvaluationContext:
    def __init__(self, cfg, config_path, adapter, config_binding, admission, resource):
        self.cfg, self.adapter = cfg, adapter
        self.root, self.results = Path(config_path).resolve().parent, Path(config_path).resolve().parent/"results"
        self.source_arm, self.real_arm = adapter["SOURCE_arm"], adapter["REAL_arm"]
        self.arms = [self.source_arm,self.real_arm]
        self.config_binding, self.admission, self.resource = config_binding, admission, resource
        self.start = time.monotonic()
        self.counter = {"HGB_attempts":0,"HGB_completed_slots":0,
            "density_description_slots":0,"density_completed_slots":0,"density_EM_attempts":0,
            "main_generation_rows":0,"repeat_noise_rows":0,"source_probe_rows":0,
            "extra_guide_probe_rows":0,"EVAL_read":0,"CAL_waveform_decode":0,
            "response_updates":0,"response_model_forward_rows_in_evaluation":0,
            "new_BASE_control_fits":0}
        self.references, self.actual_input_files = {},{}
        # Every old reference is prebound and verified before the first old read.
        for key,path in reference_paths(cfg).items():
            row = {"path":path,"sha256":cfg["reference_bindings"][path]}
            verify_file(row,"reference."+key)
            self.references[key] = row
            self.actual_input_files["reference."+key] = {**row,"verified_before_first_read":True}
        self.results.mkdir(parents=True,exist_ok=True)

    def write(self, name, value):
        write_json(self.root/name,ev.safe(value))

    def progress(self):
        self.write("EVALUATION_PROGRESS.json",{"PID":os.getpid(),"UTC":utc(),
            "counters":self.counter,"elapsed_s":time.monotonic()-self.start})

    def log(self, value):
        print(utc(),value,flush=True)

    def reference_json(self, key):
        return read_json_bound(self.references[key],"reference."+key)

    def reference_npz(self, key, keys=None):
        return read_npz_bound(self.references[key],"reference."+key,keys)

    def generated_json(self, name):
        p = Path(self.adapter["output_root"])/name
        self.actual_input_files["generated."+name] = {
            "path":str(p),"sha256":sha_file(p),
            "role":"new current-stage generation sidecar; not a frozen historical reference"}
        return json.loads(p.read_text(encoding="utf-8"))

    def generated_npz(self, generation, key, expected_path, rows):
        row = generation["output_files"][key]
        require(Path(row["path"]).resolve() == Path(expected_path).resolve() and
                row.get("rows") == rows, "current output location/population changed: "+key)
        payload = read_npz_bound(row,"current_output."+key)
        require(set(row.get("array_SHA256",{})) == set(payload),
                "current output array digest fields changed")
        for field,value in payload.items():
            require(array_sha(value) == row["array_SHA256"][field],
                    "saved current output field SHA mismatch: "+key+"."+field)
        check_payload(payload,rows)
        codes,canonical = adc_representations(payload["waveforms_V"])
        require(exact(codes,payload["ADC_integer_codes"]) and
                exact(canonical,payload["evaluation_canonical_V"]),
                "strict canonical code/representation mismatch: "+key)
        q,p = feature_Q_PSD(canonical)
        require(exact(q,payload["evaluation_canonical_Q_mVns"]) and
                exact(p,payload["evaluation_canonical_PSD"]) and
                exact(sd.raw_bins(q,self.edges),payload["evaluation_output_bin_raw"]),
                "saved canonical evaluation feature/condition mismatch: "+key)
        self.actual_input_files["generated."+key] = row
        return payload


def attach(d, edges):
    s = d["V"].astype("f8")-d["V"].astype("f8")[:,:100].mean(1,keepdims=True)
    p = np.maximum(s.max(1,keepdims=True),.0025)
    d.update(s=s,p=p,u=s/p,q=d["features"]["Qtotal_mVns"],
             b=sd.raw_bins(d["features"]["Qtotal_mVns"],edges))
    return d


def check_original_identity(payload, expected):
    for key,value in expected.items():
        require(key in payload and exact(payload[key],value),
                "original full identity/order/metadata/weight changed: "+key)


def source_probe_descriptions(payload, source_payload):
    """Recompute strict common canonical waveform/PSD descriptions from saved rows."""
    post = payload["evaluation_canonical_V"]
    reference = source_payload["evaluation_canonical_V"][payload["reference_parent_index"]]
    features,reference_features = ev.features(post),ev.features(reference)
    rms = np.sqrt(np.mean((post-reference)**2,axis=1))
    rows = {}
    for tag in np.unique(payload["probe_kind"]):
        use = payload["probe_kind"] == tag
        rows[str(tag)] = {"N":int(use.sum()),"QC":int(features["valid"][use].sum()),
            "waveform_RMS_vs_correct_reference_V":ev.summary(rms[use]),
            "PSD_vs_reference":ev.summary(features["features"]["PSD"][use]-
                                          reference_features["features"]["PSD"][use]),
            "output_PSD":ev.summary(features["features"]["PSD"][use]),
            "output_Q_mVns":ev.summary(features["features"]["Qtotal_mVns"][use])}
    same = payload["probe_kind"] == "same_input_same_noise"
    return {"rows":rows,"new_generated_rows":len(post),
        "same_input_same_noise_bitwise_observed":bool(np.array_equal(
            payload["waveforms_V"][same],
            source_payload["waveforms_V"][payload["source_parent_index"][same]])),
        "observed_layout_only_no_permutation_batch_invariance_claim":True,
        "corrected_parent_semantics_reused":True,"final_physical_signal_electronic_split":"unknown"}


def run(config_path):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    require(isinstance(cfg.get("evaluation_adapter_config"),str) and
            Path(cfg["evaluation_adapter_config"]).is_absolute(),
            "config_post requires explicit absolute sealed adapter config path")
    adapter,configuration = load_config(cfg["evaluation_adapter_config"])
    admission = preflight_training(adapter)  # JSON-only, before any NPZ read.
    validate_post_config(cfg,adapter,admission)
    resource = owned_runtime(adapter,"evaluate_saved.py",Path(__file__).resolve().parent)
    # Thread settings occur only after the registered native handshake, never on import.
    for key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"]:
        os.environ[key] = "12"
    ctx = EvaluationContext(cfg,config_path,adapter,configuration,admission,resource)
    require(not (ctx.root/"EVALUATION_STARTED.json").exists(),
            "evaluation already attempted; root must register a separate retry")
    ctx.write("EVALUATION_STARTED.json",{"UTC":utc(),"PID":os.getpid(),
        "post_config":{"path":str(config_path),"sha256":sha_file(config_path)},
        "adapter_config":configuration,"training_admission":admission,
        "preverified_reference_bindings":ctx.references,"resource_handshake":resource})
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=12):
            return evaluate(ctx)
    except BaseException as exc:
        ctx.write("EVALUATION_FAILURE.json",{"UTC":utc(),"PID":os.getpid(),
            "error":repr(exc),"traceback":traceback.format_exc(),
            "elapsed_s":time.monotonic()-ctx.start,"counters":ctx.counter,
            "counter_scope":"started HGB/EM attempts and completed description slots explicitly separated",
            "unfinished_fit_or_slot_completion":"unknown",
            "no_response_updates_or_model_forward_in_evaluation":True,
            "actual_input_files":ctx.actual_input_files,"candidate_A_to_F_pass":None})
        raise


def metrics_record(ctx, data_roles, stats, distance, local, psd, width, composition, association, historical, source_probes, joint, guide, projection, raw, identity, current_training, charge, legacy_adc_description, generation, c2st, boundary):
    """Pure assembly of measured descriptions; no fit, file, or gate call."""
    cfg = ctx.cfg
    return {
        "status":"complete","acceptance_version":5,"training_scope":cfg["recipe_description"],
        "new_training_runs_in_evaluation":0,"new_response_updates_in_evaluation":0,
        "current_training":current_training,"charge_diagnostics":charge,"REAL_identity":identity,
        "data_roles":data_roles,"domains":stats,"domain_vs_REAL":distance,"local":local,
        "PSD_main":psd,"energy_conditional_width":width,"source_composition":composition,
        "source_association_actual_preADC_input":association,
        "source_association_historical_postADC_reference":historical,
        "source_probes":source_probes,"random_decomposition":joint,"guide_probe":guide,
        "C2ST":c2st,"generation":generation["domains"],
        "boundary_differences":boundary,"explicit_constraint_projection":projection,
        "raw_voltage_moments":raw,"conditional_mean_SD":{"status":"not_requested; raw voltage moments provided"},
        "covariance_description":{"status":"current joint8 algebraic description only; no new global covariance"},
        "counters":dict(ctx.counter),"primary_DEV_population_reweighted":False,
        "EVAL_read":0,"old_reference_fits_and_RAW_RR_reused":True,
        "one_new_same_treated_REAL_control":True,
        "canonical_ADC_features_before_all_new_statistics":True,
        "raw_SOURCE_INPUT_charge_and_PSD_preserved":True,
        "RAW_RR_same_acquisition_reference_reused":True,
        "final_physical_signal_electronic_split":"unknown",
        "historical_SOURCE_CLEAN_bin_definition":"original float32 postADC charge207/182 legacy denominators; primary raw SOURCE213/181",
        "legacy_source_ADC_description":legacy_adc_description,
        "Qstar_is_scientific_target_inputQ_no_longer_conserved":True,
        "SOURCE_and_REAL_roles_never_interchanged":True,
        "all_charge_CDF_and_migrations_use_full_original_mass_not_QC_subset":True,
        "configuration_binding":ctx.config_binding,
        "actual_input_files":ctx.actual_input_files,
        "versions":{"Python":platform.python_version(),"NumPy":np.__version__,
                    "SciPy":scipy.__version__,"sklearn":sklearn.__version__}}


def evaluate(ctx):
    cfg,adapter = ctx.cfg,ctx.adapter
    old,ref = ctx.reference_json("frozen_width_metrics"),ctx.reference_json("CAL_metadata")
    ctx.valley = ctx.reference_json("valley")
    ctx.edges = np.asarray(ctx.reference_json("width_definition")["edges_logQ"],dtype=np.float64)
    require(ctx.edges.shape == (9,) and np.isfinite(ctx.edges).all() and
            (np.diff(ctx.edges)>0).all() and len(ctx.valley["bins"]) == 8,
            "frozen width/valley conditions changed")
    data_roles = ctx.reference_json("data_roles")
    actual,metadata,reference = main_inputs(adapter)
    for key in ["actual","source_metadata","REAL_packet"]:
        ctx.actual_input_files["adapter."+key] = adapter["inputs"][key]
    axes = validate_main(actual,metadata,reference,ctx.edges)
    real = read_npz_bound(adapter["inputs"]["REAL_packet"],"untreated_REAL_DEV",
                         ["real_V","real_ids","real_groups","real_QC_valid"])
    require(real["real_V"].shape == (2000,480) and len(real["real_ids"]) == 2000 and
            len(real["real_groups"]) == 2000 and len(np.unique(real["real_ids"])) == 2000,
            "untreated_REAL_DEV complete2000 population changed")
    require(real["real_QC_valid"].shape == (2000,) and
            np.isin(real["real_QC_valid"],[0,1]).all(),"frozen REAL QC flags invalid")
    ids,source = actual["source_ids"],actual["source_waveforms_V"]
    comp,pdg = metadata["composition"],metadata["pdg_inferred"]
    source_adc,legacy_adc_description = sd.legacy_source_adc(source)
    domains = {
        "REAL":aa.data(real["real_V"],real["real_ids"],real["real_groups"],"REAL",ref),
        "SOURCE_INPUT":aa.data(source,ids,np.arange(2000),"G4_SOURCE",ref,adc_domain=False),
        "SOURCE_CLEAN":aa.data(source_adc,ids,np.arange(2000),"G4_SOURCE",ref)}
    domains["REAL"]["valid"] &= real["real_QC_valid"].astype(np.bool_)
    domains["TRAIN_REFERENCE"] = aa.data(reference["reference_V"],reference["reference_ids"],
                                        reference["reference_groups"],"TRAIN_REFERENCE",ref)
    for name,key in [("BASE","BASE_output"),("INDEPENDENT_CFM","INDEPENDENT_CFM_output")]:
        a = ctx.reference_npz(key,["waveforms_V","source_ids"])
        require(exact(a["source_ids"],ids),"frozen control source IDs/order changed")
        domains[name] = aa.data(a["waveforms_V"],ids,np.arange(2000),"G4_SOURCE",ref)
    a = ctx.reference_npz("BASE_REAL_output",["waveforms_V","reference_ids","reference_groups"])
    require(exact(a["reference_ids"],reference["reference_ids"]) and
            exact(a["reference_groups"],reference["reference_groups"]),
            "frozen BASE_REAL input IDs/groups/order changed")
    domains["BASE_REAL_CONTROL"] = aa.data(a["waveforms_V"],reference["reference_ids"],
                                            reference["reference_groups"],"TRAIN_REFERENCE",ref)
    generation = ctx.generated_json("GENERATION_RECEIPT.json")
    require(generation.get("configuration_binding") == ctx.config_binding and
            generation.get("input_binding") == adapter["inputs"] and
            generation.get("training_admission",{}).get("completed_updates") == ctx.admission["completed_updates"],
            "generation not from this exact bound completed endpoint/input/config")
    ctx.counter.update(generation_counters(generation,ctx.source_arm,ctx.real_arm))
    output_root = Path(adapter["output_root"])
    source_payload = ctx.generated_npz(generation,ctx.source_arm,
                                      output_root/ctx.source_arm/"generated.npz",2000)
    real_payload = ctx.generated_npz(generation,ctx.real_arm,
                                    output_root/ctx.real_arm/"generated.npz",2000)
    check_original_identity(source_payload,identity_payload("SOURCE",actual,metadata,reference,axes))
    check_original_identity(real_payload,identity_payload("REAL",actual,metadata,reference,axes))
    probes = ctx.generated_npz(generation,"probes",
        output_root/"results"/(ctx.source_arm+"_SOURCE_PROBES_PRIVATE.npz"),832)
    original_probes = read_npz_bound(adapter["inputs"]["probes"],"original_probe_inputs")
    validate_aux(original_probes,832,actual,"probes")
    probe_parent_keys = ["source_parent_index","source_parent_ids","noise_origin_parent_index",
                         "reference_parent_index","source_waveform_modified_counterfactual","probe_kind"]
    check_original_identity(probes,{**{k:original_probes[k] for k in probe_parent_keys},
        "actual_input_V":original_probes["source_waveforms_V"],
        "actual_white":original_probes["white"],"actual_z":original_probes["z"],
        "actual_style_unused":actual["style"][original_probes["noise_origin_parent_index"]],
        "original_parent_weight":np.full(832,1/2000.,dtype=np.float64)})
    joint_payload = ctx.generated_npz(generation,"joint8",
        output_root/"results"/(ctx.source_arm+"_joint8_REPEAT_PRIVATE.npz"),1024)
    original_joint = read_npz_bound(adapter["inputs"]["joint8"],"original_joint_inputs")
    validate_aux(original_joint,1024,actual,"joint8")
    check_original_identity(joint_payload,{**original_joint,
        "actual_input_V":original_joint["source_waveforms_V"],
        "original_parent_weight":np.full(1024,1/2000.,dtype=np.float64),
        "within_parent_draw_weight":np.full(1024,1/8.,dtype=np.float64)})
    guide_payload = ctx.generated_npz(generation,"guide",
        output_root/"results"/(ctx.source_arm+"_GUIDE_PERTURBATION_PRIVATE.npz"),128)
    pick = axes["guide_pick"]
    check_original_identity(guide_payload,{
        "parent_index":pick,"source_ids":ids[pick],"original_guide_V":source[pick],
        "original_output_preADC_V":source_payload["pre_ADC_V"][pick],
        "fixed_source_Qbin":axes["support_bin"][pick],
        "actual_white":actual["white"][pick],"actual_z":actual["z"][pick],
        "actual_style_unused":actual["style"][pick],
        "original_parent_weight":np.full(128,1/2000.,dtype=np.float64)})
    for key in ["probes","joint8"]:
        ctx.actual_input_files["adapter."+key] = adapter["inputs"][key]
    domains[ctx.source_arm] = aa.data(source_payload["evaluation_canonical_V"],ids,np.arange(2000),
                                    "G4_SOURCE",ref,adc_domain=False)
    domains[ctx.real_arm] = aa.data(real_payload["evaluation_canonical_V"],reference["reference_ids"],
                                  reference["reference_groups"],"TRAIN_REFERENCE",ref,adc_domain=False)
    charge = sd.charge_diagnostics(
        {ctx.source_arm:source_payload,ctx.real_arm:real_payload},
        {ctx.source_arm:source,ctx.real_arm:reference["reference_V"]},
        domains["REAL"]["V"],ctx.edges,
        {ctx.source_arm:"original_SOURCE2000",ctx.real_arm:"same_treated_TRAIN_REFERENCE2000"})
    ctx.write("CHARGE_DISTRIBUTION_DIAGNOSTICS.json",charge)
    # Complete target/source mass is retained in the diagnostic even on failure.
    require(charge["REAL_reference"]["invalid_count"] == 0,
            "untreated full REAL target contains illegalQ; no deletion/renormalization")
    for arm in ctx.arms:
        require(all(charge["domains"][arm]["axes"][axis]["invalid_count"] == 0
                    for axis in charge["required_axes"]),
                "full candidate charge axis illegal; retained in diagnostic, no deletion")

    actual_bins = sd.raw_bins(domains["SOURCE_INPUT"]["features"]["Qtotal_mVns"],ctx.edges)
    # Historical mother bin207/182 uses the original float32 ADC BEFORE common
    # canonical evaluation; primary213/181 uses the original unquantized SOURCE.
    reference_bins = sd.raw_bins(ev.features(source_adc)["features"]["Qtotal_mVns"],ctx.edges)
    require([int(np.sum((comp=="pure_had") & (actual_bins==b))) for b in [2,3]] == [213,181],
            "original SOURCE_INPUT mother213/181 changed")
    require([int(np.sum((comp=="pure_had") & (reference_bins==b))) for b in [2,3]] == [207,182],
            "historical postADC mother207/182 changed")
    stats,distance = ev.characterize(domains,ref)
    psd = fs.psd_metrics(ctx,domains)
    for d in domains.values():
        attach(d,ctx.edges)
    c2st = fs.c2st(ctx,domains)
    width = fs.widths(ctx,domains,old)
    source_domains = {name:domains[name] for name in
                      ["REAL","SOURCE_CLEAN","SOURCE_INPUT",*CONTROLS,ctx.source_arm]}
    composition = sd.source_tables(source_domains,actual_bins,reference_bins,comp,ctx.valley,ctx.edges)
    association = {name:sd.association(
        {"SOURCE_CLEAN":domains["SOURCE_INPUT"],"TRANSLATED":domains[name]},comp,pdg,ctx.edges)
        for name in [*CONTROLS,ctx.source_arm]}
    historical = {name:sd.association(
        {"SOURCE_CLEAN":domains["SOURCE_CLEAN"],"TRANSLATED":domains[name]},comp,pdg,ctx.edges)
        for name in [*CONTROLS,ctx.source_arm]}
    local = {name:dc.native_statistics(d,ref) for name,d in domains.items()}
    feature_packet = {}
    for name,d in domains.items():
        feature_packet.update({name+"__"+k:v for k,v in {**d["features"],**d["extra"]}.items()})
        feature_packet[name+"__valid"],feature_packet[name+"__region"] = d["valid"],d["region"]
    np.savez_compressed(ctx.results/"FEATURES_PRIVATE.npz",**feature_packet)
    raw,raw_arrays = fs.waveform_moments(domains)
    np.savez_compressed(ctx.results/"RAW_VOLTAGE_MOMENTS_PRIVATE.npz",**raw_arrays)
    boundary = sd.boundary_metrics(domains,ctx.results)
    ctx.write("BOUNDARY_METRICS.json",boundary)

    source_probes = {ctx.source_arm:source_probe_descriptions(probes,source_payload)}
    ctx.write("SOURCE_PROBES_CANONICAL_EVALUATION_SUMMARY.json",source_probes)
    # These are current generation descriptions; no hidden prior model executes.
    joint = ctx.generated_json("JOINT8_SUMMARY.json")
    guide = ctx.generated_json("GUIDE_PERTURBATION_COMPLETE_SUMMARY.json")
    projection = ctx.generated_json("CONSTRAINT_PROJECTION_SUMMARY.json")
    identity = {ctx.real_arm:sd.REAL_identity(reference["reference_V"],real_payload)}
    training_receipt = read_json_bound(adapter["training"]["receipt"],"completed_current_training")
    numerical_reference = read_json_bound(adapter["training"]["numerical_reference"],"current_numeric_reference")
    current_training = {
        "config_binding":adapter["training"]["config"],"receipt_binding":adapter["training"]["receipt"],
        "completed_updates":ctx.admission["completed_updates"],
        "unique_selected_endpoint":training_receipt["unique_selected_endpoint"],
        "complete_parent_per_update":training_receipt["complete_parent_per_update"],
        "SOURCE_N":6593,"REAL_N":21047,"full_parameter_VJP_rows":training_receipt["full_parameter_VJP_rows"],
        "SOURCE_main_exposures":training_receipt["SOURCE_main_exposures"],
        "REAL_main_exposures":training_receipt["REAL_main_exposures"],
        "numeric_admission":ctx.admission,"numerical_reference":numerical_reference,
        "frozen_Q_component_and_full_shape_final_model":adapter["model"],
        "neural_training_replay_rows":training_receipt.get("neural_replay_rows","unknown"),
        "retained_first_forward_graph":training_receipt.get("retained_first_forward_graph","unknown"),
        "heldout_replay_is_separate4000":True,"new_updates_in_evaluation":0,
        "training_receipt":training_receipt}
    metrics = metrics_record(ctx, data_roles, stats, distance, local, psd, width, composition, association, historical, source_probes, joint, guide, projection, raw, identity, current_training, charge, legacy_adc_description, generation, c2st, boundary)
    require(ctx.counter["HGB_attempts"] == ctx.counter["HGB_completed_slots"] == 10 and
            ctx.counter["density_description_slots"] == ctx.counter["density_completed_slots"] == 16 and
            ctx.counter["density_EM_attempts"] <= 16,
            "original tenHGB/sixteen density slots not completed")
    ctx.write("metrics.json",metrics)
    receipt = {"PID":os.getpid(),"UTC":utc(),"status":"complete",
        "elapsed_s":time.monotonic()-ctx.start,"counters":ctx.counter,
        "training_admission":ctx.admission,"resource_handshake":ctx.resource,
        "actual_input_files":ctx.actual_input_files,
        "reference_bindings_verified_before_old_read":ctx.references,
        "metrics_file":{"path":str(ctx.root/"metrics.json"),"sha256":sha_file(ctx.root/"metrics.json")},
        "evaluation_response_forward_rows":0,"evaluation_response_updates":0,
        "old_control_new_fits":0,"A_to_F_gate_aggregation":"root only; not computed here"}
    ctx.write("EVALUATION_NUMERIC_RECEIPT.json",receipt)
    ctx.log("saved v5 full response evaluation complete "+json.dumps(ctx.counter))
    return metrics


def main():
    import argparse
    argv = None
    # Existing owned worker uses runpy and leaves the entry token.
    import sys
    if os.environ.get("OWNED_SCIENCE_NONCE") and sys.argv[1:] == ["evaluate_saved.py"]:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=Path(__file__).resolve().parent/"config_post.json")
    run(parser.parse_args(argv).config)


if __name__ == "__main__":
    main()
