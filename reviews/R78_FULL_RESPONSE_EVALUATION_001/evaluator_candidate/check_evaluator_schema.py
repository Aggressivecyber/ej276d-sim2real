"""Artificial arrays/metadata only; no PT/NPZ read, model invocation or actual fit."""
from __future__ import annotations
import ast
import contextlib
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.stats import wasserstein_distance
import source_diagnostics as sd
import evaluation_protocol as protocol
import evaluation_primitives as ev
import diagnostic_primitives as dc
import width_primitives as ew
import frozen_statistics as fs
import evaluate_saved as entry
from evaluation_io import AdmissionError, q_window, feature_Q_PSD, summary, adc_representations


def run_checks(directory):
    directory = Path(directory)
    checks = {}
    def check(name,value):
        assert bool(value),name
        checks[name] = True
    def rejected(name,fn):
        try:
            fn()
        except (AdmissionError,AssertionError,KeyError):
            checks[name] = True
        else:
            raise AssertionError(name+": expected explicit failure")
    def prohibited(*args,**kwargs):
        raise AssertionError("scientific read/actual fit forbidden in artificial suite")
    original_load = np.load
    np.load = prohibited
    try:
        # No import in this bundle can instantiate a neural model.
        import sys
        check("Torch_not_imported","torch" not in sys.modules)
        for path in directory.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            check("AST_"+path.name,tree is not None)
        derivation = json.loads((directory/"PRIMITIVE_DERIVATION.json").read_text(encoding="utf-8"))
        for name,record in derivation.items():
            original = ast.parse(Path(record["original_path"]).read_text(encoding="utf-8"))
            current = ast.parse((directory/name).read_text(encoding="utf-8"))
            a = {n.name:n for n in original.body if isinstance(n,ast.FunctionDef)}
            b = {n.name:n for n in current.body if isinstance(n,ast.FunctionDef)}
            for function in record["functions"]:
                left = copy.deepcopy(a[function])
                for node in ast.walk(left):
                    if isinstance(node,ast.ImportFrom) and node.module == "common_math":
                        node.module = "common_math_primitives"
                check("original_AST_"+name+"_"+function,
                      ast.dump(left,include_attributes=False) == ast.dump(b[function],include_attributes=False))
        edges = np.log(np.array([100,160,256,409.6,655.36,1048.576,1677.7216,2684.35456,4294.967296]))
        q = np.array([80.,121.,190.,291.,500.,750.,1100.,1800.,2700.,4400.,9000.,150.])
        weights = sd.original_uniform_weights(len(q))
        axis,bins = sd.axis_description(q,edges,weights,q[::-1],weights)
        check("axis_full_mass_11list",len(axis["counts"]) == len(axis["weights"]) == 11 and
              sum(axis["counts"]) == len(q) and abs(sum(axis["weights"])-1) < 1e-12)
        check("CDF_original_full_mass",abs(axis["empirical_CDF"]["cumulative_original_weight"][-1]-1)<1e-12)
        check("outside_mass_not_clipped",axis["outside_support_count"] == 3 and
              abs(axis["outside_support_weight"]-3/len(q)) < 1e-12)
        edge_q = np.exp(edges[-1])
        check("output_rightmost_edge_included",sd.raw_bins(np.array([edge_q]),edges)[0] == 7)
        bad = np.array([np.nan,-1.,0.,np.inf,90.,9000.])
        bad_axis,bad_bins = sd.axis_description(bad,edges,sd.original_uniform_weights(len(bad)))
        check("invalid_distinct_from_above",bad_axis["invalid_count"] == 4 and
              bad_axis["outside_support_count"] == 2 and np.all(bad_bins[:4] == -2))
        check("invalid_CDF_unknown_no_deleted_normalization",
              bad_axis["empirical_CDF"]["defined"] is False and
              bad_axis["empirical_CDF"]["unique_Q_mVns"] == [] and
              abs(sum(bad_axis["weights"])-1)<1e-12)
        for a,b in [
            (np.array([101.,160.,160.,910.]),np.array([110.,110.,225.,650.,940.,2100.,4500.])),
            (np.array([200.,200.,200.]),np.array([180.,220.])),
        ]:
            ax,_ = sd.axis_description(a,edges,sd.original_uniform_weights(len(a)),
                                      b,sd.original_uniform_weights(len(b)))
            check("unequalN_W1_"+str(len(a))+"_"+str(len(b)),
                  abs(ax["vs_REAL"]["W1"]-wasserstein_distance(a,b)) < 1e-12)
        m = sd.migration(bad_bins,bad_bins[::-1],sd.original_uniform_weights(len(bad)))
        check("migration11x11_allmass",np.shape(m["count_matrix"]) == (11,11) and
              m["count_total"] == len(bad) and abs(m["weight_total"]-1)<1e-12 and
              np.sum(np.array(m["count_matrix"])[10]) == 4)
        tamper = weights.copy();tamper[0] = 0.
        rejected("reweight_is_rejected",lambda:sd.axis_description(q,edges,tamper))
        rejected("wrong_weight_precision_is_rejected",
                 lambda:sd.axis_description(q,edges,weights.astype(np.float32)))

        def waves(charge):
            charge = np.asarray(charge,dtype=np.float64)
            v = np.zeros((len(charge),480),dtype=np.float64)
            v[:,:100] = .0002*np.sin(np.arange(100)*2*np.pi/100)[None,:]
            v[:,100:150] = .7*charge[:,None]/50000.
            v[:,150:400] = .3*charge[:,None]/250000.
            return v
        raw,pre = waves(q),waves(q*1.07).astype(np.float32)
        codes,strict = adc_representations(pre.astype(np.float64))
        targetq = q_window(pre)
        payload = {"pre_ADC_V":pre,"waveforms_V":strict,"evaluation_canonical_V":strict,
                   "predicted_target_Q_Vns":targetq,"Q0":q_window(raw),"QADC":q_window(strict),
                   "original_weight":weights}
        charge = sd.charge_diagnostics({"S":payload,"R":payload},{"S":raw,"R":raw},
                                       strict,edges,{"S":"SOURCE","R":"TRAIN_REFERENCE"})
        check("four_charge_axes_and_three_migrations",
              all(set(charge["domains"][a]["migrations"]) == set(sd.MIGRATIONS) and
                  all(k in charge["domains"][a]["axes"] for k in charge["required_axes"])
                  for a in ["S","R"]))
        check("canonical_Q_recomputed_not_pipeline_Q",
              charge["domains"]["S"]["axes"]["canonical_evaluation"]["empirical_CDF"]["unique_Q_mVns"] ==
              np.unique(feature_Q_PSD(strict)[0]).tolist())
        check("REAL_reference_top_CDF_units",
              charge["REAL_reference"]["units"] == "mV ns" and
              charge["REAL_reference"]["empirical_CDF"] ==
              charge["REAL_reference"]["axes"]["canonical_evaluation"]["empirical_CDF"])
        for arm in ["S","R"]:
            check("three_migrations_fullmass_"+arm,all(
                row["count_total"] == len(q) and abs(row["weight_total"]-1) < 1e-12
                for row in charge["domains"][arm]["migrations"].values()))
        illegal_real = strict.copy();illegal_real[0] = strict[0,:100].mean()
        illegal = sd.charge_diagnostics({"S":payload},{"S":raw},illegal_real,edges)
        check("illegal_REAL_retained_noCDF_or_W1",
              illegal["REAL_reference"]["rows"] == len(q) and
              illegal["REAL_reference"]["invalid_count"] == 1 and
              illegal["REAL_reference"]["empirical_CDF"]["defined"] is False and
              illegal["domains"]["S"]["axes"]["canonical_evaluation"]["vs_REAL"]["W1"] is None)
        identity = sd.REAL_identity(raw,payload)
        check("REAL_identity_original_pairs",
              identity["rows"] == len(q) and identity["paired_original_reference_rows"] and
              identity["preADC_RMS_difference_V"]["mean"] > 0)

        def domain(v):
            d = ev.features(v)
            return entry.attach(d,edges)
        s,out = domain(raw),domain(strict)
        comp = np.array(["pure_had"]*8+["pure_EE"]*2+["mixed"]*2)
        pdg = np.array([2112]*8+[22]*2+[0]*2)
        mother = np.array([2,2,2,3,3,3,0,1,4,5,6,7])
        historical_mother = np.roll(mother,1)
        valley = {"bins":[{"valid":True,"low":.2,"high":.4} for _ in range(8)]}
        tables = sd.source_tables({"SOURCE_INPUT":s,"CANDIDATE":out},mother,historical_mother,comp,valley,edges)
        check("source_mother_denominators_not_output_bins",
              tables["primary_pure_had_actual_input_source_Q_bins3_4"]["CANDIDATE"]["bin3"]["source_count"] == 3 and
              tables["primary_pure_had_actual_input_source_Q_bins3_4"]["CANDIDATE"]["bin4"]["source_count"] == 3)
        association = sd.association({"SOURCE_CLEAN":s,"TRANSLATED":out},comp,pdg,edges)
        check("association_original_mass_and_parent_labels",
              np.shape(association["migration_source_8bin_rows_to_output_8bin_columns_with_low_high_invalid"]) == (11,11) and
              np.sum(association["migration_source_8bin_rows_to_output_8bin_columns_with_low_high_invalid"]) == len(q) and
              association["composition_and_PDG"]["all"]["source_n"] == len(q))
        moved = out["b"] != s["b"]
        check("canonical_condition_migration_visible",bool(moved.any()))
        moments,private = fs.waveform_moments({"REAL":s,"CANDIDATE":out})
        check("full480_moments_and_current_bins",
              len(moments["CANDIDATE"]) == 9 and moments["CANDIDATE"][0]["N"] == len(q) and
              all(row["N"] == int((out["valid"] & (out["b"]==k)).sum())
                  for k,row in enumerate(moments["CANDIDATE"][1:])) and
              all(v.shape == (480,) for v in private.values()))
        boundaries = sd.boundary_metrics({"REAL":s,"CANDIDATE":out})
        check("original9_boundary_locations",len(boundaries["results"]["CANDIDATE"]) == 9)

        # Density and classifier routines below are explicit fakes; actual fit is prohibited.
        original_save,original_fit,original_posterior = np.savez_compressed,ew.fit_slice,ew.posterior
        saved = []
        np.savez_compressed = lambda *args,**kwargs:saved.append((args,kwargs))
        ew.posterior = prohibited
        components = [{"mean":.1,"sigma_PSD":.01,"detrended_sigma_PSD":.009},
                      {"mean":.3,"sigma_PSD":.02,"detrended_sigma_PSD":.019}]
        def fake_density(x,z,w):
            finite = np.isfinite(x)&np.isfinite(z)&np.isfinite(w)
            n = int(finite.sum())
            return {"fit_attempts":int(n>=40),"valid":n>=40,
                    "components":copy.deepcopy(components) if n>=40 else [],
                    "reason":None if n>=40 else "synthetic_support_below40"}
        ew.fit_slice = fake_density
        class FakeContext:
            def __init__(self):
                self.cfg = {"C2ST":protocol.HGB,"C2ST_views":protocol.VIEWS,
                            "bootstrap_repeats":500,"bootstrap_seed":2026100871,"HGB_max_fits":10}
                self.source_arm,self.real_arm,self.arms = "S","R",["S","R"]
                self.counter = {"HGB_attempts":0,"HGB_completed_slots":0,
                                "density_EM_attempts":0,"density_description_slots":0,"density_completed_slots":0}
                self.edges,self.valley,self.results = edges,valley,directory/"NOT_WRITTEN"
                self.references = {"frozen_C2ST_metrics":{"path":"artificial_R51_metadata"}}
                self.outputs = {}
            def write(self,n,v):self.outputs[n] = copy.deepcopy(v)
            def progress(self):pass
            def log(self,v):pass
        density_ctx = FakeContext()
        qmids = np.exp((edges[:-1]+edges[1:])/2)
        d = {"valid":np.ones(320,dtype=np.bool_),
             "features":{"Qtotal_mVns":np.repeat(qmids,40),"PSD":np.tile(np.linspace(.1,.3,40),8)}}
        old = {"energy_conditional_width":{"tables":{n:{"slices":[{"valid":True,"components":copy.deepcopy(components)} for _ in range(8)]}
                  for n in ["REAL","SOURCE_CLEAN","BASE_REAL_CONTROL",*protocol.CONTROLS]}}}
        width = fs.widths(density_ctx,{"S":d,"R":d},old)
        check("density16_slots_no_scan_or_actual_fit",
              density_ctx.counter["density_description_slots"] == density_ctx.counter["density_completed_slots"] == 16 and
              density_ctx.counter["density_EM_attempts"] == 16)
        check("density_fixed_Q2_to_Q7_required",
              all(row["common_bin_ids_1based"] == [2,3,4,5,6,7] and
                  row["fixed_required_bins_not_candidate_intersection"]
                  for row in width["common_support_errors"]["all_new_and_BASE"]))
        missing_ctx = FakeContext()
        missing = copy.deepcopy(d);missing["valid"][160:200] = False
        missing_width = fs.widths(missing_ctx,{"S":missing,"R":d},old)
        check("missing_required_density_null_not_zero",
              all(row["MAE"]["S"] is None and row["failed_required_bins_by_domain"]["S"] == [5]
                  for row in missing_width["common_support_errors"]["all_new_and_BASE"]))

        selections = {f"phase{p}_{j}":np.arange(0,600,dtype=np.int64) if p==0
                      else np.arange(600,850,dtype=np.int64) for p in [0,1] for j in range(3)}
        selected = protocol.validate_selection(selections)
        invalid_selection = copy.deepcopy(selections);invalid_selection["phase1_0"][0] = 0
        rejected("original_selection_no_resplit_overlap",lambda:protocol.validate_selection(invalid_selection))
        big = domain(waves(np.full(2000,700.)))
        def grouped(d,prefix):
            r = dict(d);r["groups"] = np.array([prefix+str(i) for i in range(2000)])
            r["cluster"] = np.array(["G4_SOURCE|"+str(i) for i in range(2000)])
            return r
        source_d = grouped(big,"source_")
        # SOURCE grouping must match original G4_SOURCE integer parent semantics.
        source_d["groups"] = np.arange(2000)
        real_d,treated_d = grouped(big,"dev_"),grouped(big,"reference_")
        domains = {"S":source_d,"REAL":real_d,"R":treated_d}
        hgb_ctx = FakeContext()
        hp = {}
        labels = np.r_[np.ones(250),np.zeros(250)]
        probabilities = .45+.2*labels
        real_clusters = np.array(["R|"+str(g) for g in real_d["groups"][selected[1][1]]])
        source_clusters = source_d["cluster"][selected[1][0]]
        treated_clusters = np.array(["R|"+str(g) for g in treated_d["groups"][selected[1][2]]])
        for alias,clusters in [("BASE_SOURCE",source_clusters),("BASE_REAL",treated_clusters),
                               ("RAW_REFERENCE",treated_clusters)]:
            for view in protocol.VIEWS:
                tag = alias+"__"+view+"__"
                hp.update({tag+"probability":probabilities,tag+"label":labels,
                           tag+"cluster":np.r_[clusters,real_clusters]})
        hist = {"results":{alias+"__"+view:{"BA":1.,"BA_95CI":[1.,1.]}
                          for alias in protocol.CONTROL_ALIASES.values() for view in protocol.VIEWS}}
        hgb_ctx.reference_npz = lambda name,keys=None:selections if name=="fixed_selection" else hp
        hgb_ctx.reference_json = lambda name:hist
        old_fit,old_constructor = dc.fit_one,dc.HistGradientBoostingClassifier
        dc.HistGradientBoostingClassifier = prohibited
        observed_views = []
        def fake_fit(candidate,real,selection_candidate,selection_real,view,cfg,counter):
            tr,te = selection_candidate;rt,re = selection_real
            assert len(tr)==len(rt)==600 and len(te)==len(re)==250
            a,b = dc.matrix(candidate,view),dc.matrix(real,view)
            observed_views.append((view,a.shape[1]))
            counter["HGB_attempts"] += 1
            p = {"probability":probabilities,"label":labels,
                 "cluster":np.r_[candidate["cluster"][te],real["cluster"][re]]}
            return {"balanced_accuracy":1.,"heldout_CI":{"balanced_accuracy_95CI":[1.,1.]}},p
        dc.fit_one = fake_fit
        try:
            cs = fs.c2st(hgb_ctx,domains)
        finally:
            dc.fit_one,dc.HistGradientBoostingClassifier = old_fit,old_constructor
        check("ten_fakeHGB_calls_original_views_sizes",
              len(observed_views)==10 and hgb_ctx.counter["HGB_completed_slots"]==10 and
              all(n==2 for v,n in observed_views if v=="Q_PSD_2D") and
              all(n==480 for v,n in observed_views if v in ["normalized_u480","full_voltage"]))
        check("BASE_RAW_and_same_treated_pair_contract",
              "R" in cs["same_treated_REAL_control_paired"] and
              cs["paired"]["S"]["Q_PSD_2D"]["reference"] == "BASE" and
              cs["paired"]["R"]["Q_PSD_2D"]["reference"] == "RAW_REAL_REAL" and
              cs["same_treated_REAL_control_paired"]["R"]["Q_PSD_2D"]["BA_delta_first_minus_second"] == 0)
        np.savez_compressed,ew.fit_slice,ew.posterior = original_save,original_fit,original_posterior

        # Pure current-generation description functions are extracted in memory.
        adapter_dir = Path(entry.__file__).resolve().parent.parent.parent/"R74_EXACT_LOSS_FULL_TRAIN_001"/"evaluation_candidate"
        # The caller supplies the sealed bundle through sys.path. Resolve it from its module.
        import evaluation_io
        generator = Path(evaluation_io.__file__).with_name("generate_full_response.py")
        text = generator.read_text(encoding="utf-8")
        namespace = {"np":np,"summary":summary}
        tree = ast.parse(text)
        for name in ["describe_rows","joint_description"]:
            node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(generator),"exec"),namespace)
        names = ["input_variance100_V2","lifted_baseline_variance","noise_amplitude_squared",
                 "noise_kappa2","innovation_variance100_actual_V2","initial_variance100_actual_V2",
                 "Q_relative_error_f64","Q_relative_error_f32","projection_RMS_V",
                 "constraint_projection_maxabs_V","constraint_projection_rail_sample_count",
                 "pre_constraint_out_of_range_sample_count","pre_constraint_charge_relative_error"]
        description = {name:np.linspace(0,1e-5,len(q)) for name in names}
        description.update(Qstar=targetq,range_projection_active=np.zeros(len(q),dtype=np.bool_),
                           noise_completion_triggered=np.ones(len(q),dtype=np.bool_))
        projection = {"S":namespace["describe_rows"](description,sd.raw_bins(q,edges))}
        check("projection_actual_perrow_fields_10bins",
              len(projection["S"]["by_original_bin"]) == 10 and
              sum(r["N"] for r in projection["S"]["by_original_bin"]) == len(q) and
              projection["S"]["fields"]["Q_relative_error_f64"]["N"] == len(q))
        joint_source = np.broadcast_to(raw[0],(1024,480)).copy()
        innovation = np.arange(1024)[:,None]%8 * np.sin(np.arange(480))[None,:]*1e-6
        jp = {"source_parent_index":np.repeat(np.arange(128),8),
              "source_waveforms_V":joint_source}
        jo = {"projected_colored_innovation_V":innovation,
              "pre_ADC_V":(joint_source+innovation).astype(np.float32)}
        joint = {"S":{"groups":{"joint8":namespace["joint_description"](jp,jo)}}}
        check("joint_description_algebraic_unknown_physics",
              joint["S"]["groups"]["joint8"]["actual_rows"]==1024 and
              joint["S"]["groups"]["joint8"]["physical_final_signal_electronic_split"]=="unknown")

        guide = {"S":{"rows":128,"output_nonidentical_fraction":1.,
                       "preADC_output_RMS_V":{"N":128,"mean":1e-4,"median":1e-4}}}
        generation = {"domains":{"S":{"all_source_ID_order_weights_and_composition_preserved":True,
                                     "BASE_fallback_N":0,"full_response_applied_N":2000},
                                 "R":{"reference_ID_order_groups_preserved":True}},
                      "status":"complete5984_generation_only","all_generation_rows":5984,
                      "counter":{"source_main_rows":2000,"REAL_control_rows":2000,
                        "repeat_noise_rows":1024,"source_probe_rows":832,"extra_guide_probe_rows":128,
                        "response_fit_or_update":0,"HGB_attempts":0,"density_EM_attempts":0,"final_EVAL_read":0}}
        check("generation_SOURCE_plus_REAL4000",protocol.generation_counters(generation,"S","R")["main_generation_rows"]==4000)
        bad_generation = copy.deepcopy(generation);bad_generation["counter"]["REAL_control_rows"] = 0
        rejected("generation_cannot_credit_partial_REAL",lambda:protocol.generation_counters(bad_generation,"S","R"))
        current_training = {"completed_updates":25000,"full_parameter_VJP_rows":27640*25000,
                            "neural_training_replay_rows":0,"retained_first_forward_graph":True}
        kwargs = {"data_roles":{"DEV_composition":{"mixed":65,"pure_EE":417,"pure_had":1518}},
            "stats":{"S":{"invalid":0},"R":{"invalid":0}},"distance":{},
            "local":{},"psd":{"tables":{},"vs_REAL":{}},"width":width,
            "composition":tables,"association":{"S":association},"historical":{},
            "source_probes":{"S":{"rows":{"source_roll_no_fixed_points":{"waveform_RMS_vs_correct_reference_V":{"mean":1e-4}}}}},
            "joint":joint,"guide":guide,"projection":projection,"raw":moments,
            "identity":{"R":identity},"current_training":current_training,"charge":charge,
            "legacy_adc_description":{"bits":14,"pre_ADC_finite":True}}
        contract_ctx = SimpleNamespace(cfg={"recipe_description":"ARTIFICIAL_METADATA_ONLY"},
            counter={"HGB_attempts":0,"density_EM_attempts":0},config_binding={"artificial":True},
            actual_input_files={"artificial_fixture_only":True})
        contract_ctx.source_arm,contract_ctx.real_arm = "S","R"
        contract_ctx.__dict__["generation"] = generation
        # metrics_record uses generation passed through its context-independent fixture.
        # The production builder references measured generation as an explicit input.
        kwargs["generation"] = generation
        kwargs["c2st"] = cs
        kwargs["boundary"] = boundaries
        kwargs["raw"] = {"REAL":moments["REAL"],"S":moments["CANDIDATE"],"R":moments["CANDIDATE"]}
        metrics = entry.metrics_record(contract_ctx,**kwargs)
        schema = json.loads((directory/"METRICS_SCHEMA.json").read_text(encoding="utf-8"))
        check("all_root_required_metrics_fields",set(schema["required_top_level"]) <= set(metrics))
        check("root_renderer_guide_projection_generation_moments_contract",
              metrics["guide_probe"]["S"]["rows"]==128 and
              metrics["guide_probe"]["S"]["preADC_output_RMS_V"]["median"]==1e-4 and
              metrics["explicit_constraint_projection"] is projection and
              metrics["generation"] == generation["domains"] and
              all(len(metrics["raw_voltage_moments"][a])==9 for a in ["S","R"]))
        check("charge_file_and_metrics_same_object",metrics["charge_diagnostics"] is charge)
        check("known_neural_replay_zero_distinct_fullVJP",
              metrics["current_training"]["neural_training_replay_rows"]==0 and
              metrics["current_training"]["full_parameter_VJP_rows"]==691000000)
        json.dumps(ev.safe(metrics),allow_nan=False)

        # Simulated run exercises preflight -> old reference SHA verify -> evaluate.
        # It reads only a clearly marked artificial JSON config inside this directory.
        fixture_root = directory/"ARTIFICIAL_ONLY"
        fixture_root.mkdir(exist_ok=True)
        fake_cfg = copy.deepcopy(json.loads((directory/"config_post.example.json").read_text(encoding="utf-8")))
        for key in ["R26_root","R32_root","R33_root","R39_root","canonical_R51_root","frozen_eval_input_root"]:
            fake_cfg[key] = str(fixture_root/key)
        fake_cfg["old_BASE_REAL_generated"] = str(fixture_root/"BASE_REAL.generated.fixture")
        fake_cfg["local_input_paths"]["width_definition"] = str(fixture_root/"width.fixture")
        fake_cfg.update(SOURCE_arm="S",REAL_arm="R",evaluation_adapter_config=str(fixture_root/"ADAPTER_NOT_READ.fixture"),
                        recipe_description="ARTIFICIAL_METADATA_SIMULATION_NO_SCIENCE")
        fake_cfg["reference_bindings"] = {p:"0"*64 for p in protocol.reference_paths(fake_cfg).values()}
        config_path = fixture_root/"artificial_post.json"
        config_path.write_text(json.dumps(fake_cfg),encoding="utf-8")
        trace = []
        adapter = {"SOURCE_arm":"S","REAL_arm":"R"}
        old_functions = {k:getattr(entry,k) for k in ["load_config","preflight_training","owned_runtime","verify_file","evaluate","os"]}
        entry.load_config = lambda path:(trace.append("adapter_config") or (adapter,{"artificial":True}))
        entry.preflight_training = lambda c:(trace.append("preflight_training") or {"completed_updates":25000})
        entry.owned_runtime = lambda *args:(trace.append("owned_runtime_SIMULATED") or {"artificial":True})
        entry.verify_file = lambda row,name:(trace.append("verify_"+name) or Path(row["path"]))
        entry.os = SimpleNamespace(environ={},getpid=os.getpid)
        entry.evaluate = lambda ctx:(trace.append("evaluate_SIMULATED") or metrics)
        import threadpoolctl
        old_limit = threadpoolctl.threadpool_limits
        threadpoolctl.threadpool_limits = lambda **kw:contextlib.nullcontext()
        try:
            started = fixture_root/"EVALUATION_STARTED.json"
            if started.exists(): started.unlink()
            simulated = entry.run(config_path)
            check("run_preflight_all11_preverification_before_evaluate",
                  trace[:3] == ["adapter_config","preflight_training","owned_runtime_SIMULATED"] and
                  len([t for t in trace if t.startswith("verify_reference.")]) == 11 and
                  trace[-1] == "evaluate_SIMULATED" and simulated is metrics)
        finally:
            threadpoolctl.threadpool_limits = old_limit
            for name,value in old_functions.items():setattr(entry,name,value)
        check("Torch_still_not_imported","torch" not in sys.modules)
        return {"status":"PASS_ARTIFICIAL_ONLY","checks":checks,"check_count":len(checks),
                "actual_scientific_arrays_decoded":0,"actual_HGB_fits":0,"actual_density_EM_fits":0,
                "actual_response_model_forwards":0,"actual_training_updates":0,
                "mocked_HGB_slot_calls":10,"mocked_density_slot_calls":32,
                "native_scientific_execution":"not performed","run_handshake":"metadata simulation only",
                "CUDA_model_or_generation_validation":"unknown; no model or device execution requested"}
    finally:
        np.load = original_load
