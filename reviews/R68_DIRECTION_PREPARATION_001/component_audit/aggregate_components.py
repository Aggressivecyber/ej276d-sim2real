"""Read-only aggregation of the explicitly permitted R67 endpoint JSONs.

No numerical/scientific runtime, model, waveform, RNG draw, fit, or network call.
Run with the existing Python313 executable and -B under affinity mask 2.
"""
import ctypes
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path("E:/GAN/research/sim2real_two_papers_20261008_0305")
OUT = ROOT / "reviews/R68_DIRECTION_PREPARATION_001/component_audit"
GUARD = ROOT / "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"
D = ("full_window", "coordinate_RMS", "PSD", "noise", "temporal_joint")
C = ("source_order", "source_valley", "REAL_identity", "box_correction")
ARMS = ("baseline", "full", "mini")
LIMIT = 1024 ** 3


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stats(values):
    assert len(values) == 8 and all(math.isfinite(v) for v in values)
    return {"n": 8, "mean": statistics.fmean(values),
            "sample_SD": statistics.stdev(values), "min": min(values),
            "median": statistics.median(values), "max": max(values),
            "negative_count": sum(v < 0 for v in values),
            "zero_count": sum(v == 0 for v in values),
            "positive_count": sum(v > 0 for v in values), "values": values}


def compositions(total, slots):
    if slots == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in compositions(total - first, slots - 1):
                yield (first,) + rest


def exact_empirical_bootstrap(values):
    # Only averages of the eight saved paired differences. Enumerate all 6435
    # count configurations, weighted by the number of their 8**8 ordered draws.
    weighted = []
    for counts in compositions(8, 8):
        weight = math.factorial(8) // math.prod(math.factorial(c) for c in counts)
        weighted.append((math.fsum(c * v for c, v in zip(counts, values)) / 8, weight))
    weighted.sort()
    assert len(weighted) == 6435 and sum(w for _, w in weighted) == 8 ** 8
    ranks = (math.ceil((8 ** 8) / 40), math.ceil(39 * (8 ** 8) / 40))
    endpoints = []
    for rank in ranks:
        cumulative = 0
        for value, weight in weighted:
            cumulative += weight
            if cumulative >= rank:
                endpoints.append(value)
                break
    return {"paired_mean_percentile_95_interval": endpoints,
            "includes_zero": endpoints[0] <= 0 <= endpoints[1],
            "method": "exact empirical paired bootstrap; 8 saved MC pairs resampled 8 times; inverse-CDF percentiles",
            "count_configurations": 6435, "ordered_resamples": 8 ** 8,
            "random_draws": 0,
            "scope": "post-hoc MC description conditional on these fixed models and TRAIN populations; not training/acquisition uncertainty or a new gate"}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    utc0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    assert Path(__file__).resolve().parent == OUT.resolve()
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.SetProcessAffinityMask.restype = ctypes.c_int
    process = k.GetCurrentProcess()
    if not k.SetProcessAffinityMask(process, 2):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask(2)")
    spec = importlib.util.spec_from_file_location("existing_local_resource_guard", GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    native_guard = guard.apply(OUT)
    assert native_guard["CPU_affinity_mask"] == 2 and native_guard["RSS_hard_max_bytes"] == LIMIT
    relatives = [f"work/R67_EQUAL_COST_REPAIR_001/{a}/EVALUATION.json" for a in ARMS]
    relatives += [f"work/R67_EQUAL_COST_REPAIR_001/{a}/TRAINING_RECEIPT.json" for a in ("full", "mini")]
    relatives += ["work/R67_EQUAL_COST_REPAIR_001/RESULT.json", "R63_REPORT.md", "R64_REPORT.md", "R67_REPORT.md", "PROJECT_ACCEPTANCE.json"]
    raw = {p: (ROOT / p).read_bytes() for p in relatives}
    inputs = {p: {"SHA256": sha(b), "bytes": len(b)} for p, b in raw.items()}
    missing_baseline_receipt = "work/R67_EQUAL_COST_REPAIR_001/baseline/TRAINING_RECEIPT.json"
    assert not (ROOT / missing_baseline_receipt).exists()
    guard_hash = sha(GUARD.read_bytes())
    ev = {a: json.loads(raw[f"work/R67_EQUAL_COST_REPAIR_001/{a}/EVALUATION.json"].decode("utf-8-sig")) for a in ARMS}
    tr = {a: json.loads(raw[f"work/R67_EQUAL_COST_REPAIR_001/{a}/TRAINING_RECEIPT.json"].decode("utf-8-sig")) for a in ("full", "mini")}
    result = json.loads(raw["work/R67_EQUAL_COST_REPAIR_001/RESULT.json"].decode("utf-8-sig"))
    acceptance = json.loads(raw["PROJECT_ACCEPTANCE.json"].decode("utf-8-sig"))
    assert acceptance and result["status"] == "NEGATIVE_LOCAL_SUPPORT"
    seeds = [p["seed"] for p in ev["baseline"]["probes"]]
    assert len(seeds) == len(set(seeds)) == 8
    for a in ARMS:
        assert ev[a]["arm"] == a and ev[a]["probe_count"] == 8
        assert ev[a]["SOURCE_N"] == 6593 and ev[a]["REAL_N"] == 21047
        assert ev[a]["train_streams_and_model_unchanged"] is True
        assert [p["seed"] for p in ev[a]["probes"]] == seeds
        assert all(set(p["pieces"]) == set(D + C) for p in ev[a]["probes"])
    for a in ("full", "mini"):
        assert tr[a]["model_SHA256"] == ev[a]["model_SHA256"]
        assert tr[a]["fixed_learning_rate"] == result["fixed_learning_rate"]
        assert tr[a]["first_original_step"] == 501
        assert tr[a]["source_occurrences"] == tr[a]["completed_updates"] * 320
        assert tr[a]["REAL_identity_occurrences"] == tr[a]["completed_updates"] * 64
    values, summaries = {}, {}
    max_sum_error = 0.0
    for a in ARMS:
        rows = ev[a]["probes"]
        values[a] = {key: [p["pieces"][key] for p in rows] for key in D + C}
        values[a]["D5"] = [math.fsum(p["pieces"][k] for k in D) for p in rows]
        values[a]["C4"] = [math.fsum(p["pieces"][k] for k in C) for p in rows]
        values[a]["score"] = [p["score"] for p in rows]
        values[a]["score_without_source_valley"] = [math.fsum(p["pieces"][k] for k in D + C if k != "source_valley") for p in rows]
        max_sum_error = max(max_sum_error, max(abs(math.fsum(p["pieces"].values()) - p["score"]) for p in rows))
        summaries[a] = {k: stats(v) for k, v in values[a].items()}
        assert abs(summaries[a]["score"]["mean"] - result["primary"][a + "_mean"]) < 1e-12
    assert max_sum_error < 1e-12
    paired = {}
    for new, ref in (("full", "baseline"), ("mini", "baseline"), ("full", "mini")):
        label = new + "_minus_" + ref
        paired[label] = {key: stats([x - y for x, y in zip(values[new][key], values[ref][key])]) for key in values[new]}
        denominator = paired[label]["score"]["mean"]
        for key in D + C + ("D5", "C4"):
            paired[label][key]["signed_share_of_total_score_delta_percent"] = 100 * paired[label][key]["mean"] / denominator
        assert abs(math.fsum(paired[label][k]["mean"] for k in D + C) - denominator) < 1e-12
    fm = paired["full_minus_mini"]
    exact_intervals = {key: exact_empirical_bootstrap(fm[key]["values"]) for key in ("score", "score_without_source_valley", "D5", "C4")}
    assert fm["score"]["negative_count"] == result["primary"]["full_beats_mini_probes"] == 0
    assert fm["score_without_source_valley"]["positive_count"] == 8
    harm = {}
    for check in result["harm_checks"]:
        group = harm.setdefault(check["group"], {"total": 0, "failed": 0, "failed_keys": []})
        group["total"] += 1
        if not check["pass_gate"]:
            group["failed"] += 1
            group["failed_keys"].append(check["key"])
    harm_total, harm_failed = sum(v["total"] for v in harm.values()), sum(v["failed"] for v in harm.values())
    assert (harm_total, harm_failed) == (102, 63) and len(result["failed_harm_checks"]) == 63
    auxiliary = {}
    for group, keys in (("nonnegative", ("PSD_W1_overall", "baseline_RMS_W1_overall", "baseline_ACF1_W1_overall", "projection_fraction_overall")), ("correlations", ("overall",)), ("valleys", tuple(ev["baseline"]["probes"][0]["valleys"]))):
        auxiliary[group] = {key: {a: stats([p[group][key] for p in ev[a]["probes"]]) for a in ARMS} for key in keys}
    real_control = {a: ev[a]["REAL_control"] for a in ARMS}
    real_D5 = {a: math.fsum(real_control[a]["pieces"][k] for k in D) for a in ARMS}
    timing = {"full": tr["full"], "mini": tr["mini"],
              "mini_over_full_update_ratio": tr["mini"]["completed_updates"] / tr["full"]["completed_updates"],
              "C_SOURCE_exposure_ratio": tr["mini"]["source_occurrences"] / tr["full"]["source_occurrences"],
              "C_REAL_identity_exposure_ratio": tr["mini"]["REAL_identity_occurrences"] / tr["full"]["REAL_identity_occurrences"],
              "comparison_scope": "whole-route equal-wall-time endpoint difference; update count, exposures, adaptive optimizer trajectory and D-estimator all differ; no isolated estimator causal claim"}
    output = {"stage": "R68_DIRECTION_PREPARATION_001", "status": "EXISTING_JSON_COMPONENT_AUDIT_COMPLETE",
              "evidence_level": "native CPU stdlib aggregation plus static interpretation; no model/waveform/probe execution",
              "difference_convention": "new minus reference; negative means smaller loss; signed shares may exceed 100 percent because other components offset",
              "common_weights": {k: 1 for k in D + C}, "D5_keys": D, "C4_keys": C,
              "seeds": seeds, "summaries": summaries, "paired_differences": paired,
              "exact_descriptive_empirical_MC_intervals": exact_intervals,
              "original_R67_primary_interval_unchanged": result["primary"]["descriptive_paired_noise_MC_full_minus_mini_95_interval"],
              "source_valley_concentration": {a: {"share_of_mean_score_percent": 100 * summaries[a]["source_valley"]["mean"] / summaries[a]["score"]["mean"], "largest_probe_share_of_valley_sum_percent": 100 * max(values[a]["source_valley"]) / math.fsum(values[a]["source_valley"]) if math.fsum(values[a]["source_valley"]) else None} for a in ARMS},
              "auxiliary_saved_metrics": auxiliary, "same_treated_REAL_once_per_model": {"controls": real_control, "D5_sum": real_D5, "MC_n_per_model": 1},
              "training_and_cost": timing, "harm_checks": {"total": harm_total, "failed": harm_failed, "by_group": harm},
              "arithmetic_checks": {"max_abs_score_minus_sum_nine": max_sum_error, "matches_saved_primary_means": True, "paired_seeds_identical": True, "receipt_model_hash_matches_evaluation": True},
              "missing_input": {missing_baseline_receipt: "absent; baseline has no training receipt, no substitute created"},
              "limits": ["8 noise/direction repeats conditional on one pair of fixed endpoints and TRAIN populations, not independent training/acquisition repeats", "C values are the saved common-probe pieces; do not label them full-population C losses", "without-valley analysis and exact intervals are post-hoc descriptions, not a revised acceptance criterion", "shared checkpoint500/full-D warm start and fixed learning rate only; no original-init/50000-step/general architecture claim", "scalar losses do not establish gradient alignment/conflict or unique mechanism", "mini is not an accepted final candidate; original A-F, density widths, C2ST, replay and utility remain incomplete"],
              "recommended_low_cost_mechanism": "C enforcement opportunities per GPU second: mini has 48.95x C update/exposure opportunities; investigate whether under-served C dynamics explains source-order/valley/REAL-preservation endpoint gap, as a hypothesis only",
              "do_not_repeat": "restart the retired R65/full-D 50000-step route on the premise that lower D loss or R64 estimator geometry guarantees joint equal-cost response gain"}
    dump(OUT / "DECOMPOSITION.json", output)

    lines = ["# R68 共同九项分解（仅已有结果）", "",
             "全量的五D更低，但四C更高，故在这次同500断点、固定学习率、等约98.5秒的路线比较中九项总分更差。以下是固定端点事后解释，原阴性判定及验收门保持。", "",
             "| 项 | baseline | full | mini | full−baseline | mini−baseline | full−mini |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in D + C + ("D5", "C4", "score"):
        means = [summaries[a][key]["mean"] for a in ARMS]
        deltas = [paired[label][key]["mean"] for label in ("full_minus_baseline", "mini_minus_baseline", "full_minus_mini")]
        lines.append("| " + key + " | " + " | ".join(f"{v:.9f}" for v in means + deltas) + " |")
    lines += ["", f"full−mini 总差 **{fm['score']['mean']:.9f}**：D贡献 {fm['D5']['mean']:.9f}（抵消 {abs(fm['D5']['signed_share_of_total_score_delta_percent']):.2f}%），C贡献 {fm['C4']['mean']:.9f}（{fm['C4']['signed_share_of_total_score_delta_percent']:.2f}%）。source_valley 单项贡献 {fm['source_valley']['mean']:.9f}，占净差 {fm['source_valley']['signed_share_of_total_score_delta_percent']:.2f}%；source_order 贡献 {fm['source_order']['mean']:.9f}（{fm['source_order']['signed_share_of_total_score_delta_percent']:.2f}%）。百分比是共同损失标度上的代数贡献，不能解释成因果比例。", ""]
    for a in ("full", "mini"):
        delta = paired[a + "_minus_baseline"]
        lines.append(f"baseline→{a} 总分下降 {-delta['score']['mean']:.9f}，其中五D下降 {-delta['D5']['mean']:.9f}、四C下降 {-delta['C4']['mean']:.9f}；valley 下降 {-delta['source_valley']['mean']:.9f}，解释净下降 {delta['source_valley']['signed_share_of_total_score_delta_percent']:.2f}%。")
    lines += ["", "| 已保存MC seed | full−mini D5 | full−mini C4 | full−mini 九项 | full−mini 扣valley |", "| --- | ---: | ---: | ---: | ---: |"]
    for i, seed in enumerate(seeds):
        lines.append(f"| {seed} | " + " | ".join(f"{fm[k]['values'][i]:.9f}" for k in ("D5", "C4", "score", "score_without_source_valley")) + " |")
    for key, name in (("score", "九项差"), ("score_without_source_valley", "扣valley差")):
        s, ci = fm[key], exact_intervals[key]["paired_mean_percentile_95_interval"]
        lines += ["", f"{name}：均值 {s['mean']:.9f}，样本SD {s['sample_SD']:.9f}，配对范围 [{s['min']:.9f}, {s['max']:.9f}]，full在 {s['positive_count']}/8 组更差。精确经验配对重采样95%分位区间 [{ci[0]:.9f}, {ci[1]:.9f}]，不含零。"]
    lines += ["", f"baseline 的 valley 均值 {summaries['baseline']['source_valley']['mean']:.9f}，范围 [{summaries['baseline']['source_valley']['min']:.9f}, {summaries['baseline']['source_valley']['max']:.9f}]；full 范围 [{summaries['full']['source_valley']['min']:.9f}, {summaries['full']['source_valley']['max']:.9f}]；mini 八项均为零。full最大的valley组占其valley总和 {output['source_valley_concentration']['full']['largest_probe_share_of_valley_sum_percent']:.2f}%，但扣除该项后仍8/8更差。因此总分受该项显著影响，却不是路线劣势的唯一来源。", "",
             "区间仅对八个已保存配对差枚举6435个频数配置（对应8^8个等概率经验重采样序列），没有新噪声、模型或探针。它描述固定模型/人口的MC变化；既非独立训练/采集区间，也未把删valley作为新目标或验收。原RESULT中的九项区间原样保留，有限随机bootstrap与本次精确分位端点可以不同。", "",
             f"最值得低成本查的一项机制是**单位墙钟的C保持机会不足**：full20次、mini979次，原C每更新320 SOURCE+64 REAL identity曝光保持，两者C更新及曝光机会比48.95。full的source_order/REAL_identity更高；同处理REAL单次完整控制D5合计 baseline/full/mini={real_D5['baseline']:.9f}/{real_D5['full']:.9f}/{real_D5['mini']:.9f}，identity MSE={real_control['baseline']['nonnegative']['parentwise_normalized_identity_MSE']:.9f}/{real_control['full']['nonnegative']['parentwise_normalized_identity_MSE']:.9f}/{real_control['mini']['nonnegative']['parentwise_normalized_identity_MSE']:.9f}，故差异不只来自valley的评分波动。这些观察支持优先核对C动态/单位成本保持机会；尚未证明它是原因。将来若登记一项短程干预，需要让D规则和C机会分别可控；本轮未实施任何干预。", "",
             "979次与20次还同时改变Adam轨迹、重复曝光及D估计规则。端点标量无法分离各自因果作用，不能称为独立梯度估计器效果，也不能由损失值宣称D/C梯度冲突。现有资料没有逐更新C轨迹或分项梯度，不能定位何时改善、valley为何为零、模型是否忽略噪声，或估计新噪声/新采集的失败率。", "",
             "不应再做：以D较低或R64的经验梯度几何为理由，重启已退休R65的50,000步昂贵路线。R64仅在冻结R63、固定参照下比较五D且排除了四C（报告7、20、47行），并且B128主误差更差（3、29行）；它没有提供联合训练收益保证。", "",
             "原63/102退化门失败已重算吻合；它们是相关的门，不是63项独立因果证据。两个端点的TRAIN总体PSD改善均不能证明原A–F：共同起点已有full-D500步，学习率被固定，本次没有原初始化/50,000步末点、DEV/EVAL、新采集、密度峰宽、C2ST、重放或任务效用验收；mini同样不能作为最终合格候选。", "",
             "来源限于三个EVALUATION、full/mini TRAINING_RECEIPT、RESULT、R63/R64/R67报告及PROJECT_ACCEPTANCE。baseline/TRAINING_RECEIPT不存在，明确保留缺失，未补造。D按R64报告9行分为五项；九项外权重1见R63报告92行。共同起点/等时/固定LR/证据边界见R67报告3、7、103行。C均为保存的共同probe值，未把它们冒称为完整人口C。", "",
             "可复核数值、全部八组分量/差值、配对区间算法、来源哈希及不变性检查在DECOMPOSITION.json、aggregate_components.py、SHA256_MANIFEST.json；本机既有guard与实测资源在RESOURCE_RECEIPT.json。"]
    (OUT / "COMPONENT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    unchanged = all(sha((ROOT / p).read_bytes()) == item["SHA256"] for p, item in inputs.items())
    assert unchanged and sha(GUARD.read_bytes()) == guard_hash
    memory = guard.memory()
    assert memory["peak_RSS_bytes"] <= LIMIT
    native_receipt = {"status": "AGGREGATION_COMPLETED", "UTC_started": utc0,
                      "UTC_finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "wall_elapsed_s": time.perf_counter() - wall0, "process_CPU_s": time.process_time() - cpu0,
                      "existing_guard_path": str(GUARD), "existing_guard_SHA256": guard_hash,
                      "native_guard": native_guard, "actual_memory": memory,
                      "RSS_within_1GiB": True, "specified_inputs_count": len(inputs), "specified_inputs_unchanged": unchanged,
                      "model_or_waveform_reads": 0, "scientific_training_updates": 0, "new_probes": 0,
                      "network_calls": 0, "SSH_calls": 0, "git_calls": 0,
                      "bootstrap_random_draws": 0, "write_root": str(OUT)}
    dump(OUT / "RESOURCE_RECEIPT.json", native_receipt)
    outputs = {name: {"SHA256": sha((OUT / name).read_bytes()), "bytes": (OUT / name).stat().st_size}
               for name in ("aggregate_components.py", "DECOMPOSITION.json", "COMPONENT_AUDIT.md", "RESOURCE_RECEIPT.json")}
    dump(OUT / "SHA256_MANIFEST.json", {"inputs": inputs, "allowed_guard_dependency": {"path": str(GUARD), "SHA256": guard_hash}, "outputs": outputs, "inputs_unchanged_after_analysis": unchanged, "missing_baseline_receipt": missing_baseline_receipt})
    print(json.dumps({"status": native_receipt["status"], "full_minus_mini_D5": fm["D5"]["mean"], "full_minus_mini_C4": fm["C4"]["mean"], "full_minus_mini_score": fm["score"]["mean"], "without_valley": fm["score_without_source_valley"]["mean"], "without_valley_95_interval": exact_intervals["score_without_source_valley"]["paired_mean_percentile_95_interval"], "harm_failed": harm_failed, "native_guard": native_guard, "actual_memory": memory}, ensure_ascii=False))


if __name__ == "__main__":
    main()
