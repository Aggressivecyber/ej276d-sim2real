from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def write(name, value):
    (ROOT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def brief_stat(value):
    return {k: value[k] for k in ("n", "defined_n", "undefined_n", "mean", "sample_sd") if k in value}


def native_receipt(job, tool):
    return dict(job, native_tool_chunk_id=tool)


def main():
    m = read("metrics.json")
    cfg = read("config.json")
    runs = read("RUN_HISTORY.json")
    plots = read("PLOT_RECEIPT.json")
    freeze = read("PRECOMPUTE_FREEZE.json")
    assert len(runs) == 2 and all(j["actual_worker_native_exit"] == 0 for j in runs)
    assert m["counters"] == {
        "HGB_attempts": 15, "response_updates": 0, "response_generation": 0,
        "EM_attempts": 0, "completed_projection_group_blocks": 15,
    }
    assert len(plots["figures"]) == 9
    science_s = sum(j["elapsed_s"] for j in runs)
    now = dt.datetime.now(dt.timezone.utc)
    assert now < dt.datetime.fromisoformat(cfg["deadline_UTC"])

    dmain = {}
    djoint = {}
    for domain, v in m["D_branches"]["main"].items():
        dmain[domain] = {
            k: v[k] for k in (
                "shared_normalization", "functional_branches_not_independent_physical_noise",
                "saved_pre_vs_branchsum_maxabs_V", "normalized_additive_closure_maxabs",
                "observed_Q", "source_Q",
            )
        }
        dmain[domain]["ADC_residual_V"] = brief_stat(v["ADC_residual_V"])
        dmain[domain]["ADC_row_RMS_V"] = brief_stat(v["ADC_row_RMS_V"])
    for domain, v in m["D_branches"]["joint8"].items():
        q = {k: v[k] for k in ("parent_composition_counts", "preADC_from_saved_component_sum_not_separately_saved")}
        q["ADC_residual_V"] = brief_stat(v["ADC_residual_V"])
        for space in ("raw_baseline_subtracted_V", "normalized_u", "normalized_seen", "normalized_complement"):
            q[space] = {}
            for composition, s in v[space].items():
                q[space][composition] = {k: s[k] for k in (
                    "source_parent_count", "finite_parent_count", "undefined_parent_count",
                    "repeats", "sample_variance_ddof", "closure_maxabs",
                )}
                q[space][composition]["gates"] = {
                    gate: {term: brief_stat(stat) for term, stat in terms.items()}
                    for gate, terms in s["gates"].items()
                }
        djoint[domain] = q
    compact = {
        "stage": m["stage"], "status": m["status"], "versions": m["versions"],
        "counters": m["counters"], "actual_inputs": m["actual_inputs"],
        "A_space": {k: v for k, v in m["A_space"].items() if k != "singular_values"},
        "A_coverage": m["A_coverage"],
        "B_common_support": m["B_projections"]["common_support_summary"],
        "B_definition": {k: v for k, v in m["B_projections"].items() if k not in ("blocks", "common_support_summary")},
        "B_all_bins_full_local_file": "B_PROJECTION_METRICS.json",
        "C_classifiers": m["C_classifiers"],
        "D_main": dmain, "D_joint8_summary": djoint,
        "source_Q_migration_R26_reused": m["source_Q_migration_R26_reused"],
        "private_event_payloads_published": False,
        "whole_loss_blind_space_claim": False,
        "EVAL_read": 0, "CAL_waveforms": 0, "SSH": 0,
    }
    write("METRICS_COMPACT.json", compact)

    # Known, actually inspected final images; no new fit or rendering here.
    qa = {
        "PID": os.getpid(), "UTC": now.isoformat(),
        "review_method": "nine existing PNGs actually opened with view_image by executor; ROOT separately confirmed all nine",
        "ROOT_confirmation": "parent message 2026-10-09, all nine readable; no further layout edits",
        "figures": {
            name: dict(value, executor_actually_viewed=True, labels_legends_axes_readable=True,
                       unsupported_strata_retained=True, final_image_modified_after_review=False)
            for name, value in plots["figures"].items()
        },
        "scientific_recomputation_for_QA": 0,
    }
    write("FIGURE_QA.json", qa)

    resource = {
        "stage": m["stage"], "registered_UTC": cfg["registered_UTC"],
        "first_preparation_command": {"tool_chunk_id": "968453", "PID": 54292, "native_exit": 0,
                                      "UTC": "2026-10-09T13:31:13.765523+00:00"},
        "executor_seal_deadline_UTC": cfg["deadline_UTC"],
        "ROOT_stage_deadline_UTC": cfg["absolute_deadline_UTC"],
        "jobs": [native_receipt(runs[0], "4f5f0b"), native_receipt(runs[1], "3ee4af")],
        "science_elapsed_s_including_SVD_stats_HGB_bootstrap_plots": science_s,
        "science_budget_s": cfg["science_budget"], "CPU_threads": 4,
        "actual_worker_RSS_peak_bytes": max(j["actual_worker_RSS_peak_bytes"] for j in runs),
        "RSS_limit_bytes": cfg["RSS_limit_bytes"], "memory_probe_failures": 0,
        "GPU_used": False, "GPU_allocated_peak_bytes": 0,
        "counters": m["counters"], "science_failures": [],
        "response_training_inference_generation": 0, "density_fits": 0,
        "EVAL_read": 0, "CAL_waveforms": 0, "raw_long_waveform_decode": 0,
        "SSH": 0, "installation": 0, "new_G4": 0, "migration_content_tests": 0,
        "single_owned_science_process": True, "actual_PID_guard_before_imports": True,
        "output_limit_bytes": cfg["output_limit_bytes"],
        "control_only_assembly_PID": os.getpid(), "assembly_UTC": now.isoformat(),
        "final_output_bytes_and_stop_UTC": "FINAL_RECEIPT.json; no invented pre-command native exit",
        "limitations": [
            "science charge uses actual supervisor wall times, including process startup/import/final saving",
            "closure/source-control assembly wall time is not a new numerical job",
            "no old response generation/model replay was repeated in R27",
        ],
    }
    write("RESOURCE_RECEIPT.json", resource)

    candidates = cfg["candidates"]
    names = {"R23_D1_SELECTED": "D1 selected (step0)", "HOMO_SELECTED": "HOMO selected (2750)",
             "HOMO_FINAL": "HOMO final (2950)", "R23_D1_FINAL2950": "D1 final (2950)",
             "TRAIN_REFERENCE": "REAL TRAIN–DEV"}
    arows = []
    crows = []
    brows = []
    for domain in candidates + ["TRAIN_REFERENCE"]:
        a = m["A_coverage"][domain]["observed_Q"][0]
        arows.append(f"| {names[domain]} | {a['mean_difference_sqnorm']:.6f} | {a['seen_fraction']:.3%} | {a['complement_fraction']:.3%} |")
        v = m["C_classifiers"]["results"][domain]
        crows.append(f"| {names[domain]} | {v['normalized_u480']['balanced_accuracy']:.3f} | {v['seen_reconstruction480']['balanced_accuracy']:.3f} | {v['complement_reconstruction480']['balanced_accuracy']:.3f} |")
        for feature in ("shape480", "diff479", "baseline100_V"):
            b = m["B_projections"]["common_support_summary"][feature]["domains"][domain]
            old = b["equal_mean_on_common_bins"]["ORIGINAL"]
            lo, hi = b["new_block_range"]
            brows.append(f"| {names[domain]} | {feature} | {old:.6f} | {b['new_block_mean']:.6f} | {lo:.6f}–{hi:.6f} |")
    had_rows = []
    for domain in candidates:
        q = m["D_branches"]["joint8"][domain]["normalized_complement"]["pure_had"]["gates"]["all"]
        had_rows.append(f"| {names[domain]} | {q['signal_variance']['mean']:.8g} | {q['electronic_variance']['mean']:.8g} | {q['twice_covariance']['mean']:.8g} | {q['total_variance']['mean']:.8g} |")
    report = f"""# R27 固定投影覆盖与保存波形诊断

完整波形仍明显不匹配。四个保存端点在已覆盖空间与90维正交补中都强可分，不能将剩余问题归结为仅正交补。HOMO 的原固定 shape/diff 方向显示较大改善，独立新方向揭示了保留的差距；这支持下一轮有界检验投影刷新，但尚未证明固定投影过拟合或唯一失败原因。本轮没有响应训练、推理、生成或密度拟合。

## 输入、条件与用途

复用 R23 D1 selected（G为原R21所选）、R26 HOMO selected2750、HOMO final2950、旧D1 final2950 的实际保存输出，每个2000条；所有源父记录、原顺序与主人口不删、不重加权。源组成417 pure-EE、1518 pure-had、65 mixed；推定PDG为317个22、1683个2112，沿用旧CSV/标签映射证据，不新增逐事件PDG真值读取。REAL DEV2000与保存REAL TRAIN reference2000全部QC有效。DEV已历史暴露，本轮是开发诊断；REAL–REAL是同采集TRAIN–DEV参照，人工组不代表独立采集批次。

所有比较保留总体及冻结八个观察Q区；源Q曲线/迁移单列，不能替代观察Q条件。共同合格区为bins1–7，各域每区至少40。HOMO selected/final的bin8仅36/37，明确不支持；其余箱外QC事件分别422/397，仍纳入总体。本轮没有EVAL、CAL波形、原始长采集、SSH或新G4。

## A：覆盖空间与均值残差

u=B(v)/max(max(B(v)),.0025V)，B扣前100点均值。390列覆盖 TRAIN缩放shape128、实际diff128、pooled32、前100坐标及两个积分指示向量；float64 SVD固定阈值得到rank390、正交补90维，baseline投影自身rank100。显式前100坐标意味着正交补该区域代数为零；仅消除其约1.44e-15的SVD舍入泄漏，避免分类缩放放大该已知零。正交、幂等、重建核对通过，最大投影幂等误差约8.9e-16。这只界定所列线性映射，不能称整个训练损失的盲区。

| 域 | 总体均值残差平方范数 | seen占比 | 正交补占比 |
|---|---:|---:|---:|
{chr(10).join(arows)}

总体与条件比例不同：HOMO selected在共同bins1–7的seen占比约67.1%–78.9%，不能仅引用总体90.9%。均值残差不等于全分布距离；总体混合比例与Q迁移也可影响残差，不能由此指认唯一固定时间伪影。全时间均值/中位和逐区分解保存为图及私有数组。

## B：原方向与四块独立新方向

TRAIN尺度、128个.005–.995分位及插值定义不变。新方向用四个完整PCG64种子20261009271–274，分别按shape/diff/baseline顺序取128单位列；实际矩阵和随机状态已保存。下表均为相同共同bins1–7的等权均值。

| 域 | 特征 | 原方向W1 | 四块新方向均值 | 新块范围 |
|---|---|---:|---:|---:|
{chr(10).join(brows)}

D1 selected→HOMO selected的原shape .379800→.323901，而新shape .422385→.415094；原diff .262015→.229920，但新diff .269660→.270313，未改善。HOMO selected的新/原比shape1.282、diff1.176、baseline1.084；同N REAL参照对应约.995/.982/1.014。差距不只存在于原覆盖外，新线性方向在有支持Q区揭示较大保留差异。完整逐区表及总体另存，负向例外不删。四块范围只是投影抽样参照，不能当置信区间或仅凭一次差异宣布过拟合。

## C：独立纯480时间视图

固定600+250每域、原R26实际行/分组和HGB超参数；分别只使用u480、seen重建480、正交补重建480，没有历史13个标量。下表为留出BA；完整AUC、训练分数及固定分类器的500组bootstrap区间在metrics中。

| 域 | u480 BA | seen BA | 正交补 BA |
|---|---:|---:|---:|
{chr(10).join(crows)}

所有分类器训练BA/AUC均1.0，REAL–REAL留出接近随机；两空间仍强可分。不同投影改变特征几何和分类敏感性，BA差异不构成去除某空间的因果试验，也不证明模型表达容量。区组CI仅条件于本次固定分类器和保存波形，不包含响应训练不确定性。

## D：共同归一化分支与joint8

signal/electronic均用同条pre-ADC总输出的p，扣各自前100点均值后可相加。保存pre与两分支和的最大差1.49e-8V，归一化闭合最大约5.95e-8，保留float32求和舍入。HOMO selected总体均值平方范数：signal12.368555、electronic.019569、两倍交叉项−.819891，总11.568233；正交补对应.959547、.001830、−.061278，总.900099。负交叉项不能删除；这些是分支均值能量，不是各分支对REAL残差的独立贡献。

条件均值图中固定尾段起伏在该功能分解下主要随signal出现，electronic保留平滑偏移及交叉项；无法据此称signal为物理均值或electronic为独立物理噪声。joint8用保存128父×8随机重复（EE42/had44/mixed42）；没有重生成，未独立保存的pre总量按分支float64和派生。以下为pure-had44父、正交补全480点、每父ddof1方差的均值：

| 域 | signal方差 | electronic方差 | 两倍协方差 | 总方差 |
|---|---:|---:|---:|---:|
{chr(10).join(had_rows)}

HOMO的signal正交补随机变化较大，同时负协方差抵消部分总变化。完整raw-V/归一化/seen/正交补与五gate、三组成统计、有限/undefined父数均保存。共同p依赖两个分支，会产生耦合；八次重复不会增加独立G4或真实事件数。ADC residual单列，HOMO selected样本SD约1.76e-5V，不能混入pre-ADC分支贡献。

## 资源、交付与下一问题

CPU4，主实际worker52400/native0（32.2709325s、RSS631394304B），渲染worker58504/native0（6.7575143s）。累计科学{science_s:.7f}s/600s，15/15固定HGB；response/EM/新生成0；失败0。实际PID在导入前握手并受RSS守护，监督器与worker回执全部保留。九图均已实际打开检查，ROOT也逐图确认可读；本次无需改图。资源与最终原生封存回执分别见RESOURCE_RECEIPT、FINAL_RECEIPT。完整结果、分组/预测/矩阵等私有文件保留本地；发布METRICS_COMPACT只含聚合信息。

优先下一问题：在同起点、同更新数和完全相同源/REAL/随机行流下，只刷新fine投影方向，仍按原固定J20选点，并预指定末点与R26末点比较。它检验增加训练方向覆盖是否改善独立方向及完整分布；若两空间和原有条件宽度仍失败，应保留结果，不将投影刷新或本轮分解当作物理修复证明。本轮止于诊断封存，不启动未登记下一轮。

关键图：results/figures/projection_blocks_common.png、temporal_c2st.png、conditional_residuals.png、branch_conditional_means.png、joint_variance_covariance.png；全部九图索引见FIGURE_QA.json。
"""
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    input_lines = []
    for key, value in cfg["inputs"].items():
        if isinstance(value, dict):
            input_lines.extend(f"- {key}/{sub}: `{path}`" for sub, path in value.items())
        else:
            input_lines.append(f"- {key}: `{value}`")
    reproduction = f"""# R27 reproduction

Actual local CPU interpreter: C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe. Versions: {json.dumps(m['versions'], ensure_ascii=False)}. CPU threads4, no CUDA/Torch import required. Scientific formulas and hashes were frozen before the first HGB in PRECOMPUTE_FREEZE.json and config.json/PROTOCOL.md.

Original commands (cwd this stage, -B, all temporary/cache paths under E):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:TEMP='E:/GAN/research/sim2real_two_papers_20261008_0305/.runtime-cache/tmp'
$env:TMP=$env:TEMP
& 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe' -B supervise.py numeric_worker.py 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe'
& 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe' -B supervise.py render_figures.py 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe'
```

These commands were executed once in this stage; no extra science replay was performed or required because response inference/generation is0. Saved arrays, predictions, matrices and RNG states support detailed reproduction. The .used tokens deliberately prevent a duplicate submission here. A future authorized reproduction must use a new E/D directory and its own deadline/counters; never modify closed stages or remove tokens to force a rerun. Any numerical reproduction would count its15 HGB fits and runtime separately.

Reuse the original precomputed actual phase0/1 C2ST rows and cluster identities, TRAIN scaling and four new projection seeds, rather than resplitting or fitting scalers on holdout. Main output2000 and joint128×8 from the four historical endpoints are read-only. Historical493-column classifier results were not substituted for these pure480 views. PROJECT dependencies and original definitions are preserved through INPUT_BINDINGS.json; no new broad source hash or migration-content test was performed.

Inputs:

{chr(10).join(input_lines)}

Evidence: numeric_worker.py.NATIVE_RECEIPT.json (actual52400, supervisor25396, native0), render_figures.py.NATIVE_RECEIPT.json (actual58504, supervisor60948, native0), RUN_HISTORY.json, NUMERIC_RECEIPT.json and PLOT_RECEIPT.json. Each actual worker PID was acknowledged before imports, with RSS monitoring failures0. TEMP/TMP remain E; C is used only to run the existing interpreter/system dependencies.

The dimensional span is not the full loss nullspace. New projection-block ranges are not confidence intervals. REAL TRAIN–DEV is one-acquisition descriptive sampling reference. DEV has repeated historical exposure; no EVAL/CAL waveform was opened. Shared total normalization and signal/electronic covariance prohibit a physical-independent-noise interpretation.
"""
    (ROOT / "REPRODUCTION.md").write_text(reproduction, encoding="utf-8", newline="\n")
    current_files = [p for p in ROOT.rglob("*") if p.is_file()]
    write("ASSEMBLY_RECEIPT.json", {
        "PID": os.getpid(), "UTC": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": m["stage"], "status": "control_reports_written",
        "scientific_array_decode": 0, "scientific_recompute": 0, "fit_generation_update": 0,
        "science_elapsed_s": science_s, "files_before_this_receipt": len(current_files),
        "bytes_before_this_receipt": sum(p.stat().st_size for p in current_files),
        "final_source_bindings": freeze["bindings"],
        "native_exit_evidence": "calling native tool receipt; os PID is not an exit proof",
    })
    print(json.dumps({"PID": os.getpid(), "UTC": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "status": "assembly_complete", "science_elapsed_s": science_s,
                      "compact_bytes": (ROOT / "METRICS_COMPACT.json").stat().st_size,
                      "REPORT_bytes": (ROOT / "REPORT.md").stat().st_size,
                      "no_new_science": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
