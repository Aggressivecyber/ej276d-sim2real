"""Record the actual root review after all ten R63 figures were opened."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
S = R / "work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001"
ARM = "COVERAGE_RESPONSE"
CTRL = "COVERAGE_REAL_CONTROL"
NOW = datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads(p.read_text(encoding="utf8"))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_once(p, value):
    with p.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

assert R.drive.upper() in ("E:", "D:")
assert read(S / "LOCAL_PIPELINE_CONTROL.json")["status"] == "registered_pipeline_complete_pending_root_QA"
assert not read(S / "LIVE_OBSERVATION.json")["actual_process"]["exists"]
assert read(S / "FROZEN_ARTIFACT_REVIEW.json")["pass"]
assert read(S / "TRAINING_FROZEN_AUDIT.json")["pass"]
replay = read(S / "REPLAY_RECEIPT.json")
assert replay["rows"] == 4000 and replay["pass"]
M = read(S / "METRICS_COMPACT.json")
G = read(S / "GATE_METRICS.json")
A = M["acceptance_A_C"][ARM]
assert G["decision"] == "close_fixed_R63_coverage_extension"
assert not G["primary_A_C_pass"] and all(not A[k]["pass"] for k in ("A", "B", "C"))

visual = {
    "training_and_checks.png": "Seven original traces are readable; the displayed 50000-update audit and FAIL/FAIL/FAIL outcome match saved metrics.",
    "canonical_five_view_response.png": "All five views and unchanged [0.4,0.6] band are visible; SOURCE baseline upper interval still exceeds 0.6 and all same-treated REAL intervals lie inside it.",
    "PSD_all_eight_bins.png": "All eight fixed bins appear; Q3/Q4 and edge Q8 SOURCE unqualified fits remain explicitly marked, with visible narrow histogram concentrations.",
    "PSD_centers_widths_and_W1.png": "Axes, thresholds and low/high components are legible; missing required fits remain visible gaps and are not interpolated into successful estimates.",
    "noise_and_source_association.png": "RMS/ACF distributions and the original SOURCE PSD/Q associations are visible; the title limits association claims.",
    "complete_waveform_population.png": "The full 480 samples, means, SDs, four populations and fixed time markers are visible; the SOURCE mean differs from REAL.",
    "fixed_time_boundaries.png": "All nine original boundaries and fixed limits are visible; the second difference at index150 exceeds its limit.",
    "explicit_constraint_projection.png": "Both complete 2000-event main populations have zero range-active correction and zero rail samples; TRAIN minibatch activation is separately identified.",
    "precision_order_decomposition.png": "Original, retained and omitted penalties plus retained pair mass are visible and explicitly described as training surrogates.",
    "coverage_loss_decomposition.png": "The two added objectives, temporal marginal/sliced parts and all four coordinate RMS views are visible; none is labelled scientific acceptance.",
}
figures = []
for name, observation in visual.items():
    p = S / "figures" / name
    assert p.is_file() and p.stat().st_size > 1000
    figures.append({"file": "figures/" + name, "bytes": p.stat().st_size,
                    "sha256": sha(p), "actually_opened_by_root": True,
                    "layout_pass": True, "observation": observation})
write_once(S / "FIGURE_QA.json", {
    "UTC": NOW, "reviewer": "/root", "method": "Actual local PNG opening through the image viewing tool; all panels inspected.",
    "all_figures_opened": True, "figure_count": 10, "layout_pass": True,
    "findings": "No cut-off title, axis, legend or required failed bin found. Displayed outcomes agree with the saved numerical records.",
    "scientific_acceptance_pass": False, "figures": figures,
})

next_action = (
    "封存R63固定覆盖扩展末点，保留基线和部分全窗区分度改善、但PSD与必需密度拟合退化且原A–C仍失败的结论。"
    "下一轮先预登记仅用合法TRAIN的冻结R63端点、条件参照几何及小批量梯度诊断：保持全部原父事件和区域质量、旧及新增TRAIN常数，"
    "共用固定端点检查31维基线与全窗统计残差、同量REAL抽样底噪和电荷均值约束，再以明确的有限经验参照核对4×32与128的分布梯度。"
    "每父固定噪声与原每次出现独立噪声的区别必须单列；先完成人工值/梯度/重复父/两遍VJP检查再运行。"
    "据TRAIN证据决定后续机制，不续训或择取R63检查点，不自动添加特征或增大训练批量，不改变原A–F、源人口及R60阴性结论；本候选不启动D/E。"
)
views = ["full_historical_shape", "normalized_u480", "full_voltage", "Q_PSD_2D", "baseline_local"]
stages = G["models"][ARM]
comparison = M["paired_R63_minus_frozen_R61"]
old = read(R / "work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001/METRICS_COMPACT.json")
direction = {
    "UTC": NOW, "reviewer": "/root", "stage": "R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001",
    "scientific_decision": G["decision"], "fixed_recipe_closed": True,
    "model_goal_complete": False, "application_goal_status": "active",
    "original_A_B_C": {k: A[k]["pass"] for k in ("A", "B", "C")},
    "five_view_response": A["A"]["views"],
    "PSD_width_acceptance": A["B"],
    "previous_R61_PSD_width_acceptance": old["acceptance_A_C"]["PRECISION_RESOLVED_RESPONSE"]["B"],
    "paired_BA_differences": {arm: {v: {k: comparison[arm][v][k] for k in (
        "BA_delta_first_minus_second", "BA_delta_95CI", "valid_replicates",
        "conditional_on_two_fitted_classifiers", "response_training_uncertainty")}
        for v in views} for arm in (ARM, CTRL)},
    "original_stage_gates": {k: v["pass"] for k, v in stages.items() if k.startswith("gate")},
    "failed_boundary_checks": [v for v in stages["gate4_boundary_source"]["boundaries"] if not v["pass"]],
    "actual_SOURCE_valleys": stages["gate3_source_preservation"]["actual_source_pure_had_valleys"],
    "actual_SOURCE_ranks": stages["gate3_source_preservation"]["whole_population_source_ranks"],
    "root_interpretation": [
        "The frozen extension lowers SOURCE BA in shape, normalized waveform, voltage and baseline views relative to R61; their paired BA intervals exclude zero. Q+PSD BA worsens with an interval above zero.",
        "SOURCE baseline BA 0.566 is insufficient: its unrounded upper interval 0.6077807423339705 exceeds 0.6. All five same-treated REAL intervals pass; this cannot substitute for SOURCE acceptance.",
        "Overall and common Q1-7 PSD W1 worsen from 0.010148559344896581/0.012159815007557322 to 0.01743116548718676/0.017282995403714172. Q3/Q4 fail required density descriptions; both aggregate peak/width results remain undefined and failing.",
        "Only second_center_150 fails the fixed boundary W1 rule. Source preservation, the original same-treated REAL stage gate, and numerical/replay checks pass.",
        "Matching original initialization, six consumed random streams, realized parent exposures and constants supports the registered combined-objective comparison. It does not isolate the two additions or estimate training-seed variability.",
        "The observed tradeoff does not itself establish gradient conflict, minibatch bias, a unique feature blind space or charge-induced irreducibility. R60's negative markers remain closed.",
        "The saved training traces are minibatch path measurements, not exhaustive R63 endpoint statistics. The next TRAIN-only diagnostic needs its own registration and fixed numerical definitions.",
    ],
    "next_action": next_action,
    "next_scientific_stage_registered_in_this_review": False,
    "next_scientific_job_started_in_this_review": False,
    "auxiliary_preparation": {
        "root_synthesis": "reviews/R63_WAIT_PARALLEL_001/ROOT_SYNTHESIS.json",
        "D_statistics_review": "reviews/R63_WAIT_PARALLEL_001/root_analysis/CODE_ROOT_REVIEW.json",
        "charge_bound_status": "Static algebra only, not a computed bound or A-F impossibility proof.",
        "gradient_code_status": "Artificial-data code preparation in progress; no scientific use or adoption yet.",
        "D_E_status": "Prepared partial adapters; not registered or scientifically evaluated for R63.",
    },
    "reviewed_evidence_sha256": {n: sha(S / n) for n in (
        "METRICS_COMPACT.json", "GATE_METRICS.json", "ACCEPTANCE_A_C.json",
        "PAIRED_R63_MINUS_R61.json", "TRAINING_FROZEN_AUDIT.json", "FROZEN_ARTIFACT_REVIEW.json",
        "TRAIN_SOLVER_CHECK.json", "REPLAY_RECEIPT.json", "FIGURE_QA.json")},
    "private_Git_and_D_delivery": "pending actual publication and copy receipts",
}
write_once(R / "reviews/R63_ROOT_DIRECTION_001.json", direction)
print(json.dumps({"figures_opened": len(figures), "layout_pass": True,
                  "scientific_decision": G["decision"], "model_goal_complete": False}, indent=2))
