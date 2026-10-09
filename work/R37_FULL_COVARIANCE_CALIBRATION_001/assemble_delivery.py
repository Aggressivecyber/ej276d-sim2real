"""R37 control-only delivery: use saved JSON, no scientific arrays or fits."""
import hashlib
import json
import os
import pathlib
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent
END = datetime.fromisoformat("2026-10-09T22:01:09.839435+00:00")
CAP = 512 * 1024**2
NAME = "FULL_COVARIANCE_CALIBRATION"


def utc():
    return datetime.now(timezone.utc)


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def write(name, obj):
    assert utc() < END
    data = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    assert sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) + len(data) + 256 * 1024 < CAP
    with (ROOT / name).open("xb") as f:
        f.write(data)


def text(name, value):
    assert utc() < END
    with (ROOT / name).open("xb") as f:
        f.write(value.encode("utf-8"))


def binding(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    began = time.perf_counter()
    history = read("RUN_HISTORY.json")
    compact = read("METRICS_COMPACT.json")
    gate = read("GATE_METRICS.json")
    cfg = read("config.json")
    fit = read("TRAINING_NUMERIC_RECEIPT.json")["records"][NAME]
    replay = read("REPLAY_RECEIPT.json")
    gen = read("GENERATION_RECEIPT.json")
    transfer = read("TRANSFER_RECEIPT.json")
    returned = read("RETURN_TRANSFER_RECEIPT.json")
    first = read("START_RECEIPT.json")
    prep = read("PREPARATION_RECEIPT.json")
    cov = read("DEV_COVARIANCE_DESCRIPTION.json")
    assert len(history) == 7 and all(r["actual_worker_native_exit"] == 0 and r["launcher_native_exit"] == 0 for r in history)
    assert replay["pass"] and replay["rows"] == 2000 and gate["retain"] is False
    assert all(fit["original_R33_stream_full_match"].values()) and fit["G_before"] == fit["G_after"]
    assert compact["counters"]["HGB_attempts"] == 10 and compact["counters"]["density_EM_attempts"] == 7
    science_s = sum(r["elapsed_s"] for r in history)
    assert science_s < 1800

    # These hashes bind the project execution source, not a migration test.
    freeze_checks = {}
    for receipt, key in [("PRETRAIN_FREEZE.json", "source_bindings"), ("LOCAL_PREGEN_FREEZE.json", "entries")]:
        freeze_checks[receipt] = {}
        for name, prior in read(receipt)[key].items():
            actual = binding(ROOT / name)
            assert actual == prior, (receipt, name, actual, prior)
            freeze_checks[receipt][name] = True
    sources = {p.name: binding(p) for p in sorted(ROOT.glob("*.py"))}
    for name in ["config.json", "remote_config.json", "PROTOCOL.md", "STATISTICS_INPUT_BINDING.json"]:
        sources[name] = binding(ROOT / name)
    write("SOURCE_BINDINGS.json", {
        "UTC": utc().isoformat(), "PID": os.getpid(), "project_sources": sources,
        "pre_execution_bindings_unchanged": freeze_checks,
        "frozen_BASE_weight_reference": cfg["base_G"],
        "executed_receipt_G_before": fit["G_before"], "executed_receipt_G_after": fit["G_after"],
        "eight_map_file": cfg["endpoint_weight_paths"][NAME],
        "map_and_payload_byte_hashes": "FINAL_MANIFEST.json; pure byte sealing, no array decoding",
        "new_science_or_migration_consistency_tests": 0,
    })

    figdir = ROOT / "results" / "figures"
    figure_names = sorted(read("PLOT_SUMMARY_RECEIPT.json")["figures"])
    assert len(figure_names) == 11
    figures = []
    for name in figure_names:
        info = binding(figdir / name)
        prior = (read("GUIDE_LAYOUT_REPAIR_V2_RECEIPT.json") if name == "guide_and_random_decomposition.png"
                 else read("PLOT_SUMMARY_RECEIPT.json")["figures"][name])
        assert info["sha256"] == prior["sha256"]
        figures.append({"file": "results/figures/" + name, **info, "actual_image_opened": True,
                        "axes_legends_support_gaps_readable": True,
                        "view_evidence": "599dad view_image final v2" if name == "guide_and_random_decomposition.png"
                        else "earlier in this R37 execution, before delivery; no later image changes"})
    write("FIGURE_QA.json", {
        "UTC": utc().isoformat(), "figure_count": 11, "all_final_figures_actually_opened": True,
        "pass": True, "figures": figures,
        "layout_only_repairs": [
            {"entry": "fix_guide_layout.py", "tool_native_exit": 0, "tool": "f29284", "science_elapsed_s": history[-2]["elapsed_s"],
             "reason": "initial long probe labels overlapped; first shortened version still crowded"},
            {"entry": "fix_guide_layout_v2.py", "tool_native_exit": 0, "tool": "237fce", "science_elapsed_s": history[-1]["elapsed_s"],
             "reason": "horizontal probe bars; final labels clear; both prior images retained under layout_initial"},
        ],
        "data_recomputed_fits_generation": 0,
        "numerical_fit_and_replay_success_not_scientific_acceptance": True,
    })

    tool_ids = {
        "train_server.py": {"start": "f8171e", "completion": "f0228a"},
        "generate_all.py": {"start": "2ba881", "completion": "bbdaa7"},
        "evaluate_saved.py": {"start": "5dcf3f", "completion": "597af2"},
        "replay_all.py": {"start": "7b00fc", "completion": "7b00fc"},
        "summarize_and_plot.py": {"start": "ffef40", "completion": "ba4329"},
        "fix_guide_layout.py": {"start": "f29284", "completion": "f29284"},
        "fix_guide_layout_v2.py": {"start": "237fce", "completion": "237fce"},
    }
    native = []
    for row in history:
        native.append({**row, "original_tool_chunks": tool_ids[row["entry"]], "observed_tool_native_exit": 0,
                       "receipt": ("remote_receipts/" if row["entry"] == "train_server.py" else "") + row["entry"] + ".NATIVE_RECEIPT.json"})
    lookup_errors = [
        {"tool": "b56192", "actual_tool_exit": 0, "missing_source_lookup": "R35/generate_all.py", "science_started": False,
         "resolution": "used R36 compatible generation entry; PREPARATION_RECEIPT preserved"},
        {"tool": "1ec311", "actual_tool_exit": 0, "missing_source_lookup": "R36/summarize.py", "science_started": False,
         "resolution": "located existing summarize_and_plot.py; nonterminating shell error retained in original tool"},
        {"tool": "7d65bd", "actual_tool_exit": 0, "missing_own_control_lookup": "RETRIEVAL_RECEIPT.json", "science_started": False,
         "resolution": "actual returned receipt is RETURN_TRANSFER_RECEIPT.json; original tool error retained"},
    ]
    write("NATIVE_CHAIN.json", {
        "first_actual_command": first, "science": native, "all_science_terminal": True,
        "all_scientific_process_native_exits_observed": True,
        "remote_local_launch": {"local_launcher_PID": 38592, "SSH_PID": 25160, "remote_supervisor_PID": 59539,
                                "actual_remote_worker_PID": 59540, "tool_native_exit": 0, "completion_tool": "f0228a"},
        "normal_transfers": {"upload": transfer, "return": {k: v for k, v in returned.items() if k != "remote_outputs"},
                             "return_start_tool": "6a2d54", "return_completion_tool": "eb2149", "native_exit": 0},
        "source_control_lookup_errors": lookup_errors,
        "original_stdout_stderr_and_receipts_kept": True,
        "no_inferred_unknown_PID_or_exit": True,
    })

    measures = gate["gate2_PSD_width_centers_noise"]["measures"]
    c2 = compact["C2ST"]["results"][NAME]
    stats = fit["statistics"]
    maximum = lambda key: max(s[key] for s in stats)
    pop = cov["tables"]["population"]
    source_distinct = sum(s["distinct_source_row_indices"] for s in stats)
    real_distinct = sum(s["distinct_REAL_row_indices"] for s in stats)
    all_region_cov_improved = all(
        cov["tables"][f"observedQbin{i}"][NAME]["covariance_relative_Frobenius_vs_REAL"] <
        cov["tables"][f"observedQbin{i}"]["BASE"]["covariance_relative_Frobenius_vs_REAL"]
        for i in range(1, 8))
    guide = gate["gate4_boundary_guide"]["guide_probe"]
    guide_median_mV = 1000 * guide["preADC_output_RMS_V"]["quantiles"][3]
    def row(label, key, digits=6):
        v = measures[key]
        return f"| {label} | {v['BASE']:.{digits}f} | {v['candidate']:.{digits}f} | {100*v['relative_change']:+.2f}% |"
    table = "\n".join([
        "| PSD/噪声描述 | BASE | 协方差校准 | 相对变化 |", "|---|---:|---:|---:|",
        row("全体 PSD W1", "PSD_overall_W1"),
        row("共同 Q bins1–7 等权 PSD W1", "PSD_common_Q_equal_mean_W1"),
        row("低 PSD 成分峰位 MAE（bins2–7）", "component0_peak_position_MAE"),
        row("高 PSD 成分峰位 MAE（bins2–7）", "component1_peak_position_MAE"),
        row("低 PSD 成分 raw SD 误差 MAE", "component0_raw_width_MAE"),
        row("高 PSD 成分 raw SD 误差 MAE", "component1_raw_width_MAE"),
        row("baseline RMS 分布 W1（V）", "baseline_RMS_V_overall_W1", 9),
        row("baseline ACF1 分布 W1", "ACF1_overall_W1"),
    ])
    valley = gate["gate3_source_preservation"]["actual_source_pure_had_valleys"]
    valley_text = "；".join(f"{v['actual_source_bin']}（N={v['N']}）{100*v['BASE']:.2f}%→{100*v['candidate']:.2f}%（{100*v['delta']:+.2f} pp）" for v in valley)
    report = f"""# R37 完整协方差校准：关闭固定配方

联合门槛未通过。完整波形可分性下降、部分 PSD 距离和高侧宽度改善，但两侧峰位、低侧宽度、baseline RMS 与纯 had 源条件谷区占比退化。本轮不扫描 floor、秩、bin、旋转或叠加保护。整个项目目标继续，由根代理登记下一问题。

本轮冻结 R32 CONTROL_FINAL 的完整 480 点源驱动 G；唯一新模型是 8 个 TRAIN 高斯 Monge 仿射校准矩阵，神经参数更新为 0。严格复用 R33 的 2950×32=94400 条 source/REAL/ordinal/bin/z-white/style 流，七项实验 hash 全部相同，冻结 G 前后记录一致。各域 float64 稳定中心协方差采用 ddof=1，唯一 eigenfloor=1e-6×trace(C)/480；16 份协方差的实际 floor 坐标数均为 0。中间平方根的 SVD 实现计算同一个主平方根，没有换配方。统计唯一拟合耗时 {history[0]['elapsed_s']:.6f}s，8 个矩阵正定。

94400 是重复抽样次数。实际被使用的唯一源行 {source_distinct}、唯一 REAL TRAIN 行 {real_distinct}；加载的源 TRAIN 6593 中共同 Q 支持池3753、REAL TRAIN21047 中支持池18895。它们不能视为同等数量独立采集。本轮不使用内部 val 选择矩阵、不得用 DEV 选择新参数。主输出保留原2000父事件：pure_EE417/pure_had1518/mixed65，1583行校准，417支持外保留 BASE，顺序/身份/权重不变；全部2000有效且无量化裁剪。未读取新 EVAL/CAL/长采集，未新增 G4 或神经训练。

完整 shape493 BA 从 BASE .978 降至 {c2['full_historical_shape']['balanced_accuracy']:.3f}（配对 Δ=-.054，95%区间[-.079205,-.029132]），纯时间归一化 u480 从 .978 降至 {c2['normalized_u480']['balanced_accuracy']:.3f}（Δ=-.060，[-.084645,-.034543]）；相对 R35 也改善。完整电压 BA={c2['full_voltage']['balanced_accuracy']:.3f}、baseline 局部 BA={c2['baseline_local']['balanced_accuracy']:.3f}、Q/PSD 二维 BA={c2['Q_PSD_2D']['balanced_accuracy']:.3f}，整体仍明显可区分。分类器保持每域600训练/250留出和原组；500次区组 CI 条件于固定已拟合分类器，不包括响应训练不确定性，也不是物理等效阈值。

{table}

以上两成分是密度描述，不是粒子真值；raw SD 也不是 Gaussian FWHM。峰位/宽度比较的共同合格区间统一为 bins2–7，不隐藏支持缺口：新增 bin1 两成分未分离，bin8 N=36<40 未拟合；BASE 有效 bins2–7 没有新增失效。低侧 raw SD 误差增大93.62%，两侧峰位误差分别增大503.25%/224.29%，不能以高侧宽度变好宣称 PSD 成功。

实际未量化 NN 源 Q 下 pure-had：{valley_text}。两区谷区增量均超过预设2 pp，属于源条件保真风险，不能仅凭谷区判为粒子错类。主口径213/181与历史量化 SOURCE_CLEAN 参考207/182分列。实际源 Q→输出 Q 秩相关 .999726→.999046、源 PSD→输出 PSD .418992→.408773，仍通过降幅门槛；非零源敏感性不是正确物理逐事件映射。DEV PDG22/2112 数317/1683是旧桶规则结合R2仅两种PDG证据的推定，区别于显式逐事件PDG。

TRAIN 正则协方差最大相对闭合残差 {maximum('regularized_covariance_relative_Frobenius_residual'):.3e}；未正则闭合 {maximum('unregularized_covariance_relative_Frobenius_residual'):.3e}。这只证明拟合二阶矩的数值闭合。DEV 按观察到的输出 Q 条件，bins1–7协方差相对误差均改善={all_region_cov_improved}，bin8样本不足仍保留。DEV 全体协方差相对误差却从 {pop['BASE']['covariance_relative_Frobenius_vs_REAL']:.6f} 增至 {pop[NAME]['covariance_relative_Frobenius_vs_REAL']:.6f}，非对角相关矩阵 RMS 差从 {pop['BASE']['off_diagonal_correlation_RMS_difference']:.6f} 降至 {pop[NAME]['off_diagonal_correlation_RMS_difference']:.6f}。TRAIN闭合、DEV区内改善、DEV总体协方差略差是三个不同结论；输出条件迁移与总体组成不能省略。DEV描述不参与训练或部署矩阵估计。

128条预指定 guide 只在150:400增加去均值正弦，段 RMS=1 mV，原bin/G随机量保持。实际 float32 Qfast 未变，Qtotal最大绝对漂移约1.23e-5 mV·ns、PSD约1.02e-9；所有128条输出改变。全480点 preADC 输出差 RMS 中位 **{guide_median_mV:.3f} mV**（均值10.734 mV），显示该线性层放大扰动，不能称为物理正确的响应增益。9个固定一二阶边界门槛全部通过。joint8可加代数保留交叉项、闭合约6.51e-19 V²，校准后物理信号/电子噪声分解仍unknown；模板中的flow字段只是校准增量兼容名，不是本轮CFM/ODE。

独立CPU完整 source→G→校准→一次14bit ADC重放2000条，所有保存字段逐位通过。新拟合为10 HGB、8个密度槽/7次EM，旧BASE/R33/R35/R36/RAW RR零重复拟合；旧RAW RR二维预测缺失延续unknown。7个真实科学进程全部native0，累计 {science_s:.6f}s（含两次图布局修复），实际观测RSS峰 {max(r['actual_worker_RSS_peak_bytes'] for r in history):,}B，GPU allocated峰 {fit['GPU_peak_allocated_bytes']:,}B。服务器两次实际PID RSS查询失败发生在20:51:37.699380/.799651；具体原因unknown，邻近退出不能证明成因，也不声称缺口期间连续RSS已测。原工具的3次控制路径查找错误、所有日志与两版旧布局保留，未引发科学重跑。

11张最终图均实际打开检查，图版本以FIGURE_QA.json为准。门槛1/4/5通过，2（PSD/峰位/宽度/噪声）与3（源保真）失败，**关闭此固定配方**。证据支持二阶匹配无法单独保护两带PSD中心与源条件分布；不能推出所有非线性源响应都不可行。同采集人工50记录组、历史DEV暴露和旧模型先验成本仍存在，本轮是开发诊断，不能称外部盲测或独立粒子真值验证。

产物：`GATE_METRICS.json`、`METRICS_COMPACT.json`、本地完整`metrics.json`、`FULL_COVARIANCE_CALIBRATION/CALIBRATION_MAPS.npz`、`generated.npz`/原始辅助载荷、`REPLAY_RECEIPT.json`、`DEV_COVARIANCE_DESCRIPTION.json`、11张`results/figures/*.png`、`RESOURCE_RECEIPT.json`、`NATIVE_CHAIN.json`和最终字节清单。完整逐事件数据/模型留E/D，紧凑统计只发布聚合量。
"""
    text("REPORT.md", report)

    reproduction = f"""# R37 复现入口

此固定科学配方已关闭，旧stage不可覆盖、不可删一次性标记后重跑。需要复现时在新的E/D目录复制必要代码/配置并建立新运行记录；本文件不授权新实验或新拟合。

实际统计入口为服务器 `{cfg['remote_root']}/train_server.py`，由`linux_supervise.py`和`linux_owned_worker.py`以真实PID/nonce监督执行。Python是`{cfg['server_python']}`；R33原输入只读，现场Python3.12.3/NumPy1.26.4/Torch2.12.0+cu130/SciPy1.11.4、RTX5070，四CPU线程。固定R32 G以原GPUchunk32处理2950批，生成/REAL统计流与R33七项hash匹配，随后8个float64高斯map。无新神经训练。正常SSH/scp全部成功；没有为迁移新做对等内容测试。

本机CPU入口使用`{cfg['CPU_python']}`（Python3.13.3/NumPy2.2.6/Torch2.10.0+cpu/sklearn1.7.2）；单作业顺序为`supervise.py generate_all.py <CPU_python>`、`supervise.py evaluate_saved.py <CPU_python>`、`supervise.py replay_all.py <CPU_python>`。原本机监督还需要E项目`.runtime-cache/tmp`的TEMP/TMP、-B、PYTHONUTF8、PYTHONDONTWRITEBYTECODE和线程4。重放实际worker62156、supervisor60340，tool7b00fc/native0，2000条完整链各字段逐位一致；这属于科学模型复现，不是迁移测试。

最小推理输入为完整480点`ACTUAL_GENERATION_INPUTS.npz`（实际source/ID与z-white/style）、原冻结`{cfg['base_G']}`、本轮`FULL_COVARIANCE_CALIBRATION/CALIBRATION_MAPS.npz`中的8个A/ms/mt及原TRAIN mu/scale/edges、本轮model/legacy_model/flow_runtime/covariance_core实现。实际source-Q选bin，支持外保留G；支持内原TRAIN坐标中的mean+线性map，转float32后仅一次原14bit[-.5,.5]V ADC。推理不读取REAL库、不重拟合矩阵、不修改父事件。`flow_*`名字为兼容模板字段，实际不是ODE流。

独立回放加载保存输入与map，完整重新计算G和map；不把保存的输出当答案输入。`REPLAY_RECEIPT.json`保存所有14字段核对和pre/post数组SHA，科学输入/执行源冻结分别见`PRETRAIN_FREEZE.json`、`LOCAL_PREGEN_FREEZE.json`、`STATISTICS_INPUT_BINDING.json`，最终文件版本见`FINAL_MANIFEST.json`。Linux统计与本机CPU是不同执行环境，不能称全平台浮点逐位相同；本机生成和本机独立重放逐位相同已执行。

评价复用原600/250分组与旧RAW RR/BASE/R33/R35/R36结果；没有新旧REAL重抽或旧模型重新拟合。完整metrics/逐事件预测/身份/私有矩阵留本地，METRICS_COMPACT只含聚合量。11张最终图QA记录及初图/第一次短标签布局原件均保留。报告、资源与最终manifest由纯控制脚本汇总，不再解码科学NPZ或运行模型。
"""
    text("REPRODUCTION.md", reproduction)

    all_files = [p for p in ROOT.rglob("*") if p.is_file()]
    local_size = sum(p.stat().st_size for p in all_files)
    write("RESOURCE_RECEIPT.json", {
        "stage": cfg["stage"], "UTC": utc().isoformat(), "first_actual_command": first,
        "registered_UTC": cfg["registered_UTC"], "executor_deadline_UTC": cfg["deadline_UTC"],
        "science_elapsed_s_including_plots_and_repairs": science_s, "science_budget_s": 1800,
        "science_process_count": 7, "scientific_failed_process_count": 0,
        "statistics_fit_actual_supervised_s": history[0]["elapsed_s"], "statistics_fit_budget_s": 900,
        "counts": {**compact["counters"], "unique_statistics_fit": 1, "two_domain_bin_covariance_estimates": 16,
                   "maps": 8, "fixed_G_forward_training_rows": 94400, "fixed_REAL_statistics_draws": 94400,
                   "actual_distinct_source_TRAIN_row_indices": source_distinct,
                   "actual_distinct_REAL_TRAIN_row_indices": real_distinct, "complete_CPU_replay_rows": 2000,
                   "old_reference_HGB_or_density_refits": 0, "new_neural_updates": 0, "new_G4": 0,
                   "new_long_raw_decode": 0, "new_REAL_acquisition_read": 0, "installation": 0,
                   "migration_content_consistency_tests": 0},
        "draw_counts_not_independent_collection_counts": True, "threads": 4,
        "actual_worker_RSS_max_observed_bytes": max(r["actual_worker_RSS_peak_bytes"] for r in history),
        "RSS_limit_bytes": 4 * 1024**3, "GPU_peak_allocated_bytes": fit["GPU_peak_allocated_bytes"],
        "GPU_allocated_limit_bytes": 4 * 1024**3,
        "memory_probe_failures": sum(r["memory_probe_failures"] for r in history),
        "memory_probe_failure_events": history[0]["memory_probe_failure_events"],
        "RSS_failure_cause": "unknown; near termination timestamps alone do not establish cause",
        "continuous_RSS_during_two_query_gaps": "unknown",
        "remote_worker_self_peak_RSS_bytes": history[0]["worker_self_receipt"]["self_peak_RSS_bytes"],
        "input_copy_payload_bytes": prep["copied_dependency_payload_bytes"],
        "upload_tar_payload_bytes": transfer["upload_tar_bytes"], "returned_tar_payload_bytes": returned["return_tar_bytes"],
        "duplicated_return_package_and_extracted_payload_retained": True,
        "source_control_lookup_errors": lookup_errors,
        "local_stage_bytes_before_final_resource_and_seal": local_size,
        "local_stage_files_before_final_resource_and_seal": len(all_files),
        "local_stage_total_cap_bytes": CAP, "final_stage_exact_size": "FINAL_RECEIPT.json sealing stdout; all copies/packages included",
        "general_system_dependency_loading_IO": "not measured as project data IO; actual RSS independently monitored",
        "no_per_file_read_write_framework_introduced": True,
        "all_science_native_exits": [r["actual_worker_native_exit"] for r in history],
        "new_science_started_in_report_or_seal": False,
        "scientific_gate_result": {k: v["pass"] for k, v in gate.items() if k.startswith("gate")},
        "decision": gate["decision"], "project_goal_closed": False,
    })
    write("DELIVERY_ASSEMBLY_RECEIPT.json", {
        "PID": os.getpid(), "UTC": utc().isoformat(), "elapsed_s": time.perf_counter() - began,
        "control_only_saved_JSON_summary_and_project_byte_binding": True,
        "science_arrays_models_fits_generation": 0, "science_elapsed_s_unchanged": science_s,
        "native_exit_evidence": "external exec tool result; child does not invent its future native exit",
        "final_seal_pending": True,
    })
    print(json.dumps({"PID": os.getpid(), "UTC": utc().isoformat(), "reports_written": True,
                      "science_s": science_s, "guide_figure_sha256": figures[figure_names.index('guide_and_random_decomposition.png')]['sha256']}, ensure_ascii=False))


if __name__ == "__main__":
    main()
