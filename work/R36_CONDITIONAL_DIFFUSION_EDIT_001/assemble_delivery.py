"""Control-only delivery assembly from already saved R36 JSON; no scientific imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import time

ROOT = Path(__file__).resolve().parent
START = time.monotonic()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def put(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def binding(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def main():
    cfg, metrics, gate = read("config.json"), read("metrics.json"), read("GATE_METRICS.json")
    history, replay, gen = read("RUN_HISTORY.json"), read("REPLAY_RECEIPT.json"), read("GENERATION_RECEIPT.json")
    train, prep = read("TRAINING_NUMERIC_RECEIPT.json"), read("PREPARATION_RECEIPT.json")
    assert all(not gate[f"gate{i}_" + suffix]["pass"] for i, suffix in [
        (1, "complete_response"), (2, "PSD_width_noise"), (3, "source_preservation"),
        (4, "boundary_guide"), (5, "REAL_control")])
    assert gate["gate6_replay"]["pass"] and replay["pass"] and replay["rows"] == 4000
    assert metrics["counters"]["HGB_attempts"] == 20 and metrics["counters"]["density_EM_attempts"] == 16
    assert all(row["actual_worker_native_exit"] in (0, 1) for row in history)
    science_s = sum(row["elapsed_s"] for row in history)

    figures = []
    for f in sorted((ROOT / "results/figures").glob("*.png")):
        figures.append({"path": f.relative_to(ROOT).as_posix(), **binding(f),
                        "actual_visual_review": True, "layout_readable": True,
                        "review_method": "tools.view_image; all ten inspected during the executing conversation"})
    assert len(figures) == 10
    control_figure = next(row for row in figures if row["path"].endswith("real_control_and_guide.png"))
    assert control_figure["sha256"] == read("CONTROL_LAYOUT_REPAIR_V2_RECEIPT.json")["sha256"]
    put("FIGURE_QA.json", {"PID": os.getpid(), "recorded_UTC": utc(), "figure_count": 10,
        "all_final_figures_actual_visual_review_pass": True, "figures": figures,
        "latest_control_figure_review_UTC": "2026-10-09T20:31:40+00:00",
        "retained_layout_history": {
            "results/figures/layout_initial/real_control_and_guide.png": "initial probe labels crowded",
            "results/figures/layout_initial/real_control_and_guide_short_v1.png": "first shorter labels still crowded",
            "results/figures/real_control_and_guide.png": "second label-only repair reopened and readable"},
        "initial_plot_receipt_not_overwritten": True, "science_recomputed_for_layout": False,
        "review_notes": ["complete populations and support gaps visible", "fixed boundary failures retained",
                         "matched raw voltage/time axes; no new smoothing", "negative algebraic cross covariance retained",
                         "density components and additive branches not labeled physical particle/noise truth"]})

    old_compact = ROOT / "METRICS_COMPACT.json"
    extended = ROOT / "METRICS_AGGREGATE_EXTENDED.json"
    assert not extended.exists(), "do not repeat delivery assembly"
    shutil.copyfile(old_compact, extended)
    compact = {"status": metrics["status"], "decision": gate["decision"], "gate": gate,
        "training": metrics["training"], "counters": metrics["counters"],
        "data_roles": metrics["data_roles"], "C2ST": metrics["C2ST"],
        "overall_domain_vs_REAL": {k: v["overall"] for k, v in metrics["domain_vs_REAL"].items()},
        "PSD_main_vs_REAL": metrics["PSD_main"]["vs_REAL"],
        "width_common_support_errors": metrics["energy_conditional_width"]["common_support_errors"],
        "density_support_gaps": gate["gate2_PSD_width_noise"]["all_density_support_gaps"],
        "actual_source_pure_had_bins3_4": metrics["source_composition"]["primary_pure_had_actual_input_source_Q_bins3_4"],
        "historical_postADC_reference_pure_had_bins3_4": metrics["source_composition"]["historical_postADC_reference_Q_bins3_4"],
        "guide_probe": metrics["guide_probe"], "generation": gen["domains"],
        "replay": {"rows": replay["rows"], "pass": replay["pass"], "independent_process": True},
        "privacy": "aggregates only; event identities, predictions, actual RNG streams and waveforms remain in local private files",
        "full_local_metrics": "metrics.json", "extended_aggregate": "METRICS_AGGREGATE_EXTENDED.json",
        "versions": metrics["versions"], "final_physical_signal_electronic_split": "unknown"}
    put("METRICS_COMPACT.json", compact)

    freeze = {}
    for name, field in [("PRETRAIN_FREEZE.json", "source_bindings"),
                        ("PRE_GENERATION_SOURCE_FREEZE.json", "source_bindings"),
                        ("PRE_EVALUATION_SOURCE_FREEZE.json", "sources")]:
        original = read(name)
        rows = {n: {"recorded": old, "current": binding(ROOT / n)} for n, old in original[field].items()}
        assert all(v["recorded"] == v["current"] for v in rows.values()), name
        freeze[name] = rows
    sources = [p for p in ROOT.glob("*.py")]
    artifacts = ["config.json", "remote_config.json", "PROTOCOL.md", "DIFFUSION_CONSTANTS.npz",
                 "CONDITION_CONSTANTS.json", "CONDITIONAL_DIFFUSION_EDIT/FINAL_DENOISER.pt"]
    put("SOURCE_BINDINGS.json", {"UTC": utc(), "existing_execution_freezes_still_match": freeze,
        "current_stage_code": {p.name: binding(p) for p in sorted(sources)},
        "current_model_and_recipe": {n: binding(ROOT / n) for n in artifacts},
        "task": {"path": str(ROOT.parents[1] / "tasks/R36_CONDITIONAL_DIFFUSION_EDIT_001.md"),
                 "registered_sha256": "a2faa6898cf50e62684a0653e3720815de703e2889c077645bb096d4e30f32b2"},
        "source_weight": cfg["base_G"], "old_external_weights_not_rehashed_for_migration": True,
        "repaired_evaluation_preserves_original_failure_source": "failures/evaluation_001/evaluate_saved.py"})

    process_tools = {
        "train_server.py": {"start": "598a1a", "completion": "a06145"},
        "generate_all.py": {"start": "5c2c11", "completion": "da05a1"},
        "evaluate_saved.py": {"start": "d29190", "completion": "4e7595"},
        "evaluate_saved_repair.py": {"start": "975485", "completion": "797b1b"},
        "replay_all.py": {"start": "506d76", "completion": "aaf993"},
        "summarize_and_plot.py": {"completion": "e89460"},
        "fix_control_layout.py": {"completion": "7a23ff"},
        "fix_control_layout_v2.py": {"completion": "ca4023"}}
    native_chain = [{**row, "tool_evidence": process_tools[row["entry"]]} for row in history]
    put("NATIVE_CHAIN.json", {"first_control": read("START_RECEIPT.json"),
        "processes": native_chain, "science_elapsed_s": science_s,
        "all_scientific_processes_terminal": True, "external_native_tool_exits_observed": True,
        "control_only_assembly_PID": os.getpid(), "assembly_UTC": utc(),
        "same_stage_seal_native_exit_to_be_observed_after_close_stage_process_returns": True})

    snapshot = [p for p in ROOT.rglob("*") if p.is_file()]
    packages = [p for p in snapshot if p.suffix == ".tar"]
    copy_names = ["ACTUAL_GENERATION_INPUTS.npz", "JOINT8_INPUTS_PRIVATE.npz", "PROBE_INPUTS_CORRECTED.npz",
                  "DEV_SOURCE_METADATA.npz", "REAL_DEV_AND_REFERENCE_PRIVATE.npz", "CAL_REFERENCE_METADATA.json",
                  "DATA_ROLES.json", "VALLEY_DEFINITION.json", "AFFINE_CONSTANTS.npz"]
    copied_payload = sum((ROOT / name).stat().st_size for name in copy_names)
    assert copied_payload == prep["copied_dependency_payload_bytes"]
    snapshot_total = sum(p.stat().st_size for p in snapshot)
    package_bytes = sum(p.stat().st_size for p in packages)
    assert snapshot_total < cfg["output_limit_bytes"]
    put("RESOURCE_RECEIPT.json", {"PID": os.getpid(), "UTC": utc(),
        "registered_UTC": cfg["registered_UTC"], "executor_deadline_UTC": cfg["deadline_UTC"],
        "science_elapsed_s": science_s, "science_limit_s": cfg["science_budget"],
        "training_guard_elapsed_s": history[0]["elapsed_s"],
        "training_body_elapsed_s": train["records"]["CONDITIONAL_DIFFUSION_EDIT"]["training_s"],
        "denoiser_fit_attempts": 1, "denoiser_updates": 2950, "training_sampled_REAL_rows": 94400,
        "G_D_reader_flow_updates": 0, "CPU_threads": 4,
        "actual_worker_peak_RSS_bytes": max(row["actual_worker_RSS_peak_bytes"] for row in history),
        "RSS_limit_bytes": cfg["RSS_limit_bytes"],
        "GPU_peak_allocated_bytes": train["GPU_peak_allocated_bytes"],
        "GPU_allocated_limit_bytes": cfg["GPU_allocated_limit_bytes"],
        "memory_probe_failures": sum(row["memory_probe_failures"] for row in history),
        "memory_probe_failure_events": [event for row in history for event in row["memory_probe_failure_events"]],
        "RSS_missing_event_cause": "unknown; event was after worker finally receipt and before parent observed termination; do not assert cause",
        "generation_logical_chain_rows": gen["counter"], "independent_replay_rows": 4000,
        "HGB_attempts": 20, "density_description_slots": 16, "density_EM_attempts": 16,
        "completed_projection_group_blocks": metrics["counters"]["completed_projection_group_blocks"],
        "scientific_failure": {"entry": "evaluate_saved.py", "native_exit": 1,
            "elapsed_s": history[2]["elapsed_s"], "HGB_EM_or_projection_attempts_before_failure": 0,
            "reason": "historical raw REAL-REAL Q_PSD_2D prediction never existed",
            "repair": "kept unknown; no historical refit; original source/logs/receipt preserved"},
        "pure_control_failure": {"tool": "8308f5", "PID": 1408, "SSH_PID": 36704,
            "native_exit": 1, "remote_exit": 0, "elapsed_s": 1.475336,
            "reason": "inherited preflight used old R35 leaf; current leaf verified by corrected entry before science",
            "science_started": False, "preserved_directory": "control_failures/preflight_template_001"},
        "source_path_lookup_errors": "preserved in conversation tool outputs; no response fit/generation; no claim all control reads used a per-file ledger",
        "copied_existing_dependency_payload_bytes": copied_payload,
        "normal_transfer_package_bytes": {p.name: p.stat().st_size for p in packages},
        "normal_transfer_packages_total_bytes": package_bytes,
        "all_stage_files_before_resource_report_seal": {"file_count": len(snapshot), "bytes": snapshot_total},
        "unique_scientific_outputs_conservative_upper_bound_bytes": snapshot_total - copied_payload - package_bytes,
        "upper_bound_includes_source_copies_and_duplicate_receipts": True,
        "final_all_file_bytes_authoritative_in": "FINAL_MANIFEST.json",
        "unique_output_limit_bytes": cfg["output_limit_bytes"],
        "control_finalization_time_not_scientific_computation": True,
        "new_EVAL_CAL_G4_REAL_longwave_installation": 0, "migration_content_tests": 0,
        "server_training_then_local_CPU_only": True, "versions_training": read("SERVER_PREFLIGHT.json"),
        "versions_generation_evaluation_replay": metrics["versions"], "all_processes_terminal": True})

    new = "CONDITIONAL_DIFFUSION_EDIT"
    c2st = metrics["C2ST"]["results"]
    lines = ["# R36：完整源引导的固定条件扩散修正", "",
        "本次配方未通过联合门槛，按预定规则关闭；保留部分 PSD/峰宽和局部波形收益，不提升为合格的 G4→实测响应模型。门槛1–5均失败，独立完整回放及最终图目检通过。项目研究目标继续。", "",
        "原始完整480点 G4 派生 clean 波形先进入冻结 R32 CONTROL_FINAL 网络，再以其实际 preADC 输出为 guide。只在原支持内执行 t_start=50 的50步条件 DDPM，条件为 guide 的 logQtotal/PSD；支持外保留 BASE。没有目标库检索、独立目标域随机抽样、x0裁剪、投影、平滑或积分重标。", "",
        "REAL-only 去噪训练从零一次：1,372,097参数；原R33实际2950×32=94400 REAL target/bin/ordinal行序三项哈希全匹配，原共同支持18895/21047 TRAIN用于条件统计。初态/噪声/推理种子分别20261010360/361/362，固定末点，不用DEV选checkpoint。100步cosine日程保存beta cap .999后的实际cumprod，GN/FiLM一维U-Net和低分辨率注意力按冻结配置执行。没有更新原G/D/reader/流。", "",
        "源主2000行（pure_EE417/pure_had1518/mixed65）保留父ID、顺序和权重，其中1583编辑、417原G回退；既有2000 TRAIN_REFERENCE实测guide控制1787编辑、213原ADC记录回退。所有主行有效、有限，无ADC裁剪。PDG22/2112计数317/1683仅由旧桶规则和已绑定R2 CSV两种PDG推定，区别于逐事件显式PDG。现有源波形来自沉积总量解析响应，不含新恢复的微观时序。", "",
        "固定分类器仍为每域600训练/250留出、500人工采集记录组bootstrap。CI只反映固定分类器留出抽样，不覆盖响应训练不确定性；数据来自同一采集，人工50记录组不是独立采集批次。DEV及历史模型暴露保留，不称新盲测。", "",
        "| C2ST视图：留出BA | BASE | R35 PSD_ORDERED | R36源编辑 | RAW REAL–REAL | REAL guide控制 |",
        "|---|---:|---:|---:|---:|---:|"]
    for view in ["full_historical_shape", "normalized_u480", "full_voltage", "prompt_local", "postpeak_local", "tail_local", "baseline_local", "Q_PSD_2D"]:
        def ba(domain):
            value = c2st[domain].get(view, {}).get("balanced_accuracy")
            return "unknown" if value is None else f"{value:.3f}"
        lines.append("| " + view + " | " + " | ".join(ba(domain) for domain in ["BASE", "PSD_ORDERED_CFM", new, "RAW_REAL_REAL", "REAL_GUIDE_EDIT"]) + " |")
    lines += ["", "源编辑 full493 相对 BASE 的配对 ΔBA=-.020，95%CI[-.0395423,+.00077335]，上界未低于0；纯u480 ΔBA=-.024，CI[-.0457884,-.00197248]。相对R35 full493点值恶化+.014，超过+.01容差。完整分布仍强可区分，局部prompt/噪声分类下降不能代表整体成功。", "",
        "| PSD/噪声指标（相对REAL） | BASE | R36源编辑 | 范围 |",
        "|---|---:|---:|---|",
        "| 总体PSD W1 | .02306546 | .01870572 | 全2000父，改善18.90% |",
        "| 共同Q区等权PSD W1 | .03476580 | .02901041 | 原8窄区中的1–7，改善16.55% |",
        "| 低侧raw-width MAE | .01039272 | .00579135 | 同一合格区2–7，改善44.27% |",
        "| 高侧raw-width MAE | .01305117 | .00798988 | 同一合格区2–7，改善38.78% |",
        "| 低側峰位MAE | .00585717 | .01058073 | 同一合格区2–7，恶化 |",
        "| 高側峰位MAE | .00265866 | .00679959 | 同一合格区2–7，恶化 |",
        "| baseline RMS W1（V） | .0000193584 | .0000308620 | 恶化59.42% |",
        "| ACF1 W1 | .06423497 | .03975358 | 改善38.11% |", "",
        "BASE有效密度区2–7未新增失败。R36低端bin1两成分未分离，高端bin8高侧支持不足20；这些缺口原样保留。两成分仅PSD密度描述，不是独立粒子真值。峰宽靠近而峰位偏离，不能以W1或宽度单项宣称双带成功。", "",
        "实际未量化NN source-Q的pure-had bin3/4分别213/181父。谷比例由BASE的5.63%/0.55%升到14.55%/9.94%，增加8.92/9.39个百分点，超过2pp容差；源Q/PSD全体秩关联仍约.997418/.417549，不能抵消这项条件保真风险。历史postADC SOURCE_CLEAN参考207/182另列在metrics，不混称实际输入。谷比例不是粒子标签误判真值。", "",
        "九个固定边界中 second_center_150 的 W1=.000776690V，超过预定上限.000703809V；其他八项通过。128个固定c/bin/实际noise的guide扰动均改变输出，preADC配对RMS中位.000602099V，大于1LSB=.0000610389V。扰动输入的Qfast精确不变；float32 Qtotal最大绝对变化1.2267e-5mV·ns、PSD约1.0165e-9。此证据说明完整guide有作用，不能证明逐事件物理映射或保住输出电荷/PSD。", "",
        "REAL guide控制 full493/u480/baseline 相对RAW REAL–REAL的配对ΔBA上界分别.129944/.141378/.125944，均超过.10容差；完整电压BA=.780。处理已经是实测的guide仍造成可分变形。该控制包含TRAIN样本，不能当未见采集测试。RAW REAL–REAL Q_PSD_2D旧预测从未存在，保留unknown且未补拟合；三个预定控制门槛的旧预测和同一划分完整复用。RAW W1仅作抽样参照，不另套未登记的相对阈值。", "",
        "source交换及同父变噪声探针有响应；同输入/同随机流128行逐值相同。joint8报告BASE signal/electronic、加法舍入及编辑增量和全部交叉项，编辑后物理signal/electronic分解仍unknown。非零源敏感性不等于正确物理对应。", "",
        "独立本机CPU新进程完成源2000和REAL控制2000的完整链回放：pre/postADC、实际noise hash、种子、条件、support、父身份及BASE/edit分量全部逐字节通过。兼容字段名flow_increment只是扩散编辑增量，未运行旧CFM。", "",
        f"累计科学 {science_s:.6f}s / 3000s（含首次失败9.772323s和三次绘图/布局进程）；唯一训练guard 53.429464s，训练体51.207049s。实际科学worker峰RSS1,950,699,520B，GPU allocated峰353,069,056B；服务器RSS查询1次缺测发生在退出附近，原因unknown，保留时间戳。新20HGB/16EM，旧分类器/密度/RAW RR只复用。", "",
        "首次评价因缺少旧Q_PSD_2D RR预测在任何HGB/EM前native1；原源码、日志、回执留在failures，修复入口仅将缺失描述参照写unknown，没有重训/重生成或重拟合已完成块。早期preflight旧路径模板控制失败及两次布局修复均保留。迁移使用正常传输退出码，未做迁移内容一致性测试。", "",
        "十张最终图均已实际打开：完整CDF/电压、条件均值/SD、峰位峰宽和支持缺口、源分组、边界、同父实例、实测控制及guide作用。control图只缩短标签；原初图与第一次短标签图留档。详见FIGURE_QA.json。", "",
        "交付：metrics.json为完整本地结果；METRICS_COMPACT.json为聚合发布版；GATE_METRICS.json保留预定全门槛；权重/日程/条件常数、全部实际主/辅助输入和随机关联、精确preADC与ADC输出及复现入口保留。此固定有限加噪配方关闭，不扫描强度、日程、方差、种子或架构；本结果不宣称所有扩散模型不可能。"]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    reproduction = f"""# R36 复现与保存入口

本目录已执行证据，不是再次运行许可。训练仅一次，固定最终权重；生成/评价/完整主回放均在本机CPU4执行。旧阶段只读。

训练服务器目录：`{cfg['remote_root']}`；解释器：`{cfg['server_python']}`。正常部署由deploy_training.py、launch_remote.py和linux_supervise.py/train_server.py完成；训练actual50548/super50547，原生exit0见TRAINING_NUMERIC_RECEIPT.json和remote_receipts/train_server.py.NATIVE_RECEIPT.json。服务器既有Python3.12.3/NumPy1.26.4/torch2.12.0+cu130、RTX5070；训练输入复用远端R33的TRAIN target/原实际流和仿射，不需要新G4、原长采集或EVAL。

本机解释器：`{cfg['CPU_python']}`，Python3.13.3/NumPy2.2.6/torch2.10.0+cpu/sklearn1.7.2。实际入口形式：`python -B supervise.py <entry.py> <CPU_python>`。已执行generate_all.py（PID30272/native0）、evaluate_saved_repair.py（28588/native0）、replay_all.py（53452/native0）、summarize_and_plot.py（60552/native0）；原evaluate_saved.py首次失败留档。supervise.py使用entry.used防重复提交、导入前真实PID/nonce握手和RSS守护；复现实验需复制到新的E/D目录，并明确获得新预算，不能删掉本目录used记录重跑。

部署推理最小依赖为diffusion_core.py、flow_runtime.py、model.py/legacy_model.py、flow_core.py，冻结BASE G（`{cfg['base_G']}`）、CONDITIONAL_DIFFUSION_EDIT/FINAL_DENOISER.pt、DIFFUSION_CONSTANTS.npz及保存record seeds。`Pipeline`先对完整source执行BASE，`set_seeds`后`infer`执行固定50步编辑，最后model.adc一次。它不加载REAL训练库。condition来自实际guide；支持来自原未量化source-Q。不能从扰动guide重算固定探针condition。

已有完整源输入：ACTUAL_GENERATION_INPUTS.npz（source_waveforms_V/source_ids及原z/white/style）、DEV_SOURCE_METADATA.npz；REAL control仅使用REAL_DEV_AND_REFERENCE_PRIVATE.npz既存reference_V/IDs/groups。JOINT8_INPUTS_PRIVATE.npz、PROBE_INPUTS_CORRECTED.npz及ACTUAL_DIFFUSION_SEED_MAPS_PRIVATE.npz保存辅助/种子关联。条件常数/日程在CONDITION_CONSTANTS.json和DIFFUSION_CONSTANTS.npz；核心源码冻结哈希在PRETRAIN_FREEZE/PRE_GENERATION_SOURCE_FREEZE/SOURCE_BINDINGS。

主结果：CONDITIONAL_DIFFUSION_EDIT/generated.npz与REAL_GUIDE_EDIT/generated.npz保留精确pre_ADC_V、waveforms_V、base_pre_ADC_V、flow_increment_f64_V、support、conditions、record seed、实际noise hash和身份。flow命名只是兼容编辑增量，不代表使用旧CFM。额外guide原/扰动/输出与固定c在results/GUIDE_PERTURBATION_PRIVATE.npz。原full480/source-probe/joint实际随机序列保持完整，不用ID重新抽噪声。

REPLAY_RECEIPT.json由独立CPU进程实际完成4000条全链逐字节比较，包括精确噪声哈希/条件/支持/父ID和BASE/edit分量，全部通过；未重拟合任何模型。源主preADC数组SHA b82aa8763568287917fe0cf7101471159231d681c0217b95db539bfa41c0d073、ADC数组SHA e11f1dd8072342aebb1f6eb5de4b33bf41bc509dd2cf35b9c0bd712cd6587b04。REAL控制对应数组SHA见回放回执。数组SHA是已执行回放证据，不是跨端迁移测试。

评价复用原600/250分组、旧BASE/R35/RAW REAL–REAL的预测及既有TRAIN/Q-bin/密度定义；新20HGB与16EM均已保存到results。完整metrics/预测/身份/噪声为本地私有结果，METRICS_COMPACT仅聚合。没有EVAL、新REAL或CAL波形解码、原长波形读取、安装或新增G4。

恢复到新目录时按config中明确路径恢复这些既有输入和权重、保持core/常数/实际随机映射/CPU线程和版本；更改项目根只调整路径映射，不更改科学配方。所有缓存TEMP/TMP/TORCH_HOME/CUDA_CACHE_PATH/MPLCONFIGDIR/JOBLIB_TEMP_FOLDER设E/D。不同平台训练不声称CPU/CUDA逐位一致；本次完整推理与独立推理均是已记录本机环境。

这是同采集、已历史暴露的开发诊断，不是新盲测。固定配方联合门槛失败；可复现性不等于响应物理真实性。
"""
    (ROOT / "REPRODUCTION.md").write_text(reproduction, encoding="utf-8")
    put("DELIVERY_ASSEMBLY_RECEIPT.json", {"PID": os.getpid(), "UTC": utc(),
        "elapsed_s": time.monotonic() - START, "operation": "JSON/control/byte bindings and prose only",
        "science_refit_generation_replay_or_array_decode": 0,
        "written": ["REPORT.md", "REPRODUCTION.md", "METRICS_COMPACT.json", "METRICS_AGGREGATE_EXTENDED.json",
                    "FIGURE_QA.json", "RESOURCE_RECEIPT.json", "SOURCE_BINDINGS.json", "NATIVE_CHAIN.json"],
        "science_elapsed_s_unchanged": science_s, "all_ten_figures_actual_reviewed": True})
    print(json.dumps({"PID": os.getpid(), "UTC": utc(), "science_elapsed_s": science_s,
                      "REPORT": str(ROOT / "REPORT.md"), "all_gates_pass": False,
                      "assembly_control_only": True}), flush=True)


if __name__ == "__main__":
    main()
