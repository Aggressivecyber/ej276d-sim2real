"""Control-only closure. Does not import scientific libraries or decode arrays."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def write(name, obj):
    path = ROOT / name
    assert not path.exists(), f"closure output already exists: {name}"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def text_file(name, text):
    path = ROOT / name
    assert not path.exists(), f"closure output already exists: {name}"
    path.write_text(text, encoding="utf-8")


def binding(path):
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
            n += len(block)
    return {"bytes": n, "sha256": h.hexdigest()}


def main():
    started = now()
    cfg = read("config.json")
    deadline = datetime.fromisoformat(cfg["deadline_UTC"])
    assert datetime.now(timezone.utc) < deadline
    numeric = read("NUMERIC_RECEIPT.json")
    auxiliary = read("AUXILIARY_NUMERIC_RECEIPT.json")
    formula = read("FORMULA_NUMERIC_CHECK.json")
    marker = read("DIRECTION_MARKERS.json")
    velocity = read("VELOCITY_SUMMARY.json")
    population = read("POPULATION_BACKGROUND.json")
    remote = read("REMOTE_RUN_HISTORY.json")
    local = read("RUN_HISTORY.json")
    transport = read("LOCAL_NATIVE_RECEIPT.json")
    stream = read("ACTUAL_INPUT_STREAM_BINDING.json")
    plots = read("PLOT_RECEIPT.json")
    initial_bindings = read("PRE_EXECUTION.json")
    old_inputs = read("INPUT_BINDINGS.json")
    environment = read("ENVIRONMENT.json")
    failures = read("CONTROL_FAILURES.json")
    assert len(remote) == len(local) == 1
    r, l = remote[0], local[0]
    science_s = math.fsum([r["elapsed_s"], l["elapsed_s"]])
    assert science_s <= cfg["science_budget"]
    assert numeric["status"] == "complete"
    assert numeric["attempted_forward_rows"] == numeric["committed_rows"] == numeric["network_forward_rows"] == 94400
    assert numeric["forward_calls"] == 2950 and numeric["chunk"] == 32
    assert numeric["response_updates"] == numeric["new_ODE_generation"] == numeric["HGB_EM_bootstrap_fits"] == 0
    assert numeric["weight_parameters_before_after_equal"] and formula["pass"]
    assert auxiliary["new_network_forward_rows"] == auxiliary["new_ODE_generation_training_fits"] == 0
    assert r["actual_worker_native_exit"] == l["actual_worker_native_exit"] == transport["SSH_native_exit"] == 0
    assert r["handshake_before_imports"] and l["handshake_before_imports"]
    assert marker["hits"] == auxiliary["direction_hits"] == 1 and marker["required_hits"] == 4
    assert not marker["supported"]
    assert all(stream["white_random_reconstruction_and_unused_audit_match"].values())
    assert stream["original_rng_initial_final_verified"] and stream["saved_t_no_new_draw"]
    assert len(plots["figures"]) == 4

    qa = {
        "reviewer": "same sole executor; actual image display reviewed",
        "actual_opened": True,
        "opened_via": "view_image(original) for all four final PNGs before control-only closure",
        "QA_record_UTC": now(),
        "plots": {},
        "rerender_after_success": False,
        "producer_pending_field": "PLOT_RECEIPT retains original pending flag; this later actual QA resolves it without changing the producer receipt",
    }
    notes = {
        "PSD_velocity_Q_time.png": "All 8x4 cells and sample counts are readable; target/prediction share a signed scale.",
        "PSD_MSE_geometry.png": "PSD-mode fraction and complementary MSE are readable; a small MSE fraction is not a physical irrelevance claim.",
        "population_PSD_CDF.png": "Seven saved populations remain separately labeled with fixed Q bins; shared axis and all CDF samples retained.",
        "population_means_direction_markers.png": "Mean/SD and preregistered ratio threshold are readable; only bin7 is marked HIT.",
    }
    for name, expected in plots["figures"].items():
        actual = binding(ROOT / "results" / "figures" / name)
        assert actual == expected
        qa["plots"][name] = {**actual, "actual_opened": True, "pass": True, "note": notes[name]}
    write("FIGURE_QA.json", qa)

    compact = {
        "stage": cfg["stage"],
        "decision": marker["decision"],
        "direction_markers": marker,
        "velocity": velocity,
        "population_background": population,
        "formula_checks": formula,
        "counts": {"forward_rows": 94400, "forward_calls": 2950, "loaded_SOURCE_REAL": numeric["loaded_source_REAL"],
                   "unique_consumed_SOURCE_REAL": numeric["unique_source_matched_REAL"], "response_updates": 0, "ODE": 0, "new_HGB_EM": 0},
        "scientific_elapsed_s": science_s,
        "interpretation": {"TRAIN_paths_are_not_ODE_paths": True, "same_acquisition_retrospective_DEVELOPMENT": True,
                           "mixture_or_PSD_component_is_not_particle_truth": True,
                           "one_dimensional_MSE_fraction_does_not_establish_causality": True},
        "private_row_records": "results/PATH_ROWS_PRIVATE.npy and POPULATION_BACKGROUND_PRIVATE.npz stay local; not included here",
    }
    write("METRICS_COMPACT.json", compact)

    failures = failures + [{
        "tool": "000946", "native_exit": 1, "PID": "unknown; PowerShell command did not print its PID",
        "reason": "control read used results/POPULATION_BACKGROUND.json; saved JSON is at the stage root",
        "science_s": 0, "stage_writes": 0,
        "recovery": "control tool77981c/PID56468/native0 read the existing stage-root JSON; no scientific block repeated",
    }]
    write("FAILURE_ACCOUNTING.json", {
        "scientific_failures": [], "control_failures": failures,
        "two_remote_RSS_query_gaps": {"count": r["memory_probe_failures"], "events": r["memory_probe_failure_events"],
                                      "cause": "unknown", "near_stop_timing_not_proof_of_cause": True},
        "no_retry_of_94400_forward": True,
    })

    effective_sources = {}
    with tarfile.open(ROOT / "UPLOAD.tar", "r:") as archive:
        for name in ["diagnose_frozen.py", "network.py", "source_flow_core.py", "reorder_core.py", "linux_supervise.py", "linux_owned_worker.py", "config_remote.json"]:
            member = archive.getmember(name)
            data = archive.extractfile(member).read()
            effective_sources[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "basis": "retained local deployment archive source bytes"}
    source_note = {
        "initial_PRE_EXECUTION_linux_supervise": initial_bindings["source_bindings"]["linux_supervise.py"],
        "effective_deployment_linux_supervise": effective_sources["linux_supervise.py"],
        "explanation": "PRE_EXECUTION's earlier Linux-guard source hash predates the added immediate admission check. Its original record is retained. The deployment archive records the actual final guard source; no source/destination migration comparison was run.",
    }
    write("SOURCE_BINDINGS.json", {
        "executed_remote_sources": effective_sources,
        "local_population_and_guard_sources": {n: binding(ROOT / n) for n in ["populations_and_plot.py", "diagnose_frozen.py", "supervise.py", "owned_worker.py", "config.json"]},
        "unused_historical_pre_resource_sources": ["diagnose_frozen_pre_resource_unused.py", "config_original_registered_pre_resource.json"],
        "old_scientific_input_bindings": old_inputs,
        "old_scientific_hashes": "copied from sealed R43 manifest; not a new whole-source hash audit",
        "effective_guard_binding_clarification": source_note,
        "normal_transfers": "SSH/scp native exits and actual loading; no migration consistency tests",
    })

    chain = {
        "first_actual_control": cfg["first_actual_command"],
        "server_preflight": {"tool": "54213d", "native_exit": 0, "local_PID": 28244, "server_PID": 101167, "receipt": "SERVER_PREFLIGHT.json"},
        "deploy": {"tool": "8b362f", "native_exit": 0, "create_PID": 102123, "extract_PID": 102238, "upload_archive_bytes": 61440},
        "GPU_forward": {"start_tool": "186264", "session": 58075, "completion_tool": "adffb5", "native_exit": 0,
                        "local_controller_PID": 48072, "SSH_PID": 7804, "receipt": "REMOTE_RUN_HISTORY.json", "actual": r},
        "return": {"tool": "d7c799", "native_exit": 0, "remote_tar_PID": 102760, "local_extract_PID": 36088, "archive_bytes": 17541120},
        "CPU_population_plot": {"start_tool": "e73fd7", "session": 37944, "completion_tool": "b1a1d1", "native_exit": 0,
                                "receipt": "RUN_HISTORY.json", "actual": l},
        "seal": {"PID": os.getpid(), "start_UTC": started, "native_exit": "external tool completion is authoritative; not self-certified"},
    }
    write("NATIVE_CHAIN.json", chain)

    population_table = []
    for name, record in population.items():
        overall = next(row for row in record["rows"] if row["bin_id"] is None)
        outside = next(row for row in record["rows"] if row["bin_id"] == "outside")
        population_table.append(f"|{name}|{overall['N']}|{overall['mean']:.6f}|{overall['SD_ddof1']:.6f}|{outside['N']}|")
    marker_table = [f"|{b['bin_id']}|{b['path_N']}|{b['endpoint_N']}|{b['B_signed_velocity_residual']:.6f}|{b['D_mean_abs_target_velocity']:.6f}|{b['G_endpoint_minus_consumed_PSD_mean']:.6f}|{b['abs_B_over_D']:.6f}|{'yes' if b['hit'] else 'no'}|" for b in marker["bins"]]
    o = velocity["overall"]
    report = f"""# R44 冻结 PSD 速度诊断

预定方向规则仅命中 **1/6** 电荷区（bin7），低于要求的4/6。本轮关闭“平均 TRAIN PSD 速度残差足以解释终点 PSD 偏移”这一具体解释路线；不调整 PSD 权重、不追加训练。R43 仍为已关闭的科学候选，项目完成指标没有因此通过。

唯一 GPU 作业对冻结 R43 末点在原2950×32=94400条 TRAIN 线性插值状态上前向一次，chunk32、float64坐标/几何与float32网络。原 source/target/bin/ordinal/t/white及原随机状态绑定通过，权重初末相同。载入6593 SOURCE和21047 REAL，实际消费3752与18739唯一记录；94400是重复暴露次数，不是独立样本数。没有响应更新、ODE求解、新输出生成、HGB/EM或bootstrap。

定义 q=sum(100:400)-3sum(0:100)，F=sum(100:150)-0.5sum(0:100)，signed PSD=1-F/q。速度为单位归一化t下的PSD变化；路径上q=1，PSD速度为−F(scale·v)。残差e=预测−目标，B=mean(ePSD)，D=mean(abs(targetPSDvelocity))，G=已有R43 preADC端点PSD均值−该区已消费REAL均值。命中要求abs(B)/D≥0.20且B与G同号。输出区按实际SOURCE preADC Q，REAL区按自身preADC Q；不混用历史ADC参考Q。

|bin|路径N|端点N|B|D|G|abs(B)/D|命中|
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(marker_table)}

全部路径signed残差均值 **{o['signed_mean']:.9f}**，PSD残差RMSE **{o['PSD_RMSE']:.9f}**；目标速度均值 {o['target_velocity_mean']:.9f}，预测 {o['predicted_velocity_mean']:.9f}。电荷切空间内PSD方向占标准化480维MSE能量 **{100*o['PSD_mode_energy_fraction']:.6f}%**。这是一维几何分解，不说明PSD不重要，也不能推断优化器或实际ODE的因果。公式/电荷恒等最大误差≤4.17e−16；逐行能量闭合最大误差9.10e−13。按区汇总百万量级能量存在≤9.32e−10浮点加法尾差，整体闭合差0。

人口背景独立保存，未按结果重选、删支持外事件或将两成分当粒子真值。每个Q区完整CDF与mean/SD均在JSON/图中；总体数值不能代替Q条件比较。STD使用ddof1。

|人口|N|PSD均值|SD|支持外N|
|---|---:|---:|---:|---:|
{chr(10).join(population_table)}

TRAIN消费/唯一/完整人口的总体均值相近，但支持范围和权重不同；完整TRAIN支持外2152行仍单列。R43输出preADC→postADC均值0.293287→0.293289，仍与实测右移；417条支持外SOURCE及输出全部保留。人口组成、有限耦合、局部随t变化及轨迹外行为不能由本次平均残差规则排除。诊断只覆盖原TRAIN线性路径，不等同新ODE验证或新盲测；REAL来自同次采集、DEV有历史暴露。

执行位置在任何正式前向前按用户资源修订切到RTX5070：Python3.12.3、NumPy1.26.4、Torch2.12.0+cu130；CPU12、进程树48GiB、GPU10GiB、科学3600秒、输出2GiB。CPU4仅做人群汇总和绘图。两作业native0、累计 **{science_s:.9f}秒**，GPU worker RSS1,656,750,080B，本机汇总RSS1,287,458,816B。两次远端RSS查询缺口原样保存，原因unknown；自报高水位与树采样峰分别列明。无跨平台逐位相同声明、无迁移内容一致性测试。三个纯控制错误保留，科学前向没有重复。四张最终图均已实际打开QA。初版Linux守护绑定与最终上传源的小版本差异见SOURCE_BINDINGS，原记录未覆盖。

主要入口：diagnose_frozen.py；既存行记录results/PATH_ROWS_PRIVATE.npy。汇总/图：VELOCITY_SUMMARY.json、POPULATION_BACKGROUND.json、DIRECTION_MARKERS.json、results/figures。完整逐行载荷留E/D，发布用METRICS_COMPACT.json。封存后停止阶段写入；下一科学问题由根另行登记。
"""
    text_file("REPORT.md", report)
    text_file("REPRODUCTION.md", f"""# R44 reproduction

Actual server root: `{cfg['remote_root']}`. Entry: `{cfg['server_python']} -B linux_supervise.py diagnose_frozen.py`; actual worker102441/supervisor102436, native0, 94400rows/2950calls. Frozen checkpoint/constants/coupling are in the read-only R43_REMOTE_20261009_233131_UTC directory. TRAIN_PRIVATE.npz is the cached R33 input; consumed R26 stream and R23 random-state/receipt files are the cached R32 inputs. Exact paths are in config_remote.json, input identities in INPUT_BINDINGS and effective deployed source hashes in SOURCE_BINDINGS.

In a new isolated E/D directory or a new project-owned server leaf, copy only the necessary scripts/config and map those existing inputs. Preserve chunk32, original saved t/white/coupling, float64 coordinates and float32 network, and one forward over each of the94400rows. No optimizer, ODE, new model fit or generated candidate belongs to this diagnostic. Do not rerun in this sealed directory. Normal transfer exit and actual loading were used; no migration byte-equality test was performed.

The subsequent local CPU4 entry was `{cfg['CPU_python']} -B supervise.py populations_and_plot.py` (actual worker29392/native0). It consumes the saved path rows and existing R43 SOURCE/REAL/outputs to produce backgrounds and four PNGs. Reuse saved aggregates for report/plot changes; no additional frozen-model forward is necessary. Actual formulas and resource revision are in PROTOCOL and the original task/server addendum. CUDA-vs-CPU bitwise equivalence was not tested or claimed.

NATIVE_CHAIN lists real PID/nonce/native completion and resource receipts. Original controls/errors, original upload/return tar, private rows and population arrays remain local. FINAL_MANIFEST binds the stage bytes; its own bytes and FINAL_RECEIPT are explicitly excluded to avoid self-reference. This experiment reports a closed explanation rule, not a validated physical source-to-REAL response model.
""")

    write("RESOURCE_RECEIPT.json", {
        "stage": cfg["stage"], "effective_resource_addendum": "tasks/R44_SERVER_RESOURCE_ADDENDUM_001.md",
        "first_actual_control": cfg["first_actual_command"], "executor_deadline_UTC": cfg["deadline_UTC"],
        "science_budget_s": 3600, "science_elapsed_s": science_s,
        "accounting": "Sum full supervised GPU-forward and CPU-population/plot elapsed; worker-inner and SSH transport times are nested, not added twice. Pure control read/write/seal has no scientific operations.",
        "GPU_forward_elapsed_s": r["elapsed_s"], "CPU_background_plot_elapsed_s": l["elapsed_s"],
        "server_environment": environment, "server_CPU_threads": 12, "local_CPU_threads": 4,
        "server_job_tree_RSS_limit_bytes": 48 * 1024**3, "local_RSS_limit_bytes": 4 * 1024**3,
        "GPU_limit_bytes": 10 * 1024**3,
        "server_worker_RSS_high_water_bytes": r["actual_worker_RSS_peak_bytes"],
        "server_job_tree_sampled_RSS_peak_bytes": r["job_tree_RSS_peak_bytes"],
        "RSS_basis": "worker self high-water and monitored actual-PID/tree samples; sampled tree maximum need not equal self high-water",
        "GPU_peak_allocated_bytes": numeric["GPU_peak_allocated_bytes"], "GPU_peak_reserved_bytes": numeric["GPU_peak_reserved_bytes"],
        "GPU_process_sampled_peak_bytes": r["GPU_process_memory_peak_bytes"],
        "local_actual_worker_RSS_peak_bytes": l["actual_worker_RSS_peak_bytes"],
        "RSS_gaps": r["memory_probe_failure_events"], "RSS_gap_cause": "unknown", "GPU_query_failures": r["GPU_query_failures"],
        "forward_rows": 94400, "forward_calls": 2950, "forward_repetitions": 0,
        "training_updates_ODE_generation_new_HGB_EM_bootstrap": 0,
        "new_EVAL_CAL_original_long_waveform_G4_reads": 0,
        "normal_upload_archive_bytes": 61440, "normal_return_archive_bytes": 17541120,
        "migration_consistency_tests": 0, "new_installations": 0,
        "control_failures_count": len(failures), "scientific_failures_count": 0,
        "output_limit_bytes": 2 * 1024**3, "final_output_accounting": "FINAL_MANIFEST manifested bytes plus FINAL_MANIFEST and FINAL_RECEIPT bytes; no unknown source I/O claimed zero",
        "control_read_write_exact_byte_ledger": "not required by simplified stage; no new complete data-I/O claim",
        "seal_PID": os.getpid(), "seal_start_UTC": started,
    })

    excluded = {"FINAL_MANIFEST.json", "FINAL_RECEIPT.json"}
    items = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.relative_to(ROOT).as_posix() not in excluded:
            assert not path.is_symlink()
            items.append({"path": path.relative_to(ROOT).as_posix(), **binding(path)})
    total = sum(item["bytes"] for item in items)
    assert total + 1024 * 1024 < cfg["output_limit_bytes"]
    assert datetime.now(timezone.utc) < deadline
    write("FINAL_MANIFEST.json", {
        "PID": os.getpid(), "UTC": now(), "stage": cfg["stage"],
        "files": items, "file_count": len(items), "manifested_bytes": total,
        "self_reference_exclusions": sorted(excluded),
        "scope": "New R44 stage only; hashes are stage artifact seals, not migration comparisons or new scientific source audits",
    })
    manifest = binding(ROOT / "FINAL_MANIFEST.json")
    sealed_utc = now()
    assert datetime.fromisoformat(sealed_utc) < deadline
    final = {
        "stage": cfg["stage"], "PID": os.getpid(), "status": "sealed", "last_stage_write_UTC": sealed_utc,
        "executor_deadline_UTC": cfg["deadline_UTC"], "deadline_met": True,
        "decision": marker["decision"], "direction_hits": 1, "required_hits": 4,
        "forward_rows": 94400, "scientific_elapsed_s": science_s,
        "response_training_ODE_new_HGB_EM": 0, "scientific_native_exits": [0, 0],
        "actual_QA_figure_count": 4,
        "manifest": {"path": "FINAL_MANIFEST.json", **manifest},
        "manifested_file_count": len(items), "manifested_bytes": total,
        "output_limit_bytes": cfg["output_limit_bytes"],
        "self_reference": "FINAL_MANIFEST and this FINAL_RECEIPT excluded from manifest; external tool output supplies this receipt hash/native completion",
        "post_seal_stage_writes": "none authorized or planned",
        "native_seal_exit": "external completion required",
    }
    write("FINAL_RECEIPT.json", final)
    receipt = binding(ROOT / "FINAL_RECEIPT.json")
    # No stage writes after FINAL_RECEIPT. Metadata inspection only.
    actual_last_write = datetime.fromtimestamp((ROOT / "FINAL_RECEIPT.json").stat().st_mtime, timezone.utc).isoformat()
    assert datetime.fromisoformat(actual_last_write) < deadline
    print(json.dumps({
        "seal_PID": os.getpid(), "actual_last_write_UTC": actual_last_write,
        "FINAL_MANIFEST": manifest, "FINAL_RECEIPT": receipt,
        "total_stage_files": len(items) + 2,
        "total_stage_bytes": total + manifest["bytes"] + receipt["bytes"],
        "scientific_elapsed_s": science_s, "direction_hits": 1,
        "no_further_stage_writes": True,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
