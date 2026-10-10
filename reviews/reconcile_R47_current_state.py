"""Reconcile current control fields to the already registered R47 task."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
STAGE = "R47_POPULATION_NOISE_DIAGNOSTIC_001"
NOW = datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((ROOT / name).read_text("utf-8-sig"))


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


reg = read("reviews/R47_REGISTRATION_001/RECEIPT.json")
ack = {
    "stage": STAGE,
    "recorded_UTC": NOW,
    "recording_PID": os.getpid(),
    "executor": "/root/sim2real_sol_max_executor",
    "reported_first_control_PID": 44412,
    "reported_first_control_UTC": "2026-10-10T05:37:47.5032863+00:00",
    "reported_native_tool_chunk": "a8dbb4",
    "reported_native_exit_code": 0,
    "evidence_source": "executor acknowledgment message; first control read, not scientific process start",
    "scientific_process_start_observed": False,
    "status": "executor_implementing_registered_diagnostic",
}
write("reviews/R47_EXECUTOR_ACK_001.json", ack)

for name in ("PROGRAM.json", "PROJECT_PROGRESS.json"):
    x = read(name)
    snapshot = ROOT / "reviews/R47_REGISTRATION_001" / ("before_field_reconciliation_" + name)
    if not snapshot.exists():
        snapshot.write_bytes((ROOT / name).read_bytes())
    x.update({
        "current_stage": STAGE,
        "current_task": f"tasks/{STAGE}.md",
        "current_stage_scope": "saved_population_and_functional_noise_diagnostic_no_response_training",
        "current_activity": "R47_executor_implementation",
        "current_stage_progress": "executor_control_ack_received_scientific_start_not_yet_observed",
        "current_stage_deadline_UTC": reg["stage_deadline_UTC"],
        "stage_deadline_UTC": reg["stage_deadline_UTC"],
        "executor_seal_deadline_UTC": reg["executor_seal_deadline_UTC"],
        "current_direction": "Use saved R46 outputs to distinguish charge population differences and changes at initial-noise versus learned-flow stages before defining the final method plan.",
        "latest_user_steering": "构建计划推进",
        "latest_user_steering_receipt": "reviews/R47_REGISTRATION_001/RECEIPT.json",
        "execution_plan": "PROJECT_EXECUTION_PLAN_20261010_002.md",
        "current_executor_ack": "reviews/R47_EXECUTOR_ACK_001.json",
        "acceptance_version": read("PROJECT_ACCEPTANCE.json")["version"],
        "original_fixed_charge_flow_family_closed": True,
        "current_route_research_closed": False,
    })
    x["execution_location"]["mode"] = "server_CPU12_saved_output_diagnostics_local_auxiliary"
    if name == "PROGRAM.json":
        x["updated_at_utc"] = NOW
        x["current_job"] = {"stage": STAGE, "status": ack["status"], "scientific_start_observed": False}
        x["latest_permission_status"] = "User requests a new execution plan and progress; R47 registered saved-output diagnosis with unchanged A-F."
    else:
        x["updated_UTC"] = NOW
    write(name, x)

for name, key in (("jobs.json", "jobs"), ("experiments.json", "experiments")):
    x = read(name)
    for row in x[key]:
        if row.get("stage", row.get("id")) == STAGE:
            row["status"] = ack["status"]
            row["executor_ack"] = "reviews/R47_EXECUTOR_ACK_001.json"
            row["scientific_start_observed"] = False
    write(name, x)

path = ROOT / "PROJECT_OBJECTIVE.md"
value = path.read_text("utf-8-sig")
old = "原验收与源权重保持；未登记R47。"
new = "原验收与源权重保持。R47已登记并交既有执行器实施，使用保存数据诊断人口与噪声；下一响应方法尚未登记。"
assert old in value
path.write_text(value.replace(old, new, 1), encoding="utf-8")

write("reviews/R47_STATE_RECONCILIATION_001.json", {
    "UTC": NOW, "PID": os.getpid(), "stage": STAGE,
    "status": "current_task_direction_deadlines_and_ack_reconciled",
    "registered_protocol_or_acceptance_modified": False,
    "sealed_scientific_stage_modified": False,
    "scientific_jobs_submitted_by_root": 0,
})
print(json.dumps({"status": "reconciled", "UTC": NOW, "PID": os.getpid()}, ensure_ascii=False))
