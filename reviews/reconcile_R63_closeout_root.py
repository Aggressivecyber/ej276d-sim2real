"""Keep current entrypoints explicit about the completed science and pending delivery."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
OUT = R / "reviews/R63_WAIT_PARALLEL_001/root_analysis/CLOSEOUT_DOCUMENT_RECONCILIATION.json"
assert not OUT.exists()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

changed = []
for name in ("AGENTS.md", "PLAN.md", "PROJECT_OBJECTIVE.md", "reviews/finalize_R63_root.py"):
    p = R / name
    before = sha(p)
    text = p.read_text(encoding="utf8")
    assert text.count("当前完成私有 Git 与 D 盘交付。") == 1, name
    text = text.replace("当前完成私有 Git 与 D 盘交付。", "私有 Git 与 D 盘交付待完成。", 1)
    p.write_bytes(text.encode("utf8"))
    changed.append({"path": name, "before_sha256": before, "after_sha256": sha(p)})

p = R / "README.md"
before = sha(p)
text = p.read_text(encoding="utf8")
old = "当前R63仍按原冻结50000步末点及后处理协议运行。"
assert text.count(old) == 1
text = text.replace(old, "R63已完成原冻结50000步末点及全部后处理，原A/B/C仍未通过；后续科学工作仍须单独预登记。", 1)
entry = ("# 当前入口：R63科学终点完成，长期目标仍 active\n\n"
         "R63的50000步训练、数值检查、完整评价与4000条独立逐位重放已完成，10张图已实际打开复核。"
         "原A/B/C均未通过：基线和部分全窗区分度改善，PSD及Q3/Q4必需密度描述仍失败。"
         "本次固定方案已关闭；下一项工作为预登记的冻结TRAIN端点、条件参照与小批量梯度诊断。"
         "实际交付状态以回执为准。\n\n"
         "[R63报告](R63_REPORT.md) · [科学回执](work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/FINAL_RECEIPT.json) · "
         "[后续方向](reviews/R63_ROOT_DIRECTION_001.json) · [验收标准](PROJECT_ACCEPTANCE.md)\n\n")
p.write_bytes((entry + text).encode("utf8"))
changed.append({"path": "README.md", "before_sha256": before, "after_sha256": sha(p)})
for name in ("reviews/finalize_R63_root.py", "reviews/deliver_R63_root.py"):
    compile((R / name).read_text(encoding="utf8"), str(R / name), "exec")

receipt = {"UTC": datetime.now(timezone.utc).isoformat(), "status": "current_entrypoints_reconciled",
           "changed_files": changed, "scientific_method_weights_or_acceptance_changed": False,
           "D_auxiliary_delivery": "PowerShell launcher source included in Git selection; complete non-cache auxiliary outputs included in D payload.",
           "whitespace_check": "Use the repository delivery command including cr-at-eol for existing Windows line endings.",
           "model_goal_complete": False}
OUT.write_bytes((json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf8"))
print(json.dumps({"status": receipt["status"], "changed_files": len(changed)}, indent=2))
