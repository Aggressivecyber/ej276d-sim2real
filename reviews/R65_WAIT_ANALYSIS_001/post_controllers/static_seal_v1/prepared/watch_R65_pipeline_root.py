"""Advance registered post entries only after each genuine native success.

No failed/unknown attempt is resubmitted. A local stop request never kills the
remote worker. Training and recovery are exclusively managed by root elsewhere.
"""
import json
import time
from r65_controller_common import parser, run_owned, remote_call, launch_entry, fetch_entry, write_new, utc


def watch(owner):
    write_new(owner.root / "PIPELINE_WATCH_STARTED.json", {"UTC": utc(), "operation": owner.operation,
               "poll_interval_seconds": 30, "automatic_retry": False, "training_launch": False})
    while True:
        if (owner.root / "PIPELINE_STOP_REQUEST.json").exists():
            result = {"UTC": utc(), "state": "LOCAL_WATCH_STOPPED", "remote_worker_killed": False}
            write_new(owner.root / "PIPELINE_WATCH_STOPPED.json", result)
            return result
        result = remote_call(owner, "observe")
        print(json.dumps({"UTC": utc(), "state": result["state"], "entry": result["active"]["entry"],
                          "actual_worker_native_exit": result["active"].get("actual_worker_native_exit"),
                          "ready_entry": result.get("ready_entry")}), flush=True)
        if result["state"] == "RUNNING_OR_FINALIZING":
            time.sleep(30)
            continue
        if result["state"] == "ATTENTION":
            raise RuntimeError(result["reason"])
        if result["state"] == "READY":
            launch_entry(owner, result["ready_entry"])
            time.sleep(30)
            continue
        if result["state"] == "COMPLETE":
            fetched = fetch_entry(owner, "summarize_and_plot.py")
            final = {"UTC": utc(), "state": "POST_PIPELINE_COMPLETE", "fetched_files": fetched["verified_files"],
                     "completed_chain": result["completed_chain"], "scientific_model_goal_complete": False}
            write_new(owner.root / "PIPELINE_WATCH_COMPLETE.json", final)
            return final
        raise RuntimeError("Unknown controller state; no automatic action")


def main():
    args = parser(__doc__).parse_args()
    result = run_owned(args, "watch", watch)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
