"""Record the user's latest execution preference without changing R65 science."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os

R = Path(__file__).resolve().parents[1]
S = R / 'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
O = R / 'reviews/R65_THROUGHPUT_OPTIMIZATION_001/user_steering'
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(), 8)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

assert not O.exists(), 'Do not replace a prior steering receipt'
O.mkdir(parents=True)
(O/'before').mkdir()
review = json.loads((S/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))
frozen = {row['name']: row['sha256'] for row in review['files']}
for n, h in frozen.items():
    assert sha(S/n) == h, n
acceptance = {n: sha(R/n) for n in ['PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json']}
stamp = datetime.now(timezone.utc).isoformat()
policy = {
    'UTC': stamp, 'root_leads_and_reviews': True,
    'analysis_subagents_allowed': True, 'analysis_model': 'gpt-6.1-sol',
    'analysis_reasoning_effort': 'max', 'scientific_terminal_updates': 50000,
    'prioritize_numerically_equivalent_implementation_speedups': True,
    'four_hour_early_stop_authorized_only_if_quality_impact_is_evidenced_small': True,
    'evidence_supporting_four_hour_early_stop': None,
    'current_early_stop_decision': 'Keep all 50000 updates; no evidence that a four-hour stop preserves quality.',
    'if_safe_speedup_unavailable': 'Continue the original training to its fixed endpoint.',
    'scheduled_wakeup_requested': True, 'schedule_status': 'pending_measured_estimate',
    'original_A_F_changed': False, 'new_local_files': 'E/D only',
    'resource_limits_unchanged': {'server_CPU': 12, 'server_RSS_GiB': 48, 'server_GPU_GiB': 10,
                                  'local_total_CPU': 4, 'local_total_RSS_GiB': 4},
}
header = '''# Current user instruction: optimize R65 and keep its 50000-step endpoint

The latest user instruction supersedes the earlier root-only / identical-model restrictions: auxiliary analysis uses gpt-6.1-sol with max reasoning, with root leading and reviewing. Optimize throughput while preserving the computation, source population, original A-F and fixed 50000 updates. If a safe speedup is unavailable, let the original run finish. A four-hour early stop is conditional on evidence that it has little quality cost; that evidence is currently absent, so no early stop is scheduled. Arrange an estimated follow-up wake after measured throughput is available. Server CPU12/RSS48GiB/GPU10GiB, one active scientific computation, shared local CPU4/RSS4GiB, and new local files only on E/D remain in force. The steering receipt is reviews/R65_THROUGHPUT_OPTIMIZATION_001/user_steering/RECEIPT.json.

## Earlier records

'''
changed = []
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json']:
    p = R/name
    original = p.read_bytes()
    (O/'before'/name).write_bytes(original)
    if name.endswith('.md'):
        p.write_bytes(header.encode('utf8')+original)
    else:
        value = json.loads(original.decode('utf-8-sig'))
        value['subagents_allowed'] = True
        value['latest_user_execution_policy'] = policy
        value['latest_user_revision'] = 'reviews/R65_THROUGHPUT_OPTIMIZATION_001/user_steering/RECEIPT.json'
        if name == 'PROGRAM.json':
            value['analysis_subagent_model'] = 'gpt-6.1-sol'
            value['analysis_subagent_reasoning_effort'] = 'max'
            value['execution_coordination'] = 'root_leads_and_reviews_scoped_gpt_6_1_sol_max_analysis'
            value['latest_permission_status'] = 'User authorizes scoped gpt-6.1-sol/max analysis and equivalent optimization; retain 50000 updates if speedup unavailable.'
        else:
            value['application_goal_status'] = 'paused'
            value['application_goal_status_observed_UTC'] = '2026-10-11T04:35:41+00:00'
            value['application_goal_status_note'] = 'App goal tool currently reports paused; current user request authorizes optimization and scheduled follow-up; remote training continues.'
        p.write_bytes((json.dumps(value, indent=2, ensure_ascii=False)+'\n').encode('utf8'))
    changed.append({'name': name, 'before_SHA256': hashlib.sha256(original).hexdigest(), 'after_SHA256': sha(p)})
for n, h in frozen.items():
    assert sha(S/n) == h, n
assert {n: sha(R/n) for n in acceptance} == acceptance
receipt = dict(policy, recorded_by_PID=os.getpid(), changed_files=changed,
               verified_frozen_files_unchanged=len(frozen), original_acceptance_SHA256=acceptance)
(O/'RECEIPT.json').write_bytes((json.dumps(receipt, indent=2, ensure_ascii=False)+'\n').encode('utf8'))
print(json.dumps({'steering_recorded': True, 'frozen_files_unchanged': len(frozen),
                  'endpoint': 50000, 'four_hour_stop_scheduled': False}))
