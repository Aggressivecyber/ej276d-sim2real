"""Inspect only session metadata and model/effort settings; never export messages."""
from pathlib import Path
from datetime import datetime, timezone
import json

BASE = Path('C:/Users/10629/.codex/sessions')
HERE = Path(__file__).resolve().parent
THREADS = {
    'root_original': '2026/10/03/rollout-2026-10-03T11-14-52-01a0ffc1-e68d-7123-b09b-0c86a7253724.jsonl',
    'root_current': '2026/10/08/rollout-2026-10-08T03-03-33-01a0ffc1-e68d-7123-b09b-0c86a7253724_01a117bf-e1e4-72b1-80ab-d2d8b73f2158.jsonl',
    'child_090520': '2026/10/11/rollout-2026-10-11T09-05-20-01a1287e-2d71-73e0-bcfb-1ae4e671d9aa.jsonl',
    'child_090532': '2026/10/11/rollout-2026-10-11T09-05-32-01a1287e-5d30-78b3-a160-9cb8d7c5e001.jsonl',
    'child_090542': '2026/10/11/rollout-2026-10-11T09-05-42-01a1287e-85e5-7021-ad71-157e1f565c41.jsonl',
}

rows = []
for label, relative in THREADS.items():
    row = {'label': label, 'session_file_name': Path(relative).name,
           'first_configuration': None, 'last_configuration': None,
           'distinct_model_effort': [], 'turn_context_count': 0}
    configs = set()
    with (BASE / relative).open('r', encoding='utf8') as stream:
        for line in stream:
            if '"turn_context"' not in line[:200] and '"session_meta"' not in line[:200]:
                continue
            item = json.loads(line)
            data = item.get('payload', {})
            if item.get('type') == 'session_meta':
                row['session_id'] = data.get('id')
                row['agent_nickname'] = data.get('agent_nickname')
                row['agent_role'] = data.get('agent_role')
                # Source may contain only the subagent spawn lineage; omit other forms.
                source = data.get('source')
                if isinstance(source, dict) and 'subagent' in source:
                    row['subagent_lineage'] = source['subagent']
            elif item.get('type') == 'turn_context':
                configuration = {'timestamp': item.get('timestamp'),
                                 'model': data.get('model'),
                                 'effort': data.get('effort', data.get('reasoning_effort'))}
                row['turn_context_count'] += 1
                if row['first_configuration'] is None:
                    row['first_configuration'] = configuration
                row['last_configuration'] = configuration
                configs.add((configuration['model'], configuration['effort']))
    row['distinct_model_effort'] = [{'model': m, 'effort': e} for m, e in sorted(configs, key=str)]
    rows.append(row)

root = next(x for x in rows if x['label'] == 'root_current')['last_configuration']
assert root and root['model'] and root['effort'], 'Current root configuration unavailable'
children = [x for x in rows if x['label'].startswith('child_')]
for child in children:
    child['all_recorded_turns_match_current_root'] = bool(child['turn_context_count']) and all(
        x['model'] == root['model'] and x['effort'] == root['effort']
        for x in child['distinct_model_effort'])
receipt = {'UTC': datetime.now(timezone.utc).isoformat(),
           'scope': 'Recorded effective model and reasoning effort for the current root and the three R63 auxiliary agents',
           'root_current_configuration': root,
           'all_three_children_match': all(x['all_recorded_turns_match_current_root'] for x in children),
           'sessions': rows,
           'session_messages_exported': False,
           'settings_changed': False,
           'scientific_jobs_started_or_stopped': 0}
(HERE / 'AGENT_MODEL_PARITY_OBSERVATION.json').write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
