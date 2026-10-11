from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
S = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
P = R/'reviews/R64_PREPARATION_001/worker_review/REVIEW.json'
assert not (S/'ROOT_STATIC_REVIEW_DECISION.json').exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
review = json.loads(P.read_text('utf8'))
assert review['scientific_logic_blocking_findings'] == []
assert all(row['pass_static'] for row in review['checked_invariants'])
for row in review['static_checks']['manifest']:
    assert sha(S/row['name']) == row['sha256'], row['name']
assert {row['id'] for row in review['prelaunch_pending']} == {'ROOT_SEAL_001'}
value = dict(UTC=datetime.now(timezone.utc).isoformat(), owner='/root',
             ready_for_registered_dispatch=True, worker_sha256=sha(S/'diagnose_train.py'),
             render_sha256=sha(S/'render_diagnostic.py'), independent_review=dict(path=str(P), sha256=sha(P)),
             static_review_manifest_unchanged=True, remaining_scientific_logic_blockers=[],
             root_seal_closure='deploy_R64_root.py creates ROOT_IMPLEMENTATION_REVIEW.json with all package hashes before upload and launch; worker verifies every entry before artificial tests',
             root_reviewed_definitions=['original full-parent source and REAL population and separate masses',
                 'direct projected endpoint; frozen model train mode and disabled MHA fastpath',
                 'paired region-major 4x32/128 same occurrence construction',
                 'five original D terms only; canonical and Box derivative each once',
                 'full original 64+1 SOURCE replay on every pass and bitwise pre-VJP checks',
                 'named float64 CPU parameter accumulation with no extra divisor',
                 '32-macro Welford and shared fixed bootstrap; no population-bias assertion',
                 'complete saved vectors/identities/random streams and input immutability',
                 'original A-F unchanged; failure evidence retained; native and figure receipts required'],
             numeric_mass_boundary_disclosed=True, original_acceptance_unchanged=True,
             no_new_scientific_arrays_or_checkpoints_opened_by_root_review=True,
             execution_not_yet_started=True, server_artificial_CPU_CUDA_pass_still_required=True,
             result_JSON_alone_not_completion=True)
(S/'ROOT_STATIC_REVIEW_DECISION.json').write_bytes((json.dumps(value, indent=2)+'\n').encode())
print(json.dumps(dict(ready_for_registered_dispatch=True, reviewed_file_bindings=len(review['static_checks']['manifest']),
                     worker_sha256=value['worker_sha256'])))
