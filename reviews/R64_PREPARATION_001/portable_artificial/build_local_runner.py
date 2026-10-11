"""Assemble a local resource-owned runner without executing an artificial test."""
from pathlib import Path

root = Path(__file__).resolve().parent
guard_source = (root.parent / "endpoint_geometry" / "artificial_check.py").read_text("utf8")
guard = guard_source.split("import numpy as np", 1)[0]
guard = guard.replace('"""Artificial-only checks.  Run with Python -B; writes only beside this script."""',
                      '"""One CPU port check under a separate Windows resource owner; not the portable API."""')
tail = '''
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.mha.set_fastpath_enabled(False)
import portable_fixture

receipt_path = ROOT / 'LOCAL_CPU_RECEIPT.json'
if receipt_path.exists() or (ROOT / 'PORTABLE_CPU_ARTIFICIAL.json').exists():
    raise RuntimeError('This task permits one local portable CPU invocation; no repeat')
adapter = ROOT.parent / 'distribution_gradient'
legacy = json.loads((adapter / 'ARTIFICIAL_RECEIPT.json').read_text('utf-8-sig'))
binding = json.loads((ROOT / 'PORT_SOURCE_BINDING.json').read_text('utf8'))
try:
    result = portable_fixture.run_check('cpu', adapter_root=adapter)
    comparisons = {}
    fields = ('fixture_arrays_sha256', 'tolerance_sha256_before_run', 'fixture',
              'original_nine_piece_values_32', 'D_F_values', 'D_A_four32_values', 'D_B128_values',
              'D_F_parent_gradient_norms', 'D_F_parameter_gradients',
              'D_A_B_endpoint_gradient_distance', 'same_schedule_replay_checks', 'tests')
    for field in fields:
        comparisons[field] = {'exact_equal_to_original_receipt': result[field] == legacy[field]}
    all_equal = all(v['exact_equal_to_original_receipt'] for v in comparisons.values())
    protected = []
    for record in binding['protected_existing_files']:
        current = hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()
        protected.append({'path': record['path'], 'sha256': current, 'unchanged': current == record['sha256']})
    end_memory = memory()
    assert end_memory['peak_RSS_bytes'] <= LIMIT
    assert all_equal, comparisons
    assert all(v['unchanged'] for v in protected)
    receipt = {
        'status': 'PORTABLE_CPU_69_CHECKS_AND_ORIGINAL_RECEIPT_EXACT_PARITY_PASSED',
        'UTC': datetime.now(timezone.utc).isoformat(), 'PID': os.getpid(),
        'local_portable_CPU_invocations': 1, 'CUDA_executed': False,
        'checks': len(result['tests']), 'original_receipt_comparisons': comparisons,
        'all_12_key_fields_exact_equal': all_equal,
        'fixture_arrays_sha256': result['fixture_arrays_sha256'],
        'tolerance_sha256': result['tolerance_sha256_before_run'],
        'caller_state_preserved': result['caller_state_preserved'],
        'wall_seconds_including_imports': time.monotonic() - START,
        'resource': {'CPU_affinity_mask': one_cpu_mask, 'CPU_affinity_core_count': 1,
                     'torch_intraop_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
                     'process_commit_limit_bytes': LIMIT, 'RSS_watchdog_limit_bytes': LIMIT,
                     'RSS_watchdog_poll_seconds': .05, **end_memory, 'TEMP': str(TEMP),
                     'Python_B': bool(sys.dont_write_bytecode)},
        'protected_existing_files_after_execution': protected,
        'source': {'portable_fixture_sha256': hashlib.sha256((ROOT / 'portable_fixture.py').read_bytes()).hexdigest(),
                   'original_fixture_sha256': portable_fixture.ORIGINAL_FIXTURE_SOURCE_SHA256,
                   'original_receipt_sha256': portable_fixture.ORIGINAL_CPU_RECEIPT_SHA256},
        'scope': {'scientific_arrays_read': 0, 'scientific_weights_read': 0, 'DEV_EVAL_CAL_reads': 0,
                  'scientific_forward_or_gradient_runs': 0, 'optimizer_updates': 0,
                  'root_metadata_or_Git_writes': 0},
    }
    (ROOT / 'PORTABLE_CPU_ARTIFICIAL.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\\n', encoding='utf8')
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\\n', encoding='utf8')
    print(json.dumps({'status': receipt['status'], 'checks': receipt['checks'], 'all_12_key_fields_exact_equal': all_equal,
                      'peak_RSS_bytes': end_memory['peak_RSS_bytes'], 'wall_seconds': receipt['wall_seconds_including_imports']}))
except Exception as exc:
    failure = getattr(exc, 'receipt', {'error_type': type(exc).__name__, 'error': str(exc)})
    (ROOT / 'LOCAL_CPU_FAILURE.json').write_text(json.dumps(failure, indent=2, allow_nan=False)+'\\n', encoding='utf8')
    raise
finally:
    stop_guard.set()
'''
text = guard + tail
compile(text, str(root / "run_cpu_once.py"), "exec")
(root / "run_cpu_once.py").write_text(text, encoding="utf8")
print("Local one-shot CPU runner assembled; no test executed.")
