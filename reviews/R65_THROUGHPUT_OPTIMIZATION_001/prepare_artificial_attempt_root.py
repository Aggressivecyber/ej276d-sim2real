"""Freeze a root-reviewed, synthetic-only old/new CUDA comparison."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import ctypes
import hashlib
import json

P = Path(__file__).resolve().parent
project = P.parents[1]
stage = project/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
candidate = P/'agent_candidate'
out = P/'attempt_001'
cfg = json.loads((stage/'config.json').read_text('utf8'))
remote = cfg['remote_root']+'/engineering_benchmarks/attempt_001'
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(), 8)
assert not out.exists()
assert hashlib.sha256((candidate/'benchmark_cuda_artificial.py').read_bytes()).hexdigest() == '68aea374314eff2418a029d244509b3b3992f5b9db90915c689c2a5b4d2fdd89'
cpu = json.loads((candidate/'CPU_ARTIFICIAL_RECEIPT.json').read_text('utf8'))
assert cpu['status'] == 'passed' and cpu['resources']['peak_RSS_bytes'] <= 1024**3
out.mkdir()
files = []
for name, source in [('gradient_pack_candidate.py', candidate), ('artificial_common.py', candidate),
                     ('benchmark_cuda_artificial.py', candidate), ('remote_artificial_benchmark_guard.py', P)]:
    body = (source/name).read_bytes()
    ast.parse(body.decode('utf8'))
    (out/name).write_bytes(body)
    files.append({'name':name, 'sha256':hashlib.sha256(body).hexdigest()})
manifest = {
    'registered_UTC':datetime.now(timezone.utc).isoformat(), 'owner':'/root',
    'root_reviewed':True, 'purpose':'Measure an exactly comparable implementation candidate with artificial data on the registered GPU.',
    'remote_directory':remote, 'entry':'benchmark_cuda_artificial.py',
    'arguments':['--source-root',cfg['remote_root'],'--output',remote+'/CUDA_ARTIFICIAL_RECEIPT.json',
                 '--threads','12','--samples','7','--pack-repetitions','20','--max-seconds','150'],
    'maximum_benchmark_seconds':180, 'maximum_artificial_compute_seconds':150,
    'scientific_data_reads':0, 'scientific_updates':0, 'saved_weight_reads':0,
    'frozen_training_file_changes':0, 'original_A_F_changes':0,
    'training_isolation':'SIGSTOP verified owned PID291399; always SIGCONT in finally; same process and optimizer retained.',
    'combined_GPU_limit_bytes':10*1024**3, 'combined_RSS_limit_bytes':48*1024**3,
    'benchmark_CPU_affinity':list(range(12)), 'root_local_CPU_affinity':[3],
    'files':files, 'root_review_scope':['Candidate parameter order and float64 CPU accumulation unchanged',
        'Finite checks and mixed dtype/device fallback preserve failure behavior',
        'Signed zero/subnormal byte comparisons, actual 64-row native artificial endpoint+VJP',
        'TF32 disabled, deterministic algorithms, MHA fastpath disabled, intra/inter threads12',
        'Alternating old/new timing order; microbenchmark and forward+VJP reported separately',
        'No inference of scientific quality or full-run speed from a microbenchmark'],
    'result_files':['GUARD_STARTED.json','GUARD_RECEIPT.json','CUDA_ARTIFICIAL_RECEIPT.json',
                    'BENCHMARK.stdout.log','BENCHMARK.stderr.log'],
}
(out/'BENCHMARK_REGISTRATION.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
print(json.dumps({'registered':True, 'root_reviewed':True, 'files':len(files), 'maximum_seconds':180,
                  'scientific_arrays_or_weights_read':False, 'remote_directory':remote}))
