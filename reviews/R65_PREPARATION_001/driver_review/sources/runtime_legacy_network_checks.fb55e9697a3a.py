"""Actual frozen-network architecture, artificial inputs/state; no science I/O."""
from pathlib import Path
import hashlib
import json
import sys
import time


def state_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check(device='cpu', chunk=4):
    import numpy as np
    import torch
    from network import ResponseNetwork
    from source_flow_core import endpoint, canonical_adc_torch, q_torch

    assert chunk in (4, 64)
    dev = torch.device(device)
    started = time.monotonic()
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    tolerances = {'gradient_relative_L2': 2e-5, 'gradient_max_absolute': 2e-6,
                  'gradient_max_relative_to_reference_max': 2e-5,
                  'nonzero_reference_gradient_norm_above': 1e-9}
    fork_devices = [dev.index or 0] if dev.type == 'cuda' else []
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(20261011643)
            model = ResponseNetwork()
            with torch.no_grad():
                model.final.weight.copy_(torch.randn(model.final.weight.shape) * .015)
                model.final.bias.copy_(torch.randn(model.final.bias.shape) * .005)
            model.to(dev).train()
            model.requires_grad_(True)
            before = state_sha(model)
            params = tuple(model.parameters())
            n = chunk + 1
            t = np.arange(480, dtype='f8')
            row = np.arange(n, dtype='f8')[:, None]
            pulse = np.where((t >= 100) & (t < 400), .05*np.exp(-(t-100)/58.), 0.)
            x = (pulse[None, :] * (1 + row/(10*n)) + .001*np.sin(t[None, :]/7 + row)).astype('f4').astype('f8')
            mu = np.zeros((8, 480), dtype='f8')
            mu[:, 100:400] = 1/300.
            scale = np.full((8, 480), .2, dtype='f8')
            b = np.arange(n, dtype='i8') % 8
            q = x[:, 100:400].sum(1) - 3*x[:, :100].sum(1)
            noise = .007*np.sin(t[None, :]/13 + row*.31)
            noise[0] *= 110.
            nq = noise[:, 100:400].sum(1) - 3*noise[:, :100].sum(1)
            noise -= mu[b]*nq[:, None]
            h = np.linspace(-.7, .8, n, dtype='f8')
            data = [torch.as_tensor(a, device=dev) for a in (x, noise, q, b, h, mu, scale)]
            xt, nt, qt, bt, ht, mt, st = data
            weights = torch.as_tensor(np.cos(t[None, :]/17 + row*.07)/(480*n), device=dev)

            def forward(sl):
                return endpoint(model, xt[sl], nt[sl], qt[sl], bt[sl], ht[sl], mt, st)

            def loss_from_out(y):
                v = canonical_adc_torch(y, True)
                return (weights*torch.sin(3*v)).sum() + .01*v.square().mean()

            schedule = [(j, min(j+chunk, n)) for j in range(0, n, chunk)]
            single_parts = [forward(slice(lo, hi))[0] for lo, hi in schedule]
            single_y = torch.cat(single_parts)
            single_loss = loss_from_out(single_y)
            single_gy = torch.autograd.grad(single_loss, single_y, retain_graph=True)[0].detach()
            single_parameters = torch.autograd.grad(single_loss, params)
            reference = torch.cat([v.detach().reshape(-1).cpu().double() for v in single_parameters])
            reference_y = single_y.detach().clone()
            reference_value = float(single_loss.detach())
            del single_y, single_parts, single_loss, single_parameters
            with torch.no_grad():
                cache = torch.cat([forward(slice(lo, hi))[0] for lo, hi in schedule])
            assert torch.equal(cache, reference_y), 'no_grad collection differs from grad-enabled forward'
            leaf = cache.clone().requires_grad_(True)
            leaf_loss = loss_from_out(leaf)
            gy = torch.autograd.grad(leaf_loss, leaf)[0].detach()
            assert torch.equal(gy, single_gy)
            assert float(leaf_loss.detach()) == reference_value
            result = torch.zeros_like(reference)
            checks = []
            for lo, hi in schedule:
                out, raw = forward(slice(lo, hi))
                expected = cache[lo:hi]
                same_y = torch.equal(out.detach(), expected)
                same_codes = torch.equal(canonical_adc_torch(out.detach(), False),
                                         canonical_adc_torch(expected, False))
                same_free = torch.equal((out.detach() > -.5) & (out.detach() < .5),
                                        (expected > -.5) & (expected < .5))
                assert same_y and same_codes and same_free
                values = torch.autograd.grad(out, params, grad_outputs=gy[lo:hi])
                result += torch.cat([v.detach().reshape(-1).cpu().double() for v in values])
                checks.append({'start': lo, 'stop': hi, 'bitwise_endpoint': same_y,
                               'canonical_codes': same_codes, 'Box_free_mask': same_free})
                del out, raw, values
            difference = result-reference
            norm = float(torch.linalg.vector_norm(reference))
            relative = float(torch.linalg.vector_norm(difference))/norm
            max_error = float(difference.abs().max())
            max_reference = float(reference.abs().max())
            assert norm > tolerances['nonzero_reference_gradient_norm_above']
            assert relative <= tolerances['gradient_relative_L2']
            assert max_error <= tolerances['gradient_max_absolute'] + tolerances['gradient_max_relative_to_reference_max']*max_reference
            assert state_sha(model) == before and all(p.grad is None for p in params)
            charge_error = float((q_torch(cache)-qt).abs().max())
            rail_count = int((cache.abs() == .5).sum())
            assert rail_count > 0, 'fixture did not exercise an active Box face'
            return {'pass': True, 'device': str(dev), 'torch': torch.__version__,
                    'numpy': np.__version__, 'architecture': 'original R63 ResponseNetwork',
                    'state_origin': 'fixed artificial nonzero head; no checkpoint read',
                    'artificial_state_SHA256': before, 'state_unchanged': True,
                    'training_mode': True, 'attention_dropout': 0., 'MHA_fastpath': False,
                    'chunk': chunk, 'rows_per_pass': n, 'passes': 3,
                    'neural_rows': 3*n, 'chunk_checks': checks,
                    'loss': reference_value, 'reference_parameter_gradient_L2': norm,
                    'relative_parameter_gradient_L2_error': relative,
                    'max_absolute_parameter_gradient_error': max_error,
                    'max_absolute_reference_parameter_gradient': max_reference,
                    'canonical_leaf_gradient_bitwise': True,
                    'active_Box_rail_samples': rail_count,
                    'max_preADC_charge_absolute_error': charge_error,
                    'tolerances_frozen_before_execution': tolerances,
                    'scientific_arrays_or_checkpoints_read': 0, 'scientific_updates': 0,
                    'elapsed_s': time.monotonic()-started,
                    'limitations': 'Architecture replay test with a scalar artificial STE loss. Integrated five-term and unequal-mass objective tests are recorded separately.'}
    finally:
        torch.backends.mha.set_fastpath_enabled(old_fastpath)


if __name__ == '__main__':
    import os
    import ctypes
    import threading
    source_root, output = Path(sys.argv[1]), Path(sys.argv[2])
    device = sys.argv[3] if len(sys.argv) > 3 else 'cpu'
    chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    assert source_root.is_dir() and not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent/'network_artificial_cache'
    cache.mkdir(exist_ok=True)
    for k in ['TMP', 'TEMP', 'TMPDIR', 'TORCH_HOME', 'MPLCONFIGDIR']:
        os.environ[k] = str(cache)
    for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[k] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_root))
    guard_stop = threading.Event()
    resource = {'actual_worker_PID': os.getpid(), 'parent_PID': os.getppid(),
                'actual_worker_peak_RSS_bytes': 0, 'RSS_limit_bytes': 1288490188,
                'actual_worker_monitor_before_scientific_imports': True}
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        handle = kernel.GetCurrentProcess()
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        assert kernel.GetProcessAffinityMask(handle, ctypes.byref(mask), ctypes.byref(system_mask))
        resource['actual_worker_CPU_affinity_mask'] = mask.value
        assert mask.value.bit_count() == 1

        def probe():
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            assert psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), ctypes.sizeof(counters))
            resource['actual_worker_peak_RSS_bytes'] = max(resource['actual_worker_peak_RSS_bytes'], counters.PeakWorkingSetSize)
            if counters.WorkingSetSize > resource['RSS_limit_bytes']:
                failure = dict(resource, status='actual_worker_RSS_limit_exceeded')
                output.with_suffix('.resource_failure.json').write_text(json.dumps(failure), 'utf8')
                os._exit(91)

        def monitor():
            while not guard_stop.wait(.1):
                try:
                    probe()
                except BaseException:
                    os._exit(92)
        probe()
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    else:
        raise RuntimeError('Standalone wrapper is Windows-only; server imports check under the owned supervisor')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    value = check(device, chunk)
    probe()
    guard_stop.set()
    thread.join()
    value['resource'] = resource
    output.write_bytes((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())
    print(json.dumps(value, indent=2, allow_nan=False))
