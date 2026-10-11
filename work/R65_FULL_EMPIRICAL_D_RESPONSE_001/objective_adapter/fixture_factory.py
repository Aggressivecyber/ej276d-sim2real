"""Deterministic R64 artificial factory; body copied from bound portable fixture lines 230-314."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch

def build_fixture(dg,R63,device,bound_root):
    ROOT=Path(bound_root)
    source_counts = np.array([3, 4, 5, 2, 6, 3, 4, 5, 2, 4], dtype=np.int64)
    real_counts = np.array([2, 5, 3, 6, 4, 2, 5, 3, 6, 4], dtype=np.int64)
    source_regions = np.repeat(dg.CATEGORIES, source_counts)
    real_regions = np.repeat(dg.CATEGORIES, real_counts)
    t = np.arange(480, dtype=np.float64)

    def waveform(region, parent, real=False):
        if region == -1:
            code = np.full(480, 8192, dtype=np.float64)
            code[100:150], code[150:400] = (10100, 8750)
            return -0.5 + code / 16383
        j = parent if parent < 2 else parent % 2
        base = 0.0008 * np.sin(t * (0.11 + 0.013 * region) + j) + 0.00035 * np.cos(t * 0.19 + region)
        pulse = np.zeros(480)
        amp = 0.11 + 0.012 * region + 0.004 * j + (0.005 if real else 0.0)
        tau = 27.0 + 2.5 * region + 1.5 * j + (4.0 if real else 0.0)
        pulse[100:400] = amp * np.exp(-np.arange(300) / tau)
        return (base + pulse).astype(np.float32).astype(np.float64)
    source_np = np.stack([waveform(int(r), j) for r, n in zip(dg.CATEGORIES, source_counts) for j in range(n)])
    real_np = np.stack([waveform(int(r), j, True) for r, n in zip(dg.CATEGORIES, real_counts) for j in range(n)])
    source = torch.from_numpy(source_np).to(device)
    real = torch.from_numpy(real_np).to(device)
    q = R63.core.q_torch(source)
    bins = torch.from_numpy(np.clip(source_regions, 0, 7)).to(device)
    condition = torch.from_numpy(source_regions.astype(np.float64)).to(device)
    mu = torch.zeros((8, 480), dtype=torch.float64, device=device)
    mu[:, 100:400] = 1 / 300
    scale = torch.full((8, 480), 0.012, dtype=torch.float64, device=device)
    noise = 0.0004 * torch.sin(torch.arange(len(source), dtype=torch.float64, device=device)[:, None] + torch.arange(480, device=device)[None, :] * 0.131)
    noise = R63.core.project(noise, mu[bins])
    noise[condition < 0] = 0

    class ToyNetwork(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.coefficients = torch.nn.Parameter(torch.tensor([0.08, 0.035, 0.4, 0.018], dtype=torch.float32, device=device))
            basis = torch.sin(torch.arange(480, dtype=torch.float32, device=device) * 0.041)
            basis[450], basis[451] = (28.0, -24.0)
            self.register_buffer('basis', basis)

        def forward(self, noisy, b, guide, h):
            y, g, hh = (noisy.float(), guide.float(), h.float()[:, None])
            c = self.coefficients
            return (hh >= 0).float() * (c[0] * y + c[1] * g + c[2] * self.basis[None, :] + c[3] * (b.float()[:, None] + hh))
    net = ToyNetwork().train()
    schedule = tuple(((k, min(k + 7, len(source))) for k in range(0, len(source), 7)))

    def endpoint_chunk(a, b):
        return R63.core.endpoint(net, source[a:b], noise[a:b], q[a:b], bins[a:b], condition[a:b], mu, scale)

    def endpoint_all():
        pairs = [endpoint_chunk(a, b) for a, b in schedule]
        return (torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs]))
    cfg = json.loads((ROOT / 'r63_original/config.json').read_text('utf8'))
    stats = {'source_counts': source_counts, 'REAL_counts': real_counts, 'noise_scale': np.array([0.0012, 0.2]), 'valley_bounds': np.array([[0.2, 0.7], [0.2, 0.7]]), 'source_h_one_fraction': np.ones(2), 'source_raw_valley_fraction': np.zeros(2)}
    for name, dim, value in zip(dg.NAMES, (480, 480, 479, 478), (0.15, 0.5, 0.25, 0.25)):
        stats[name + '_center'] = np.zeros((10, dim))
        stats[name + '_scale'] = value * (1 + 0.08 * np.arange(10)[:, None]) * (1 + 0.05 * np.sin(np.arange(dim)[None, :] * 0.03))
    stats['temporal_center'] = np.zeros((10, 31))
    stats['temporal_scale'] = np.tile(np.r_[0.005, np.full(30, 0.35)], (10, 1)) * (1 + 0.04 * np.arange(10)[:, None])
    rng = np.random.Generator(np.random.PCG64(640032128))
    directions = []
    for dim in (480, 480, 479, 478):
        p = rng.standard_normal((dim, 64))
        p /= np.linalg.norm(p, axis=0, keepdims=True)
        directions.append(p.astype(np.float32))
    tp = rng.standard_normal((31, 64))
    tp /= np.linalg.norm(tp, axis=0, keepdims=True)
    tp = tp.astype(np.float32)
    objective = dg.FrozenDistributionObjective(stats, cfg, directions, tp, source_regions, real_regions, device=device)
    full = dg.ParentLayout.full_reference(source_regions, real_regions)
    sslots = np.stack([rng.choice(np.flatnonzero(source_regions == r), size=(4, 32), replace=True) for r in dg.CATEGORIES])
    rslots = np.stack([rng.choice(np.flatnonzero(real_regions == r), size=(4, 32), replace=True) for r in dg.CATEGORIES])
    alayouts, blayout = dg.paired_layouts(sslots, rslots, source_regions, real_regions)
    fixture_hash = hashlib.sha256()
    for a in (source_np, real_np, source_counts, real_counts, noise.detach().cpu().numpy(), mu.detach().cpu().numpy(), scale.detach().cpu().numpy(), q.detach().cpu().numpy(), bins.detach().cpu().numpy(), condition.detach().cpu().numpy(), net.coefficients.detach().detach().cpu().numpy(), net.basis.detach().cpu().numpy(), *(np.asarray(stats[k]) for k in sorted(stats)), *directions, tp, sslots, rslots):
        fixture_hash.update(str(a.shape).encode())
        fixture_hash.update(str(a.dtype).encode())
        fixture_hash.update(a.tobytes())
    with torch.no_grad():
        cached_y, cached_raw = endpoint_all()
    assert bool((cached_y.abs() == 0.5).any()), 'Fixture must exercise active Box rails'
    assert bool((cached_y.abs() < 0.5).any()), 'Fixture must retain free coordinates'
    assert bool((q > 0).all())
    return locals()
