"""Five frozen-reference D terms and parent endpoint gradients; preparation only.

Endpoint/features stay on the explicitly selected CPU/CUDA device. SOURCE endpoints are float64 AFTER Box projection and
BEFORE canonical ADC. REAL references are fixed raw stored waveforms, float64.
Masses, scales, directions, parent labels and all row orders are frozen inputs.
No file contains or loads scientific data, checkpoints, optimizer, or runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Mapping

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from original_bindings import R63
from weighted_empirical_w1 import FrozenMass, weighted_w1_numpy

CATEGORIES = tuple([-1, *range(8), 8])
NAMES = tuple(R63.old_loss.NAMES)
D_NAMES = ('full_window', 'PSD', 'noise', 'coordinate_RMS', 'temporal_joint')
C_NAMES = ('source_order', 'source_valley', 'REAL_identity', 'box_correction')


def _indices(x, size, name):
    a = np.asarray(x)
    if a.ndim != 1 or not a.size or a.dtype.kind not in 'iu':
        raise ValueError(name + ': nonempty integer vector required')
    if (a < 0).any() or (a >= size).any():
        raise ValueError(name + ': parent index outside original population')
    return tuple(int(i) for i in a)


@dataclass(frozen=True)
class ParentLayout:
    """Explicit fixed row order, with ragged regional lists and duplicates.

    Construct with full_reference or region_major. Full global laws retain the
    original parent row order. Sampled global laws retain region-major slot
    order, with microbatch then slot order inside each region. No value dedup.
    """
    source: tuple[tuple[int, ...], ...]
    real: tuple[tuple[int, ...], ...]
    source_global: tuple[int, ...]
    real_global: tuple[int, ...]
    kind: str

    @classmethod
    def full_reference(cls, source_regions, real_regions):
        s, r = np.asarray(source_regions), np.asarray(real_regions)
        sr = tuple(tuple(map(int, np.flatnonzero(s == b))) for b in CATEGORIES)
        rr = tuple(tuple(map(int, np.flatnonzero(r == b))) for b in CATEGORIES)
        result = cls(sr, rr, tuple(range(len(s))), tuple(range(len(r))), 'full_reference')
        result.validate(s, r)
        return result

    @classmethod
    def region_major(cls, source_slots, real_slots, source_regions, real_regions):
        s, r = np.asarray(source_slots), np.asarray(real_slots)
        if s.ndim not in (2, 3) or r.ndim != s.ndim or s.shape[0] != 10 or r.shape[0] != 10:
            raise ValueError('Require [10, slots] or [10, microbatch, slot] region-major arrays')
        ss = tuple(_indices(z.reshape(-1), len(source_regions), 'SOURCE') for z in s)
        rr = tuple(_indices(z.reshape(-1), len(real_regions), 'REAL') for z in r)
        result = cls(ss, rr, tuple(i for z in ss for i in z),
                     tuple(i for z in rr for i in z), 'region_major')
        result.validate(source_regions, real_regions)
        return result

    def validate(self, source_regions, real_regions):
        for name, lists, global_rows, regions in (
            ('SOURCE', self.source, self.source_global, np.asarray(source_regions)),
            ('REAL', self.real, self.real_global, np.asarray(real_regions)),
        ):
            if regions.ndim != 1 or regions.dtype.kind not in 'iu' or not np.isin(regions, CATEGORIES).all():
                raise ValueError(name + ': invalid original region labels')
            if len(lists) != 10:
                raise ValueError(name + ': exactly ten regional lists required')
            for label, rows in zip(CATEGORIES, lists):
                rows = _indices(rows, len(regions), name)
                if not np.all(regions[list(rows)] == label):
                    raise ValueError(name + ': wrong region-major layout / parent region')
            _indices(global_rows, len(regions), name)
            flat = tuple(i for group in lists for i in group)
            if sorted(flat) != sorted(global_rows):
                raise ValueError(name + ': global and conditional occurrences differ')
            if self.kind == 'full_reference':
                if global_rows != tuple(range(len(regions))):
                    raise ValueError(name + ': full reference requires every parent once in original order')
                for label, rows in zip(CATEGORIES, lists):
                    if rows != tuple(map(int, np.flatnonzero(regions == label))):
                        raise ValueError(name + ': full regional reference has been truncated or reordered')
            elif self.kind == 'region_major':
                if global_rows != flat:
                    raise ValueError(name + ': sampled global rows must be region-major')
            else:
                raise ValueError('Unknown layout kind')


class _ColumnW1(torch.autograd.Function):
    """Analytic selected first-order W1 branch, exact integer CDF segmentation."""
    @staticmethod
    def forward(ctx, a, b, ma, mb):
        aa, bb = a.detach().double().cpu().numpy(), b.detach().double().cpu().numpy()
        w = np.empty(aa.shape[1], dtype=np.float64)
        ga, gb = np.empty_like(aa), np.empty_like(bb)
        for j in range(aa.shape[1]):
            result = weighted_w1_numpy(aa[:, j], ma, bb[:, j], mb)
            w[j] = result['value']
            ga[:, j], gb[:, j] = result['source_gradient'], result['target_gradient']
        ctx.save_for_backward(torch.from_numpy(ga).to(a.device), torch.from_numpy(gb).to(b.device))
        ctx.a_dtype, ctx.b_dtype = a.dtype, b.dtype
        return torch.from_numpy(w).to(a.device)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        ga, gb = ctx.saved_tensors
        # Multiply the f64 reduction coefficient BEFORE the f32 feature cast.
        return (ga * grad[None, :]).to(ctx.a_dtype), (gb * grad[None, :]).to(ctx.b_dtype), None, None


@lru_cache(maxsize=128)
def _uniform_plan(ns, nr):
    """Exact positive quantile-overlap lengths, dependent only on both counts."""
    i = j = edge = 0
    ix, iy, lengths = [], [], []
    while i < ns and j < nr:
        right = min((i + 1) * nr, (j + 1) * ns)
        ix.append(i); iy.append(j); lengths.append(right - edge)
        edge = right
        if (i + 1) * nr == right:
            i += 1
        if (j + 1) * ns == right:
            j += 1
    assert edge == ns * nr and sum(lengths) == ns * nr
    return tuple(ix), tuple(iy), tuple(z / (ns * nr) for z in lengths)


def column_w1(a, b, ma, mb, *, feature_chunk=64, use_uniform_fastpath=True):
    """Block coordinates only; every sort sees all rows of its complete law."""
    if a.device != b.device or a.device.type not in ('cpu', 'cuda') or a.dtype not in (torch.float32, torch.float64) or b.dtype not in (torch.float32, torch.float64):
        raise ValueError('Same CPU/CUDA device and float32/float64 features required')
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not a.shape[1]:
        raise ValueError('Coordinate arrays must be [nSOURCE,d] and [nREAL,d]')
    if len(ma) != len(a) or len(mb) != len(b) or feature_chunk < 1:
        raise ValueError('Mass/feature length or feature_chunk mismatch')
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('Finite feature values required')
    uniform = len(set(ma.ticks)) == 1 and len(set(mb.ticks)) == 1 and ma.ticks[0] > 0 and mb.ticks[0] > 0
    if use_uniform_fastpath and uniform:
        ix, iy, lengths = _uniform_plan(len(a), len(b))
        ix = torch.tensor(ix, device=a.device, dtype=torch.int64)
        iy = torch.tensor(iy, device=a.device, dtype=torch.int64)
        lengths = torch.tensor(lengths, device=a.device, dtype=torch.float64)
        result = []
        for k in range(0, a.shape[1], feature_chunk):
            sx = torch.sort(a[:, k:k+feature_chunk], dim=0, stable=True).values.double()
            sy = torch.sort(b[:, k:k+feature_chunk], dim=0, stable=True).values.double()
            result.append((lengths[:, None] * (sx[ix] - sy[iy]).abs()).sum(0))
        return torch.cat(result)
    return torch.cat([_ColumnW1.apply(a[:, k:k+feature_chunk], b[:, k:k+feature_chunk], ma, mb)
                      for k in range(0, a.shape[1], feature_chunk)])


def _uniform(n, label):
    return FrozenMass.from_normalized(np.full(n, 1.0 / n), label=label)


def _frozen_tensor(value, shape, *, dtype, name, positive=False, device='cpu'):
    if getattr(value, 'requires_grad', False):
        raise ValueError(name + ': frozen input required')
    result = torch.as_tensor(value, device=device, dtype=dtype).clone().detach()
    if tuple(result.shape) != tuple(shape) or not bool(torch.isfinite(result).all()):
        raise ValueError(name + ': shape or finite-value contract failed')
    if positive and not bool((result > 0).all()):
        raise ValueError(name + ': positive frozen scale required')
    return result


@dataclass
class DResult:
    pieces: dict[str, torch.Tensor]
    total: torch.Tensor
    diagnostics: dict


class FrozenDistributionObjective:
    """No fitted constants: callers must supply frozen original statistics.

    statistics keys: source_counts, REAL_counts, {V480,U480,DU479,D2U478}_{center,scale}
    (10xd), temporal_center/scale (10x31), noise_scale (2). cfg requires
    PSD_loss_scale, temporal_power_floor, projection_directions=64,
    temporal_projection_directions=64 and all five D loss_weights=1.
    directions is four f32 d-by-64 arrays; temporal_directions is f32 31-by-64.
    Explicit parent region vectors use [-1,0,...,8], independent of network bins.
    """
    def __init__(self, statistics: Mapping, cfg: Mapping, directions, temporal_directions,
                 source_regions, real_regions, *, feature_chunk=64, device='cpu', uniform_fastpath=True):
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Only explicit CPU/CUDA numerical paths are supported')
        self.uniform_fastpath = bool(uniform_fastpath)
        self.source_regions, self.real_regions = np.asarray(source_regions).copy(), np.asarray(real_regions).copy()
        ParentLayout.full_reference(self.source_regions, self.real_regions)
        self.source_counts = np.asarray(statistics['source_counts']).copy()
        self.real_counts = np.asarray(statistics['REAL_counts']).copy()
        for counts, regions, name in ((self.source_counts, self.source_regions, 'SOURCE'), (self.real_counts, self.real_regions, 'REAL')):
            actual = np.array([np.sum(regions == b) for b in CATEGORIES], dtype=np.int64)
            if counts.shape != (10,) or not np.array_equal(counts, actual) or not (counts > 0).all():
                raise ValueError(name + ': masses must bind exactly to all original parent counts')
        self.source_mass = self.source_counts.astype(np.float64) / int(sum(self.source_counts))
        self.real_mass = self.real_counts.astype(np.float64) / int(sum(self.real_counts))
        self.stats = {}
        dims = (480, 480, 479, 478)
        for name, dim in zip(NAMES + ('temporal',), dims + (31,)):
            for kind in ('center', 'scale'):
                key = name + '_' + kind
                self.stats[key] = _frozen_tensor(statistics[key], (10, dim), dtype=torch.float64,
                                                 name=key, positive=kind == 'scale', device=self.device)
        self.stats['noise_scale'] = _frozen_tensor(statistics['noise_scale'], (2,), dtype=torch.float64, name='noise_scale', positive=True, device=self.device)
        if cfg['projection_directions'] != 64 or cfg['temporal_projection_directions'] != 64:
            raise ValueError('Frozen R63 requires 64 full-window and 64 temporal directions')
        if cfg['PSD_loss_scale'] != .01 or cfg['temporal_power_floor'] != 1e-20:
            raise ValueError('Original R63 PSD scale and temporal floor must be preserved')
        if any(cfg['loss_weights'][k] != 1 for k in D_NAMES):
            raise ValueError('Frozen R63 D coefficients must all remain one')
        self.cfg = dict(cfg)
        if len(directions) != 4:
            raise ValueError('Four original full-window direction matrices required')
        self.directions = tuple(_frozen_tensor(p, (d, 64), dtype=torch.float32, name='directions', device=self.device) for p, d in zip(directions, dims))
        self.temporal_directions = _frozen_tensor(temporal_directions, (31, 64), dtype=torch.float32, name='temporal_directions', device=self.device)
        self.feature_chunk = int(feature_chunk)
        if self.feature_chunk < 1:
            raise ValueError('feature_chunk must be positive')

    def _w1(self, *args):
        return column_w1(*args, feature_chunk=self.feature_chunk, use_uniform_fastpath=self.uniform_fastpath)

    def _mass(self, layout, domain):
        lists, rows, population_mass, regions = ((layout.source, layout.source_global, self.source_mass, self.source_regions)
            if domain == 'SOURCE' else (layout.real, layout.real_global, self.real_mass, self.real_regions))
        if layout.kind == 'full_reference':
            return _uniform(len(rows), domain + '_full_original_parents')
        per_region = {label: population_mass[k] / len(lists[k]) for k, label in enumerate(CATEGORIES)}
        return FrozenMass.from_normalized([per_region[int(regions[i])] for i in rows], label=domain + '_global')

    def __call__(self, source_endpoint, reference, layout: ParentLayout, *, terms=D_NAMES):
        terms = tuple(terms)
        if not terms or len(set(terms)) != len(terms) or not set(terms).issubset(D_NAMES):
            raise ValueError('Select unique D terms only; C terms are excluded')
        layout.validate(self.source_regions, self.real_regions)
        for name, value, n in (('source_endpoint', source_endpoint, len(self.source_regions)), ('reference', reference, len(self.real_regions))):
            if not isinstance(value, torch.Tensor) or value.device != self.device or value.dtype != torch.float64 or value.shape != (n, 480):
                raise ValueError(name + ': parent-level float64 [N,480] on configured device required')
            if not bool(torch.isfinite(value).all()):
                raise ValueError(name + ': finite waveform values required')
        if reference.requires_grad:
            raise ValueError('REAL empirical reference is frozen; no reference graph allowed')
        if bool((source_endpoint.abs() > .5).any()):
            raise ValueError('SOURCE endpoint must already have passed original Box projection')
        candidate = R63.core.canonical_adc_torch(source_endpoint, True)
        ref = R63.core.canonical_REAL_torch(reference)
        if not bool((R63.core.q_torch(candidate) > 0).all() and (R63.core.q_torch(ref) > 0).all()):
            raise ValueError('Positive canonical charge required; no row deletion')
        conditional_masses = [(_uniform(len(s), 'SOURCE_region_' + str(k)), _uniform(len(r), 'REAL_region_' + str(k)))
                              for k, (s, r) in enumerate(zip(layout.source, layout.real))]
        full, rms, regional_rms, pieces, diagnostics = [], [], [], {}, {}
        families = zip(NAMES, R63.old_loss.feature_vectors(candidate), R63.old_loss.feature_vectors(ref), self.directions) if {'full_window', 'coordinate_RMS'}.intersection(terms) else ()
        for name, fy, fr, direction in families:
            fw_r, rms_r = [], []
            for k, (si, ri) in enumerate(zip(layout.source, layout.real)):
                a = ((fy[list(si)] - self.stats[name + '_center'][k]) / self.stats[name + '_scale'][k]).float()
                b = ((fr[list(ri)] - self.stats[name + '_center'][k]) / self.stats[name + '_scale'][k]).float()
                ma, mb = conditional_masses[k]
                marginal = self._w1(a, b, ma, mb)
                if 'full_window' in terms:
                    sliced = self._w1(a @ direction, b @ direction, ma, mb)
                    fw_r.append(.5 * (marginal.mean() + sliced.mean()))
                if 'coordinate_RMS' in terms:
                    # torch's vector_norm chooses the exact zero gradient at zero.
                    rms_r.append(torch.linalg.vector_norm(marginal) / math.sqrt(a.shape[1]))
            mass = torch.as_tensor(self.source_mass, dtype=torch.float64, device=self.device)
            if 'full_window' in terms:
                full.append((torch.stack(fw_r) * mass).sum())
            if 'coordinate_RMS' in terms:
                rms.append((torch.stack(rms_r) * mass).sum())
                regional_rms.append(torch.stack(rms_r).detach())
        if 'full_window' in terms:
            pieces['full_window'] = torch.stack(full).mean()
        if 'coordinate_RMS' in terms:
            pieces['coordinate_RMS'] = torch.stack(rms).mean()
            diagnostics['coordinate_RMS_by_region'] = torch.stack(regional_rms)
        sg, rg = list(layout.source_global), list(layout.real_global)
        global_sm, global_rm = self._mass(layout, 'SOURCE'), self._mass(layout, 'REAL')
        if 'PSD' in terms:
            yp, rp = R63.old_loss.psd(candidate), R63.old_loss.psd(ref)
            overall = self._w1(yp[sg, None], rp[rg, None], global_sm, global_rm)[0]
            conditional = [self._w1(yp[list(layout.source[k]), None], rp[list(layout.real[k]), None], *conditional_masses[k])[0]
                           for k in range(1, 8)]
            pieces['PSD'] = .5 * (overall + torch.stack(conditional).mean()) / self.cfg['PSD_loss_scale']
            diagnostics.update(overall_PSD_W1=overall.detach(), Q1_Q7_mean_PSD_W1=torch.stack(conditional).mean().detach())
        if 'noise' in terms:
            yn, rn = R63.old_loss.noise_features(candidate), R63.old_loss.noise_features(ref)
            pieces['noise'] = (self._w1(yn[sg], rn[rg], global_sm, global_rm) / self.stats['noise_scale']).mean()
        if 'temporal_joint' in terms:
            tfy = R63.loss.temporal_features(candidate, self.cfg['temporal_power_floor'])
            tfr = R63.loss.temporal_features(ref, self.cfg['temporal_power_floor'])
            temporal = []
            for k, (si, ri) in enumerate(zip(layout.source, layout.real)):
                a = ((tfy[list(si)] - self.stats['temporal_center'][k]) / self.stats['temporal_scale'][k]).float()
                b = ((tfr[list(ri)] - self.stats['temporal_center'][k]) / self.stats['temporal_scale'][k]).float()
                ma, mb = conditional_masses[k]
                w = self._w1(a, b, ma, mb)
                p = self._w1(a @ self.temporal_directions, b @ self.temporal_directions, ma, mb)
                temporal.append(.5 * (w.mean() + p.mean()))
            pieces['temporal_joint'] = (torch.stack(temporal) * torch.as_tensor(self.source_mass, device=self.device)).sum()
        diagnostics.update(SOURCE_global_mass=global_sm.to_dict(), REAL_global_mass=global_rm.to_dict(),
                           layout_kind=layout.kind, included_terms=terms, excluded_terms=C_NAMES,
                           numerical_device=str(self.device), uniform_fastpath=self.uniform_fastpath)
        return DResult(pieces, sum(pieces.values()), diagnostics)

    def endpoint_gradients(self, source_endpoint, reference, layout, *, terms=D_NAMES):
        """Return each D value and dD/dY [N_SOURCE,480] in ORIGINAL parent order.

        Y is projected pre-ADC; these gradients already contain canonical STE,
        all global/conditional masses, all reductions and duplicate scatter-add.
        Replay original endpoint and inject into out exactly once. Do not inject
        into raw, divide again, clip gradients, or run Box/STE a second time.
        """
        gradients, values, diagnostics = {}, {}, {}
        for name in terms:
            leaf = source_endpoint.detach().clone().requires_grad_(True)
            result = self(leaf, reference, layout, terms=(name,))
            gradients[name] = torch.autograd.grad(result.pieces[name], leaf)[0].detach()
            values[name] = float(result.pieces[name].detach())
            diagnostics.update(result.diagnostics)
        return {
            'values': values,
            'total_value': sum(values.values()),
            'gradients': gradients, 'total_gradient': sum(gradients.values()),
            'gradient_coordinate': 'original SOURCE parent order; projected endpoint before ADC',
            'diagnostics': dict(diagnostics, included_terms=tuple(terms)),
        }


def paired_layouts(source_slots_10x4x32, real_slots_10x4x32, source_regions, real_regions):
    """A=mean of four D32; B=D128 of SAME region/microbatch/slot occurrences."""
    s, r = np.asarray(source_slots_10x4x32), np.asarray(real_slots_10x4x32)
    if s.shape != (10, 4, 32) or r.shape != (10, 4, 32):
        raise ValueError('Paired layout requires explicit [region=10,microbatch=4,slot=32]')
    a = tuple(ParentLayout.region_major(s[:, k, :], r[:, k, :], source_regions, real_regions) for k in range(4))
    b = ParentLayout.region_major(s, r, source_regions, real_regions)
    return a, b
