"""Full-parent, actual-output-condition nine-term loss.

No data files, network inference, random draws, resampling or domain flags.
Endpoint values/physical statistics use float64; preADC leaves may be float32.
Caller supplies the already canonical SOURCE and processed REAL endpoints.
"""
from __future__ import annotations
from collections import OrderedDict
import math
import copy
import numpy as np
import torch

VIEWS = ("V480", "U480", "DU479", "D2U478")
REGIONS = tuple(range(-1, 9))
TERM_NAMES = ("full_window", "coordinate_RMS", "PSD", "temporal_joint",
              "noise", "source_order", "source_valley", "REAL_identity",
              "box_correction")
D_TERMS = TERM_NAMES[:5]
TEMPORAL_NAMES = (["RMS_V"] + [f"ACF_lag{k}" for k in range(1, 21)] +
                  [f"rFFT_power_bin{1+5*k}_{5+5*k}_fraction" for k in range(10)])


def charge(v):
    return v[:, 100:400].sum(1) - 3.0 * v[:, :100].sum(1)


def signed_psd(v):
    return 1.0 - (v[:, 100:150].sum(1) - .5 * v[:, :100].sum(1)) / charge(v)


def feature_vectors(v):
    centered = v - v[:, :100].mean(1, keepdim=True)
    u = centered / centered.amax(1, keepdim=True).clamp_min(.0025)
    return (v, u, torch.diff(u, dim=1), torch.diff(u, n=2, dim=1))


def noise_features(v, power_floor=1e-20):
    b = v[:, :100] - v[:, :100].mean(1, keepdim=True)
    power = b.square().mean(1).clamp_min(power_floor)
    return torch.stack((power.sqrt(), (b[:, 1:] * b[:, :-1]).mean(1) / power), 1)


def temporal_features(v, power_floor=1e-20):
    b = v[:, :100] - v[:, :100].mean(1, keepdim=True)
    power = b.square().mean(1).clamp_min(power_floor)
    acf = torch.stack([(b[:, k:] * b[:, :-k]).mean(1) / power
                       for k in range(1, 21)], 1)
    spectrum = torch.fft.rfft(b, dim=1)[:, 1:]
    frequency_power = (spectrum.real.square() + spectrum.imag.square()) / 10000.0
    multiplicity = frequency_power.new_full((50,), 2.0)
    multiplicity[-1] = 1.0
    bands = ((frequency_power * multiplicity).reshape(len(v), 10, 5).sum(2)
             / power[:, None])
    return torch.cat((power.sqrt()[:, None], acf, bands), 1)


def _tensor(value, name, device, *, shape=None, integer=False):
    if not isinstance(value, torch.Tensor) or value.device != device:
        raise TypeError(name + " must be a tensor on the common device")
    if integer:
        if value.dtype != torch.int64:
            raise TypeError(name + " must be int64")
    elif value.dtype != torch.float64:
        raise TypeError(name + " must be float64")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(name + " shape/count changed")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(name + " is nonfinite; no row deletion")
    return value


class UniformW1Cache:
    """Exact finite uniform quantile cells, with bounded device-local LRU.

    Counts determine exact rational boundaries. No approximate quantile grid.
    columns() batches feature dimensions and returns one W1 per column.
    """
    def __init__(self, device, max_entries=32, max_cells=262144):
        if max_entries < 1 or max_cells < 1:
            raise ValueError("cache caps must be positive")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.max_entries, self.max_cells = int(max_entries), int(max_cells)
        self._cache = OrderedDict()
        self.cached_cells = self.hits = self.misses = 0

    def _plan(self, n, m):
        if n <= 0 or m <= 0:
            raise ValueError("W1 requires both nonempty finite laws")
        key = (int(n), int(m))
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        denominator = math.lcm(n, m)
        step_a, step_b = denominator // n, denominator // m
        edges = sorted(set(range(0, denominator + 1, step_a)).union(
                       range(0, denominator + 1, step_b)))
        left, right = edges[:-1], edges[1:]
        plan = (torch.tensor([x // step_a for x in left], device=self.device,
                             dtype=torch.int64),
                torch.tensor([x // step_b for x in left], device=self.device,
                             dtype=torch.int64),
                torch.tensor([(b - a) / denominator for a, b in zip(left, right)],
                             device=self.device, dtype=torch.float64))
        cells = len(left)
        if cells <= self.max_cells:
            while self._cache and (len(self._cache) >= self.max_entries or
                                    self.cached_cells + cells > self.max_cells):
                _, previous = self._cache.popitem(last=False)
                self.cached_cells -= previous[2].numel()
            self._cache[key] = plan
            self.cached_cells += cells
        return plan

    def columns(self, a, b, *, b_sorted=False, check=True):
        if a.ndim == 1:
            a = a[:, None]
        if b.ndim == 1:
            b = b[:, None]
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError("W1 feature dimensions differ")
        if check:
            _tensor(a, "W1 candidate", self.device)
            _tensor(b, "W1 reference", self.device)
        ia, ib, mass = self._plan(len(a), len(b))
        sa = torch.sort(a, dim=0, stable=True).values
        sb = b if b_sorted else torch.sort(b, dim=0, stable=True).values
        return ((sa[ia] - sb[ib]).abs() * mass[:, None]).sum(0)

    def receipt(self):
        return {"entries": len(self._cache), "cells": self.cached_cells,
                "max_entries": self.max_entries, "max_cells": self.max_cells,
                "hits": self.hits, "misses": self.misses}


def _fenwick_counts(rank, low, high, low_order, high_order):
    """Strict resolved-inversion integer derivatives; Python/Numba identical."""
    n = len(rank)
    size = int(rank.max()) + 1
    result = np.zeros(n, dtype=np.int64)
    tree = np.zeros(size + 1, dtype=np.int64)
    p = 0
    for index in range(n):
        j = low_order[index]
        while p < n and high[high_order[p]] < low[j]:
            position = int(rank[high_order[p]]) + 1
            while position <= size:
                tree[position] += 1
                position += position & -position
            p += 1
        stop, count = int(rank[j]) + 1, 0
        while stop:
            count += tree[stop]
            stop -= stop & -stop
        result[j] -= p - count
    tree[:] = 0
    p = 0
    for index in range(n - 1, -1, -1):
        i = high_order[index]
        while p < n and low[low_order[n - 1 - p]] > high[i]:
            position = int(rank[low_order[n - 1 - p]]) + 1
            while position <= size:
                tree[position] += 1
                position += position & -position
            p += 1
        stop, count = int(rank[i]), 0
        while stop:
            count += tree[stop]
            stop -= stop & -stop
        result[i] += count
    return result


try:
    from numba import njit
except ImportError:
    _fenwick_counts_compiled = None
else:
    _fenwick_counts_compiled = njit(cache=False, fastmath=False)(_fenwick_counts)
NUMBA_AVAILABLE = _fenwick_counts_compiled is not None


class PrecisionOrderPlan:
    """Fixed original-PSD precision intervals, O(N log N) inversion sweeps.

    For each unordered resolved pair i,j with high_i < low_j, the old two
    directed hinge terms equal 2*max(y_i-y_j,0). The original uniform 1/N**2
    numerator/denominator cancel, leaving denominator N**2-sum(tie_count**2).
    Fenwick queries count STRICT output inversions. If g is their signed
    per-parent derivative, -cumsum(g in ascending-output order) is the count
    crossing each adjacent output gap. Summing nonnegative gap*crossings
    yields the same hinge value without subtracting large float sums.
    """
    def __init__(self, original_psd, low, high):
        self.original = np.array(original_psd, dtype=np.float64, copy=True)
        self.low = np.array(low, dtype=np.float64, copy=True)
        self.high = np.array(high, dtype=np.float64, copy=True)
        if (self.original.ndim != 1 or len(self.original) == 0 or
                self.low.shape != self.original.shape or
                self.high.shape != self.original.shape):
            raise ValueError("source precision arrays must have matching nonempty shapes")
        if not (np.isfinite(self.original).all() and np.isfinite(self.low).all()
                and np.isfinite(self.high).all()):
            raise ValueError("source precision arrays must be finite")
        if not ((self.low <= self.original).all() and
                (self.original <= self.high).all()):
            raise ValueError("precision intervals must enclose the original PSD")
        _, count = np.unique(self.original, return_counts=True)
        self.nonzero_pair_count = int(len(self.original)**2 - np.dot(count, count))
        self.low_order = np.argsort(self.low, kind="stable")
        self.high_order = np.argsort(self.high, kind="stable")
        self.backend = "numba" if NUMBA_AVAILABLE else "python"

    def value_gradient(self, prediction, scale, *, backend=None):
        y = np.asarray(prediction, dtype=np.float64)
        n = len(self.original)
        if y.shape != (n,) or not np.isfinite(y).all() or not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid order prediction/scale")
        if self.nonzero_pair_count == 0:
            return 0.0, np.zeros(n, dtype=np.float64)
        _, rank = np.unique(y, return_inverse=True)
        selected_backend = self.backend if backend is None else backend
        if selected_backend == "numba":
            if _fenwick_counts_compiled is None:
                raise RuntimeError("Numba backend requested but unavailable")
            function = _fenwick_counts_compiled
        elif selected_backend == "python":
            function = _fenwick_counts
        else:
            raise ValueError("unknown exact source-order backend")
        derivative_count = function(rank, self.low, self.high,
                                    self.low_order, self.high_order)

        order = np.argsort(y, kind="stable")
        crossings = -np.cumsum(derivative_count[order], dtype=np.int64)[:-1]
        if derivative_count.sum() != 0 or (crossings < 0).any():
            raise AssertionError("order sweep conservation failed")
        gaps = np.diff(y[order])
        hinge_sum = float(np.dot(gaps, crossings.astype(np.float64)))
        factor = 2.0 / self.nonzero_pair_count / float(scale)
        return hinge_sum * factor, derivative_count.astype(np.float64) * factor


class _PrecisionOrder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, plan, scale):
        value, derivative = plan.value_gradient(prediction.detach().cpu().numpy(), scale)
        gradient = torch.as_tensor(derivative, device=prediction.device,
                                   dtype=prediction.dtype)
        ctx.save_for_backward(gradient)
        return prediction.new_tensor(value)

    @staticmethod
    def backward(ctx, incoming):
        gradient, = ctx.saved_tensors
        return incoming * gradient, None, None


class FullWaveformLoss:
    """Full populations; nine unit-weight terms; current canonical-Q bins.

    Reference/features/statistics are cloned frozen constants. Each iter_terms
    yield is (name, scalar, diagnostics) from a fresh endpoint graph. Use detached
    endpoint leaves when accumulating one term's six-leaf gradient at a time.
    __call__ returns (total, terms, diagnostics) for an ordinary complete graph.
    """
    def __init__(self, real_reference_canonical, edges_logQ, source_psd,
                 source_interval_low, source_interval_high, source_bin_original,
                 source_h, valley_bounds, config):
        if not isinstance(real_reference_canonical, torch.Tensor):
            raise TypeError("reference must be a tensor")
        self.device = real_reference_canonical.device
        if real_reference_canonical.ndim != 2 or real_reference_canonical.shape[1] != 480 or len(real_reference_canonical) == 0:
            raise ValueError("reference must have nonempty (N,480) shape")
        _tensor(real_reference_canonical, "canonical REAL reference", self.device)
        self.reference = real_reference_canonical.detach().clone()
        self.real_count = len(self.reference)
        self.edges = _tensor(edges_logQ, "edges_logQ", self.device,
                             shape=(9,)).detach().clone()
        if not bool((self.edges[1:] > self.edges[:-1]).all()):
            raise ValueError("REAL edges must be strictly increasing")
        if source_psd.ndim != 1 or len(source_psd) == 0:
            raise ValueError("source_psd must be nonempty")
        self.source_count = len(source_psd)
        for value, name in ((source_psd, "source_psd"),
                            (source_interval_low, "source_interval_low"),
                            (source_interval_high, "source_interval_high")):
            _tensor(value, name, self.device, shape=(self.source_count,))
        self.source_bins = _tensor(source_bin_original, "source_bin_original",
                                   self.device, shape=(self.source_count,),
                                   integer=True).detach().clone()
        if not bool(((self.source_bins >= -1) & (self.source_bins <= 8)).all()):
            raise ValueError("original SOURCE raw bins must be -1..8")
        if not isinstance(source_h, torch.Tensor) or source_h.device != self.device or source_h.shape != (self.source_count,) or not bool(torch.isfinite(source_h).all()):
            raise ValueError("original source_h must retain all finite parent values")
        self.source_h = source_h.detach().clone()
        self.valley_bounds = _tensor(valley_bounds, "valley_bounds", self.device,
                                    shape=(2, 2)).detach().clone()
        if not bool((self.valley_bounds[:, 1] > self.valley_bounds[:, 0]).all()):
            raise ValueError("valley bounds must have low < high")

        self.config = copy.deepcopy({k: config[k] for k in (
            "feature_scale_floor", "noise_scale_floor", "temporal_scale_floor",
            "temporal_power_floor", "PSD_loss_scale", "valley_smoothing",
            "valley_delta", "loss_weights")})
        if (set(self.config["loss_weights"]) != set(TERM_NAMES) or
                any(float(w) != 1.0 for w in self.config["loss_weights"].values())):
            raise ValueError("all nine declared loss weights must equal one")
        if (float(self.config["PSD_loss_scale"]) != .01 or
                float(self.config["valley_delta"]) != .02):
            raise ValueError("fixed PSD=.01 and valley delta=.02 required")
        for key in ("temporal_power_floor", "valley_smoothing"):
            if not math.isfinite(float(self.config[key])) or float(self.config[key]) <= 0:
                raise ValueError(key + " must be positive")
        self.w1 = UniformW1Cache(self.device)
        self.order_plan = PrecisionOrderPlan(source_psd.detach().cpu().numpy(),
                         source_interval_low.detach().cpu().numpy(),
                         source_interval_high.detach().cpu().numpy())
        self.reference_bins = self.bins(self.reference)
        self.reference_counts = torch.bincount(self.reference_bins + 1, minlength=10)
        self._reference_counts = self.reference_counts.cpu().tolist()
        for region in range(7):
            if self._reference_counts[region + 1] == 0:
                raise ValueError(f"required REAL PSD bin {region} empty")
        self.stats = {}
        self._reference_features, self._reference_sorted = {}, {}
        for name, features in zip(VIEWS, feature_vectors(self.reference)):
            floor = float(self.config["feature_scale_floor"][name])
            if not math.isfinite(floor) or floor <= 0:
                raise ValueError("feature scale floors must be finite positive")
            self._prepare_reference(name, features, features.new_tensor(floor))
        tf = temporal_features(self.reference, float(self.config["temporal_power_floor"]))
        floor = torch.as_tensor(self.config["temporal_scale_floor"], device=self.device,
                                dtype=torch.float64)
        if floor.shape != (31,) or not bool(torch.isfinite(floor).all() and (floor > 0).all()):
            raise ValueError("31 finite positive temporal floors required")
        self._prepare_reference("temporal", tf, floor)
        nf = noise_features(self.reference)
        nfloor = torch.as_tensor(self.config["noise_scale_floor"], device=self.device,
                                 dtype=torch.float64)
        if nfloor.shape != (2,) or not bool(torch.isfinite(nfloor).all() and (nfloor > 0).all()):
            raise ValueError("two finite positive noise floors required")
        self.stats["noise_mean"] = nf.mean(0)
        self.stats["noise_scale"] = torch.maximum(nf.std(0, unbiased=False), nfloor)
        self._noise_sorted = torch.sort(
            (nf - self.stats["noise_mean"]) / self.stats["noise_scale"],
            dim=0, stable=True).values
        rp = signed_psd(self.reference)
        self._reference_psd_sorted = torch.sort(rp, stable=True).values
        self._reference_psd_by_bin = [
            torch.sort(rp[self.reference_bins == region], stable=True).values
            if self._reference_counts[region + 1] else None for region in REGIONS]
        self._identity_scale = torch.empty_like(self.reference)
        for region in REGIONS:
            use = self.reference_bins == region
            if self._reference_counts[region + 1]:
                self._identity_scale[use] = self.stats["V480_scale"][region + 1]
        self._valley_masks, raw_fraction = [], []
        for row, region in enumerate((2, 3)):
            use = (self.source_bins == region) & (self.source_h == 1)
            if not bool(use.any()):
                raise ValueError(f"original bin {region} h==1 valley denominator empty")
            self._valley_masks.append(use)
            low, high = self.valley_bounds[row]
            raw_fraction.append(((source_psd[use] >= low) &
                                 (source_psd[use] <= high)).to(torch.float64).mean())
        self.stats["source_raw_valley_fraction"] = torch.stack(raw_fraction).detach()
        self.stats["valley_bounds"] = self.valley_bounds
        self.stats["REAL_counts"] = self.reference_counts

    def bins(self, v):
        q = charge(v).detach()
        if not bool(torch.isfinite(q).all() and (q > 0).all()):
            raise ValueError("canonical Q must be finite positive; no row deletion")
        logq = torch.log(1000.0 * q)
        raw = torch.searchsorted(self.edges, logq.contiguous(), right=True) - 1
        return torch.where(logq == self.edges[-1], torch.full_like(raw, 7), raw)

    def _prepare_reference(self, name, features, floor):
        centers, scales, reference, sorted_reference = [], [], [], []
        for region in REGIONS:
            count = self._reference_counts[region + 1]
            if not count:
                centers.append(None); scales.append(None)
                reference.append(None); sorted_reference.append(None)
                continue
            selected = features[self.reference_bins == region]
            center = selected.mean(0)
            scale = torch.maximum(selected.std(0, unbiased=False), floor)
            if not bool(torch.isfinite(center).all() and torch.isfinite(scale).all() and (scale > 0).all()):
                raise ValueError("invalid fitted legal REAL statistics")
            normalized = (selected - center) / scale
            centers.append(center.detach()); scales.append(scale.detach())
            reference.append(normalized.detach())
            sorted_reference.append(torch.sort(normalized, dim=0, stable=True).values.detach())
        self.stats[name + "_center"] = centers
        self.stats[name + "_scale"] = scales
        self._reference_features[name] = reference
        self._reference_sorted[name] = sorted_reference

    def _context(self, source_canonical, real_processed_canonical,
                 source_preADC, REAL_preADC, source_raw, REAL_raw,
                 directions4, temporal_directions):
        values = (source_canonical, real_processed_canonical, source_preADC,
                  REAL_preADC, source_raw, REAL_raw)
        for i, value in enumerate(values):
            n = self.source_count if i % 2 == 0 else self.real_count
            if not isinstance(value, torch.Tensor) or value.device != self.device or value.shape != (n, 480):
                raise ValueError("endpoint leaf shape/device/parent count changed")
            if i in (2, 3):
                if value.dtype not in (torch.float32, torch.float64):
                    raise TypeError("preADC leaves must be float32 or float64")
            elif value.dtype != torch.float64:
                raise TypeError("canonical and raw endpoints must be float64")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite endpoint; no deletion/masking")
        sb, rb = self.bins(source_canonical), self.bins(real_processed_canonical)
        counts = [torch.bincount(b + 1, minlength=10).cpu().tolist() for b in (sb, rb)]
        for label, count in zip(("SOURCE", "REAL_processed"), counts):
            for region in range(7):
                if count[region + 1] == 0:
                    raise ValueError(f"required {label} PSD bin {region} empty")
            for slot, n in enumerate(count):
                if n and self._reference_counts[slot] == 0:
                    raise ValueError(f"occupied {label} raw bin {slot-1} has empty REAL reference")
        if len(directions4) != 4:
            raise ValueError("four public sliced-direction tensors required")
        directions = []
        for dim, direction in zip((480, 480, 479, 478), directions4):
            if not isinstance(direction, torch.Tensor) or direction.device != self.device or direction.shape != (dim, 64) or direction.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(direction).all()):
                raise ValueError("invalid full-window sliced directions")
            directions.append(direction.detach().to(torch.float64))
        if not isinstance(temporal_directions, torch.Tensor) or temporal_directions.device != self.device or temporal_directions.shape != (31, 64) or temporal_directions.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(temporal_directions).all()):
            raise ValueError("invalid temporal sliced directions")
        diagnostics = {"SOURCE_bin_counts": counts[0], "REAL_processed_bin_counts": counts[1],
                       "SOURCE_bin_mass": [n / self.source_count for n in counts[0]],
                       "REAL_processed_bin_mass": [n / self.real_count for n in counts[1]],
                       "raw_regions": list(REGIONS), "full_original_parent_uniform_weights": True,
                       "hard_bins_use_current_canonical_output_Q": True}
        return values, (sb, rb), counts, directions, temporal_directions.detach().to(torch.float64), diagnostics

    def _conditional_features(self, v, bins, counts, directions, mode):
        if mode == "temporal":
            arrays = (temporal_features(v, float(self.config["temporal_power_floor"])),)
            names, direction_list = ("temporal",), (directions,)
        else:
            arrays = feature_vectors(v)
            names, direction_list = VIEWS, directions
        result = v.new_zeros(())
        for name, values, direction in zip(names, arrays, direction_list):
            feature_term = v.new_zeros(())
            for slot, n in enumerate(counts):
                if not n:  # Precisely zero CURRENT population mass; not a failed nonempty comparison.
                    continue
                selected = values[bins == slot - 1]
                a = ((selected - self.stats[name + "_center"][slot])
                     / self.stats[name + "_scale"][slot])
                marginal_vector = self.w1.columns(a, self._reference_sorted[name][slot], b_sorted=True, check=False)
                if mode == "coordinate_RMS":
                    regional = torch.linalg.vector_norm(marginal_vector) / math.sqrt(values.shape[1])
                else:
                    b = self._reference_features[name][slot]
                    sliced = self.w1.columns(a @ direction, b @ direction, check=False).mean()
                    regional = .5 * (marginal_vector.mean() + sliced)
                feature_term = feature_term + (n / len(v)) * regional
            result = result + feature_term
        return result / len(names)

    def _domain_PSD(self, v, bins):
        p = signed_psd(v)
        global_w1 = self.w1.columns(p, self._reference_psd_sorted, b_sorted=True, check=False).mean()
        conditional = torch.stack([
            self.w1.columns(p[bins == region], self._reference_psd_by_bin[region + 1],
                            b_sorted=True, check=False).mean() for region in range(7)]).mean()
        return .5 * (global_w1 + conditional) / .01

    def _domain_noise(self, v):
        values = ((noise_features(v) - self.stats["noise_mean"]) / self.stats["noise_scale"])
        return self.w1.columns(values, self._noise_sorted, b_sorted=True, check=False).mean()

    def _compute_term(self, name, context):
        values, bins, counts, directions, tdirections, _ = context
        sc, rc, sp, rp, sr, rr = values
        if name in ("full_window", "coordinate_RMS", "temporal_joint"):
            mode = "temporal" if name == "temporal_joint" else name
            direction = tdirections if mode == "temporal" else directions
            return .5 * sum(self._conditional_features(v, b, c, direction, mode)
                             for v, b, c in zip((sc, rc), bins, counts))
        if name == "PSD":
            return .5 * (self._domain_PSD(sc, bins[0]) + self._domain_PSD(rc, bins[1]))
        if name == "noise":
            return .5 * (self._domain_noise(sc) + self._domain_noise(rc))
        if name == "source_order":
            return _PrecisionOrder.apply(signed_psd(sc), self.order_plan, .01)
        if name == "source_valley":
            p = signed_psd(sc)
            terms = []
            for row, use in enumerate(self._valley_masks):
                low, high = self.valley_bounds[row]
                probability = (torch.sigmoid((p[use] - low) / self.config["valley_smoothing"])
                               * torch.sigmoid((high - p[use]) / self.config["valley_smoothing"])).mean()
                excess = torch.relu(probability - self.stats["source_raw_valley_fraction"][row] - .02)
                terms.append(excess.square() / (.02 ** 2))
            return torch.stack(terms).mean()
        if name == "REAL_identity":
            return ((rc - self.reference) / self._identity_scale).square().mean()
        if name == "box_correction":
            return .5 * ((sp.to(torch.float64) - sr).square().mean()
                         + (rp.to(torch.float64) - rr).square().mean()) / (.5 ** 2)
        raise KeyError(name)

    def iter_terms(self, source_canonical, real_processed_canonical,
                   source_preADC, REAL_preADC, source_raw, REAL_raw,
                   directions4, temporal_directions):
        context = self._context(source_canonical, real_processed_canonical,
                    source_preADC, REAL_preADC, source_raw, REAL_raw,
                    directions4, temporal_directions)
        for name in TERM_NAMES:
            value = self._compute_term(name, context)
            if not bool(torch.isfinite(value)):
                raise FloatingPointError("nonfinite " + name + "; no masking")
            yield name, value, context[-1]

    def __call__(self, *args):
        terms, diagnostics = {}, None
        for name, value, diagnostics in self.iter_terms(*args):
            terms[name] = value
        total = sum(terms.values())
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("nonfinite total")
        return total, terms, diagnostics

    def stats_serializable(self):
        def plain(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value
        return {"source_count": self.source_count, "REAL_count": self.real_count,
                "raw_regions": list(REGIONS), "edges_logQ": plain(self.edges),
                "stats": {k: plain(v) for k, v in self.stats.items()},
                "config": self.config, "temporal_names": TEMPORAL_NAMES,
                "source_original_bin_counts": torch.bincount(
                     self.source_bins + 1, minlength=10).cpu().tolist(),
                "source_valley_h1_counts": [int(v.sum().item()) for v in self._valley_masks],
                "source_order_all_original_nonzero_directed_pair_count":
                    self.order_plan.nonzero_pair_count,
                "source_order_algorithm": "two strict-interval Fenwick sweeps O(N log N); CPU",
                "source_order_backend": self.order_plan.backend,
                "numba_available": NUMBA_AVAILABLE,
                "numba_cache": False, "numba_fastmath": False,
                "quantile_cache": self.w1.receipt()}
