"""480-D spline coupling flows; nflows 0.14 provides the spline/Jacobian code.

The convolutional conditioner and symmetric likelihood training are project
adaptations of Neural Spline Flows and Flows for Flows, not their published
low-dimensional architecture. All coordinates and physical scale are retained.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from nflows.transforms import CompositeTransform, PiecewiseRationalQuadraticCouplingTransform
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow


class ResidualConv(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        return (x + self.conv(torch.nn.functional.silu(x))) / math.sqrt(2.0)


class WaveConditioner(nn.Module):
    def __init__(self, in_features, out_features, mask, channels, bins, dilations):
        super().__init__()
        assert len(mask) % 2 == 0 and in_features == len(mask) // 2
        self.parameters_per_sample = out_features // in_features
        assert self.parameters_per_sample == 3 * bins - 1
        time = torch.linspace(-1., 1., len(mask))
        self.register_buffer('input_time', time[mask <= 0])
        self.register_buffer('output_time', time[mask > 0])
        self.input = nn.Conv1d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualConv(channels, d) for d in dilations])
        self.output = nn.Conv1d(channels, self.parameters_per_sample, 1)
        # Equal bin widths/heights and derivative exactly one give identity.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[2*bins:].fill_(math.log(math.expm1(1. - 1e-3)))

    def forward(self, x, context=None):
        n = len(x)
        h = torch.stack([x, self.input_time.expand(n, -1), self.output_time.expand(n, -1)], dim=1)
        h = self.blocks(self.input(h))
        out = self.output(torch.nn.functional.silu(h))
        return out.transpose(1, 2).reshape(n, -1)


def make_transform(cfg, features=480, layers=None):
    layers = cfg['layers'] if layers is None else layers
    blocks = []
    for index in range(layers):
        mask = ((torch.arange(features) + index) % 2).float()
        def factory(a, b, mask=mask):
            return WaveConditioner(a, b, mask, cfg['channels'], cfg['bins'], cfg['dilations'])
        blocks.append(PiecewiseRationalQuadraticCouplingTransform(
            mask=mask, transform_net_create_fn=factory, num_bins=cfg['bins'],
            tails='linear', tail_bound=cfg['tail_bound'],
            min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3))
    return CompositeTransform(blocks)


def make_density(cfg):
    return Flow(make_transform(cfg), StandardNormal([480]))


def symmetric_loss(transform, source_density, real_density, x, y):
    tx, logdet = transform(x)
    iy, inv_logdet = transform.inverse(y)
    # log p_T(x) = log p_REAL(T(x)) + log |det DT(x)| and reverse.
    source_term = -(real_density.log_prob(tx) + logdet).mean() / x.shape[1]
    target_term = -(source_density.log_prob(iy) + inv_logdet).mean() / y.shape[1]
    return .5 * (source_term + target_term), source_term, target_term


def engineering_checks(cfg, device):
    """Numerical inverse/Jacobian and gradients; no scientific fitting."""
    torch.manual_seed(cfg['seed'] + 900)
    toy = make_transform(cfg['model'], features=8, layers=2).double().to(device)
    x = torch.randn(3, 8, dtype=torch.float64, device=device) * .3
    with torch.no_grad():
        identity, identity_ld = toy(x)
        identity_error = float((identity - x).abs().max())
        for block in toy._transforms:
            block.transform_net.output.weight.normal_(0., .004)
    y, ld = toy(x)
    back, ild = toy.inverse(y)
    inv_error = float((back - x).abs().max().detach())
    ld_error = float((ld + ild).abs().max().detach())
    one = x[0].detach().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(lambda z: toy(z[None])[0][0], one)
    actual = torch.linalg.slogdet(jac)[1]
    jac_error = float((actual - ld[0]).abs().detach())
    (y.square().mean() + ld.mean()).backward()
    grads = [p.grad for p in toy.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    # Identity bias was initialized in float32 before the double precision test.
    assert identity_error < 1e-7 and inv_error < 1e-8 and ld_error < 1e-8 and jac_error < 1e-8 and finite
    return {'identity_maxabs': identity_error, 'inverse_maxabs': inv_error,
            'logdet_cancellation_maxabs': ld_error, 'autograd_logdet_maxabs': jac_error,
            'nontrivial_transform': True, 'finite_gradients': finite}
