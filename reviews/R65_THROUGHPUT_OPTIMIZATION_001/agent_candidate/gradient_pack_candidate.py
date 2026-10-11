"""R65 candidate only: native-dtype pack -> one CPU copy -> CPU f64 cast.

No random draws, parameter mutation, data I/O, optimizer changes, or caching.
The trainer's original 64-row VJP schedule and CPU f64 addition stay untouched.
"""
from __future__ import annotations


def _flatten_legacy(gradients, parameters):
    import torch
    flat = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None or gradient.shape != parameter.shape or gradient.dtype != parameter.dtype:
            raise RuntimeError("Missing, reshaped or differently typed parameter gradient")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Nonfinite parameter gradient")
        flat.append(gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float64))
    return torch.cat(flat)


def flatten_gradients_cpu_f64(gradients, parameters):
    """Bit-preserving fast path for homogeneous real f32/f64 gradients.

    All elements retain original parameter order. A same-dtype cat does no
    reduction. f32 -> f64 is exact for finite IEEE values. Mixed devices/dtypes
    and other dtypes fall back to the old operation sequence. CUDA subnormal
    behavior must still be tested against the original fused .to(cpu, f64).
    """
    import torch
    gradients, parameters = tuple(gradients), tuple(parameters)
    if len(gradients) != len(parameters) or not parameters:
        raise ValueError("Every supplied parameter must have one gradient")
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None or gradient.shape != parameter.shape or gradient.dtype != parameter.dtype:
            raise RuntimeError("Missing, reshaped or differently typed parameter gradient")
    first = gradients[0]
    if first.dtype not in (torch.float32, torch.float64) or any(
        g.device != first.device or g.dtype != first.dtype or g.layout != torch.strided
        for g in gradients
    ):
        return _flatten_legacy(gradients, parameters)
    packed = torch.cat([g.detach().reshape(-1) for g in gradients])
    # Explicitly copy native dtype first; cast on CPU, never reduce on GPU.
    native_cpu = packed.to(device="cpu")
    result = native_cpu.to(dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Nonfinite parameter gradient")
    return result


def parameter_vjp_cpu_f64(output, parameters, gradient):
    import torch
    parameters = tuple(parameters)
    if output.shape != gradient.shape or output.dtype != torch.float64 or gradient.dtype != torch.float64:
        raise ValueError("Output and GY must be matching float64 tensors")
    if output.device != gradient.device or gradient.requires_grad:
        raise ValueError("Detached GY must be on the replay output device")
    values = torch.autograd.grad(output, parameters, grad_outputs=gradient,
                                 create_graph=False, retain_graph=False, allow_unused=False)
    return flatten_gradients_cpu_f64(values, parameters)


def scalar_parameter_gradient_cpu_f64(scalar, parameters, *, retain_graph=False):
    import torch
    parameters = tuple(parameters)
    if scalar.ndim != 0:
        raise ValueError("Scalar objective required")
    values = torch.autograd.grad(scalar, parameters, create_graph=False,
                                 retain_graph=retain_graph, allow_unused=False)
    return flatten_gradients_cpu_f64(values, parameters)
