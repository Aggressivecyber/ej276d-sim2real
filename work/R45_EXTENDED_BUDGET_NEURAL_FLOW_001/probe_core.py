from __future__ import annotations
import hashlib
import numpy as np
import torch
from source_flow_core import initial, q_torch


def fast(x):
    return x[..., 100:150].sum(-1) - .5 * x[..., :100].sum(-1)


def build_probe(coupled, source, real):
    pick_rng = np.random.Generator(np.random.PCG64(20261010453))
    pick_initial = pick_rng.bit_generator.state
    pos = []
    for b in range(8):
        available = np.flatnonzero(coupled['bin'] == b)
        if len(available) < 512:
            raise ValueError(('fixed probe selection unsupported', b, len(available)))
        pos.append(pick_rng.choice(available, 512, replace=False))
    position = np.concatenate(pos).astype('i8')
    rt = np.random.Generator(np.random.PCG64(20261010454))
    rw = np.random.Generator(np.random.PCG64(20261010455))
    t_initial, w_initial = rt.bit_generator.state, rw.bit_generator.state
    t = rt.uniform(0, 1, 4096).astype('f4')
    white = rw.standard_normal((4096, 480)).astype('f4')
    si = coupled['source'][position].astype('i8')
    ri = coupled['matched_target'][position].astype('i8')
    package = dict(original_consumed_position=position, source_TRAIN_index=si,
                   matched_target_TRAIN_index=ri, bin=coupled['bin'][position].astype('i8'),
                   source_V=source[si].copy(), target_V=real[ri].copy(), t=t, white=white)
    binding = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in package.items()}
    info = {
        'rows': 4096, 'rows_per_bin': 512,
        'selection': 'one PCG64(20261010453), bins0..7 in order; choice512 without replacement in each occurrence pool',
        'unique_SOURCE_REAL': [len(np.unique(si)), len(np.unique(ri))],
        'not_unseen_parent_validation': True, 'selection_initial_RNG': pick_initial,
        'selection_final_RNG': pick_rng.bit_generator.state,
        't_initial_RNG': t_initial, 't_final_RNG': rt.bit_generator.state,
        'white_initial_RNG': w_initial, 'white_final_RNG': rw.bit_generator.state,
        'payload_sha256': binding,
    }
    return package, info


@torch.no_grad()
def fixed_probe(model, field, package, mu, scale, initial_parameters, global_step, device):
    was_training = model.training
    model.eval()
    rows = []
    max_path = max_field = 0.
    for j in range(0, 4096, 32):
        sl = slice(j, j + 32)
        b = torch.as_tensor(package['bin'][sl], dtype=torch.int64, device=device)
        x = torch.as_tensor(package['source_V'][sl], device=device)
        w = torch.as_tensor(package['white'][sl], device=device)
        a, _, _ = initial(x, w, mu[b], scale[b])
        target = torch.as_tensor(package['target_V'][sl], dtype=torch.float64, device=device)
        c = (target / q_torch(target)[:, None] - mu[b]) / scale[b]
        t = torch.as_tensor(package['t'][sl], dtype=torch.float64, device=device)
        y = (1 - t[:, None]) * a + t[:, None] * c
        v = c - a
        pred = field(y, t, b)
        error = pred - v
        ePSD = -fast(scale[b] * error)
        targetPSD = -fast(scale[b] * v)
        predictedPSD = -fast(scale[b] * pred)
        values = torch.stack([error.square().mean(1), ePSD, targetPSD, predictedPSD], 1)
        if not torch.isfinite(values).all():
            raise FloatingPointError('nonfinite fixed probe')
        rows.append(values.cpu().numpy())
        max_path = max(max_path, float(q_torch(scale[b] * v).abs().max()))
        max_field = max(max_field, float(q_torch(scale[b] * pred).abs().max()))
    all_rows = np.concatenate(rows)
    def summary(mask):
        v = all_rows[mask]
        if not len(v):
            return {'N': 0, 'MSE480': None, 'PSD_RMSE': None, 'PSD_signed_mean': None}
        return {'N': len(v), 'MSE480': float(v[:, 0].mean()),
                'PSD_RMSE': float(np.sqrt(np.mean(v[:, 1]**2))), 'PSD_signed_mean': float(v[:, 1].mean()),
                'target_PSD_velocity_mean': float(v[:, 2].mean()), 'predicted_PSD_velocity_mean': float(v[:, 3].mean())}
    time_bin = np.minimum((package['t'].astype('f8') * 4).astype('i8'), 3)
    by_q = [{'bin_id': b + 1, **summary(package['bin'] == b)} for b in range(8)]
    by_q_time = [{'bin_id': b + 1, 'time_bin0': tb, **summary((package['bin'] == b) & (time_bin == tb))}
                 for b in range(8) for tb in range(4)]
    square = 0.; max_delta = 0.; changed = 0
    for name, parameter in model.named_parameters():
        d = parameter.detach().to('cpu', dtype=torch.float64) - initial_parameters[name].to(torch.float64)
        square += float(d.square().sum()); max_delta = max(max_delta, float(d.abs().max()))
        changed += int(torch.count_nonzero(d))
    if was_training: model.train()
    result = {'global_step': global_step, 'overall': summary(np.ones(4096, dtype=bool)),
              'by_Q': by_q, 'by_Q_time': by_q_time, 'chunk': 32,
              'max_path_q_residual': max_path, 'max_field_q_residual': max_field,
              'parameter_delta_from_inherited2950': {'L2': square**.5, 'maxabs': max_delta, 'changed_values': changed},
              'new_fit': False, 'used_for_checkpoint_selection': False, 'ODE_error_claim': False}
    return result, all_rows
