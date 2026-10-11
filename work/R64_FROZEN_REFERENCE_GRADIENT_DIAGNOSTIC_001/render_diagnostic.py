"""Render registered, cached R64 summaries without new scientific computation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(root):
    root = Path(root)
    read = lambda name: json.loads((root/name).read_text('utf8'))
    geometry = read('ENDPOINT_GEOMETRY_SUMMARY.json')
    p, e = read('PARAMETER_GRADIENT_SUMMARY.json'), read('ENDPOINT_GRADIENT_SUMMARY.json')
    support = read('SUPPORTING_SUMMARY.json')
    out = root/'figures'
    out.mkdir(exist_ok=True)
    files = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 140, 'savefig.dpi': 170})
    palette = ['#3066be', '#eb8b23', '#159b88', '#b64d6c', '#7255a6', '#2b2e34']

    def save(fig, name, title, subtitle):
        fig.suptitle(title, fontsize=16, fontweight='bold', y=.985)
        fig.text(.5, .022, subtitle, ha='center', va='bottom', fontsize=9, color='#465267')
        fig.tight_layout(rect=[0, .075, 1, .94], h_pad=2.1, w_pad=2.)
        path = out/name
        fig.savefig(path, facecolor='white')
        plt.close(fig)
        files.append(str(path.relative_to(root)))

    regions = [row['scope'].replace('region', '') for row in geometry['rows'][1:]]
    gx = np.arange(10)
    terms = ['full_window', 'coordinate_RMS', 'temporal_joint']
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))
    for ax, term in zip(axes.flat[:3], terms):
        vals = np.array([row['SOURCE_vs_complete_REAL']['terms'][term] for row in geometry['rows'][1:]])
        control = np.array([row['same_treated_REAL_vs_complete_REAL']['terms'][term] for row in geometry['rows'][1:]])
        floor = np.array([row['matched_N_REAL_pair_floor']['terms'][term]['values'] for row in geometry['rows'][1:]])
        ax.fill_between(gx, floor.min(1), floor.max(1), color='#b9c4d3', alpha=.45, label='32 REAL pairs: min-max')
        ax.plot(gx, floor.mean(1), 's--', color='#64748b', label='REAL pair mean')
        ax.plot(gx, vals, 'o-', color=palette[0], label='SOURCE vs complete REAL')
        ax.plot(gx, control, '^-', color=palette[1], label='same-treated REAL control')
        ax.set(title=term.replace('_', ' '), xticks=gx, xticklabels=regions, xlabel='Original raw charge region', ylabel='Frozen standardized distance')
        ax.set_yscale('log')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    viewnames = [v['view'] for v in geometry['rows'][1]['SOURCE_vs_complete_REAL']['views']]
    heat = np.array([[v['coordinate_RMS'] for v in row['SOURCE_vs_complete_REAL']['views']] for row in geometry['rows'][1:]])
    image = axes[1, 1].imshow(np.log10(np.maximum(heat.T, 1e-15)), aspect='auto', cmap='magma')
    axes[1, 1].set(title='SOURCE coordinate RMS by view', xticks=gx, xticklabels=regions,
                   yticks=np.arange(5), yticklabels=viewnames, xlabel='Original raw charge region')
    fig.colorbar(image, ax=axes[1, 1], label='log10 standardized coordinate RMS')
    save(fig, '01_endpoint_geometry.png', 'R64 | Complete TRAIN endpoint geometry',
         'Frozen direct endpoint; original SOURCE regional masses. REAL pairs describe finite sampling, not an acceptance limit.')

    names = p['reported_order']
    x = np.arange(6)
    labels = ['Full window', 'PSD', 'Noise', 'Coord. RMS', 'Temporal', 'Total D']
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))
    for col, (stat, coordinate) in enumerate([(p, 'Parameter'), (e, 'Parent endpoint')]):
        ax = axes[0, col]
        for dx, arm, color in [(-.18, 'A', palette[0]), (.18, 'B', palette[1])]:
            values = [stat['arms'][arm][name]['relative_mean_squared_l2_error'] for name in names]
            values = [np.nan if v is None else v for v in values]
            ax.bar(x+dx, values, width=.35, color=color, label='A: mean four D32' if arm == 'A' else 'B: one D128')
        ax.set(title=coordinate+' squared gradient error', xticks=x, xticklabels=labels,
               ylabel='Mean squared error / squared reference norm')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
        ax = axes[1, col]
        for dx, arm, color in [(-.1, 'A', palette[0]), (.1, 'B', palette[1])]:
            values = [stat['arms'][arm][name]['mean_cosine_reference'] for name in names]
            ax.plot(x+dx, [np.nan if v is None else v for v in values], 'o', color=color, label=arm)
        ax.axhline(0, color='#9ca3af', linewidth=.8)
        ax.set(title=coordinate+' mean cosine to full reference', xticks=x, xticklabels=labels,
               ylabel='Mean of 32 macro cosines', ylim=(-1.05, 1.05))
        ax.grid(axis='y', alpha=.2)
    save(fig, '02_gradient_comparison.png', 'R64 | Paired finite-reference gradient comparison',
         'Identical parent IDs, multiplicities, fixed noise and directions in both arms. Four constraint terms are excluded.')

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.7))
    for col, (stat, coordinate) in enumerate([(p, 'Parameter'), (e, 'Parent endpoint')]):
        ax = axes[col]
        means, low, high = [], [], []
        for name in names:
            item = stat['paired'][name]
            means.append(item['mean_B_minus_A_squared_l2_error'])
            low.append(item['bootstrap_percentile_95_interval'][0])
            high.append(item['bootstrap_percentile_95_interval'][1])
        for j in range(6):
            ax.plot([low[j], high[j]], [j, j], color=palette[j], linewidth=3)
            ax.plot(means[j], j, 'o', color=palette[j], markersize=6)
        ax.axvline(0, color='#64748b', linewidth=1)
        nonzero = np.abs(np.r_[low, high, means])
        nonzero = nonzero[nonzero > 0]
        ax.set_xscale('symlog', linthresh=float(nonzero.min()) if len(nonzero) else 1.)
        ax.set(title=coordinate+' gradient error: B minus A', yticks=x, yticklabels=labels,
               xlabel='Squared Euclidean error change (symmetric log scale)')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    save(fig, '03_paired_error_intervals.png', 'R64 | Paired macro bootstrap, fixed 1,000 resamples',
         'Total D parameter error is primary; components are descriptive. Negative intervals do not establish better training or A-F performance.')

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))
    bounds = [row['charge_mean_bound'] for row in geometry['rows'][1:]]
    weight = np.asarray(geometry['population']['source_mass'])
    lower = np.array([b['output_LB_V480_two_term_contribution'] for b in bounds])*weight
    measured = np.array([b['observed_V480_two_coordinate_contributions'] for b in bounds])*weight
    axes[0, 0].bar(gx-.17, measured, .34, color=palette[0], label='Measured V coordinate contribution')
    axes[0, 0].bar(gx+.17, lower, .34, color=palette[1], label='Output charge-mean lower bound')
    axes[0, 0].set(title='Original SOURCE mass weighted V contribution', xticks=gx, xticklabels=regions, ylabel='Full-window mean/8 + coordinate RMS/4')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(fontsize=8)
    rows = support['rows']
    bx = np.arange(11)
    for name, color, label in [('SOURCE_raw_vs_REAL_raw_PSD_W1', palette[2], 'Raw SOURCE / raw REAL'),
                             ('SOURCE_canonical_vs_REAL_canonical_PSD_W1', palette[0], 'SOURCE endpoint / REAL'),
                             ('REAL_control_vs_REAL_canonical_PSD_W1', palette[1], 'Same-treated REAL / REAL')]:
        axes[0, 1].plot(bx, [r[name] for r in rows], 'o-', color=color, label=label)
    axes[0, 1].set(title='Complete TRAIN PSD description', xticks=bx, xticklabels=['all']+regions, ylabel='PSD W1')
    axes[0, 1].legend(fontsize=8)
    with np.load(root/'WAVEFORM_STATISTICS_PRIVATE.npz', allow_pickle=False) as z:
        for name, color, label in [('SOURCE_raw', palette[2], 'Raw SOURCE'), ('SOURCE_canonical', palette[0], 'SOURCE endpoint'),
                                 ('REAL_canonical', '#273447', 'REAL reference'), ('REAL_control_canonical', palette[1], 'Same-treated REAL')]:
            for ax, statistic in zip(axes[1], ['mean', 'SD']):
                ax.plot(z[name+'__all__'+statistic], color=color, linewidth=1.3, label=label)
                ax.set(title='Complete-population waveform '+statistic, xlabel='Waveform coordinate', ylabel='Voltage')
        axes[1, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    save(fig, '04_charge_and_waveform.png', 'R64 | Charge constraint and complete waveform descriptions',
         'Charge bounds use nominal float64 arithmetic for this endpoint only; they do not prove original A-F infeasibility.')
    return files
