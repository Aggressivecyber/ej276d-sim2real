from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS']:
    os.environ[key] = '4'
import pathlib, json, time, datetime, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
START = time.monotonic()
m = json.loads((ROOT / 'metrics.json').read_text('utf8'))
arms = ['INDEPENDENT_CFM', 'OT_CFM']
names = ['BASE', *arms]
labels = {'REAL': 'REAL DEV', 'BASE': 'Frozen base', 'INDEPENDENT_CFM': 'Independent CFM', 'OT_CFM': 'OT CFM', 'TRAIN_REFERENCE': 'REAL TRAIN ref'}
colors = {'REAL': '#222222', 'BASE': '#777777', 'INDEPENDENT_CFM': '#2474b5', 'OT_CFM': '#d66b16', 'TRAIN_REFERENCE': '#57a36c'}

def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', 'utf8')

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight')
    plt.close(fig)

def ratio_change(new, old):
    return new / old - 1 if old is not None and new is not None and old > 0 else None

# All gate arithmetic uses previously saved statistics. No fitted evaluator, response,
# random draw or solver call is made in this entry.
views = ['full_historical_shape', 'normalized_u480']
base_pairs = {v: m['C2ST']['paired']['OT_minus_BASE'][v] for v in views}
ind_pairs = {v: m['C2ST']['paired']['OT_minus_INDEPENDENT'][v] for v in views}
g1 = {'pass': all(r['BA_delta_95CI'][1] < 0 for r in base_pairs.values()), 'paired_OT_minus_BASE': base_pairs}
g2 = {'point_not_worse_both': all(r['BA_delta_first_minus_second'] <= 0 for r in ind_pairs.values()),
      'at_least_one_paired_upper_below_zero': any(r['BA_delta_95CI'][1] < 0 for r in ind_pairs.values()),
      'paired_OT_minus_INDEPENDENT': ind_pairs}
g2['pass'] = g2['point_not_worse_both'] and g2['at_least_one_paired_upper_below_zero']
psd = m['PSD_main']
common_psd = [i for i in range(8) if all(psd['tables'][n]['bins'][i]['N'] >= 40 for n in ['REAL', *names])]
common_psd_values = {n: float(np.mean([psd['vs_REAL'][n]['bins'][i]['W1'] for i in common_psd])) if common_psd else None for n in names}
measures = {}
for field in ['PSD', 'baseline_RMS_V', 'ACF1']:
    values = {n: m['domain_vs_REAL'][n]['overall'][field]['W1'] for n in names}
    change = ratio_change(values['OT_CFM'], values['BASE'])
    measures[field + '_overall_W1'] = {'values': values, 'OT_relative_change_vs_BASE': change, 'no_more_than_10pct_worse': change is not None and change <= .10}
change = ratio_change(common_psd_values['OT_CFM'], common_psd_values['BASE'])
measures['PSD_common_Q_equal_mean_W1'] = {'values': common_psd_values, 'common_bin_ids_1based': [i + 1 for i in common_psd],
    'minimum_each_domain_N': 40, 'OT_relative_change_vs_BASE': change, 'no_more_than_10pct_worse': change is not None and change <= .10,
    'descriptive_original_PSD_table_support_rule_unchanged': True}
width_common = m['energy_conditional_width']['common_support_errors']['all_final_endpoints']
for row in width_common:
    if row['field'] == 'sigma_PSD':
        change = ratio_change(row['MAE']['OT_CFM'], row['MAE']['BASE'])
        measures[f'component{row["component"]}_raw_width_MAE'] = {'values': row['MAE'], 'common_bin_ids_1based': row['common_bin_ids_1based'],
            'OT_relative_change_vs_BASE': change, 'no_more_than_10pct_worse': change is not None and change <= .10}
gaps = {n: [{'bin_id_1based': r['bin'] + 1, 'N': r['N'], 'failure_reasons': r['failure_reasons']} for r in m['energy_conditional_width']['tables'][n]['slices'] if not r['valid']] for n in ['REAL', *names]}
g3 = {'pass': all(r['no_more_than_10pct_worse'] for r in measures.values()) and not any(gaps.values()),
      'measures': measures, 'density_support_gaps': gaps, 'missing_support_is_not_a_pass': True}
primary = m['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4']
valley_rows = []
for b in ['bin3', 'bin4']:
    values = {n: primary[n][b]['valley_fraction'] for n in names}
    delta = values['OT_CFM'] - values['BASE']
    valley_rows.append({'source_bin_1based': int(b[-1]), 'source_N': primary['BASE'][b]['N'], 'values': values,
        'OT_minus_BASE_fraction': delta, 'increase_no_more_than_2_percentage_points': delta <= .02})
rank_rows = []
for field in ['source_Q_to_output_Q', 'source_PSD_to_output_PSD']:
    values = {n: m['source_association_actual_preADC_input'][n]['composition_and_PDG']['all'][field]['Spearman'] for n in names}
    drop = values['BASE'] - values['OT_CFM']
    rank_rows.append({'measure': field, 'N': 2000, 'values': values, 'BASE_minus_OT_drop': drop, 'drop_no_more_than_0_05': drop <= .05})
technical = {}
for n in arms:
    gg = m['generation'][n]
    technical[n] = {'all_IDs_preserved': gg['all_original_source_ids_preserved'], 'QC_valid': m['domains'][n]['QC_valid'],
        'invalid': m['domains'][n]['invalid'], 'ADC': gg['ADC'], 'flow_applied_N': gg['flow_applied_N'], 'base_fallback_N': gg['unsupported_source_Q_base_fallback_N'],
        'source_roll_RMS_mean_V': m['source_probes'][n]['rows']['source_roll_no_fixed_points']['waveform_RMS_vs_correct_reference_V']['mean']}
source_ok = all(r['all_IDs_preserved'] and r['QC_valid'] == 2000 and r['invalid'] == 0 and r['ADC']['pre_ADC_finite'] and r['ADC']['clip_count'] == 0 and r['source_roll_RMS_mean_V'] > 0 for r in technical.values())
g4 = {'pass': source_ok and all(r['increase_no_more_than_2_percentage_points'] for r in valley_rows) and all(r['drop_no_more_than_0_05'] for r in rank_rows),
      'primary_source_definition': 'actual unADC 480-point NN input; BASE and both arms use identical input features',
      'technical': technical, 'pure_had_valley': valley_rows, 'whole_population_rank': rank_rows,
      'same_composition': m['data_roles']['DEV_composition'], 'nonzero_source_sensitivity_is_not_physical_correctness': True,
      'frozen_valley_membership_is_not_particle_error_truth': True}
replay = json.loads((ROOT / 'REPLAY_RECEIPT.json').read_text('utf8'))
g5 = {'pass': all(r['pass'] for r in m['solver_check'].values()) and replay['pass'] and replay['rows'] == 4000,
      'solver_check': m['solver_check'], 'replay_receipt': 'REPLAY_RECEIPT.json', 'replay_rows': replay['rows']}
gate = {'preset_recipe': 'fixed OT-CFM', 'retain': all(r['pass'] for r in [g1, g2, g3, g4, g5]),
        'decision': 'close_this_fixed_OT_CFM_recipe', 'gate1_full_response_vs_BASE': g1, 'gate2_OT_over_INDEPENDENT': g2,
        'gate3_PSD_width_noise_guardrails': g3, 'gate4_source_guardrails': g4, 'gate5_solver_and_replay': g5,
        'changes_to_prespecified_rules': False, 'new_fit_generation_or_projection_blocks': 0}
write('GATE_METRICS.json', gate)

compact = {'status': m['status'], 'gate': gate, 'data_roles': m['data_roles'], 'counters': m['counters'], 'generation': m['generation'],
    'C2ST': m['C2ST'], 'PSD_main': m['PSD_main'], 'domain_vs_REAL': m['domain_vs_REAL'],
    'density_support': gaps, 'width_common_support_errors': width_common, 'projection_common_support_summary': m['projection_distances']['common_support_summary'],
    'conditional_u_common_mean_MSE': m['conditional_mean_SD']['common_bin_equal_mean_full480_MSE'],
    'raw_voltage_moments': m['raw_voltage_moments'], 'source_primary_actual_Q': primary,
    'source_historical_ADC_reference_Q': m['source_composition']['historical_postADC_reference_Q_bins3_4'],
    'primary_source_rank': rank_rows, 'training': {n: {k: r[k] for k in ['updates', 'actual_rows', 'training_s', 'transport_solver_s', 'selection', 'GPU_peak_allocated_bytes', 'G_frozen', 'G_parameters_updated']} for n, r in m['training'].items()},
    'training_trace': {n: json.loads((ROOT/n/'TRAINING_TRACE.json').read_text('utf8')) for n in arms},
    'solver_check': m['solver_check'], 'replay_pass': replay['pass'], 'physical_final_signal_electronic_split': 'unknown',
    'EVAL_read': 0, 'new_G4': 0, 'all_source_parents_kept': True,
    'privacy': 'aggregate only; no event IDs/selected row arrays/NN predictions/RNG streams'}
write('METRICS_COMPACT.json', compact)

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 150})
features = load(ROOT/'results/FEATURES_PRIVATE.npz')
waves = load(ROOT/'results/PLOT_WAVEFORMS_PRIVATE.npz')
edges = np.asarray(psd['edges_logQ']); tt = np.arange(480)-120

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for n in arms:
    trace = compact['training_trace'][n]; step = [r['step'] for r in trace]
    ax[0].plot(step, [r['CFM_loss'] for r in trace], 'o-', color=colors[n], label=labels[n])
    ax[1].plot(step, [r['coupling_mean480_squared_cost'] for r in trace], 'o-', color=colors[n], label=labels[n])
    ax[2].plot(step, [r['prediction_RMS'] for r in trace], 'o-', color=colors[n], label=labels[n])
    ax[2].plot(step, [r['velocity_label_RMS'] for r in trace], '--', color=colors[n], lw=1)
for a, title, ylabel in zip(ax, ['CFM training loss: actual final only', 'Within-bin coupling cost', 'Velocity RMS: solid prediction, dashed target'], ['mean squared velocity error', 'mean standardized squared distance', 'standardized velocity RMS']):
    a.set(title=title, xlabel='updates', ylabel=ylabel); a.grid(alpha=.15)
ax[0].legend(fontsize=8); fig.tight_layout(); save(fig, 'training_flow.png')

views_all = list(m['C2ST']['results']['OT_CFM'])
fig, ax = plt.subplots(figsize=(14, 4))
for j, n in enumerate(names):
    rows = [m['C2ST']['results'][n][v] for v in views_all]
    y = np.array([r['balanced_accuracy'] for r in rows]); ci = np.array([r['heldout_CI']['balanced_accuracy_95CI'] for r in rows]); x = np.arange(len(rows))+(j-1)*.24
    ax.bar(x, y, .23, color=colors[n], label=labels[n]); ax.errorbar(x,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='none',color='black',lw=.6,capsize=2)
short = {'full_historical_shape':'shape493', 'full_voltage':'voltage480', 'prompt_local':'prompt', 'postpeak_local':'postpeak', 'tail_local':'tail', 'baseline_local':'baseline', 'Q_PSD_2D':'Q/PSD', 'normalized_u480':'pure u480', 'seen_reconstruction480':'seen\n480 points', 'complement_reconstruction480':'complement\n480 points'}
ax.set_xticks(np.arange(len(views_all)), [short[v] for v in views_all],fontsize=8)
ax.axhline(.5,color='gray',ls=':'); ax.set(ylim=(.4,1.04),ylabel='holdout balanced accuracy',title='Fixed 600/250 rows per domain; 500 fixed-classifier cluster resamples')
ax.legend(loc='lower right',ncol=3,fontsize=8); fig.tight_layout(); save(fig,'c2st.png')

plotnames = ['REAL', *names]
fig, ax = plt.subplots(1,4,figsize=(14,3.6),sharex=True,sharey=True)
positive = np.concatenate([features[n+'__Qtotal_mVns'][features[n+'__Qtotal_mVns']>0] for n in plotnames]); xmin,xmax=positive.min(),positive.max()
for a,n in zip(ax,plotnames):
    q=features[n+'__Qtotal_mVns']; v=features[n+'__PSD']; ok=features[n+'__valid']&(q>0)
    a.scatter(q[ok],v[ok],s=3,alpha=.28,color=colors[n],rasterized=True);a.set_xscale('log');a.set(xlim=(xmin*.95,xmax*1.05),ylim=(-.1,.6),title=labels[n],xlabel='Q total (mV ns)');a.grid(alpha=.12)
    a.text(.03,.03,f'QC {ok.sum()}; outside y-view {np.sum(ok&((v<-.1)|(v>.6)))}',transform=a.transAxes,fontsize=7)
ax[0].set_ylabel('signed PSD');fig.suptitle('Common axes and complete populations; source-Q flow support does not restrict this plot',y=1.02);fig.tight_layout();save(fig,'charge_psd.png')

sections=load(ROOT/'results/PSD_SECTIONS_PRIVATE.npz')
fig,ax=plt.subplots(2,4,figsize=(13.2,6),sharex=True,sharey=True)
for i,a in enumerate(ax.flat):
    counts=[]
    for n in plotnames:
        x=sections[n+'__bin'+str(i+1)+'__PSD'];counts.append(len(x));a.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=colors[n],label=labels[n],lw=1.1)
    a.set(title=f'Observed-Q bin {i+1}',xlim=(-.1,.6),ylim=(0,1));a.text(.02,.03,'N R/B/I/OT: '+ '/'.join(map(str,counts)),transform=a.transAxes,fontsize=6.9);a.grid(alpha=.12)
    if i not in common_psd:a.text(.03,.92,'N<40: outside common gate',transform=a.transAxes,fontsize=7,color='#666')
for a in ax[1]:a.set_xlabel('signed PSD')
for a in ax[:,0]:a.set_ylabel('empirical CDF')
ax[0,0].legend(fontsize=7);fig.tight_layout();save(fig,'conditional_psd_CDF.png')

fig,ax=plt.subplots(2,3,figsize=(12,6),sharex=True)
qmid=np.exp((edges[:-1]+edges[1:])/2)
for n in plotnames:
    slices=m['energy_conditional_width']['tables'][n]['slices']
    for c in [0,1]:
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            val=[r['components'][c][field] if r['valid'] else np.nan for r in slices];ax[c,j].plot(qmid,val,'o-',color=colors[n],label=labels[n],ms=3)
for c in [0,1]:
    for j,title in enumerate(['component center','raw width','detrended width']):
        a=ax[c,j];a.set_xscale('log');a.set(title=('low'if c==0 else'high')+' PSD '+title,ylabel='PSD units');a.grid(alpha=.12)
        if c:a.set_xlabel('observed-Q midpoint (mV ns)')
ax[0,0].legend(fontsize=7);fig.suptitle('Gaps retain failed/support-limited fits; joint REAL/base/Independent/OT bins = 2,3,6',y=1.02);fig.tight_layout();save(fig,'energy_peak_width.png')

def observed_bin(n):
    q=features[n+'__Qtotal_mVns'];logq=np.log(np.maximum(q,1e-30));b=np.searchsorted(edges,logq,side='right')-1;b[logq==edges[-1]]=7;return b

fig,ax=plt.subplots(2,4,figsize=(13.2,6),sharex=True)
for i,a in enumerate(ax.flat):
    counts=[]
    for n in plotnames:
        use=features[n+'__valid']&(observed_bin(n)==i);v=waves[n][use];counts.append(len(v))
        if len(v):a.plot(tt,np.median(v,axis=0)*1000,color=colors[n],lw=1.1,label=labels[n],ls='-'if len(v)>=40 else':')
    a.set(title=f'Observed-Q bin {i+1}',xlim=(-120,359));a.text(.03,.03,'N R/B/I/OT: '+ '/'.join(map(str,counts)),transform=a.transAxes,fontsize=6.9);a.grid(alpha=.12)
for a in ax[1]:a.set_xlabel('time relative to saved index120 (ns)')
for a in ax[:,0]:a.set_ylabel('median voltage (mV)')
ax[0,0].legend(fontsize=7);fig.suptitle('Full 480 points, no smoothing or separate per-row normalization',y=1.02);fig.tight_layout();save(fig,'conditional_waveforms.png')

raw=load(ROOT/'results/RAW_VOLTAGE_MOMENTS_PRIVATE.npz')
fig,ax=plt.subplots(2,4,figsize=(13.2,6),sharex=True)
for col,(b,title)in enumerate([(None,'population'),(1,'observed Q bin2'),(2,'observed Q bin3'),(5,'observed Q bin6')]):
    for n in plotnames:
        for row,key in enumerate(['mean_V','SD_V']):ax[row,col].plot(tt,raw[n+'__'+str(b)+'__'+key]*1000,color=colors[n],label=labels[n],lw=1.1)
    ax[0,col].set_title(title)
    for a in ax[:,col]:a.grid(alpha=.12);a.set_xlim(-120,359)
    ax[1,col].set_xlabel('time relative to index120 (ns)')
ax[0,0].set_ylabel('mean voltage (mV)');ax[1,0].set_ylabel('pointwise SD (mV; ddof1)');ax[0,0].legend(fontsize=7)
fig.suptitle('Raw voltage moments: population and the three common density-qualified Q bins',y=1.02);fig.tight_layout();save(fig,'raw_mean_SD.png')

fig,ax=plt.subplots(1,4,figsize=(13,3.4))
for a,field,title,factor in zip(ax,['baseline_RMS_V','ACF1','prompt_curvature_RMS_V','postpeak_curvature_RMS_V'],['baseline RMS (mV)','baseline lag1 ACF','prompt100:150 curvature (mV)','postpeak150:220 curvature (mV)'],[1000,1,1000,1000]):
    for n in plotnames:
        x=features[n+'__'+field];x=np.sort(x[np.isfinite(x)])*factor;a.step(x,np.arange(1,len(x)+1)/len(x),color=colors[n],label=labels[n],lw=1.1)
    a.set(title=title,ylabel='empirical CDF');a.grid(alpha=.12)
ax[0].legend(fontsize=7);fig.tight_layout();save(fig,'noise_curvature.png')

fig,ax=plt.subplots(1,3,figsize=(12.5,3.6))
proj=m['projection_distances']['common_support_summary']
for a,group in zip(ax,['shape480','diff479','baseline100_V']):
    ds=[*names,'TRAIN_REFERENCE'];rows=proj[group]['domains'];x=np.arange(4);orig=np.array([rows[n]['equal_mean_on_common_bins']['ORIGINAL']for n in ds]);fresh=np.array([rows[n]['new_block_mean']for n in ds]);rng=np.array([rows[n]['new_block_range']for n in ds])
    a.bar(x-.17,orig,.32,label='original block',color='#aaa');a.bar(x+.17,fresh,.32,label='4 exposed R27 blocks',color='#666');a.errorbar(x+.17,fresh,yerr=np.maximum(0,np.vstack([fresh-rng[:,0],rng[:,1]-fresh])),fmt='none',color='black',capsize=2)
    a.set_xticks(x,[labels[n].replace(' ','\n')for n in ds],fontsize=7);a.set(title=group+'; common bins1–7',ylabel='conditional quantile W1');a.grid(axis='y',alpha=.12)
ax[0].legend(fontsize=7);fig.suptitle('Ranges describe fixed direction blocks, not confidence intervals or new hidden directions',y=1.02);fig.tight_layout();save(fig,'projection_directions.png')

fig,ax=plt.subplots(1,4,figsize=(14,3.8))
for i,n in enumerate(names):
    for b in [3,4]:
        r=primary[n]['bin'+str(b)];h=m['source_composition']['historical_postADC_reference_Q_bins3_4'][n]['bin'+str(b)];pos=i*3+(b-3)
        for a,key in zip(ax[:3],['mean','sample_sd','valley_fraction']):
            actual=r['output_PSD'][key]if key!='valley_fraction'else r[key];hist=h['output_PSD'][key]if key!='valley_fraction'else h[key]
            a.bar(pos-.14,hist,.26,color=colors[n],alpha=.4);a.bar(pos+.14,actual,.26,color=colors[n])
    q=m['source_association_actual_preADC_input'][n]['composition_and_PDG']['all'];ax[3].bar(i-.13,q['source_Q_to_output_Q']['Spearman'],.25,color=colors[n],alpha=.4);ax[3].bar(i+.13,q['source_PSD_to_output_PSD']['Spearman'],.25,color=colors[n])
for a,title in zip(ax[:3],['Pure-had PSD mean','Pure-had sample SD','Pure-had valley fraction']):
    a.set(title=title);a.set_xticks([0,1,3,4,6,7],['B3','B4','I3','I4','OT3','OT4']);a.grid(axis='y',alpha=.12)
ax[3].set(title='Actual source rank: light Q, dark PSD',ylabel='Spearman correlation',ylim=(0,1.05));ax[3].set_xticks(np.arange(3),['Base','Independent','OT']);ax[3].grid(axis='y',alpha=.12)
fig.suptitle('Left: light historical ADC-reference Q (N207/182), dark actual NN-input Q (N213/181); valley ≠ particle truth',y=1.02);fig.tight_layout();save(fig,'source_preservation.png')

fig,ax=plt.subplots(2,3,figsize=(13,7))
comps=['pure_EE','pure_had','mixed']
for c,comp in enumerate(comps):
    for j,n in enumerate(arms):
        rr=m['random_decomposition'][n]['groups']['joint8']['by_composition'][comp]
        covariance=rr['three_additive_algebraic_components_covariance_V2']['full480']
        terms=['base_signal__base_signal','base_electronic__base_electronic','flow_increment__flow_increment','base_signal__base_electronic','base_signal__flow_increment','base_electronic__flow_increment']
        vals=np.array([covariance[k]['mean']for k in terms])*1e6;vals[3:]*=2
        ax[0,c].bar(np.arange(6)+(j-.5)*.35,vals,.34,color=colors[n],label=labels[n])
        p=rr['within_parent_PSD_SD'];levels=np.asarray(p['quantile_levels']);q=np.asarray(p['quantiles']);ax[1,c].plot(q,levels,'o-',color=colors[n],label=labels[n],ms=3)
    ax[0,c].set(title=comp+'; 2×cov cross terms retained',ylabel='mean within-parent contribution (mV²)');ax[0,c].set_xticks(np.arange(6),['base S','base E','flow','2cov\nS,E','2cov\nS,F','2cov\nE,F'],fontsize=7);ax[0,c].axhline(0,color='gray',lw=.6);ax[0,c].grid(axis='y',alpha=.12)
    ax[1,c].set(title=comp+'; 8 repeats per parent',xlabel='within-parent PSD SD',ylabel='saved type7 quantile level');ax[1,c].grid(alpha=.12)
ax[0,0].legend(fontsize=7);ax[1,0].legend(fontsize=7);fig.suptitle('Algebraic decomposition only; final physical signal/electronic split unknown',y=1.02);fig.tight_layout();save(fig,'random_components.png')

files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in sorted(OUT.glob('*.png'))}
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'figures':files,'figure_count':len(files),
    'fit_response_generation_solver_calls':0,'source':'saved R33 output features/statistics; no new random draw or evaluation fit', 'gate_decision':gate['decision'],'retain':gate['retain']}
write('PLOT_SUMMARY_RECEIPT.json',receipt)
print(json.dumps({'receipt':receipt,'gate_passes':[r['pass']for r in [g1,g2,g3,g4,g5]],'common_PSD_mean':common_psd_values,'rank':rank_rows,'valley':valley_rows}),flush=True)
