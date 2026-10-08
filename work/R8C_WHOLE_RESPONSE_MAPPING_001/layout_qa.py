from __future__ import annotations
import pathlib,json,hashlib,datetime,os,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'results'/'figures'
OLD=OUT/'QA_original'
OLD.mkdir(exist_ok=False)
A='FULL_WAVE_STOCHASTIC';B='PULSE_STOCHASTIC_OFF'
NS=['REAL','SIM','R3B','R8_JOINT_SIGNAL',A,B]
COLORS={'REAL':'#151515','SIM':'#9a9a9a','R3B':'#dc8c21','R8_JOINT_SIGNAL':'#82ac78',A:'#2871b6',B:'#b33059'}
SHORT={'REAL':'REAL','SIM':'SIM','R3B':'R3B','R8_JOINT_SIGNAL':'R8 SIGNAL',A:'FULL',B:'OFF'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()
metrics=json.loads((ROOT/'metrics.json').read_text(encoding='utf-8'))
original={}
for name in ['training_validation.png','response_features.png']:
    p=OUT/name
    original[name]={'sha256':sha(p),'bytes':p.stat().st_size}
    shutil.copyfile(p,OLD/name)
plt.rcParams.update({'font.size':9})
fig,axes=plt.subplots(1,2,figsize=(14,5.5))
for name in [A,B]:
    rows=metrics['training'][name]['validation_trace']
    steps=[r['step'] for r in rows]
    scores=[r['score'] for r in rows]
    assert min(scores)>0
    axes[0].plot(steps,scores,'o-',ms=3,color=COLORS[name],label=SHORT[name])
    selected=metrics['training'][name]['selected_step']
    selected_score=next(r['score'] for r in rows if r['step']==selected)
    axes[0].scatter([selected],[selected_score],marker='*',s=130,color=COLORS[name],edgecolor='black',linewidth=.4,zorder=5)
    for field,style in [('SW','-'),('CDF','--'),('baseline',':')]:
        values=[r['parts'][field] for r in rows]
        assert min(values)>0
        axes[1].plot(steps,values,style,color=COLORS[name],label=SHORT[name]+' '+field)
spike=next(r for r in metrics['training'][A]['validation_trace'] if r['step']==1900)
axes[0].annotate(f"FULL step 1900: {spike['score']:.4f}",(1900,spike['score']),xytext=(-145,-16),textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'->','linewidth':.6})
axes[0].set_ylabel('TRAIN internal validation score (log scale)')
axes[1].set_ylabel('Unweighted validation term (log scale)')
for ax in axes:
    ax.set_yscale('log');ax.set_xlabel('Common D/G update round');ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.suptitle('TRAIN-only checkpoint selection; stars = selected checkpoints; finite spike retained')
fig.tight_layout();fig.savefig(OUT/'training_validation.png',dpi=140,bbox_inches='tight');plt.close(fig)

fields=['PSD','Qtotal_mVns','peak_V','rise_10_90_ns','tail_charge_fraction','baseline_RMS_V','ACF1','baseline_RMS_over_peak']
fig,axes=plt.subplots(2,4,figsize=(18,9))
with np.load(ROOT/'results'/'PER_EVENT_FEATURES.npz',allow_pickle=False) as saved:
    for field,ax in zip(fields,axes.ravel()):
        data={}
        for name in NS:
            value=saved[name+'__'+field]
            good=saved[name+'__valid']&np.isfinite(value)
            data[name]=value[good]
        all_values=np.concatenate(list(data.values()))
        if not len(all_values):continue
        lo,hi=np.quantile(all_values,[.001,.999])
        bins=np.linspace(lo,hi,60) if hi>lo else np.linspace(lo-1,hi+1,60)
        # Same raw bins/counts as the original figure; only RMS display units change.
        scale=1000 if field=='baseline_RMS_V' else 1
        for name,value in data.items():
            counts,_=np.histogram(value,bins)
            density=counts/max(1,len(value))/np.diff(bins)/scale
            ax.step((bins[:-1]+bins[1:])/2*scale,density,where='mid',color=COLORS[name],label=SHORT[name])
        ax.set_xlabel('Baseline RMS [mV]' if field=='baseline_RMS_V' else field)
        ax.set_ylabel('Density (full valid denominator)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5));ax.tick_params(axis='x',labelsize=8);ax.grid(alpha=.1)
axes[0,0].legend(fontsize=7)
fig.suptitle('Saved full-population feature marginals; display tails clipped only visually')
fig.tight_layout();fig.savefig(OUT/'response_features.png',dpi=140,bbox_inches='tight');plt.close(fig)

self_permutation={}
for name in [A,B]:
    path=ROOT/'arms'/name/'SOURCE_CONTROLS.npz'
    # Open identity/position members only; do not decode any control waveforms.
    with np.load(path,allow_pickle=False) as saved:
        targets=saved['target_ids'];donors=saved['shuffle_donor_ids'];positions=saved['target_position']
        assert len(targets)==len(donors)==len(positions)==64
        same=targets==donors
        assert len(set(targets.tolist()))==64
        assert sorted(targets.tolist())==sorted(donors.tolist())
    self_permutation[name]={
        'identity_self_permutation_n':int(same.sum()),
        'local_control_positions_self_permuted':np.flatnonzero(same).tolist(),
        'DEV_parent_positions_self_permuted':positions[same].tolist(),
        'saved_waveform_unchanged_n':metrics['source_input_controls'][name]['shuffle']['unchanged_n'],
        'distinct_source_waveform_unchanged_n_inferred_from_deterministic_same_bundle_same_noise':metrics['source_input_controls'][name]['shuffle']['unchanged_n']-int(same.sum()),
        'interpretation':'Same identity implies the same full source bundle and noises in the frozen deterministic inference path. A fixed point accounts for an unchanged shuffle row; no new waveform comparison or inference was run.',
        'saved_control_npz_sha256':sha(path),'control_waveform_members_decoded':0
    }
    assert self_permutation[name]['identity_self_permutation_n']==1
    assert self_permutation[name]['saved_waveform_unchanged_n']==1
receipt={
    'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'purpose':'Saved-result layout correction and identity-only permutation clarification',
    'original_figures':original,
    'revised_figures':{n:{'sha256':sha(OUT/n),'bytes':(OUT/n).stat().st_size} for n in original},
    'changes':['Validation axes logarithmic, original finite anomaly retained and annotated; selected TRAIN checkpoints marked.','Saved-feature histograms preserve raw bins/counts; baseline RMS display is mV and x ticks reduced.'],
    'source_self_permutation':self_permutation,
    'saved_metrics_sha256':sha(ROOT/'metrics.json'),
    'saved_feature_metrics_sha256':sha(ROOT/'results'/'PER_EVENT_FEATURES.npz'),
    'new_fit_attempts':0,'response_updates':0,'model_forward':0,'waveform_science_decode':0,'EVAL_read':0,'SSH':0
}
(ROOT/'LAYOUT_QA_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,allow_nan=False))
