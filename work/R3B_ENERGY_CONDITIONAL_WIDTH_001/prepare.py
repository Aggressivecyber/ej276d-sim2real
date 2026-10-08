"""Reuse immutable R3 interfaces; freeze the single R3B recipe before data work."""
from pathlib import Path
import json, hashlib, datetime, os, ast

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'R3_FLEXIBLE_TEMPORAL_RESPONSE_001'
start=json.loads((ROOT/'START.json').read_text())
cfg=json.loads((OLD/'config.json').read_text())
cfg.update(start_UTC=start['start_UTC'],deadline_UTC=start['stage_deadline_UTC'],seed=2026100836,
           generation_seeds=[2026100837,2026100839,2026100840],batch_size=256,
           learning_rate=.001,numeric_wall_limit_s=1800,training_wall_s=900,
           R3_checkpoint={'path':str(OLD/'model.pt'),'sha256':'0b97fece53e470eabbfd144eb6d967839d2071b0693a8e17b9fad9e352b44963'},
           R3_generated_input=str(OLD/'generated.npz'),R3_metrics=str(OLD/'metrics.json'),
           R3_fixed_pairs=str(OLD/'results/FIXED_CONDITION_PAIRS.json'))
cfg['model']['initialization']='Bound R3 checkpoint; Torch/NumPy2026100836 before any constructor; initial and all explicit generator states saved.'
cfg['model']['validation']='Same R2/R3 isolated TRAIN roles and384 validation draw2026100821; R3 distribution loss plus fixed validation-only599 conditional CDF/mass; check every100; no DEV.'
cfg['fixed_condition_pairs']={'rule':'Exact R3 preselected source indices reused; no reselection or output selection',
                             'shared_latent_seed':2026100835,'source':cfg['R3_fixed_pairs']}
cfg['conditional_loss']={'centers':'Eight TRAIN-fit logQ5–95% equal-width interval centers',
                         'window':'Gaussian in logQ; sigma=one interval width',
                         'PSD_grid_n':128,'PSD_grid_range':[-.1,.6],'sigmoid_temperature':.0025,
                         'CDF_coefficient':4.,'occupancy_coefficient':1.,'reduction':'mean absolute discrepancy; eight equally weighted windows',
                         'window_mass':'mean unnormalized Gaussian weight; not hard-bin count or a normalized partition',
                         'denominator_min':1e-8,'empty_windows_excluded':False,
                         'target':'Complete2592 fit once; complete599 validation target separately; full overlap retained',
                         'no_width_or_FoM_direct_optimization':True,'all_R3_losses_retained':True}
cfg['energy_diagnostic']={'bins':8,'axis':'log signed Qtotal in mV ns; not measured true energy',
                         'edges':'Only REAL TRAIN fit2592 type7 quantiles5–95%, equal log widths; final edge included',
                         'EM':{'max_updates':500,'parameter_atol':1e-8,'variance_floor':1e-8,
                               'init':'weights0.5/0.5, type7PSDq25/q75, shared full variance',
                               'component_order':'ascending mean each update','no_restarts':True,
                               'min_component_N_times_weight':20,'min_gap_over_pooled_sigma':1.,
                               'floor_or_nonconvergence_width_undefined':True},
                         'trend_removal':'Within each narrow interval, final PSD responsibilities weight linear regression PSD~logQ; population residual RMS; no truth assignments',
                         'bootstrap':{'repeats':200,'seed':2026100838,'unit':'whole observed original_index//50 artificial groups, within each REAL split',
                                      'init':'same fixed initialization at every refit','CI':'type7 percentile2.5–97.5 among qualified replicates; require>=100/200',
                                      'not_independent_physical_runs':True},
                         'global_width_trend':'OLS log(width) versus logQ interval center over>=3 qualified bins; descriptive, no1/sqrt law',
                         'source_views':'R2/R3/mainR3B three existing or prespecified views; within-parent SD separate from density component sigma',
                         'not_CAL_refit_or_particle_truth':True}
cfg['evaluation']['question']='R3B fixed conditional-CDF recipe; complete response and narrow charge-conditioned density widths'
cfg['evaluation'].pop('start_UTC',None);cfg['evaluation'].pop('deadline_UTC',None)
cfg['evaluation'].pop('no_new_response_training',None)
cfg['evaluation']['numeric_wall_limit_s']=1800
cfg['version']='R3B single fixed recipe2026100836'
(ROOT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

for name in ['common_math.py','model.py','replay.py','launch.py','evaluation.py']:
    text=(OLD/name).read_text(encoding='utf8')
    if name=='launch.py':text=text.replace('R3','R3B')
    if name=='evaluation.py':
        text=text.replace("'R3'","'R3B'").replace('C2ST R3 ','C2ST R3B ').replace("f'R3 view","f'R3B view")
        text=text.replace(";axes.ravel()[-1].set_visible(False)",'')
        text=text.replace('plt.subplots(1,4,figsize=(19,5))','plt.subplots(1,5,figsize=(21,5))')
        text=text.replace("enumerate(['SIM','GEN_W0','R2','R3B'])","enumerate(['SIM','GEN_W0','R2','R3','R3B'])")
    (ROOT/name).write_text(text,encoding='utf8')

text=(OLD/'run.py').read_text(encoding='utf8')
def change(a,b):
    global text
    assert a in text,repr(a)
    text=text.replace(a,b)
change('import evaluation as ev','import evaluation as ev\nimport energy_width as ew')
change("ev.NAMES=['SIM','GEN_W0','R2','R3','REAL']","ev.NAMES=['SIM','GEN_W0','R2','R3','R3B','REAL']")
change("'R3':'#e65c24'","'R3':'#bda0cb','R3B':'#e65c24'")
change("r2_wave=z['waveforms_V'];dc=z['G4_condition']", "r2_wave=z['waveforms_V'];dc=z['G4_condition'];r2views=[z[k].copy() for k in ['waveforms_V','independent_view_2_V','independent_view_3_V']]\n    with load_npz(CFG['R3_generated_input']) as z:\n        assert np.array_equal(z['sim_event_ids'],meta['sim_event_ids'])\n        r3_wave=z['waveforms_V'].copy();r3views=[z[k].copy() for k in ['waveforms_V','independent_view_2_V','independent_view_3_V']]")
change("'R2_checkpoint_is_external_input':CFG['R2_checkpoint']", "'R3_checkpoint_is_external_input':CFG['R3_checkpoint']")
change('return meta,historic,r2_wave,real[:2000],qcreason[:2000],groups[:2000],orig[:2000],ref,train,rt,rv,c,q,gt,dc,dq',
       'return meta,historic,r2_wave,r3_wave,r2views,r3views,real[:2000],qcreason[:2000],groups[:2000],orig[:2000],ref,train,tg,rt,rv,c,q,gt,dc,dq')
begin=text.index('    energy=meta[',text.index('def select_condition_pairs'))
end=text.index('\n\n\ndef condition_pair_examples',begin)
text=text[:begin]+"    PAIR_SELECTION=json.loads(pathlib.Path(CFG['R3_fixed_pairs']).read_text())\n    assert [p['indices'] for p in PAIR_SELECTION['pairs']]==[[1245,793],[598,956]]\n    PAIR_SELECTION['reused_R3_selection_without_change']=True\n    write('FIXED_CONDITION_PAIRS.json',PAIR_SELECTION)\n"+text[end:]
change('def train_model(real,rt,rv,c,q,gt):','def train_model(real,rt,rv,c,q,gt,edges):')
change("p=pathlib.Path(CFG['R2_checkpoint']['path']);b=p.read_bytes()", "p=pathlib.Path(CFG['R3_checkpoint']['path']);b=p.read_bytes()")
change("CFG['R2_checkpoint']['sha256'],'R2 weight binding'", "CFG['R3_checkpoint']['sha256'],'R3 weight binding'")
change("missing,unexpected=model.load_state_dict(prior['state_dict'],strict=False)\n    assert not unexpected and all(k.startswith(('temporal_','time_')) for k in missing)",
       "model.load_state_dict(prior['state_dict'],strict=True)")
change("'external_R2_checkpoint_sha256':CFG['R2_checkpoint']['sha256']", "'external_R3_checkpoint_sha256':CFG['R3_checkpoint']['sha256']")
change("    scale=(torch.quantile(rs", "    centers=torch.tensor((edges[:-1]+edges[1:])/2,dtype=torch.float32);width=float(edges[1]-edges[0])\n    target_fit=ew.conditional_targets(tr,centers,width);target_validation=ew.conditional_targets(val,centers,width)\n    write('TARGET_STATISTICS.json',{'fit':{k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in target_fit.items()},'validation':{k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in target_validation.items()},'edges_logQ':edges,'DEV_target_events':0})\n    scale=(torch.quantile(rs")
change('loss,parts=distribution_loss(fake,val[ri],scale,projection)', "loss,parts=distribution_loss(fake,val[ri],scale,projection)\n            extra,cp=ew.conditional_loss(fake,target_validation);loss=loss+extra;parts.update(cp)")
change('total=loss+conditional+regular+temporal_smooth', 'extra,conditional_parts=ew.conditional_loss(fake,target_fit)\n        total=loss+conditional+regular+temporal_smooth+extra')
change("'composition_cov':float(composition_cov.detach()),", "'composition_cov':float(composition_cov.detach()),'training_conditional_parts':{k:float(v.detach()) for k,v in conditional_parts.items()},")

# Replace only the orchestration; shared features/model formula remain intact.
begin=text.index('def main():');end=text.index("if __name__=='__main__':",begin)
text=text[:begin]+'''def main():
    log(f'R3B worker PID={os.getpid()} torch={torch.__version__} CPU_threads={torch.get_num_threads()}')
    meta,historic,r2_wave,r3_wave,r2views,r3views,real,sourceqc,groups,orig,ref,realtrain,tg,rt,rv,c,q,gt,dc,dq=inputs()
    fit_features=ev.features(realtrain[rt]);val_features=ev.features(realtrain[rv])
    assert fit_features['valid'].all() and val_features['valid'].all()
    lim=np.quantile(np.log(fit_features['features']['Qtotal_mVns']),[.05,.95],method='linear')
    edges=np.linspace(lim[0],lim[1],9)
    assert np.all(np.diff(edges)>0)
    write('DIAGNOSTIC_DEFINITION.json',{'config':CFG['energy_diagnostic'],'EM':ew.EM,'edges_logQ':edges,'edges_mVns':np.exp(edges),'derived_only_from_TRAIN_fit2592':True,'frozen_before_any_new_density_fit':True})
    domains={'SIM':ev.features(meta['raw'].astype(float)*meta['normalization_peak_V'][:,None]),'GEN_W0':ev.features(historic.astype(float)*meta['normalization_peak_V'][:,None]),'R2':ev.features(r2_wave),'R3':ev.features(r3_wave),'REAL':ev.features(real)}
    domains['REAL']['valid'] &= sourceqc==0
    diagnostic_domains={'REAL_TRAIN_fit':fit_features,'REAL_TRAIN_validation':val_features,**{k+'_DEV':v for k,v in domains.items()}}
    pre={}
    for name,d in diagnostic_domains.items():
        good=d['valid'];pre[name]=ew.section(d['features']['PSD'][good],d['features']['Qtotal_mVns'][good],edges)
        log(f'Pre-fit charge widths: {name}, qualified slices={sum(r["valid"] for r in pre[name]["slices"])}/8')
    boots={}
    for name,d,g in [('REAL_TRAIN_fit',fit_features,tg[rt]),('REAL_TRAIN_validation',val_features,tg[rv]),('REAL_DEV',domains['REAL'],groups)]:
        good=d['valid'];boots[name]=ew.grouped_bootstrap(d['features']['PSD'][good],d['features']['Qtotal_mVns'][good],g[good],edges)
        log(f'Group bootstrap200 complete {name}')
    write('WIDTH_PRETRAIN.json',pre);write('REAL_GROUP_BOOTSTRAP.json',boots)
    parent_before={'R2':ew.parent_width([ev.features(w) for w in r2views],edges),'R3':ew.parent_width([ev.features(w) for w in r3views],edges)}
    model,training=train_model(realtrain,rt,rv,c,q,gt,edges)
    dc_t=torch.tensor(dc);dq_t=torch.tensor(dq)
    waves=[];params=[];latent=[]
    for seed in CFG['generation_seeds']:
        w,p,z=sample(model,dc_t,dq_t,2000,seed);waves.append(w);params.append(p);latent.append(z)
    assert all(np.all(np.isfinite(w)) for w in waves)
    np.savez_compressed(ROOT/'generated.npz',waveforms_V=waves[0],independent_view_2_V=waves[1],independent_view_3_V=waves[2],sim_event_ids=meta['sim_event_ids'],edep_ee_MeV=meta['edep_ee_MeV'],edep_hadron_MeV=meta['edep_hadron_MeV'],visible_energy_proxy_MeV=meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV'],visible_hadron_fraction=meta['physical_r_h'],G4_condition=dc,latent_z=latent[0],**{'main_'+k:v for k,v in params[0].items()})
    domains['R3B']=ev.features(waves[0])
    for name,d in domains.items():
        d['ids']=meta['real_event_ids'] if name=='REAL' else meta['sim_event_ids'];d['groups']=groups.astype(int) if name=='REAL' else np.arange(2000)
        d['region']=np.searchsorted(ref['CAL_charge_quantile_edges_mV_ns'],d['features']['Qtotal_mVns'],side='right').astype('u1')
    newstats,newdist=ev.characterize({'R3B':domains['R3B'],'REAL':domains['REAL']},ref)
    old=json.loads(pathlib.Path(CFG['R3_metrics']).read_text())
    stats={**{k:old['domains'][k] for k in ['SIM','GEN_W0','R2','R3']},**newstats}
    comparisons={**{k:old['domain_vs_REAL'][k] for k in ['SIM','GEN_W0','R2','R3']},**newdist}
    newc=ev.run_new_c2st(domains['R3B'],domains['REAL'])
    c2st={**old['C2ST'],'R3B_vs_REAL':newc}
    associations={}
    for k,v in [('visible_energy_proxy_MeV',meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV']),('visible_hadron_fraction',meta['physical_r_h'])]:
        associations[k]={f:float(ev.spearmanr(v,domains['R3B']['features'][f]).statistic) for f in ['PSD','Qtotal_mVns','peak_V','rise_10_90_ns']}
    viewfeatures=[ev.features(w) for w in waves]
    random_diag={k:{'median_within_parent_SD':float(np.median(np.std(np.stack([d['features'][k] for d in viewfeatures]),axis=0,ddof=1))),'mean_within_parent_SD':float(np.mean(np.std(np.stack([d['features'][k] for d in viewfeatures]),axis=0,ddof=1)))} for k in ['PSD','Qtotal_mVns','peak_V','rise_10_90_ns','baseline_RMS_V']}
    good=domains['R3B']['valid'];post=ew.section(domains['R3B']['features']['PSD'][good],domains['R3B']['features']['Qtotal_mVns'][good],edges)
    widths={'definition':CFG['energy_diagnostic'],'edges_logQ':edges,'edges_mVns':np.exp(edges),'before':pre,'R3B_DEV':post,'REAL_bootstrap':boots,'fixed_parent_width':{**parent_before,'R3B':ew.parent_width(viewfeatures,edges)}}
    width_errors={}
    for name,data in {**pre,'R3B_DEV':post}.items():
        if name.startswith('REAL_'):continue
        errors=[]
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                pairs=[(r['components'][k][field],s['components'][k][field]) for r,s in zip(data['slices'],pre['REAL_DEV']['slices']) if len(r['components'])==2 and len(s['components'])==2 and r['components'][k]['width_defined'] and s['components'][k]['width_defined'] and r['components'][k][field] is not None and s['components'][k][field] is not None]
                errors.append({'component':k,'field':field,'joint_qualified_bins':len(pairs),'mean_absolute_difference':float(np.mean([abs(a-b) for a,b in pairs])) if pairs else None})
        width_errors[name]=errors
    widths['descriptive_errors_vs_REAL_DEV']=width_errors
    write('ENERGY_WIDTH_METRICS.json',widths)
    conditional=condition_pair_examples(model,dc_t,dq_t,meta)
    metric={'status':'complete','training':training,'domains':stats,'domain_vs_REAL':comparisons,'C2ST':c2st,'REAL_group_reference':old['REAL_group_reference'],'parent_association_R3B':associations,'three_view_within_parent_variation':random_diag,'fixed_latent_condition_examples':conditional,'energy_width_diagnostic':'ENERGY_WIDTH_METRICS.json','energy_width_errors':width_errors,'generation':{'main_seed':CFG['generation_seeds'][0],'extra_views_not_independent_samples':True,'all_same_2000_R1_G4_parents':True,'physical_voltage':'learned TRAIN charge response;not independent sensor calibration','padding_count':int(params[0]['padding'].sum()),'ADC_clip_count':int(params[0]['quantization_clip'].sum())},'CAL_charge_edges_mVns':ref['CAL_charge_quantile_edges_mV_ns'],'CAL_reference_valid':[s['valid'] for s in ref['slots']],'q4_no_fallback':True,'historically_exposed_DEV':True,'current_model_fit_DEV_events':0,'EVAL_read':0,'new_G4':0,'CPU_threads':4,'worker_PID':os.getpid(),'old_metrics_reused_not_refit':True,'Torch_seed_set_before_model_constructor':True,'R3_fixed_checkpoint_input_SHA256':CFG['R3_checkpoint']['sha256'],'elapsed_before_plots_s':time.monotonic()-START}
    write('metrics.json',metric);write('INPUTS.json',INPUTS)
    ev.plots(domains,ref,stats,comparisons,c2st,meta);ev.plot_stochastic_examples(domains,meta,waves)
    colors={'REAL_DEV':'#181818','SIM_DEV':'#2878b5','GEN_W0_DEV':'#a33ea2','R2_DEV':'#bb9b48','R3_DEV':'#bda0cb','R3B_DEV':'#e65c24'}
    ew.plot_widths(pre,post,boots,edges,colors,ev.FIG)
    metric['numeric_elapsed_s']=time.monotonic()-START;write('metrics.json',metric)
    (ROOT/'metrics.json').write_text((OUT/'metrics.json').read_text(),encoding='utf-8')
    log(f'R3B complete wall={metric["numeric_elapsed_s"]:.3f}s updates={training["optimizer_updates"]} selected={training["selected_update"]}')

'''+text[end:]
(ROOT/'run.py').write_text(text,encoding='utf8')

protocol='''# R3B fixed recipe and diagnosis

All data-derived charge boundaries and response loss targets use only REAL TRAIN fit2592. Validation599 has its own conditional CDF/mass target. DEV2000 is diagnostic only; EVAL0. Warm start exact sealed R3 checkpoint, same architecture, all R3 losses retained; one seed2026100836, batch256, lr.001, <=2000 updates. Checkpoints selected every100 on fixed isolated TRAIN validation. Same main2000 parents, seed2026100837, two additional predeclared random views share parents.

Eight equal logQ intervals span type7 TRAIN-fit5–95%; last edge inclusive. All out-of-range/unknown events stay in complete evaluation and are counted. Axis is signed measured charge proxy, not physical incident/deposited energy. Two Gaussian density components are descriptive only. Frozen initial0.5 weights, PSDq25/q75 means and full variance;500 updates, max parameter delta<=1e-8, variance floor1e-8; no restarts. Components ordered by mean every update. Width qualified only if converged, no floor, N*weight>=20, and mean gap / pooled sigma>=1. Raw sigma and FWHM2.35482*sigma; local responsibility-weighted PSD~logQ residual RMS reports de-trended width. Raw data histograms and failed fit parameters stay saved. Eight narrow slices and trend removal distinguish curved-band mixing from random width; they do not establish particle truth.

REAL fit/validation/DEV each receive200 whole artificial-group bootstrap replicates using same fixed initialization, group-original_index//50; percentile95% intervals require>=100 defined replicates, otherwise unknown plus failure counts. Global trend is descriptive OLS log(width) vs logQ center using>=3 qualified intervals. No1/sqrt law or monotonic constraint is imposed. No q4 label repair/global fallback. Additional stochastic views form within-parent PSD SD diagnostics, not independent events.

New loss: Gaussian soft logQ windows at eight centers, width one interval, complete target CDF128 fixed PSDgrid[-.1,.6], sigmoid temperature.0025, each CDF normalized by own Gaussian weight sum(min1e-8); L1 mean equally across eight windows/grids with coefficient4. Occupancy is mean unnormalized Gaussian weight per window, L1 coefficient1; no empty-window exclusion. Both generator PSD and logQ are differentiable common waveform integrals. Fit targets are complete2592 once, validation complete599 separately, outside-grid counts saved. Original unpaired distribution, covariance, composition, temporal regularity losses remain. Not particle-label training/FoM maximization.

CPU4threads, numeric1800s total, stage90min, peak observed RSS4GiB, outputs512MiB. Local supervisor monitors own child and stage/aggregate wall/RSS/output. One main science worker plus a later fresh saved-weight/RNG main2000 replay, no optimizer/evaluator replay. No SSH/G4/EVAL. All files/cache E/D; sealed R3 immutable. R2 unknown historical initialization remains an inherited checkpoint provenance limit; R3B initial and explicit RNG states saved before construction.
'''
(ROOT/'PROTOCOL.md').write_text(protocol,encoding='utf8')
files={}
for p in sorted(ROOT.glob('*.py')):
    b=p.read_bytes();ast.parse(b.decode('utf8'),filename=p.name)
    files[p.name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
receipt={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),
         'AST_pass':True,'source':files,'config_sha256':hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),
         'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
         'numeric_yet':False,'recipe':cfg['version'],'EVAL_read':0,'SSH':0}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt))
