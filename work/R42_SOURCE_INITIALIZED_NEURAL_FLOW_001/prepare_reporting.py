from pathlib import Path
import ast,json,datetime,os
R=Path(__file__).resolve().parent
P=R.parent/'R41_PATCH_CONTENT_ABLATION_001'
s=(P/'summarize_and_plot.py').read_text('utf8')
s=s.replace('NO_PATCH_NCE_FINAL','SOURCE_NEURAL_FLOW_FINAL').replace('NO_PATCH_NCE_REAL_CONTROL','SOURCE_NEURAL_FLOW_REAL_CONTROL')
s=s.replace('No-PatchNCE G on REAL','Source neural flow on REAL').replace('No-PatchNCE G','Source neural flow').replace('NO-NCE','FLOW')
s=s.replace("['BASE','COV','TIME']","['BASE','COV','FLOW']")
start=s.index("record=json.loads((ROOT/A/'TRAINING_RECEIPT.json')")
end=s.index("limits={'full_historical_shape'",start)
s=s[:start]+'''record=json.loads((ROOT/A/'TRAINING_RECEIPT.json').read_text('utf8'))
flow=json.loads((ROOT/'MATCHED_TRAINING_FLOW_RECEIPT.json').read_text('utf8'))
solver=json.loads((ROOT/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))
stats={k:record[k]for k in ['updates','actual_rows','training_s','selection','stream_hashes','original_7_audit_streams_match','actual_unique_source_REAL','loaded_source_REAL','marginal_preservation_checks','max_path_q_residual','max_field_q_residual','parameter_count','GPU_peak_allocated_bytes','network_training_forward_rows','artificial_zero_check_rows','old_G_D_reader_forwards_updates']}
stats.update(new_model_no_pretrained_weights=True,white_used_z_style_NCE_unused=True,
    objective='only full480 tangent velocity MSE; field=float64 projection of float32 network',
    fixed_final2950_only=True,no_J20_D_reader_fresh_or_PatchNCE_participation=True,
    numerical_TRAIN_check=solver,original_flow_match=flow)
write('TRAINING_COMPACT.json',stats)

''' +s[end:]
s=s.replace("'all_outside_rows_use_new_G':gen['BASE_fallback_N']==0 and gen['G_applied_N']==2000", "'all_outside_rows_use_new_flow':gen['BASE_fallback_N']==0 and gen['standalone_flow_applied_N']==2000")
s=s.replace("g3['all_outside_rows_use_new_G']", "g3['all_outside_rows_use_new_flow']")
s=s.replace("'pre_registered_R41_same_treated_scope'", "'pre_registered_R42_same_treated_scope'")
s=s.replace("'comparison':'new G REAL control minus BASE G REAL control'", "'comparison':'new SOURCE flow on REAL minus frozen BASE G on REAL'")
start=s.index("g6={'unique_G_D_reader_trajectory_each_2950'")
end=s.index("gates={'gate1_complete_response'",start)
s=s[:start]+'''charge={}
for name in [A,'SOURCE_NEURAL_FLOW_REAL_CONTROL']:
    d=load(ROOT/name/'generated.npz',keys=['preADC_charge_relative_error_f64','preADC_charge_relative_error_f32','source_q_Vns','waveforms_V','neural_flow_applied_all_rows','flow_applied'])
    raw=d['waveforms_V'].astype('f8');postq=raw[:,100:400].sum(1)-3*raw[:,:100].sum(1)
    charge[name]={'rows':len(raw),'f64_max_relative_error':float(np.max(d['preADC_charge_relative_error_f64'])),
        'actual_f32_preADC_max_relative_error':float(np.max(d['preADC_charge_relative_error_f32'])),
        'postADC_relative_charge_error':ev.summary((postq-d['source_q_Vns'])/d['source_q_Vns']),
        'postADC_not_claimed_exact':True,'all_rows_neural_flow':bool(d['neural_flow_applied_all_rows'].all()),
        'flow_applied_compatibility_field_is_TRAIN_support_only':True,'inside_TRAIN_support_rows':int(d['flow_applied'].sum())}
write('CHARGE_CONSERVATION_SUMMARY.json',charge)
g6={'unique_new_velocity_model_2950':stats['updates']==2950,'no_old_response_initial_weights':stats['new_model_no_pretrained_weights'],
    'no_old_G_D_reader_forward_update':stats['old_G_D_reader_forwards_updates']==0,
    'all_seven_original_audit_streams_match':all(flow['same_7_audit_streams'].values()),
    'original_white_and_t_preserved':flow['original_t_SHA_match'] and all(stats['original_7_audit_streams_match'].values()),
    'all2950_source_target_marginals_preserved':flow['marginals_preserved_each_update'] and stats['marginal_preservation_checks']==2950,
    'TRAIN_fixed32_vs64_pass':solver['pass'],'TRAIN_no_DEV_read_or_generated':solver['DEV_read_or_generated']==0,
    'all_preADC_f32_charge_error_le1e_5':all(v['actual_f32_preADC_max_relative_error']<=1e-5 for v in charge.values()),
    'all_main_rows_processed':all(v['all_rows_neural_flow']for v in charge.values()),
    'charge_error_summaries':charge,'independent_CPU_full_chain_replay_rows':replay['rows'],'bitwise_pass':replay['pass'],
    'not_applicable':['old G initialization equivalence','J20','D/reader updates','PatchNCE','fresh projections','style use'],
    'formula_gradient_preservation_status':'static differentiable construction; no separate executed autograd test',
    'actual_new_figure_QA_required':'FIGURE_QA.json','actual_resource_and_failures_required':'RESOURCE_RECEIPT.json'}
g6['pass']=all(g6[k]for k in ['unique_new_velocity_model_2950','no_old_response_initial_weights','no_old_G_D_reader_forward_update',
    'all_seven_original_audit_streams_match','original_white_and_t_preserved','all2950_source_target_marginals_preserved',
    'TRAIN_fixed32_vs64_pass','TRAIN_no_DEV_read_or_generated','all_preADC_f32_charge_error_le1e_5','all_main_rows_processed','bitwise_pass']) and replay['rows']==4000

''' +s[end:]
s=s.replace("'Original complete CNN, PatchNCE actual computation/objective removed only, source G final2950'", "'Standalone source-initialized unit-charge convolutional neural flow, fixed final2950/32Heun'")
s=s.replace('close_this_fixed_no_PatchNCE_recipe','close_this_fixed_source_initialized_neural_flow_recipe')
start=s.index("trace=json.loads((ROOT/A/'TRAINING_TRACE.json')")
end=s.index("cov=load(ROOT/'results/DEV_COVARIANCE_DESCRIPTION_PRIVATE.npz'",start)
s=s[:start]+'''trace=json.loads((ROOT/A/'TRAINING_TRACE.json').read_text('utf8'))
xx=[r['step']for r in trace]
dense=load(ROOT/A/'LOSS_BY_UPDATE_PRIVATE.npz')['values']
fig,axs=plt.subplots(2,2,figsize=(13,7.2))
axs[0,0].plot(np.arange(1,len(dense)+1),dense[:,0],lw=.55,color=colors[A],alpha=.65,label='actual per-update velocity MSE')
axs[0,0].plot(xx,[r['velocity_MSE']for r in trace],'o',ms=3,color=colors[A]);axs[0,0].set(title='Only full480 tangent velocity regression',xlabel='update',ylabel='MSE');axs[0,0].legend(fontsize=7)
for k,tag in [(1,'target'),(2,'prediction')]:axs[0,1].plot(np.arange(1,len(dense)+1),dense[:,k],lw=.55,label=tag)
axs[0,1].set(title='Standardized velocity RMS',xlabel='update',ylabel='RMS');axs[0,1].legend(fontsize=7)
axs[1,0].plot(np.arange(1,len(dense)+1),dense[:,3],lw=.55,color=colors[A]);axs[1,0].axhline(5.,ls=':',color='gray');axs[1,0].set(title='Global gradient norm before clip5',xlabel='update',ylabel='L2 norm')
for k,tag in [(4,'target path'),(5,'actual field')]:axs[1,1].plot(np.arange(1,len(dense)+1),dense[:,k],lw=.55,label=tag)
axs[1,1].set(title='Unit-charge tangent residual',xlabel='update',ylabel='absolute q residual');axs[1,1].legend(fontsize=7)
for ax in axs.flat:ax.grid(alpha=.12)
fig.suptitle('New 1,371,841-parameter model: no old G/D/reader weights, sole final2950')
fig.tight_layout();save(fig,'training_source_neural_flow.png')

num=load(ROOT/'TRAIN_SOLVER_CHECK_PRIVATE.npz',keys=['RMS_difference_V'])
fig,axs=plt.subplots(1,3,figsize=(13.4,4.2))
err=np.sort(num['RMS_difference_V']*16383.)
axs[0].plot(err,np.arange(1,len(err)+1)/len(err),color=colors[A]);axs[0].axvline(.25,color='gray',ls=':');axs[0].axvline(1.,color='gray',ls='--');axs[0].set(title='TRAIN128:32 vs64 steps only',xlabel='RMS in ADC LSB',ylabel='CDF')
for name in [A,'SOURCE_NEURAL_FLOW_REAL_CONTROL']:
    d=load(ROOT/name/'generated.npz',keys=['preADC_charge_relative_error_f32','source_q_Vns','waveforms_V'])
    q=d['source_q_Vns'];w=d['waveforms_V'].astype('f8');postq=w[:,100:400].sum(1)-3*w[:,:100].sum(1)
    axs[1].scatter(1000*q,d['preADC_charge_relative_error_f32'],s=3,alpha=.3,color=colors[name],label=labels[name])
    axs[2].scatter(1000*q,(postq-q)/q,s=3,alpha=.3,color=colors[name],label=labels[name])
for ax in axs[1:]:ax.set_xscale('log');ax.set_xlabel('input Q (mV ns)')
axs[1].set(title='Actual float32 preADC',ylabel='relative charge error');axs[2].set(title='After one14-bitADC: not exact',ylabel='signed relative charge error')
axs[1].legend(fontsize=7)
for ax in axs:ax.grid(alpha=.12)
fig.tight_layout();save(fig,'solver_and_charge.png')

# Deterministic source-side-only example selection; REAL examples are illustrative, unpaired.
actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz',keys=['source_waveforms_V','source_ids']);meta=load(ROOT/'DEV_SOURCE_METADATA.npz',keys=['composition'])
source=actual['source_waveforms_V'];sq=source.astype('f8')[:,100:400].sum(1)-3*source.astype('f8')[:,:100].sum(1)
examples=[];fig,axs=plt.subplots(3,3,figsize=(13.8,9))
for row,cat in enumerate(['pure_EE','pure_had','mixed']):
    candidates=np.flatnonzero(meta['composition']==cat);idx=int(candidates[np.argmin(np.abs(sq[candidates]-np.median(sq[candidates])))])
    ruse=np.flatnonzero(features['REAL__valid']);ridx=int(ruse[np.argmin(np.abs(features['REAL__Qtotal_mVns'][ruse]-1000*sq[idx]))])
    vals=[source[idx],waves[A][idx],waves['REAL'][ridx]]
    lo=min(float(v.min())for v in vals)*1000;hi=max(float(v.max())for v in vals)*1000;pad=max((hi-lo)*.05,1.)
    for j,(v,title,color)in enumerate(zip(vals,['Full SOURCE input','Neural-flow output','Unpaired REAL example'],['#6e839e',colors[A],colors['REAL']])):
        axs[row,j].plot(tt,v*1000,color=color,lw=.9);axs[row,j].set(title=cat+' | '+title,xlim=(-120,359),ylim=(lo-pad,hi+pad));axs[row,j].grid(alpha=.12)
        if row==2:axs[row,j].set_xlabel('relative to source saved index120 (ns)')
        if j==0:axs[row,j].set_ylabel('raw voltage (mV)')
    examples.append({'composition':cat,'source_parent_index':idx,'source_id':str(actual['source_ids'][idx]),'source_Q_mVns':float(1000*sq[idx]),'REAL_unpaired_example_index':ridx})
fig.suptitle('Source-Q median selection; common row axes; no smoothing, normalization or true pairing')
fig.tight_layout();save(fig,'source_output_REAL_examples.png');write('EXAMPLE_SELECTION_PRIVATE.json',{'selection':'closest source-Q to within-composition median, lower input index wins ties; REAL closest rawQ illustrative only','rows':examples})

''' +s[end:]
s=s.replace("'baseS','baseE','round','delta','all2cov'", "'SOURCE','initGauss','round','flow','all2cov'")
s=s.replace('Full-SOURCE sensitivity and functional G components; not physical branch truth','Full SOURCE sensitivity and functional flow components; no physical branch truth')
s=s.replace('REAL may receive added stochastic response','REAL also passes through source-initialized flow')
s=s.replace('New minus BASE same-treated REAL','Neural flow minus BASE on same REAL')
ast.parse(s)
(R/'summarize_and_plot.py').write_text(s,'utf8')
mapping={
 'frozen_remote_training_sources_preserved':True,
 'flow_applied':'compatibility field = inside original TRAIN Q support; NOT whether flow was executed',
 'neural_flow_applied_all_rows':'authoritative execution flag; true for all rows including417 support-outside SOURCE',
 'white_prefix':'TRAIN32/64 uses first128 consumed-row white via ordinal[:128]; original noise-generation ordinal saved',
 'formula_check_field_projection_preserves_gradients':'static differentiable Torch construction; not a separately executed autograd verification',
 'functional_components':{'base_signal':'actual SOURCE','base_electronic':'projected1mV initial Gaussian, not electronics',
 'base_sum_rounding':'float32 initial sum residue','flow_increment_f64':'actual f32 final minus f32 initial, functional only',
 'learned_flow_increment_f64_V':'float64 ODE change; depends on initial random perturbation',
 'final_cast_rounding_f64_V':'final float32 minus float64 ODE output'},
 'physical_signal_electronic_independence':False}
(R/'SEMANTIC_FIELD_MAP.json').write_text(json.dumps(mapping,indent=2)+'\n','utf8')
p=R/'generate_all.py';g=p.read_text('utf8')
g=g.replace("'standalone_flow_applied_N':2000,", "'standalone_flow_applied_N':2000,'G_applied_N':2000,'G_applied_N_compatibility_means_new_standalone_flow':True,")
g=g.replace("'TRAIN_support_outside_N':int((~support).sum()),", "'TRAIN_support_outside_N':int((~support).sum()),'flow_applied_field_is_TRAIN_support_only':True,")
g=g.replace("'unused_random_inputs':['z','style']", "'unused_random_inputs':['z','style'],'field_forward_rows_main_source_plus_auxiliary':pipeline.total_field_rows,'field_forward_rows_REAL_control':ctrl.total_field_rows")
g=g.replace("'probe_kind':'fullSOURCE_input_to_G not postG guide'", "'probe_kind':'fullSOURCE_input_to_standalone_flow not postG guide'")
ast.parse(g);p.write_text(g,'utf8')
p=R/'flow_diagnostics.py';d=p.read_text('utf8')
d=d.replace("'flow_applied_N':int(detail['flow_applied'][use].sum())", "'inside_TRAIN_support_N':int(detail['flow_applied'][use].sum()),'flow_applied_N_compatibility_is_support_only':True,'neural_flow_applied_N':int(detail['neural_flow_applied_all_rows'][use].sum())")
ast.parse(d);p.write_text(d,'utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'AST':True,'new_scientific_computation':0,'reporting_entry':str(R/'summarize_and_plot.py')}))
