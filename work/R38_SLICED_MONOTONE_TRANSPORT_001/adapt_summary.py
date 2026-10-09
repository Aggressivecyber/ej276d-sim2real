from pathlib import Path
import ast,json,os,datetime
ROOT=Path(__file__).resolve().parent
src=(ROOT.parent/'R37_FULL_COVARIANCE_CALIBRATION_001/summarize_and_plot.py').read_text('utf8')
src=src.replace("A='FULL_COVARIANCE_CALIBRATION'","A='SLICED_MONOTONE_TRANSPORT'")
src=src.replace("'CONDITIONAL_DIFFUSION_EDIT']","'CONDITIONAL_DIFFUSION_EDIT','FULL_COVARIANCE_CALIBRATION']",1)
src=src.replace("A:'Full covariance map'","'FULL_COVARIANCE_CALIBRATION':'Gaussian covariance map',A:'Sliced monotone map'")
src=src.replace("A:'#0867ad'","'FULL_COVARIANCE_CALIBRATION':'#ce7d31',A:'#0867ad'")
src=src.replace("vs35={v:cs[A][v]['balanced_accuracy']-cs['PSD_ORDERED_CFM'][v]['balanced_accuracy']for v in full}","vs37={v:cs[A][v]['balanced_accuracy']-cs['FULL_COVARIANCE_CALIBRATION'][v]['balanced_accuracy']for v in full}")
src=src.replace('vs35','vs37').replace('BA_point_delta_minus_R35','BA_point_delta_minus_R37').replace('both_R35_point_delta_le_0_01','both_R37_point_delta_le_0_01')
src=src.replace("['CALIBRATION_vs_BASE']","['SLICED_vs_BASE']")
src=src.replace("'eight_map_checks':stats['all_numeric_checks_pass']","'all256_groups_4096_scalar_maps':stats['all_numeric_checks_pass'] and stats['statistical_fits']==256 and stats['maps']==4096 and stats['direction_gradient_steps']==5120")
src=src.replace("g5['eight_map_checks']","g5['all256_groups_4096_scalar_maps']")
src=src.replace('TRAIN full480 Gaussian covariance calibration after frozen source G','TRAIN 8bin x32layer x16direction sliced monotone transport after frozen source G')
src=src.replace('close_this_fixed_covariance_calibration_recipe','close_this_fixed_sliced_monotone_recipe')
src=src.replace("['BASE','PSD_ORDERED_CFM','CONDITIONAL_DIFFUSION_EDIT',A]","['BASE','PSD_ORDERED_CFM','FULL_COVARIANCE_CALIBRATION',A]")
src=src.replace("['REAL','BASE','PSD_ORDERED_CFM','CONDITIONAL_DIFFUSION_EDIT',A,'SOURCE_INPUT']","['REAL','BASE','PSD_ORDERED_CFM','FULL_COVARIANCE_CALIBRATION',A,'SOURCE_INPUT']")
src=src.replace("['REAL','BASE','PSD_ORDERED_CFM',A]","['REAL','BASE','FULL_COVARIANCE_CALIBRATION',A]")
src=src.replace("['REAL','BASE','CONDITIONAL_DIFFUSION_EDIT',A]","['REAL','BASE','FULL_COVARIANCE_CALIBRATION',A]")
src=src.replace("names=['BASE','PSD_ORDERED_CFM',A]","names=['BASE','FULL_COVARIANCE_CALIBRATION',A]")
src=src.replace("['BASE','ORDER','COV']","['BASE','COV','SLICE']")
src=src.replace('Covariance map minus BASE','Sliced monotone map minus BASE')
start=src.index("fit=load(ROOT/A/'CALIBRATION_MAPS.npz')")
stop=src.index("cov=load(ROOT/'results/DEV_COVARIANCE_DESCRIPTION_PRIVATE.npz'",start)
replacement='''fig,axs=plt.subplots(2,2,figsize=(13,7))
for k in range(8):
    rr=[r for r in stats['layer_summaries']if r['bin_id_1based']==k+1]
    x=np.arange(1,33)
    axs[0,0].semilogy(x,[r['J_initial']for r in rr],lw=.8,ls=':',alpha=.7)
    axs[0,0].semilogy(x,[r['J_final']for r in rr],lw=.9,label=f'bin{k+1}')
    axs[0,1].semilogy(x,[r['full_projected_W2_squared_before']for r in rr],ls=':',lw=.8,alpha=.7)
    axs[0,1].semilogy(x,[r['full_projected_W2_squared_after']for r in rr],lw=.9)
    axs[1,0].semilogy(x,[r['minimum_x_knot_difference']for r in rr],lw=.8,ls=':')
    axs[1,0].semilogy(x,[r['minimum_y_knot_difference']for r in rr],lw=.9)
    axs[1,1].semilogy(x,[r['segment_slope_max']for r in rr],lw=.9)
    axs[1,1].semilogy(x,[r['segment_slope_min']for r in rr],ls=':',lw=.8)
axs[0,0].set(title='20-step direction ascent: dotted initial, solid final',ylabel='subset squared projected W2')
axs[0,0].legend(ncol=4,fontsize=7)
axs[0,1].set(title='Selected directions: dotted before, solid after map',ylabel='full-bin squared projected W2')
axs[1,0].set(title='Strict knot increments: dotted X, solid Y',ylabel='minimum adjacent knot difference')
axs[1,1].set(title='Scalar slopes: dotted minimum, solid maximum',ylabel='segment slope')
for ax in axs.flat:ax.set_xlabel('layer (1..32)');ax.grid(alpha=.12)
fig.suptitle('4096 scalar maps; marginal closure is not complete distribution matching')
fig.tight_layout();save(fig,'training_sliced_maps.png')

'''
src=src[:start]+replacement+src[stop:]
old="rows=m['source_probes'][A]['rows'];tags=list(rows);axs[0,2].bar(range(len(tags)),[rows[k]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for k in tags],color=colors[A]);axs[0,2].set(title='Full source-chain probes',ylabel='RMS to correct reference (mV)')\naxs[0,2].set_xticks(range(len(tags)),[t.replace('_','\\n')for t in tags],fontsize=6.3)"
new="rows=m['source_probes'][A]['rows'];tags=list(rows);axs[0,2].barh(range(len(tags)),[rows[k]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for k in tags],color=colors[A]);axs[0,2].set(title='Full source-chain probes',xlabel='RMS to correct reference (mV)')\nprobe_labels=['Amp x0.5','Amp x1.5','Had fraction0','Had fraction.25','Had fraction.5','Had fraction.75','Had fraction1','NewNoise','Same input/noise','Rolled source']\nassert len(tags)==len(probe_labels)\naxs[0,2].set_yticks(range(len(tags)),probe_labels,fontsize=7);axs[0,2].invert_yaxis()"
assert old in src
src=src.replace(old,new).replace('calibrated physical branch split unknown','mapped physical branch split unknown')
ast.parse(src)
(ROOT/'summarize_and_plot.py').write_text(src,'utf8')
failure={'tool_chunk':'d299c7','PID':'unknown','native_exit':1,'wall_s':0.2239794,'command_error':'retrieve_server.py absent; actual entry is retrieve.py','scientific_execution':False,'new_fit_generation':0,'preserved':True}
(ROOT/'CONTROL_FAILURE_001.json').write_text(json.dumps(failure,indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'summary_AST':True,'bytes':len(src.encode('utf8'))}))
