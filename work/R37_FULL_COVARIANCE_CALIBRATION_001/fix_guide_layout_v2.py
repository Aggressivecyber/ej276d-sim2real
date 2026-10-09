from pathlib import Path
import ast,os,json,shutil,datetime,time,hashlib
R=Path(__file__).resolve().parent;START=time.monotonic()
old=R/'results/figures/guide_and_random_decomposition.png';history=old.parent/'layout_initial';history.mkdir(exist_ok=True)
shutil.copyfile(old,history/'guide_and_random_decomposition_short_v1.png')
source=(R/'summarize_and_plot.py').read_text('utf8')
start=source.index("gp=load(ROOT/'results/GUIDE_PERTURBATION_PRIVATE.npz')");end=source.index('files={',start)
block=source[start:end];p0=block.index("rows=m['source_probes']");p1=block.index("terms=['base_signal'",p0)
replacement="""rows=m['source_probes'][A]['rows'];tags=list(rows)
names={'amplitude_x0.5':'Amp x0.5','amplitude_x1.5':'Amp x1.5','analytic_fast_slow_0':'Fast/slow 0','analytic_fast_slow_0.25':'Fast/slow .25','analytic_fast_slow_0.5':'Fast/slow .5','analytic_fast_slow_0.75':'Fast/slow .75','analytic_fast_slow_1':'Fast/slow 1','fixed_source_changed_noise':'Fixed source, new noise','same_input_same_noise':'Same input, same noise','source_roll_no_fixed_points':'Rolled source, fixed noise'}
axs[0,2].barh(range(len(tags)),[rows[k]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for k in tags],color=colors[A])
axs[0,2].set(title='Full source-chain probes',xlabel='RMS to correct reference (mV)')
axs[0,2].set_yticks(range(len(tags)),[names[k]for k in tags],fontsize=6.8);axs[0,2].invert_yaxis()
"""
block=block[:p0]+replacement+block[p1:]
prefix="""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=R;OUT=R/'results/figures';A='FULL_COVARIANCE_CALIBRATION';colors={A:'#0867ad'}
m=json.loads((R/'metrics.json').read_text('utf8'));tt=np.arange(480)-120
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def save(fig,n):fig.savefig(OUT/n,bbox_inches='tight');plt.close(fig)
"""
code=prefix+block;ast.parse(code);exec(compile(code,'saved_results_guide_layout_only_v2','exec'))
row={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
    'only_probe_display_horizontal_bars':True,'prior_plot':str(history/'guide_and_random_decomposition_short_v1.png'),
    'new_HGB_EM_generation':0,'bytes':old.stat().st_size,'sha256':hashlib.sha256(old.read_bytes()).hexdigest()}
(R/'GUIDE_LAYOUT_REPAIR_V2_RECEIPT.json').write_text(json.dumps(row,indent=2)+'\n','utf8');print(json.dumps(row),flush=True)
