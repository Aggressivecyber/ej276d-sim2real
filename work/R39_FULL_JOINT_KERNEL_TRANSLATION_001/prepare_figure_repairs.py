from __future__ import annotations
import pathlib, ast, hashlib, json
ROOT=pathlib.Path(__file__).resolve().parent
s=(ROOT/'summarize_and_plot_recover.py').read_text('utf8')
prefix=s[:s.index('paired=m[')]
prefix += "paired=m['C2ST']['paired'][A];cs=m['C2ST']['results']\nstats=json.loads((ROOT/'TRAINING_COMPACT.json').read_text('utf8'));cal=json.loads((ROOT/'INITIAL_GRADIENT_SCALE_COMPACT.json').read_text('utf8'))\ncontrol_pair=json.loads((ROOT/'SAME_TREATED_REAL_CONTROL_PAIRED.json').read_text('utf8'))\nplt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})\n"
c2=s[s.index("short=['shape493'"):s.index("chartnames=['REAL'")]
c2=c2.replace("figsize=(14,7.6)","figsize=(14,8.3)")
c2=c2.replace("for ax in axs:ax.set_xticks(range(10),short);ax.legend(fontsize=8,ncol=3);ax.grid(axis='y',alpha=.12)","for ax in axs:ax.set_xticks(range(10),short);ax.grid(axis='y',alpha=.12)\naxs[0].legend(fontsize=8,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.12))\naxs[1].legend(fontsize=8,loc='upper left')")
tr=s[s.index("trace=json.loads"):s.index("cov=load",s.index("trace=json.loads"))]
tr=tr.replace("x=[0]+[r['step']for r in trace];y=[stats['step0_J20']]+[r['validation']['J']for r in trace]", "trace=[r for r in trace if 'step'in r]\nx=[0]+[r['step']for r in trace]+[2950];y=[stats['step0_J20']]+[r['validation']['J']for r in trace]+[stats['final_J20']]")
tr=tr.replace("axs[0,0].set(title=", "axs[0,0].annotate(f'final2950 J={stats[\"final_J20\"]:.5f}',(2950,stats['final_J20']),xytext=(-90,-20),textcoords='offset points',fontsize=7,arrowprops={'arrowstyle':'-','color':'gray'})\naxs[0,0].set(title=")
co=s[s.index("fig,axs=plt.subplots(2,2,figsize=(13,7.4))"):s.index('files={p.name:')]
co=co.replace("shortcv=['shape493','pure u480','baseline','V480']", "shortcv=['shape493','pure u480','baseline','V480\\n(descriptive)']")
co=co.replace("axs[0,1].axhline(.01,ls=':',color='red',label='point limit +.01');axs[0,1].axhline(.10,ls='--',color='gray',label='95% upper limit +.10');", "axs[0,1].hlines(.01,-.4,2.4,linestyles=':',color='red',label='first3: point limit +.01');axs[0,1].hlines(.10,-.4,2.4,linestyles='--',color='gray',label='first3: 95% upper +.10');axs[0,1].axvspan(2.6,3.4,color='#eeeeee',alpha=.4);")
co=co.replace("fig.suptitle('REAL controls are processing comparisons, not a stochastic identity test')", "fig.suptitle('REAL processing controls; gates apply to shape/u/baseline only, V480 is descriptive')")
backup="""
old_dir=ROOT/'results/figures/layout_initial';old_dir.mkdir(exist_ok=True)
for n in ['c2st.png','training_joint_kernel.png','same_treated_REAL_controls.png']:
    p=OUT/n;q=old_dir/n
    if q.exists():raise RuntimeError('repair already attempted')
    q.write_bytes(p.read_bytes())
"""
ending="""
names=['c2st.png','training_joint_kernel.png','same_treated_REAL_controls.png']
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
    'new_fits_generation_updates':0,'only_display_repair':True,'saved_old_versions':'results/figures/layout_initial',
    'figures':{n:{'bytes':(OUT/n).stat().st_size,'sha256':hashlib.sha256((OUT/n).read_bytes()).hexdigest()}for n in names},
    'J2950_from_saved_TRAINING_COMPACT':stats['final_J20'],'V480_descriptive_only':True,'actual_visual_QA_pending':True}
write('FIGURE_REPAIR_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
"""
result=prefix+backup+c2+tr+co+ending
ast.parse(result)
(ROOT/'fix_final_figures.py').write_text(result,'utf8')
print(json.dumps({'source':'fix_final_figures.py','AST':True,'sha256':hashlib.sha256(result.encode()).hexdigest()}))
