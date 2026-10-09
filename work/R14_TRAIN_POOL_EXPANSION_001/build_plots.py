from pathlib import Path
import json,ast
ROOT=Path(__file__).resolve().parent
s=(ROOT.parent/'R13_FACTORIZED_CONDITIONAL_DENSITY_001/plotting.py').read_text(encoding='utf-8')
a=s.index('COLORS=');b=s.index('\ndef plots',a)
s=s[:a]+"COLORS={'REAL':'black','R3B':'#999999','SMALL_GLOBAL':'#80b6d4','SMALL_CONDITIONAL':'#cc79a7','UPDATES_GLOBAL':'#0072b2','UPDATES_CONDITIONAL':'#e69f00','EXPOSURES_GLOBAL':'#009e73','EXPOSURES_CONDITIONAL':'#6c4c9b','REAL_REAL':'#777777'}\n"+s[b:]
s=s.replace(".12*(j-.5)",".08*(j-2.5)")
s=s.replace("['Tail | GMM','Tail | FLOW','Head | GLOBAL','Head | CONDITIONAL'][j]","['U global−small','U cond−small','E global−U','E cond−U','U cond−global','E cond−global'][j]")
s=s.replace("['#e69f00','#009e73','#0072b2','#8b5bb3'][j]","['#0072b2','#e69f00','#009e73','#6c4c9b','#aa5511','#cc79a7'][j]")
a=s.index("    fig,axes=plt.subplots(1,3,");b=s.index("    fig,axes=plt.subplots(2,3,",a)
s=s[:a]+'''    fig,axes=plt.subplots(1,3,figsize=(19,5),layout='constrained')
    for ax,kind in zip(axes[:2],['head','tail']):
        row=m['training'][kind];trace=row['trace'];ax.plot([r['step'] for r in trace],[r['NLL_nat_event'] for r in trace],lw=1.4)
        for budget,color in [('UPDATES','#e69f00'),('EXPOSURES','#009e73')]:
            chosen=next(r for r in trace if r['step']==row['selected_steps'][budget]);ax.scatter([chosen['step']],[chosen['NLL_nat_event']],marker='*' if budget=='UPDATES' else 'D',s=100,color=color,label=budget+' selected'+str(chosen['step']))
        ax.axvline(2000,color='gray',ls='--');ax.set_title(kind+' | N='+str(row['training_N'])+' updates='+str(row['actual_updates']));ax.set_xlabel('Continuous update');ax.set_ylabel('All599 val corresponding NLL[nat/event]');ax.legend(fontsize=7)
    ns=list(m['validation']);axes[2].bar(np.arange(len(ns)),[m['validation'][n]['complete_NLL'] for n in ns],color=[COLORS[n] for n in ns]);axes[2].set_xticks(np.arange(len(ns)),[n.replace('_','\\n') for n in ns],rotation=0);axes[2].set_ylabel('CPUfloat64 full480 val NLL[nat/event]');axes[2].set_title('Frozen selected checkpoints; full Jacobian')
    save(fig,'training_likelihood')
''' + s[b:]
a=s.index('    with np.load(root/');s=s[:a]+'''    with np.load(root/'results/NN_ORIGINAL_PRIVATE.npz',allow_pickle=False) as original,np.load(root/'results/NN_EXPANDED_PRIVATE.npz',allow_pickle=False) as expanded,np.load(__import__('pathlib').Path(cfg['remote_R13_root'])/'results/NN_PRIVATE.npz',allow_pickle=False) as old13:
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        for row,data in enumerate([original,expanded]):
            for col,view in enumerate(['V','own_shape']):
                ax=axes[row,col]
                for n in cfg['models']:
                    cdf(ax,data[n+'__'+view+'__distance'],n)
                if row==0:
                    for n,oldn in [('SMALL_GLOBAL','FLOW_GLOBAL'),('SMALL_CONDITIONAL','FLOW_CONDITIONAL')]:cdf(ax,old13[oldn+'__'+view+'__distance'],n)
                ax.set_xscale('log');ax.set_xlabel('Nearest TRAIN RMS distance '+('[V]' if view=='V' else '[dimensionless]'));ax.set_ylabel('CDF');ax.set_title('Reference original2592' if row==0 else 'Reference expanded'+str(m['data_roles']['expanded_fit_QC']));ax.legend(fontsize=6)
        fig.suptitle('All2000 each new arm; reference-cardinality changes are not memory proof');save(fig,'nearest_neighbours')
'''
ast.parse(s);(ROOT/'plotting.py').write_text(s,encoding='utf-8')
c=json.loads((ROOT/'config.json').read_text());c['contrasts']=[['UPDATES_GLOBAL','SMALL_GLOBAL'],['UPDATES_CONDITIONAL','SMALL_CONDITIONAL'],['EXPOSURES_GLOBAL','UPDATES_GLOBAL'],['EXPOSURES_CONDITIONAL','UPDATES_CONDITIONAL'],['UPDATES_CONDITIONAL','UPDATES_GLOBAL'],['EXPOSURES_CONDITIONAL','EXPOSURES_GLOBAL']];(ROOT/'config.json').write_text(json.dumps(c,indent=2)+'\n')
print('R14 plots AST ready',len(s.encode()))
