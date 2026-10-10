import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CATS=[-1,*range(8),8];BLUE='#176b9c';ORANGE='#b56a28';GREY='#333333'
def label(b):return 'All TRAIN' if b is None else 'Below Q1' if b==-1 else 'Above Q8' if b==8 else 'Q'+str(b+1)
def save(fig,root,name):fig.savefig(root/name,dpi=140,bbox_inches='tight');plt.close(fig)
def ecdf(ax,values,weights,name,color):
    order=np.argsort(values,kind='stable');x=values[order];y=np.cumsum(weights[order])/weights.sum();ax.step(np.r_[x[0],x],np.r_[0,y],where='post',label=name,color=color,lw=1.1)

def render(root,result,population,data):
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False})
    keys=data['group_keys'];counts=data['group_parent_count'];b=data['source_bin0'];rb=data['REAL_bin0'];candidate=data.get('x_group_atom');atoms=data['target_PSD'];verified=result['verification']
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained');xx=np.arange(10)
    axes[0].bar(xx-.18,[np.mean(b==c) for c in CATS],.34,label='Original SOURCE',color=ORANGE);axes[0].bar(xx+.18,[np.mean(rb==c) for c in CATS],.34,label='Original REAL',color=GREY)
    axes[0].set_xticks(xx,[label(c) for c in CATS],rotation=35,ha='right');axes[0].set(ylabel='Population mass',title='Original populations retained');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].axis('off');info=['R58: one joint-distribution LP','SOURCE parents: '+str(result['all_source_parents']),'REAL parents: '+str(result['all_REAL_parents']),'SOURCE groups: '+str(len(counts)),'Target PSD atoms: '+str(len(atoms)),'Atom compression W1: %.7f'%result['target_quantization_W1'],'Solver: '+str(result['solver']['status'])+' | '+result['solver']['message'],'','No teacher waveform or neural training.','Finite-family numerical result; not A-F acceptance.']
    axes[1].text(.01,.95,'\n'.join(info),va='top',fontsize=10,wrap=True);fig.suptitle('R58 | Statistical compatibility with all original parents')
    save(fig,root,'population_and_solver.png')
    fig,axes=plt.subplots(3,4,figsize=(17,11),layout='constrained')
    for ax,bin0 in zip(axes.flat,[None,*CATS]):
        ir=np.ones(len(rb),bool) if bin0 is None else rb==bin0;isource=np.ones(len(b),bool) if bin0 is None else b==bin0
        ecdf(ax,data['REAL_PSD'][ir],np.ones(ir.sum()),'Original REAL',GREY);ecdf(ax,data['source_PSD'][isource],np.ones(isource.sum()),'Original SOURCE',ORANGE)
        if candidate is not None and verified['validation']['strict_nonnegative_probability_law']:
            gm=np.ones(len(keys),bool) if bin0 is None else keys[:,0]==bin0;m=candidate[gm].sum(0);ecdf(ax,atoms,m,'Optimized PSD law',BLUE)
        ax.set(title=label(bin0),xlabel='Signed PSD (full observed range)',ylabel='Weighted CDF',ylim=(-.025,1.025));ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7,loc='lower right');axes.flat[-1].axis('off')
    text='All ten SOURCE regions remain.\nReference REAL uses its own original Q region.\nNo generated waveform is shown.'
    if verified is not None and verified['PSD'] is not None:text+='\n\nOverall W1: %.7f\nCommon Q1-Q7 mean W1: %.7f\nNumerical tolerance: 1e-7'%(verified['PSD']['overall_W1'],verified['PSD']['common_Q1_Q7_equal_mean_W1'])
    else:text+='\n\nNo verified probability law available.'
    axes.flat[-1].text(.03,.92,text,va='top',transform=axes.flat[-1].transAxes);fig.suptitle('R58 | Original distributions and the one fixed LP solution')
    save(fig,root,'joint_PSD_CDF.png')
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    if candidate is not None and verified['PSD'] is not None:
        z=candidate/counts[:,None];im=axes[0].imshow(z,aspect='auto',origin='lower',interpolation='nearest',cmap='viridis');fig.colorbar(im,ax=axes[0],label='Conditional probability / atom');axes[0].set(xlabel='PSD atom index (ascending)',ylabel='SOURCE group index',title='Complete conditional law, no row deletion')
        rows=verified['valleys'];q=np.arange(2);axes[1].bar(q-.18,[r['original_fraction'] for r in rows],.34,label='Original SOURCE h=1',color=ORANGE);axes[1].bar(q+.18,[r['candidate_fraction'] for r in rows],.34,label='LP conditional law',color=BLUE)
        for j,r in enumerate(rows):axes[1].plot([j-.4,j+.4],[r['original_fraction']+.02]*2,'k--',label='Original + 0.02' if j==0 else None)
        axes[1].set_xticks(q,['Q'+str(r['bin']) for r in rows]);axes[1].set(ylabel='Valley probability',title='Frozen valley definition');axes[1].legend(fontsize=7);axes[1].grid(alpha=.2);axes[1].margins(y=.2)
        rho=verified['weighted_SOURCE_PSD_rho'];axes[2].bar([0],[rho],color=BLUE,label='R58 TRAIN joint law');axes[2].axhline(result['original_DEV_rho_floor_context'],color=GREY,ls='--',label='Original DEV floor (context only)');axes[2].set_xticks([0],['TRAIN']);axes[2].set(ylabel='Weighted SOURCE/output PSD Spearman',title='Numerical maximum in this finite family',ylim=(min(0.,rho)-.05,max(1.,rho)+.05));axes[2].legend(fontsize=7);axes[2].text(.03,.04,'Different populations; this is not a C-gate pass.',transform=axes[2].transAxes,fontsize=7)
    else:
        for ax in axes:ax.axis('off');ax.text(.05,.8,'No verified joint probability law.\nSolver/verification outcome is retained.\nNo substitute solve or parameter scan.',va='top',transform=ax.transAxes)
    fig.suptitle('R58 | Source association and composition constraints are explicit')
    save(fig,root,'source_association_and_valleys.png')
    fig,axes=plt.subplots(1,2,figsize=(15,5.5),layout='constrained');adc=result['ADC_integral_bounds']
    if adc is not None:
        rows=adc['rows'][1:]
        axes[0].bar(xx-.18,[r['original_input_overrange_parents']/r['source_parents'] for r in rows],.34,color=ORANGE,label='Original SOURCE overrange parent fraction');axes[0].bar(xx+.18,[r['impossible_conditional_probability'] for r in rows],.34,color=BLUE,label='LP mass impossible in observed baseline range')
        axes[0].set_xticks(xx,[label(c) for c in CATS],rotation=35,ha='right');axes[0].set(ylabel='Conditional fraction',title='ADC integral necessary conditions');axes[0].legend(fontsize=7);axes[0].grid(alpha=.2);axes[0].margins(y=.2)
        a=adc['rows'][0];axes[1].axis('off');axes[1].text(.02,.93,'Observed REAL baseline mean range [V]:\n[%.8g, %.8g]\n\nSOURCE Q range: [%.5g, %.5g]\nFree-PSD Q upper bound in this baseline range: %.5g\nAbsolute all-sample ADC Q upper bound: 300\n\nImpossible joint probability mass: %.7f\nFrom originally in-range SOURCE: %.7f\nFrom originally overrange SOURCE: %.7f\n\nPossible integral sums do not prove realistic waveform,\nnoise, peak width, boundary or DEV acceptance.'%(adc['observed_REAL_baseline_mean_min_V'],adc['observed_REAL_baseline_mean_max_V'],adc['source_Q_min'],adc['source_Q_max'],adc['observed_baseline_free_PSD_max_Q'],a['impossible_global_mass'],a['originally_in_range_impossible_global_mass'],a['originally_overrange_impossible_global_mass']),va='top',fontsize=9)
    else:
        for ax in axes:ax.axis('off');ax.text(.03,.8,'No verified probability law available for ADC mass.\nOriginal failed solver state is preserved.',va='top',transform=ax.transAxes)
    fig.suptitle('R58 | ADC bounds are separate TRAIN descriptors, not waveform construction')
    save(fig,root,'ADC_integral_bounds.png')
