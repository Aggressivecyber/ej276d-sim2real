import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CATEGORIES=[-1,*range(8),8]
BLUE='#176b9c';ORANGE='#b56a28';PURPLE='#8046a2';GREY='#333333'

def label(b):
    return 'All TRAIN' if b is None else 'Below Q1' if b==-1 else 'Above Q8' if b==8 else 'Q'+str(b+1)

def ecdf(ax,x,w,name,color,style='-'):
    if not len(x) or not np.isfinite(x).all() or w.sum()<=0:return
    ix=np.argsort(x,kind='stable');xx=x[ix];yy=np.cumsum(w[ix])/w.sum()
    ax.step(np.r_[xx[0],xx],np.r_[0,yy],where='post',label=name,color=color,ls=style,lw=1.1)

def save(fig,root,name):
    fig.savefig(root/name,dpi=140,bbox_inches='tight');plt.close(fig)

def render(root,result,construction,p,examples):
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained');x=np.arange(10)
    for dx,key,col,lab in [(-.18,'global_bin_mass',BLUE,'SOURCE / teacher mass'),(.18,'REAL_original_bin_mass',GREY,'Full equal-parent REAL mass')]:
        axes[0].bar(x+dx,[r[key] for r in construction],width=.34,color=col,label=lab)
    axes[0].set(ylabel='Population probability',title='All original parents retained; population masses differ')
    for dx,key,col,lab in [(-.18,'SOURCE_N',ORANGE,'SOURCE parents'),(.18,'REAL_N',GREY,'REAL parents')]:
        axes[1].bar(x+dx,[r[key] for r in construction],width=.34,color=col,label=lab)
    axes[1].set(ylabel='Parent count',title='6593 SOURCE + 21047 REAL; exact integer margins')
    for ax in axes:ax.set_xticks(x,[label(b) for b in CATEGORIES],rotation=35,ha='right');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('R57 | Equal-parent TRAIN teacher, 16 fixed charge groups per region')
    save(fig,root,'population_and_margins.png')

    w=p['weight'];sw=np.ones(len(p['SOURCE__PSD']));rw=np.ones(len(p['REAL__PSD']))
    cats=[None,*CATEGORIES];fig,axes=plt.subplots(3,4,figsize=(17,11),layout='constrained')
    for ax,b in zip(axes.flat,cats):
        masks={k:np.ones(len(p[k+'__PSD']),bool) if b is None else p[k+'__rawbin']==b for k in ['SOURCE','REAL','REAL_CANONICAL','TEACHER_PRE','TEACHER_CANONICAL']}
        for key,ww,lab,col,style in [('REAL',rw,'REAL raw',GREY,'-'),('REAL_CANONICAL',rw,'REAL canonical','#777777',':'),('SOURCE',sw,'Original SOURCE',ORANGE,'-'),('TEACHER_PRE',w,'Teacher preADC',BLUE,'-'),('TEACHER_CANONICAL',w,'Teacher canonical',PURPLE,'--')]:
            mask=masks[key];ecdf(ax,p[key+'__PSD'][mask],ww[mask],lab,col,style)
        ax.set(title=label(b),xlabel='Signed PSD (full observed range)',ylabel='Weighted CDF',ylim=(-.025,1.025));ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7,loc='lower right');axes.flat[-1].axis('off')
    summary=result['PSD_summary']
    values=tuple(summary[k] for k in ['preADC_overall_W1','canonical_overall_W1','preADC_common_Q1_Q7_mean_W1','canonical_common_Q1_Q7_mean_W1'])
    axes.flat[-1].text(.03,.9,'Each representation uses its own Q bins.\nAll tails and out-of-range parents retained.\n\nOverall PSD W1 vs full REAL:\npreADC = %.6f\ncanonical = %.6f\n\nCommon Q1-Q7 mean W1:\npreADC = %.6f\ncanonical = %.6f'%values,va='top',transform=axes.flat[-1].transAxes)
    fig.suptitle('R57 | Complete weighted distributions; TRAIN descriptors, not A-F acceptance')
    save(fig,root,'TRAIN_PSD_CDF.png')

    fig,axes=plt.subplots(1,2,figsize=(15,5.4),layout='constrained');rows=result['ADC_support']
    axes[0].bar(x-.18,[r['source_input_range_exceeded_parents']/r['source_parents'] for r in rows],.34,label='Original SOURCE parent fraction',color=ORANGE)
    axes[0].bar(x+.18,[r['teacher_range_exceeded_conditional_fraction'] for r in rows],.34,label='Teacher weighted edge fraction',color=BLUE)
    axes[0].set_xticks(x,[label(b) for b in CATEGORIES],rotation=35,ha='right');axes[0].set(ylabel='Fraction outside [-0.5, 0.5] V',title='ADC support by original SOURCE region');axes[0].legend(fontsize=8)
    for j,row in enumerate(rows):
        if row['teacher_range_exceeded_edges']:axes[0].text(j+.18,row['teacher_range_exceeded_conditional_fraction'],str(row['teacher_range_exceeded_edges'])+' edges',fontsize=7,rotation=45,va='bottom',ha='left')
    y=np.arange(2);v=result['valleys']
    for dx,key,col,lab in [(-.23,'original_source_fraction',ORANGE,'Original SOURCE h=1'),(0,'teacher_preADC_fraction',BLUE,'Teacher preADC'),(.23,'teacher_canonical_fraction',PURPLE,'Teacher canonical')]:axes[1].bar(y+dx,[r[key] for r in v],.22,label=lab,color=col)
    for j,row in enumerate(v):axes[1].plot([j-.37,j+.37],[row['original_source_fraction']+row['limit']]*2,'k--',lw=1.1,label='Original + 0.02' if j==0 else None)
    axes[1].set_xticks(y,['Q'+str(r['bin']) for r in v]);axes[1].set(ylabel='Conditional valley probability',title='Frozen REAL TRAIN valley; SOURCE composition proxy h=1');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.margins(y=.22)
    fig.suptitle('R57 | Predeclared TRAIN constraints; no failed mass removed')
    save(fig,root,'ADC_and_valley_conditions.png')

    for kind in ['V','U']:
        fig,axes=plt.subplots(3,4,figsize=(17,11),layout='constrained')
        for ax,b in zip(axes.flat,cats):
            j=10 if b is None else CATEGORIES.index(b)
            for key,col,lab in [('SOURCE',ORANGE,'SOURCE'),('REAL',GREY,'REAL'),('TEACHER',BLUE,'Teacher')]:
                mu=p[key+'__wave_'+kind+'_mean'][j];sd=p[key+'__wave_'+kind+'_SD'][j];xx=np.arange(480)
                ax.fill_between(xx,mu-sd,mu+sd,color=col,alpha=.09);ax.plot(xx,mu,color=col,label=lab,lw=1)
            ax.set(title=label(b),xlabel='Sample',ylabel='Voltage [V]' if kind=='V' else 'Peak-normalized waveform')
            for cut in [100,150,400]:ax.axvline(cut,color='#999999',lw=.5,ls=':')
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7);axes.flat[-1].axis('off')
        axes.flat[-1].text(.03,.9,'Lines: full 480-sample weighted means.\nBands: mean +/- population SD.\n\nTeacher groups follow original SOURCE Q.\nREAL groups follow original REAL Q.\nOriginal SOURCE parents are equal-weight.\nFull observed ranges remain visible.\n\nThese moments are descriptive;\nthey do not establish response equivalence.',va='top',transform=axes.flat[-1].transAxes)
        fig.suptitle('R57 | Complete TRAIN '+('voltage' if kind=='V' else 'normalized shape')+' moments')
        save(fig,root,'TRAIN_full_waveform_'+kind+'.png')

    fig,axes=plt.subplots(2,5,figsize=(18,7),layout='constrained')
    for ax,b in zip(axes.flat,CATEGORIES):
        e=examples['bin_'+str(b)]
        for key,col,lab in [('source',ORANGE,'SOURCE'),('real',GREY,'REAL'),('teacher',BLUE,'Teacher')]:ax.plot(e[key],color=col,label=lab,lw=.95)
        for cut in [100,150,400]:ax.axvline(cut,color='#999999',lw=.5,ls=':')
        ax.set(title=label(b)+' | fixed edge '+str(e['edge']),xlabel='Sample',ylabel='Voltage [V]');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7);fig.suptitle('R57 | First constructed edge in every region; no visual example selection')
    save(fig,root,'fixed_teacher_examples.png')
