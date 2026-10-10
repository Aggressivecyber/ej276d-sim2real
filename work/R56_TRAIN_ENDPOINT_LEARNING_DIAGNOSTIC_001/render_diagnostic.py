import numpy as np

def render(root,coverage,path,markers,z):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    rows=coverage['by_raw_bin'];x=np.arange(10);n=np.array([a['SOURCE_parents'] for a in rows]);seen=np.array([a['SOURCE_seen'] for a in rows])
    axs[0].bar(x,seen,label='Seen at least once',color='#166a95');axs[0].bar(x,n-seen,bottom=seen,label='Never sampled',color='#d28b3b')
    axs[0].set(xticks=x,xticklabels=['below',*[str(j) for j in range(1,9)],'above'],xlabel='Original charge bin',ylabel='Unique SOURCE parents',title='All 6593 TRAIN source parents retained');axs[0].legend()
    axs[1].bar(x,[a['SOURCE_occurrences'] for a in rows],color='#166a95');axs[1].set(xticks=x,xticklabels=['below',*[str(j) for j in range(1,9)],'above'],xlabel='Original charge bin',ylabel='Actual training occurrences',title='Fixed 1.6 million exposures; repeats are not independent')
    fig.savefig(root/'coverage.png',dpi=160);plt.close(fig)
    names=['REAL','full_cycle_TEACHER','R54_PRE','SOURCE'];labels=['All legal REAL TRAIN','Frozen teacher exposures','Frozen R54 (all source parents)','Original source'];colors=['#222222','#d28b3b','#176b9c','#929292']
    fig,axs=plt.subplots(2,4,figsize=(15,8))
    for b,ax in enumerate(axs.flat):
        for name,label,col in zip(names,labels,colors):
            bins=z[name+'__rawbin'];v=z[name+'__PSD'][bins==b];v=np.sort(v[np.isfinite(v)])
            ax.step(v,np.arange(1,len(v)+1)/max(len(v),1),where='post',label=label,color=col,lw=1)
        ax.set(title='Q'+str(b+1),xlabel='Signed PSD',ylabel='Empirical CDF')
    handles,labs=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labs,loc='upper center',bbox_to_anchor=(.5,.965),ncol=2)
    fig.suptitle('TRAIN populations differ in parent weights; own preADC charge bins',fontsize=12,y=.998);fig.tight_layout(rect=(0,0,1,.90))
    fig.savefig(root/'TRAIN_PSD_CDF.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,key,title in zip(axs.flat,['B_signed_mean','PSD_RMSE','MSE480','PSD_mode_energy_fraction'],['Signed PSD velocity residual','PSD velocity RMSE','Original 480-coordinate MSE','PSD tangent share of residual energy']):
        a=np.array([r.get(key,np.nan) for r in path['SOURCE']['by_Q_time']]).reshape(8,4)
        lim=float(np.max(np.abs(a)))
        im=ax.imshow(a,origin='lower',aspect='auto',cmap='coolwarm' if key=='B_signed_mean' else 'viridis',vmin=-lim if key=='B_signed_mean' else 0,vmax=lim)
        ax.set(xticks=range(4),xticklabels=['0-.25','.25-.50','.50-.75','.75-1'],yticks=range(8),yticklabels=range(1,9),xlabel='TRAIN interpolation time',ylabel='Original Q bin',title=title)
        for b in range(8):
            for t in range(4):ax.text(t,b,f'{a[b,t]:.3g}',ha='center',va='center',fontsize=7,color='white' if abs(a[b,t])>.55*lim else 'black')
        fig.colorbar(im,ax=ax)
    fig.suptitle('Frozen final R54 on original first-cycle TRAIN paths; not integrated residuals')
    fig.savefig(root/'TRAIN_path_residual.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,key,title in zip(axs.flat,['V_mean','V_SD','U_mean','U_SD'],['Voltage mean','Voltage SD','Peak-normalized shape mean','Peak-normalized shape SD']):
        for name,label,col in zip(names,labels,colors):
            a=z[name+'__wave_'+key][8]
            ax.plot(np.arange(480),a*(1000 if key.startswith('V') else 1),label=label,color=col,lw=1)
        ax.set(title=title,xlabel='Sample',ylabel='mV' if key.startswith('V') else 'Normalized amplitude')
    handles,labs=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labs,loc='upper center',bbox_to_anchor=(.5,.965),ncol=2)
    fig.suptitle('All 480 samples; population-weighted descriptive moments',y=.998);fig.tight_layout(rect=(0,0,1,.90))
    fig.savefig(root/'TRAIN_full_waveform.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=[m['bin'] for m in markers]
    for key,label in [('teacher_REAL_W1','Teacher vs REAL'),('R54_teacher_W1','R54 vs teacher'),('R54_REAL_W1','R54 vs REAL')]:axs[0].plot(x,[m[key] for m in markers],'o-',label=label)
    axs[0].axhline(.01,color='gray',ls='--');axs[0].set(xlabel='Fixed Q2-Q7',ylabel='Signed PSD W1',title='Original TRAIN exposure weights');axs[0].legend()
    axs[1].plot(x,[m['abs_B_over_D'] for m in markers],'o-');axs[1].axhline(.2,color='gray',ls='--')
    for m in markers:axs[1].annotate('HIT' if m['learning_marker'] else 'no',(m['bin'],m['abs_B_over_D']),xytext=(0,7),textcoords='offset points',ha='center')
    axs[1].set(xlabel='Fixed Q2-Q7',ylabel='abs(mean residual) / mean abs target velocity',title='Predefined direction markers; sign condition also required')
    fig.savefig(root/'direction_markers.png',dpi=160);plt.close(fig)
