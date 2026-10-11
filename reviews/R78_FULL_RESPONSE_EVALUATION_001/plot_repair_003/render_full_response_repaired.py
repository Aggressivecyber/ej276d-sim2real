"""Frozen reporting figures from completed metrics; no fit, inference, selection."""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acceptance_gates import VIEWS, BOUNDARIES, score_aliases

COLORS={'REAL':'#252b32','SOURCE':'#007b83','REAL_CONTROL':'#d27720','BASE':'#8a8e96','TARGET':'#8c4da2','PRE':'#4774b8'}
VIEW_NAMES=['Shape + 13 scalars','Normalized full 480','Voltage full 480','Charge + signed PSD','Baseline shape / noise']
AXIS_NAMES={'input_Q0':'Original input Q','target_Qstar':'Predicted target Q*',
            'preADC_window':'Saved pre-ADC Q','canonical_evaluation':'Canonical output Q'}
AXIS_COLORS={'input_Q0':'#89909b','target_Qstar':'#8c4da2','preADC_window':'#4774b8','canonical_evaluation':'#007b83'}


def run(root,c,m,gate,acceptance,evidence):
    root=Path(root);out=root/'figures_repaired_003';out.mkdir(exist_ok=False)
    s,r=c['SOURCE_arm'],c['REAL_arm'];files=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'svg.fonttype':'none'})
    prefix='ARTIFICIAL QA - 'if c.get('artificial_figure_QA')else''

    def finish(fig,name,title):
        fig.suptitle(prefix+title,fontsize=15,fontweight='bold',x=.04,ha='left')
        subtitle=('Artificial fixture only; no scientific data or model execution.'if c.get('artificial_figure_QA')else
            'Fixed R77 endpoint; historically exposed development data. All original parent weights retained.')
        fig.text(.04,.012,subtitle,fontsize=8,color='#59616b')
        fig.tight_layout(rect=[.015,.035,.99,.95],h_pad=2.2,w_pad=2.1)
        for extension in ['png','svg']:
            p=out/(name+'.'+extension);fig.savefig(p,dpi=170)
            files.append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
        plt.close(fig)

    # Ten primary CIs, with the fixed acceptance band shared by both domains.
    cs=m['C2ST']['results'];fig,axs=plt.subplots(2,1,figsize=(11.5,7.5),sharex=True)
    for ax,n,title,color,base in [(axs[0],s,'Original SOURCE parents',COLORS['SOURCE'],'BASE'),
                                 (axs[1],r,'Same response on REAL reference parents',COLORS['REAL_CONTROL'],'BASE_REAL_CONTROL')]:
        ax.axvspan(.4,.6,color='#c4e9de',alpha=.7,label='Required CI interval [0.40, 0.60]')
        for i,view in enumerate(VIEWS):
            row=score_aliases(cs[n][view]);ci=row.get('BA_95CI');ba=row['balanced_accuracy']
            if ci is not None:
                ax.plot(ci,[i,i],color=color,lw=2.2);ax.scatter([ba],[i],color=color,s=35,zorder=3)
                ax.text(1.02,i,f'{ba:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]',va='center',ha='left',fontsize=9,
                    transform=ax.get_yaxis_transform(),clip_on=False)
            if base in cs:ax.scatter([cs[base][view]['balanced_accuracy']],[i+.15],marker='|',s=70,color=COLORS['BASE'])
        ax.set_yticks(np.arange(5),VIEW_NAMES);ax.invert_yaxis();ax.set_xlim(.35,1.01)
        ax.set_title(title+'\nGreen band: required CI interval; gray tick: frozen BASE control',loc='left',fontsize=10)
        ax.grid(axis='x',alpha=.17)
    axs[1].set_xlabel('Held-out balanced accuracy and group-bootstrap 95% CI')
    finish(fig,'01_c2st','Two-domain, five-view response assessment')

    charge=m['charge_diagnostics'];fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    target=charge['REAL_reference']['empirical_CDF']
    for ax,n,title in [(axs[0],s,'SOURCE'),(axs[1],r,'Same-treated REAL')]:
        for key,label in AXIS_NAMES.items():
            row=charge['domains'][n]['axes'][key];cdf=row['empirical_CDF']
            ax.step(cdf['unique_Q_mVns'],cdf['cumulative_original_weight'],where='post',color=AXIS_COLORS[key],lw=1.5,label=label)
        ax.step(target['unique_Q_mVns'],target['cumulative_original_weight'],where='post',color=COLORS['REAL'],ls='--',lw=1.5,label='Untreated REAL target')
        ax.set_xscale('log');ax.set_ylim(0,1.02);ax.set_title(title,loc='left');ax.set_xlabel('Charge (mV ns)');ax.grid(alpha=.15)
    axs[0].set_ylabel('Cumulative original parent weight');axs[1].legend(fontsize=8,loc='lower right')
    finish(fig,'02_charge_cdf','Original, predicted, and realized charge')

    fig,axs=plt.subplots(3,2,figsize=(11,12))
    labels=['below']+[str(i)for i in range(1,9)]+['above','invalid']
    migration_titles={'input_to_Qstar':'Original Q to predicted Q*','Qstar_to_preADC':'Predicted Q* to pre-ADC',
                      'preADC_to_canonical':'Pre-ADC to canonical output'}
    migration_max=max(np.asarray(charge['domains'][n]['migrations'][key]['count_matrix']).max()
        for n in [s,r]for key in migration_titles)
    for j,n in enumerate([s,r]):
        for i,(key,title)in enumerate(migration_titles.items()):
            ax=axs[i,j];row=charge['domains'][n]['migrations'][key];a=np.asarray(row['count_matrix'])
            im=ax.imshow(np.log1p(a),cmap='Blues',vmin=0,vmax=np.log1p(max(migration_max,1)),aspect='equal')
            for y,x in zip(*np.nonzero(a)):
                ax.text(x,y,str(a[y,x]),ha='center',va='center',fontsize=7,color='white'if np.log1p(a[y,x])>im.norm.vmax*.57 else'#243244')
            ax.set_xticks(range(11),labels,rotation=50,ha='right');ax.set_yticks(range(11),labels)
            ax.set_xlabel('Destination bin');ax.set_ylabel('Origin bin')
            ax.set_title(('SOURCE'if j==0 else'Same-treated REAL')+' | '+title+'\n'+
                f"N={row['count_total']}; total weight={row['weight_total']:.6f}",loc='left',fontsize=10)
    finish(fig,'03_charge_migration','All parent migration, including support tails')

    width=m['energy_conditional_width'];tables=width['tables'];fig,axs=plt.subplots(2,4,figsize=(14,7))
    for i,ax in enumerate(axs.flat):
        failures=[]
        for n,label,color in [('REAL','REAL',COLORS['REAL']),('BASE','BASE',COLORS['BASE']),(s,'Response',COLORS['SOURCE'])]:
            tab=tables[n];row=tab['slices'][i];edges=np.asarray(tab['histogram_edges_PSD'])
            if 'histogram_counts' in row:
                counts=np.asarray(row['histogram_counts'])
                mass=counts/max(float(row['N']),1.)/np.diff(edges)
                ax.stairs(mass,edges,color=color,lw=1.5,label=label+' histogram')
            elif n == 'REAL' and row['valid']:
                # Frozen historical REAL stores fitted density parameters but no histogram.
                # Evaluate the saved curve for display only: no event reads, refit, or new statistic.
                xx=np.linspace(-.1,.6,601)  # frozen displayed PSD axis, not new statistical bins
                yy=sum(comp['weight']/ (np.sqrt(2*np.pi)*comp['sigma_PSD']) *
                    np.exp(-.5*((xx-comp['mean'])/comp['sigma_PSD'])**2) for comp in row['components'])
                ax.plot(xx,yy,color=color,lw=1.5,ls='--',label='REAL saved mixture fit')
            else:
                failures.append(label+' histogram unavailable')
            if not row['valid']:failures.append(label+' fit failed')
        ns=tables[s]['slices'][i]['N'];nr=tables['REAL']['slices'][i]['N']
        ax.set_title(f'Q{i+1} | SOURCE {ns:g}, REAL {nr:g}',loc='left',fontsize=10)
        if failures:ax.text(.02,.98,'\n'.join(failures),transform=ax.transAxes,va='top',fontsize=8,color='#ad3333')
        ax.set_xlim(-.1,.6);ax.grid(alpha=.13);ax.set_xlabel('Signed PSD')
    axs[0,0].set_ylabel('Density using full-bin denominator');axs[1,0].set_ylabel('Density using full-bin denominator')
    handles,labels=axs[0,3].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.97,.925),ncol=3,fontsize=8)
    fig.text(.04,.94,'Dashed REAL: previously saved mixture fit (historical histogram unavailable); other curves: saved histograms.',fontsize=8,color='#59616b')
    finish(fig,'04_psd_eight_bins','PSD in all eight canonical output charge bins')

    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for row,field in enumerate(['mean','sigma_PSD']):
        for component in [0,1]:
            ax=axs[row,component];ax.axvspan(1.5,7.5,color='#ecf5f2',zorder=0)
            for n,label,color in [('REAL','REAL',COLORS['REAL']),('BASE','BASE',COLORS['BASE']),
                                   (s,'SOURCE response',COLORS['SOURCE']),(r,'REAL response',COLORS['REAL_CONTROL'])]:
                values=[]
                for slot in tables[n]['slices']:
                    value=slot['components'][component].get(field)if slot['valid']and len(slot['components'])==2 else None
                    values.append(float(value)if value is not None else np.nan)
                ax.plot(range(1,9),values,'o-',ms=4,lw=1.4,color=color,label=label)
                for i,value in enumerate(values):
                    if not np.isfinite(value):ax.text(i+1,.02,'x',color=color,transform=ax.get_xaxis_transform(),ha='center')
            ax.set_xticks(range(1,9),['Q'+str(i)for i in range(1,9)]);ax.set_xlabel('Canonical output bin')
            ax.set_ylabel('PSD peak position'if field=='mean'else'Raw PSD standard deviation')
            ax.set_title(('Low-PSD'if component==0 else'High-PSD')+' density component'+(' (x = failed fit)'if component==0 else''),loc='left')
            ax.grid(alpha=.15)
    axs[0,1].legend(fontsize=8)
    finish(fig,'05_conditional_peaks','Conditional peaks and raw widths; required Q2-Q7 shaded')

    moment_path=root/'results/RAW_VOLTAGE_MOMENTS_PRIVATE.npz'
    with np.load(moment_path,allow_pickle=False)as packet:moment={k:packet[k].copy()for k in packet.files}
    fig,axs=plt.subplots(2,2,figsize=(12,7))
    for n,label,color in [('REAL','REAL',COLORS['REAL']),('BASE','BASE',COLORS['BASE']),(s,'SOURCE response',COLORS['SOURCE']),(r,'REAL response',COLORS['REAL_CONTROL'])]:
        for row,field in enumerate(['mean_V','SD_V']):
            values=1000*moment[n+'__None__'+field]
            axs[row,0].plot(range(480),values,label=label,color=color,lw=1.3)
            axs[row,1].plot(range(100),values[:100],label=label,color=color,lw=1.3)
    for row in range(2):
        for col in range(2):
            ax=axs[row,col];ax.set_ylabel(('Mean'if row==0 else'Standard deviation')+' (mV)');ax.set_xlabel('Saved-window sample');ax.grid(alpha=.15)
    axs[0,0].set_title('Complete 480 samples',loc='left');axs[0,1].set_title('Original 100-sample baseline',loc='left');axs[0,1].legend(fontsize=8)
    finish(fig,'06_full_window','Complete voltage response and baseline moments')

    gm=gate['models'][s];fig,axs=plt.subplots(1,2,figsize=(12,5.5))
    rows=gm['gate4_boundary_source']['boundaries'];y=np.arange(len(rows));values=[]
    for row in rows:values.append(row['candidate_W1_V']/row['limit_V']if row['candidate_W1_V']is not None and row['limit_V']else np.nan)
    axs[0].barh(y,values,color=['#007b83'if row['pass']else'#bf5144'for row in rows]);axs[0].axvline(1,color='#252b32',ls='--',lw=1)
    axs[0].set_yticks(y,[x['location'].replace('first_','d1 ').replace('second_center_','d2 @ ')for x in rows]);axs[0].invert_yaxis()
    axs[0].set_xlabel('W1 / fixed per-location limit');axs[0].set_title('All nine boundaries',loc='left');axs[0].grid(axis='x',alpha=.15)
    for i,value in enumerate(values):
        if not np.isfinite(value):axs[0].text(.02,i,'undefined / failed',va='center',color='#ad3333',fontsize=8)
    measures=gm['gate2_PSD_width_centers_noise']['measures'];order=list(measures)
    vals=[measures[k]['relative_change']for k in order];short={'PSD_overall_W1':'PSD overall','baseline_RMS_V_overall_W1':'Baseline RMS',
        'ACF1_overall_W1':'Baseline ACF1','PSD_common_Q_equal_mean_W1':'PSD common Q1-Q7',
        'component0_peak_position_MAE':'Low peak MAE','component1_peak_position_MAE':'High peak MAE',
        'component0_raw_width_MAE':'Low raw width MAE','component1_raw_width_MAE':'High raw width MAE'}
    xx=[v*100 if v is not None else 0 for v in vals]
    axs[1].barh(range(len(vals)),xx,color=['#007b83'if measures[k]['pass']else'#bf5144'for k in order]);axs[1].axvline(10,color='#252b32',ls='--',lw=1)
    axs[1].set_yticks(range(len(vals)),[short[k]for k in order]);axs[1].invert_yaxis();axs[1].set_xlabel('Change from BASE (%)');axs[1].set_title('Eight original side-effect rules',loc='left')
    for i,value in enumerate(vals):
        if value is None:axs[1].text(0,i,' undefined / failed',va='center',color='#ad3333',fontsize=8)
    axs[1].grid(axis='x',alpha=.15)
    finish(fig,'07_boundaries_and_side_effects','Boundary continuity, PSD, and noise safeguards')

    fig,axs=plt.subplots(1,3,figsize=(13,4.6));g3=gm['gate3_source_preservation']
    for j,key in enumerate(['actual_source_pure_had_valleys','whole_population_source_ranks']):
        rows=g3[key];candidate=[x['candidate']if x['candidate']is not None else np.nan for x in rows]
        base=[x['BASE']if x['BASE']is not None else np.nan for x in rows];x=np.arange(len(rows))
        axs[j].bar(x-.17,base,width=.34,color=COLORS['BASE'],label='BASE');axs[j].bar(x+.17,candidate,width=.34,color=COLORS['SOURCE'],label='Response')
        axs[j].set_xticks(x,[('Original Q3\nN=213','Original Q4\nN=181')[i]if j==0 else('Q rank','PSD rank')[i]for i in range(2)])
        axs[j].set_title('Original pure-had valleys'if j==0 else'Whole-source association',loc='left');axs[j].set_ylabel('Valley fraction'if j==0 else'Spearman correlation');axs[j].grid(axis='y',alpha=.15)
        for i,value in enumerate(candidate):
            if not np.isfinite(value):axs[j].text(i,.02,'undefined',ha='center',color='#ad3333',fontsize=8,transform=axs[j].get_xaxis_transform())
    axs[0].legend(fontsize=8)
    checks=gm['gate6_numerical_replay']['checks'];text=['Numerical and replay checks']
    check_labels={
        'fixed_completed_training_and_native_exit':f"Fixed {c['updates']:,}-update training receipt",
        'independent_noniterative_NumPy_reference':'Independent NumPy precision reference',
        'training_full_population_and_endpoint_binding':'Full-parent training and frozen endpoint',
        'all_preADC_f32_Qstar_error_le1e_5':'Actual pre-ADC relative Q* error <= 1e-5',
        'all_main_rows_processed':'All 4,000 main parents processed',
        'row_metadata_and_charge_trace_verified':'Identity, weight, and charge trace',
        'main4000_independent_replay':'Independent 4,000-row bitwise replay',
        'generation_replay_evaluation_native_exit0':'Native exits and measured resources',
        'same_bound_public_SOURCE_REAL_operator':'Same public SOURCE/REAL response',
        'charge_migration_full_mass_accounted':'Full charge-distribution accounting'}
    text += [('[PASS] 'if value else'[FAIL] ')+check_labels[key]for key,value in checks.items()]
    axs[2].axis('off');axs[2].text(0,1,'\n\n'.join(text),va='top',fontsize=7.8,wrap=True,transform=axs[2].transAxes)
    finish(fig,'08_source_and_replay','Source information and numerical evidence')
    return files
