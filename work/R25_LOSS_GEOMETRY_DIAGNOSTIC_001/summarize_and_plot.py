from __future__ import annotations
import os,time,json
START=time.monotonic()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostic_support import ROOT,CFG,write,check_output,log
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.dpi':120,'savefig.dpi':160})
a=json.loads((ROOT/'A_METRICS.json').read_text('utf8'))
b=json.loads((ROOT/'B_GRADIENTS_PRIVATE.json').read_text('utf8'))
receipt=json.loads((ROOT/'B_NUMERIC_RECEIPT.json').read_text('utf8'))
terms=b['loss_names'];short=['adv','NCE','identity','Q rel','Q rank','PSD rank','domain SW','positive Q','fine','style L1']
figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
groups=CFG['A']['groups']+['original_distribution_combination'];group_short=['physical','pooled shape','shape480','diff479','baseline100 V','original combined term']
def summary(x):
    x=np.asarray(x,dtype='f8');f=x[np.isfinite(x)]
    return {'defined_n':int(len(f)),'undefined_n':int(x.size-len(f)),'mean':float(f.mean()) if len(f) else None,'median':float(np.median(f)) if len(f) else None,'q05_q95':np.quantile(f,[.05,.95]).tolist() if len(f) else None,'min':float(f.min()) if len(f) else None,'max':float(f.max()) if len(f) else None}
def num(x):return np.asarray(x,dtype='f8')
def finite_mean(x,axis=0):
    x=num(x);ok=np.isfinite(x);count=ok.sum(axis=axis);s=np.where(ok,x,0).sum(axis=axis)
    return np.divide(s,count,out=np.full(np.shape(s),np.nan),where=count>0)
states=[]
for state in b['states']:
    rows=state['batches'];norm=num([r['G_term_norm'] for r in rows]);dots=num([r['G_term_dot_total'] for r in rows]);aligned=num([r['G_term_cos_total'] for r in rows]);gn=num([r['G_total_norm'] for r in rows]);combined=num([r['G_reader_total_norm'] for r in rows]);ratio=gn/norm.sum(axis=1)
    groupsumm={}
    for key in rows[0]['parameter_groups']:
        norms=num([r['parameter_groups'][key]['term_norm'] for r in rows]);total=num([r['parameter_groups'][key]['total_norm'] for r in rows])
        groupsumm[key]={'term_norm_summary':{term:summary(norms[:,j]) for j,term in enumerate(terms)},'total_norm_summary':summary(total),'mean_term_norm_shares':finite_mean(norms,axis=0)/finite_mean(norms,axis=0).sum() if norms.sum()>0 else np.full(10,np.nan)}
    entry={'state':state['state'],'initial_D_untrained':state['initial_D_untrained'],'G_parameter_count':state['G_parameter_count'],'reader_parameter_count':state['reader_parameter_count'],'parameters_unchanged':state['parameters_unchanged'],'gradient_sum_check_first_batch':state['gradient_sum_check_first_batch'],'batch_n':32,'terms':{},'groups':groupsumm,'G_total_norm':summary(gn),'G_reader_total_norm':summary(combined),'G_cancellation_ratio_norm_sum_over_sum_norms':summary(ratio),'clip5_reference_scale':summary([r['clip5_reference_scale'] for r in rows]),'clip5_would_apply_batches':sum(r['clip5_reference_scale']<1 for r in rows),'G_cosine_matrix_mean_defined':finite_mean([r['G_cosine_matrix'] for r in rows],axis=0),'G_cosine_matrix_defined_n':np.isfinite(num([r['G_cosine_matrix'] for r in rows])).sum(axis=0)}
    for j,term in enumerate(terms):
        defined=np.isfinite(aligned[:,j]);entry['terms'][term]={'unweighted_loss':summary([r['unweighted_loss'][j] for r in rows]),'weighted_loss':summary([r['weighted_loss'][j] for r in rows]),'G_weighted_gradient_norm':summary(norm[:,j]),'G_dot_total':summary(dots[:,j]),'G_cosine_total':summary(aligned[:,j]),'negative_dot_defined_batches':int(((dots[:,j]<0)&defined).sum()),'nonzero_gradient_batches':int(defined.sum()),'negative_dot_fraction_defined':float(np.mean(dots[defined,j]<0)) if defined.any() else None}
    states.append(entry)
write('B_METRICS_COMPACT.json',{'receipt':receipt,'loss_names':terms,'weights':b['weights'],'states':states,'interpretation':'Local weighted raw gradients; not Adam updates. INITIAL critic is untrained. Zero norms remain undefined.'})
write('METRICS_COMPACT.json',{'A':a,'B':{'receipt':receipt,'loss_names':terms,'weights':b['weights'],'states':states},'no_per_event_indices_or_inputs':True,'new_response_result':False})
figures=[]
def save(fig,name):
    fig.savefig(figdir/name,bbox_inches='tight');plt.close(fig);figures.append(name)
for stat,filename,title in [('W1_quantile','contraction_W1.png','Projected small-batch quantile L1: controlled contraction only'),('energy_U','contraction_energy_U.png','Projected energy U: all three diagonals excluded')]:
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,group,label in zip(axes.flat,groups,group_short):
        for regime,color in [('MIXED','#2563eb'),('HOMOGENEOUS','#ea580c')]:
            rows=a['summary'][regime][group][stat];yy=[r['absolute']['mean'] for r in rows];err=[1.96*r['absolute']['MC_SE'] for r in rows]
            ax.errorbar(CFG['A']['c'],yy,yerr=err,marker='o',color=color,capsize=3,label=regime)
        ax.axvline(1,color='.5',ls=':',lw=1);ax.set_title(label);ax.set_xlabel('c (within-bin projected deviations)');ax.set_ylabel(stat+' (original projection units)')
        if stat=='energy_U':ax.axhline(0,color='.6',lw=.8)
        ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle(title+'; bars = fixed-pool MC 95% intervals');fig.tight_layout(rect=(0,0,1,.94));save(fig,filename)
fig,axes=plt.subplots(2,3,figsize=(13,7))
for ax,group,label in zip(axes.flat,groups,group_short):
    for regime,color in [('MIXED','#2563eb'),('HOMOGENEOUS','#ea580c')]:
        rows=a['summary'][regime][group]['W1_quantile'];ax.errorbar(CFG['A']['c'],[r['paired_minus_c1']['mean'] for r in rows],yerr=[1.96*r['paired_minus_c1']['MC_SE'] for r in rows],marker='o',capsize=3,label=regime,color=color)
    ax.axhline(0,color='.5',lw=.8);ax.set_title(label);ax.set_xlabel('c');ax.set_ylabel('paired W1(c) - W1(1)');ax.grid(alpha=.2)
axes[0,0].legend();fig.suptitle('Paired change from the unchanged empirical distribution; no waveform generation');fig.tight_layout(rect=(0,0,1,.94));save(fig,'contraction_paired_W1.png')
colors=['#64748b','#2563eb','#ea580c'];labels=['initial (untrained D)','D1 final2950','D5 final2950']
fig,ax=plt.subplots(figsize=(12,5))
for st,label,color in zip(states,labels,colors):
    med=[st['terms'][t]['G_weighted_gradient_norm']['median'] for t in terms]
    lo=[st['terms'][t]['G_weighted_gradient_norm']['q05_q95'][0] for t in terms];hi=[st['terms'][t]['G_weighted_gradient_norm']['q05_q95'][1] for t in terms]
    y=np.maximum(med,1e-8);ax.errorbar(np.arange(10),y,yerr=[y-np.maximum(lo,1e-8),np.maximum(hi,1e-8)-y],marker='o',capsize=3,label=label,color=color)
ax.set_yscale('log');ax.set_xticks(np.arange(10),short,rotation=25,ha='right');ax.set_ylabel('weighted G gradient L2 (median, 5–95% batch range)');ax.set_title('Original ten weighted components; positive-Q zero plotted at display floor 1e-8');ax.legend();ax.grid(alpha=.2);fig.tight_layout();save(fig,'weighted_gradient_norms.png')
fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,st,label in zip(axes,states,labels):
    im=ax.imshow(np.ma.masked_invalid(num(st['G_cosine_matrix_mean_defined'])),cmap='coolwarm',vmin=-1,vmax=1)
    ax.set_xticks(np.arange(10),short,rotation=60,ha='right');ax.set_yticks(np.arange(10),short);ax.set_title(label)
fig.subplots_adjust(left=.06,right=.91,bottom=.27,wspace=.35,top=.88);fig.colorbar(im,cax=fig.add_axes([.93,.3,.015,.5]),label='mean defined G-gradient cosine');fig.suptitle('Local weighted G gradient interactions; blank = zero norm / undefined');save(fig,'G_gradient_cosines.png')
groupnames=['encoder','NCE_project','decoder_mean','style_FiLM','electronic','reader']
fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,st,label in zip(axes,states,labels):
    image=num([st['groups'][g]['mean_term_norm_shares'] for g in groupnames]);im=ax.imshow(np.ma.masked_invalid(image),cmap='viridis',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(np.arange(10),short,rotation=60,ha='right');ax.set_yticks(np.arange(6),['encoder','NCE project','decoder / mean','style FiLM','electronic','reader']);ax.set_title(label)
fig.subplots_adjust(left=.08,right=.91,bottom=.27,wspace=.35,top=.88);fig.colorbar(im,cax=fig.add_axes([.93,.3,.015,.5]),label='mean L2 norm / sum of term mean norms');fig.suptitle('Parameter-group dominance (norm share, not update or causal attribution)');save(fig,'parameter_group_norm_shares.png')
fig,ax=plt.subplots(figsize=(12,5))
for i,(st,label,color) in enumerate(zip(states,labels,colors)):
    y=[st['terms'][t]['negative_dot_fraction_defined'] for t in terms];ax.bar(np.arange(10)+(i-1)*.25,num(y),width=.25,label=label,color=color)
ax.set_xticks(np.arange(10),short,rotation=25,ha='right');ax.set_ylim(0,1.05);ax.set_ylabel('fraction of defined batches with g_term · g_total < 0');ax.set_title('Negative dot = local component ascent under raw -g_total, not Adam');ax.legend();fig.tight_layout();save(fig,'negative_alignment_rates.png')
fig,axes=plt.subplots(1,3,figsize=(13,4))
for j,st in enumerate(b['states']):
    rows=st['batches'];norm=num([r['G_term_norm'] for r in rows]);ratio=num([r['G_total_norm'] for r in rows])/norm.sum(1)
    axes[0].plot(np.arange(1,33),ratio,label=labels[j],color=colors[j]);axes[1].plot(np.arange(1,33),[r['clip5_reference_scale'] for r in rows],color=colors[j]);axes[2].plot(np.arange(1,33),[r['parameter_groups']['reader']['total_norm'] for r in rows],color=colors[j])
axes[0].set_title('G cancellation ratio');axes[0].set_ylabel('||sum g|| / sum ||g||');axes[0].legend(fontsize=8)
axes[1].set_title('Clip5 hypothetical scale (G + reader)');axes[1].set_ylabel('min(1, 5 / total norm)')
axes[2].set_title('Reader total gradient, separately');axes[2].set_ylabel('reader L2 norm')
for ax in axes:ax.set_xlabel('fixed actual batch');ax.grid(alpha=.2)
fig.tight_layout();save(fig,'cancellation_clip_reader.png')
rec={'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'figures':figures,'metric_fits':0,'response_updates':0,'scientific_forward_rows':0,'inputs':'Only saved R25 A/B results; no original arrays/models reopened'}
write('PLOT_RECEIPT.json',rec);check_output();log(rec)
print(json.dumps({'A_combined':{reg:{stat:[{'c':r['c'],'mean':r['absolute']['mean'],'delta':r['paired_minus_c1']} for r in a['summary'][reg]['original_distribution_combination'][stat]] for stat in ['W1_quantile','energy_U']} for reg in ['MIXED','HOMOGENEOUS']},'B_key':[{'state':s['state'],'G_total':s['G_total_norm'],'combined_total':s['G_reader_total_norm'],'clip':s['clip5_reference_scale'],'cancellation':s['G_cancellation_ratio_norm_sum_over_sum_norms'],'sum_check':s['gradient_sum_check_first_batch'],'dominant_norm_mean':{k:v['G_weighted_gradient_norm']['mean'] for k,v in s['terms'].items()},'negative_rates':{k:v['negative_dot_fraction_defined'] for k,v in s['terms'].items()}} for s in states]},default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x),flush=True)
