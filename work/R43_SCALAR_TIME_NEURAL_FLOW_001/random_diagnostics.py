from __future__ import annotations
import numpy as np
import evaluation as ev
from model import infer,adc

def window_variation(y):
    variance=np.var(np.asarray(y,dtype='f8'),axis=1,ddof=1)
    return {k:np.sqrt(np.mean(variance[:,a:b],axis=1))for k,a,b in [
        ('full480',0,480),('baseline100',0,100),('prompt100_150',100,150),
        ('postpeak150_220',150,220),('tail220_400',220,400)]}

def summarize(post,signal,electronic,parent,n,comp,edges,source,VALLEY):
    f=ev.features(post);p=f['features']['PSD'].reshape(-1,n);q=f['features']['Qtotal_mVns'].reshape(-1,n);valid=f['valid'].reshape(-1,n)
    pick=parent.reshape(-1,n)[:,0];assert np.all(parent.reshape(-1,n)==pick[:,None])
    c=comp[pick];sf=ev.features(source[pick]);sb=np.searchsorted(edges,np.log(sf['features']['Qtotal_mVns']),side='right')-1
    waves={'output_post_ADC':window_variation(post.reshape(-1,n,480)),
           'signal_pre_ADC':window_variation(signal.reshape(-1,n,480)),
           'electronic_pre_ADC':window_variation(electronic.reshape(-1,n,480))}
    rows=[]
    for j in range(len(pick)):
        x=p[j][valid[j]];y=q[j][valid[j]];b=int(sb[j]);de=VALLEY['bins'][b]if 0<=b<8 else None
        rows.append({'source_parent_index':int(pick[j]),'composition':str(c[j]),'source_Qbin_1based':b+1 if 0<=b<8 else None,
          'valid_draws':len(x),'output_PSD':ev.summary(x),'output_Q_mVns':ev.summary(y),
          'both_sides_of_frozen_valley':bool(de and de['valid']and len(x)and(x<de['low']).any()and(x>de['high']).any())})
    by={}
    for kind in np.unique(c):
        mask=c==kind
        by[str(kind)]={'parents':int(mask.sum()),'within_parent_PSD_SD':ev.summary(np.asarray([r['output_PSD']['sample_sd']for r in rows if r['composition']==kind and r['output_PSD']['sample_sd']is not None])),
          'within_parent_Q_SD_mVns':ev.summary(np.asarray([r['output_Q_mVns']['sample_sd']for r in rows if r['composition']==kind and r['output_Q_mVns']['sample_sd']is not None])),
          'parents_straddling_valley':sum(r['both_sides_of_frozen_valley']for r in rows if r['composition']==kind),
          'within_parent_waveform_RMS_V':{component:{window:ev.summary(v[mask])for window,v in windows.items()}for component,windows in waves.items()}}
    return {'parent_count':len(pick),'draws_each':n,'rows':rows,'by_composition':by,
      'variance_denominator':n-1,'signal_is_not_conditional_expectation':True,
      'electronic_branch_indirectly_affected_by_style':True,'repeats_not_independent_G4_events':True},f

def repeats(model,source,comp,edges,cfg,root,out,VALLEY,counter):
    with np.load(root/'REPEAT_INPUTS.npz',allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
    parent=a['source_parent_index'];pick=parent[::8];c=comp[pick]
    chosen=np.concatenate([np.flatnonzero(c=='pure_EE')[:16],np.flatnonzero(c=='pure_had')[:16]])
    assert len(chosen)==32 and np.sum(c[chosen]=='pure_EE')==16 and np.sum(c[chosen]=='pure_had')==16
    selected=pick[chosen];rng=np.random.default_rng(cfg['style_repeat_seed']);style=rng.normal(size=(1024,8)).astype('f4')
    modes={'joint8':{'parent':parent,'z':a['z'],'white':a['white'],'style':style,'n':8}}
    # First draw in each decomposition is the actual joint reference; other draws are independent.
    refz=a['z'][chosen*8];refw=a['white'][chosen*8];refs=style[chosen*8]
    style32=np.concatenate([refs[:,None,:],rng.normal(size=(32,31,8)).astype('f4')],axis=1)
    rng_e=np.random.default_rng(np.random.SeedSequence(cfg['style_repeat_seed'],spawn_key=(1,)))
    z32=np.concatenate([refz[:,None,:],rng_e.normal(size=(32,31,8)).astype('f4')],axis=1)
    w32=np.concatenate([refw[:,None,:],rng_e.normal(size=(32,31,480)).astype('f4')],axis=1)
    modes['style_only32']={'parent':np.repeat(selected,32),'z':np.repeat(refz,32,axis=0),'white':np.repeat(refw,32,axis=0),'style':style32.reshape(1024,8),'n':32}
    modes['electronic_only32']={'parent':np.repeat(selected,32),'z':z32.reshape(1024,8),'white':w32.reshape(1024,480),'style':np.repeat(refs,32,axis=0),'n':32}
    summary={};reference=None
    for mode,v in modes.items():
        x=source[v['parent']];pre,signal,electronic=infer(model,x,v['z'],v['white'],v['style'],cfg['generation_chunk']);post,ad=adc(pre)
        counter['repeat_noise_rows']+=len(post)
        summary[mode],f=summarize(post,signal,electronic,v['parent'],v['n'],comp,edges,source,VALLEY)
        summary[mode]['ADC']=ad
        if mode=='joint8':reference=post.reshape(128,8,480)[chosen,0]
        else:summary[mode]['first_draw_equals_actual_joint_reference']=bool(np.array_equal(post.reshape(32,32,480)[:,0],reference))
        np.savez_compressed(out/(mode+'_REPEAT_PRIVATE.npz'),source_parent_index=v['parent'],source_waveforms_V=x,z=v['z'],white=v['white'],style=v['style'],waveforms_V=post,
          signal_component_V=signal,electronic_component_V=electronic,PSD=f['features']['PSD'],Q_mVns=f['features']['Qtotal_mVns'],valid=f['valid'],parent_composition=comp[v['parent'].reshape(-1,v['n'])[:,0]])
    return {'groups':summary,'planned_rows':3072,'actual_rows':sum(len(v['parent'])for v in modes.values()),
      'decomposition_selection':'first16 pure_EE and first16 pure_had in the unchanged original128-parent order',
      'decomposition_parents_per_group':32,'source_parent_union':128,'new_independent_source_parents':0,
      'seed_style':cfg['style_repeat_seed'],'electronic_random_stream':'SeedSequence(style_repeat_seed,spawn_key=(1,))',
      'first_decomposition_draw_is_saved_joint_reference':True,'no_new_density_fit':True}
