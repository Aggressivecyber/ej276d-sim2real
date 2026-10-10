from pathlib import Path
import ast
R=Path(__file__).resolve().parents[1]
S=R/'work/R53_PULSE_ONLY_TEACHER_FLOW_001'
text=(R/'work/R49_FROZEN_HEUN128_RESPONSE_001/evaluate_saved.py').read_text('utf8')

def replace(old,new):
    global text
    assert old in text,old
    text=text.replace(old,new)

replace('from mean_description import describe as describe_means\n','')
replace('import endpoint_common as common,temporal_common as temporal','import endpoint_common as common')
replace("ROOT/'config_cpu.json'","ROOT/'config_post.json'")
replace("REAL_ARMS=['BASE_REAL_CONTROL','GUIDED_FIXED_NOISE_REAL_CONTROL','GUIDED_COMPLETION_REAL_CONTROL']", "REAL_ARMS=CFG['REAL_arms']")
replace('for mod in [common,temporal]:','for mod in [common]:')

start=text.index('def raw_RR(')
end=text.index('\ndef widths(',start)
text=text[:start]+'''
def c2st(domains,old,real):
    saved=load(pathlib.Path(CFG['R26_root'])/'results/C2ST_PRIVATE.npz',keys=[f'phase{p}_{j}'for p in[0,1]for j in range(3)])
    selected={p:tuple(saved[f'phase{p}_{j}']for j in range(3))for p in[0,1]}
    histroot=pathlib.Path(CFG['canonical_R51_root'])
    hist=json.loads((histroot/'CANONICAL_C2ST.json').read_text('utf8'))
    hp=load(histroot/'CANONICAL_PREDICTIONS_PRIVATE.npz')
    names={'BASE':'BASE_SOURCE','BASE_REAL_CONTROL':'BASE_REAL','RAW_REAL_REAL':'RAW_REFERENCE',
           'R49_COMPLETION_SOURCE':'R49_COMPLETION_SOURCE','R49_COMPLETION_REAL':'R49_COMPLETION_REAL'}
    views=CFG['C2ST_views'];result={};paired={};private={}
    def old_prediction(name,view):
        tag=names[name]+'__'+view+'__'
        return {k:hp[tag+k] for k in ['probability','label','cluster']}
    for name,oldname in names.items():
        result[name]={}
        for view in views:
            row=dict(hist['results'][oldname+'__'+view])
            row.update(balanced_accuracy=row['BA'],heldout_CI={'balanced_accuracy_95CI':row['BA_95CI']},
                       reused_from=str(histroot/'CANONICAL_C2ST.json'),new_fit=False)
            result[name][view]=row
    paired['BASE_REAL_CONTROL']={v:dict(aa.paired_bootstrap(old_prediction('BASE_REAL_CONTROL',v),old_prediction('RAW_REAL_REAL',v),CFG),reference='RAW_REAL_REAL') for v in views}
    previous=dc.matrix
    def matrix(d,view):
        return d['u'].copy() if view=='normalized_u480' else previous(d,view)
    dc.matrix=matrix
    for name in ARMS:
        d=dict(domains[name]);r=dict(domains['REAL']);j=2 if name in REAL_ARMS else 0
        tr,te=selected[0][j],selected[1][j];rt,re=selected[0][1],selected[1][1]
        assert not set(d['groups'][tr])&set(d['groups'][te]) and not set(r['groups'][rt])&set(r['groups'][re])
        r['cluster']=np.array(['R|'+str(g)for g in r['groups']])
        if name in REAL_ARMS:d['cluster']=np.array(['R|'+str(g)for g in d['groups']])
        result[name]={};paired[name]={}
        for view in views:
            slot=name+'__'+view
            def persist(row,p):
                np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**p)
                write('results/C_'+slot+'.json',row);progress()
            dc.PERSIST_HGB_CALLBACK=persist
            row,p=dc.fit_one(d,r,(tr,te),(rt,re),view,CFG,COUNTER)
            row.update(original_fixed_rows_retained=True,canonical_ADC_representation=True,
                candidate_train_QC_n=int(d['valid'][tr].sum()),candidate_holdout_QC_n=int(d['valid'][te].sum()))
            persist(row,p);result[name][view]=row
            private.update({slot+'__'+k:p[k]for k in ['probability','label','cluster']})
            refname='RAW_REAL_REAL' if name in REAL_ARMS else 'BASE'
            paired[name][view]=dict(aa.paired_bootstrap(p,old_prediction(refname,view),CFG),reference=refname)
            write('C2ST_RUNNING.json',{'results':result,'paired':paired,'counters':COUNTER})
            log('C2ST '+slot+' BA '+str(row['balanced_accuracy']))
    same_treated={};historical_delta={}
    for name in ARMS:
        histname='R49_COMPLETION_REAL' if name in REAL_ARMS else 'R49_COMPLETION_SOURCE'
        historical_delta[name]={}
        if name in REAL_ARMS:same_treated[name]={}
        for view in views:
            pp={k:private[name+'__'+view+'__'+k]for k in ['probability','label','cluster']}
            historical_delta[name][view]=dict(aa.paired_bootstrap(pp,old_prediction(histname,view),CFG),reference=histname)
            if name in REAL_ARMS:
                same_treated[name][view]=dict(aa.paired_bootstrap(pp,old_prediction('BASE_REAL_CONTROL',view),CFG),reference='BASE_REAL_CONTROL')
    write('SAME_TREATED_REAL_CONTROL_PAIRED.json',same_treated)
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{f'phase{p}_{j}':a for p,v in selected.items()for j,a in enumerate(v)})
    return {'results':result,'paired':paired,'same_treated_REAL_control_paired':same_treated,
        'R53_minus_frozen_R49_completion':historical_delta,'original_fixed_selection_no_resplit':True,
        'fixed_classifier_CI_not_response_training_uncertainty':True,'historical_control_fits_reused_from_R51':True,
        'all_five_original_acceptance_views':True,'new_fits':COUNTER['HGB_attempts'],
        'REAL_REAL_reference':'same-acquisition disjoint TRAIN reference/DEV artificial groups; not physical equivalence'}
''' +text[end:]
replace("COUNTER['density_EM_attempts']+attempt<=32 and COUNTER['density_description_slots']<32", "COUNTER['density_EM_attempts']+attempt<=16 and COUNTER['density_description_slots']<16")
replace("'two_SOURCE_arms_and_BASE'", "'SOURCE_and_BASE'")
replace("'SOURCE_INPUT':aa.data(source,ids,np.arange(2000),'G4_SOURCE',ref)",
        "'SOURCE_INPUT':aa.data(source,ids,np.arange(2000),'G4_SOURCE',ref,adc_domain=False)")
replace("reference_bins=qb(domains['SOURCE_CLEAN']['features']['Qtotal_mVns'])", "reference_bins=qb(ev.features(source_adc)['features']['Qtotal_mVns'])")
replace("    maps=load(pathlib.Path(CFG['R27_root'])/'results/PROJECTION_SPACE_PRIVATE.npz',keys=['Pseen','Pcomplement']);seen=maps['Pseen'];complement=maps['Pcomplement']\n    for d in domains.values():d['seen']=d['u']@seen;d['complement']=d['u']@complement\n", '')
replace("'training_scope':'historical R48 only','new_training_runs':0,'new_response_updates':0", "'training_scope':'R53 sole final50000; historical R48 stream matched','new_training_runs':1,'new_response_updates':50000")
replace("'two_same_treated_REAL_controls':True", "'one_new_same_treated_REAL_control':True,'canonical_ADC_features_before_all_new_statistics':True,'raw_SOURCE_INPUT_charge_and_PSD_preserved':True")
start=text.index('    compare={}\n')
end=text.index("    write('metrics.json',metrics)",start)
text=text[:start]+"    metrics['paired_R53_minus_historical_R49_completion']=cs['R53_minus_frozen_R49_completion']\n"+text[end:]
replace("COUNTER['HGB_attempts']==40 and COUNTER['density_description_slots']==32 and COUNTER['density_EM_attempts']<=32", "COUNTER['HGB_attempts']==10 and COUNTER['density_description_slots']==16 and COUNTER['density_EM_attempts']<=16")
replace("log('R48 saved evaluation complete '","log('R53 saved evaluation complete '")
ast.parse(text)
(S/'evaluate_saved.py').write_text(text,'utf8')
print('R53 evaluate_saved.py prepared; 10 canonical HGB fits, 16 fixed density slots, original input bins retained')
