import numpy as np,json,hashlib,copy,pathlib

def reconstruct_and_reorder(cfg,root):
    # Reuse the actual R26 row trajectory, not a newly chosen permutation.
    with np.load(cfg['consumed_R26_stream'],allow_pickle=False) as z:result={k:z[k].copy() for k in z.files}
    assert len(result['source'])==94400 and len(result['NCE_positions'])==8850
    perm=result['original_row_ordinal'];assert np.array_equal(np.sort(perm),np.arange(94400))
    rs=json.loads(pathlib.Path(cfg['actual_training_random_states']).read_text('utf8'))
    old=json.loads(pathlib.Path(cfg['actual_training_receipt']).read_text('utf8'))
    r26=json.loads(pathlib.Path(cfg['consumed_R26_receipt']).read_text('utf8'))
    for k in ['source','target','bin','original_row_ordinal','NCE_positions']:
        assert hashlib.sha256(result[k].tobytes()).hexdigest()==r26['stream_hashes'][k],k
    rn=np.random.default_rng();rn.bit_generator.state=copy.deepcopy(rs['noise_initial'])
    rt=np.random.default_rng();rt.bit_generator.state=copy.deepcopy(rs['style_initial'])
    zz=np.empty((94400,8),'f4');ww=np.empty((94400,480),'f4');ss=np.empty((94400,8),'f4');hn=hashlib.sha256();hs=hashlib.sha256()
    for step in range(2950):
        sl=slice(step*32,(step+1)*32);zz[sl]=rn.normal(size=(32,8)).astype('f4');ww[sl]=rn.normal(size=(32,480)).astype('f4');ss[sl]=rt.normal(size=(32,8)).astype('f4')
        hn.update(zz[sl].tobytes());hn.update(ww[sl].tobytes());hs.update(ss[sl].tobytes())
    assert hn.hexdigest()==old['stream_hashes']['noise_stream'] and hs.hexdigest()==old['stream_hashes']['style_stream']
    assert rn.bit_generator.state==rs['noise_final'] and rt.bit_generator.state==rs['style_final']
    result.update(original_noise={'z':zz,'white':ww},original_style=ss,
                  original_rng_initial={'noise':rs['noise_initial'],'style':rs['style_initial']},
                  original_rng_final={'noise':rs['noise_final'],'style':rs['style_final']})
    nb=result['bin'].reshape(-1,32);homogeneous=np.all(nb==nb[:,:1],axis=1)
    assert int(homogeneous.sum())==2946
    rec={'rows':94400,'blocks':2950,'homogeneous_blocks':2946,'mixed_blocks':4,
         'same_actual_R26_order_and_all_row_fields':True,'same_R26_NCE_step_position_rows':True,
         'reconstruction_original_noise_style_match':True,'new_permutation_selected':False,
         'input_actual_stream':cfg['consumed_R26_stream'],'original_white_memory_bytes':ww.nbytes,
         'binding_rule':'algorithmic consumed-row/random-stream check, not migration-content testing'}
    (root/'REORDER_CONSTRUCTION_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8')
    return result
