import numpy as np,json,hashlib,copy,pathlib

def reconstruct_and_reorder(cfg,root):
    with np.load(cfg['actual_training_stream'],allow_pickle=False)as z:old={k:z[k].copy()for k in z.files}
    assert len(old['source'])==94400 and len(old['NCE_positions'])==8850
    rs=json.loads(pathlib.Path(cfg['actual_training_random_states']).read_text('utf8'));receipt=json.loads(pathlib.Path(cfg['actual_training_receipt']).read_text('utf8'))
    rng=np.random.Generator(np.random.PCG64(cfg['reorder_seed']));initial=copy.deepcopy(rng.bit_generator.state);blocks=[];left=[];counts=[]
    for b in range(8):
        ix=np.flatnonzero(old['bin']==b);ix=rng.permutation(ix);n=len(ix)//32
        blocks.extend(ix[:n*32].reshape(n,32));left.extend(ix[n*32:].tolist());counts.append({'bin_id_1based':b+1,'rows':len(ix),'homogeneous_blocks':n,'remainder':len(ix)-n*32})
    assert len(left)%32==0
    tails=np.asarray(left,dtype='i8').reshape(-1,32);blocks.extend(tails)
    blocks=np.asarray(blocks,dtype='i8');blocks=blocks[rng.permutation(len(blocks))];perm=blocks.flatten();assert len(perm)==94400 and np.array_equal(np.sort(perm),np.arange(94400))
    rn=np.random.default_rng();rn.bit_generator.state=copy.deepcopy(rs['noise_initial']);rt=np.random.default_rng();rt.bit_generator.state=copy.deepcopy(rs['style_initial'])
    zz=np.empty((94400,8),'f4');ww=np.empty((94400,480),'f4');ss=np.empty((94400,8),'f4');hn=hashlib.sha256();hs=hashlib.sha256()
    for step in range(2950):
        sl=slice(step*32,(step+1)*32);zz[sl]=rn.normal(size=(32,8)).astype('f4');ww[sl]=rn.normal(size=(32,480)).astype('f4');ss[sl]=rt.normal(size=(32,8)).astype('f4');hn.update(zz[sl].tobytes());hn.update(ww[sl].tobytes());hs.update(ss[sl].tobytes())
    assert hn.hexdigest()==receipt['stream_hashes']['noise_stream']and hs.hexdigest()==receipt['stream_hashes']['style_stream']
    assert rn.bit_generator.state==rs['noise_final']and rt.bit_generator.state==rs['style_final']
    result={k:(v.copy()if k=='NCE_positions'else v[perm])for k,v in old.items()};result['original_row_ordinal']=perm
    result.update(original_noise={'z':zz,'white':ww},original_style=ss,original_rng_initial={'noise':rs['noise_initial'],'style':rs['style_initial']},original_rng_final={'noise':rs['noise_final'],'style':rs['style_final']})
    nb=result['bin'].reshape(-1,32);homogeneous=np.all(nb==nb[:,:1],axis=1)
    audit={'seed':cfg['reorder_seed'],'initial_PC64':initial,'final_PC64':rng.bit_generator.state,'rows':len(perm),'blocks':len(blocks),'homogeneous_blocks':int(homogeneous.sum()),'mixed_blocks':int((~homogeneous).sum()),'bin_construction':counts,'tail_blocks_before_shuffle':len(tails),'permutation_sha256':hashlib.sha256(perm.astype('<i8').tobytes()).hexdigest(),'full_original_ordinal_multiset_preserved':True,'all_original_row_fields_preserved':{k:bool(np.array_equal(result[k][np.argsort(perm)],old[k]))for k in old if k!='NCE_positions'},'NCE_original_step_positions_unchanged':bool(np.array_equal(result['NCE_positions'],old['NCE_positions'])),'original_noise_style_reconstruction_hash_matches':True,'original_noise_stream_sha256':hn.hexdigest(),'original_style_stream_sha256':hs.hexdigest(),'full_white_arrays_saved':False,'in_memory_white_bytes':ww.nbytes,'same_rows_does_not_mean_same_perstep_batches':True}
    np.savez_compressed(root/'ROW_PERMUTATION_PRIVATE.npz',original_row_ordinal=perm.astype('<i4'),bin=nb,homogeneous_block=homogeneous)
    (root/'REORDER_CONSTRUCTION_RECEIPT.json').write_text(json.dumps(audit,indent=2)+'\n','utf8')
    return result
