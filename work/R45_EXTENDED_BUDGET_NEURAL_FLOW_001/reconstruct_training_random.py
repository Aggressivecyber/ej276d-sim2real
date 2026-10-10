"""Streaming reproduction of R45's new Monte Carlo draws; no network or optimizer."""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def stream(initial_t_state):
    rt=np.random.Generator(np.random.PCG64());rt.bit_generator.state=initial_t_state
    rw=np.random.Generator(np.random.PCG64(20261010452))
    for global_step in range(2951,50001):
        t=rt.uniform(0,1,32).astype('f4')
        white=rw.standard_normal((32,480)).astype('f4')
        yield global_step,(global_step-1)%2950,t,white

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--global-step',type=int,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 2951<=args.global_step<=50000:raise ValueError('new updates are2951..50000')
    rec=json.loads((ROOT/'EXTENDED_FLOW_FINAL/RANDOM_STATES.json').read_text('utf8'))
    expected=np.load(ROOT/'EXTENDED_FLOW_FINAL/BATCH_HASHES_PRIVATE.npy',mmap_mode='r',allow_pickle=False)
    for k,b,t,w in stream(rec['training_t_initial']):
        if k!=args.global_step:continue
        th=hashlib.sha256(t.tobytes()).hexdigest();wh=hashlib.sha256(w.tobytes()).hexdigest()
        e=expected[k-2951]
        names=expected.dtype.names
        # Saved structured field names are authoritative and disclosed below.
        result={'global_step':k,'original_batch0':b,'t_sha256':th,'white_sha256':wh,
                'NumPy':np.__version__,'training_NumPy':'1.26.4','saved_hash_fields':list(names),
                'saved_hash_record':{name:(e[name].decode() if e.dtype[name].kind=='S' else int(e[name])) for name in names},
                'new_response_forward_update':0,'draws_recreated_through_requested_step':k-2950}
        result['same_saved_training_draw']=bool(int(e['global_step'])==k and int(e['original_batch0'])==b
            and e['t_sha256'].decode()==th and e['white_sha256'].decode()==wh)
        if args.output:
            if args.output.drive.upper() not in ['E:','D:'] and not str(args.output).startswith('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/'):
                raise ValueError('data output must be E/D or this project server directory')
            args.output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.output,t=t,white=w,global_step=np.array(k),original_batch0=np.array(b))
        print(json.dumps(result));return

if __name__=='__main__':main()
