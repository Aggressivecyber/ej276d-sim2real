from __future__ import annotations
import torch, numpy as np, hashlib
from model import OriginalTranslator

def check_initial(model,reader,critic,xs,xt,actual,cfg,root):
    device=xs.device
    before_cpu=torch.get_rng_state().clone()
    before_cuda=torch.cuda.get_rng_state_all()if torch.cuda.is_available()else []
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        old=OriginalTranslator().to(device)
        old.load_state_dict(torch.load(cfg['G_initial'],map_location=device,weights_only=True))
    assert torch.equal(before_cpu,torch.get_rng_state())
    assert all(torch.equal(a,b)for a,b in zip(before_cuda,torch.cuda.get_rng_state_all()if torch.cuda.is_available()else []))
    assert all(torch.equal(model.state_dict()[k],v)for k,v in old.state_dict().items())
    parameter_names=list(model.state_dict())
    new_names=sorted(set(parameter_names)-set(old.state_dict()))
    assert new_names==sorted(['time_mixer.0.weight','time_mixer.0.bias','time_mixer.2.weight','time_mixer.2.bias'])
    torch.save({k:v.detach().cpu().clone()for k,v in model.state_dict().items()},root/'INITIAL_FULL_TRANSLATOR.pt')
    torch.save({k:v.detach().cpu().clone()for k,v in model.time_mixer.state_dict().items()},root/'INITIAL_TIME_MIXER.pt')
    torch.save({k:v.detach().cpu().clone()for k,v in reader.state_dict().items()},root/'INITIAL_STYLE_READER.pt')
    torch.save({k:v.detach().cpu().clone()for k,v in critic.state_dict().items()},root/'INITIAL_CRITIC.pt')
    ordinal=actual['original_row_ordinal'][:64]
    arrays=[actual['original_noise']['z'][ordinal],actual['original_noise']['white'][ordinal],actual['original_style'][ordinal]]
    random=[torch.from_numpy(np.asarray(a,dtype='f4')).to(device)for a in arrays]
    rows={}
    with torch.no_grad():
        for name,pool,index in [('SOURCE',xs,actual['source'][:64]),('REAL',xt,actual['target'][:64])]:
            x=pool[index.astype('i8')]
            a=old(x,*random);b=model(x,*random)
            equal={n:bool(torch.equal(u,v))for n,u,v in zip(['preADC','signal','electronic'],a,b)}
            delta_zero=bool(torch.count_nonzero(model.last_time_mixer_output)==0)
            assert all(equal.values())and delta_zero,(name,equal,delta_zero)
            rows[name]={'rows':64,'all_components_bitwise_equal':equal,'mixer_delta_exact_zero':delta_zero,
                'old_and_new_component_SHA256':{n:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()for n,v in zip(['preADC','signal','electronic'],b)}}
    del old
    return {'extra_model_forward_rows':256,'selected_before_any_update':True,'first_actual_SOURCE_REAL_64_rows':True,
        'source_target_ordinals_sha256':{k:hashlib.sha256(actual[k][:64].tobytes()).hexdigest()for k in ['source','target','original_row_ordinal']},
        'original_torch_CPU_and_CUDA_RNG_preserved':True,'new_parameter_names':new_names,'new_parameters':123488,
        'rows':rows,'initial_W2_exact_zero':True,'old_parameters_exact_equal':True}
