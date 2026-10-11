"""Supplementary cross-device ADC representation audit on saved TRAIN preADC."""
from pathlib import Path
import json,os,sys,time,resource
R=Path(__file__).resolve().parent
C=json.loads((R/'config.json').read_text());S=Path(C['R71_root'])
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory)
sys.path.insert(0,str(S))
from physical_operators import canonical_adc_torch
start=time.monotonic();result={}
for domain in ['SOURCE','REAL']:
    with np.load(S/('LEARNED_'+domain+'_PRIVATE.npz'),allow_pickle=False)as z:pre=z['pre_ADC_V'][:64].copy()
    v=pre.astype('f4')
    legacy=(-np.float32(.5)+np.rint((np.clip(v,-.5,.5)+np.float32(.5))*np.float32(16383))/np.float32(16383)).astype('f4')
    independent=-.5+np.clip(np.rint((legacy.astype('f8')+.5)*16383),0,16383)/16383
    observed=canonical_adc_torch(torch.tensor(pre,device='cuda')).cpu().numpy()
    codesA=np.rint((independent+.5)*16383).astype('i8');codesB=np.rint((observed+.5)*16383).astype('i8')
    result[domain]=dict(N=64,samples=64*480,float64_voltage_arrays_bitwise=bool(np.array_equal(independent,observed)),
        different_float64_voltage_values=int(np.count_nonzero(independent!=observed)),
        max_voltage_abs_difference=float(np.max(np.abs(independent-observed))),
        max_voltage_difference_LSB=float(np.max(np.abs(independent-observed))*16383),
        ADC_integer_codes_bitwise=bool(np.array_equal(codesA,codesB)),different_integer_codes=int(np.count_nonzero(codesA!=codesB)))
    assert result[domain]['ADC_integer_codes_bitwise']
answer=dict(purpose='Interpret original ADC_code_equal field which compared float64 voltages, not integer ADC indices',
    original_main_results_unchanged=True,model_forward_rows=0,new_training_updates=0,quantizer_only_rows=128,
    primary_preADC_numpy_reference_already_bitwise=True,results=result,elapsed_s=time.monotonic()-start,
    allocated_GPU_peak_bytes=torch.cuda.max_memory_allocated(),reserved_GPU_peak_bytes=torch.cuda.max_memory_reserved())
(R/'ADC_REPRESENTATION_AUDIT.json').write_text(json.dumps(answer,indent=2)+'\n')
print(json.dumps(answer),flush=True)
