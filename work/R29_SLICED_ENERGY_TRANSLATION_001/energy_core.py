from __future__ import annotations
import numpy as np, torch, json, pathlib, hashlib
from fine_core import FreshFineLoss, NAMES

def energy_u(a,b):
    if a.ndim!=2 or b.shape!=a.shape or len(a)<2:raise ValueError('energy U requires equal n-by-k arrays and n>=2')
    n=len(a);off=~torch.eye(n,device=a.device,dtype=torch.bool)
    # Both generated argument positions stay in the autograd graph.
    ab=(a[:,None,:]-b[None,:,:]).abs()[off].mean(dim=0)
    aa=(a[:,None,:]-a[None,:,:]).abs()[off].mean(dim=0)
    bb=(b[:,None,:]-b[None,:,:]).abs()[off].mean(dim=0)
    per_direction=2*ab-aa-bb
    return per_direction.mean(), {'twice_cross':2*ab.mean(),'generated_self':aa.mean(),'REAL_self':bb.mean(),
        'negative_direction_fraction':(per_direction<0).to(a.dtype).mean(),'min_direction_U':per_direction.min(),'max_direction_U':per_direction.max()}

class EnergyFineLoss(FreshFineLoss):
    def __init__(self,constants,device,seed,root,expected_stream):
        super().__init__(constants,device,seed,root)
        self.expected=json.loads(pathlib.Path(expected_stream).read_text('utf8'))
        assert self.initial_rng==self.expected['initial_state']
        self.energy_rows=[];self.context_rows=[]

    def training(self,y,real):
        arrays=[];rows=[]
        for name,d in zip(NAMES,[480,479,100]):
            raw=self.rng.normal(size=(d,128));unit=raw/np.linalg.norm(raw,axis=0,keepdims=True);p=unit.astype('f4');arrays.append(p)
            rows.append({'group':name,'shape':[d,128],'actual_float32_sha256':hashlib.sha256(p.tobytes()).hexdigest(),'float64_unit_error':float(np.max(np.abs(np.linalg.norm(unit,axis=0)-1))),'float32_unit_error':float(np.max(np.abs(np.linalg.norm(p.astype('f8'),axis=0)-1)))})
        ordinal=self.calls
        expected=self.expected['per_call_actual_float32_versions'][ordinal]
        for new,old in zip(rows,expected['matrices']):
            assert new['actual_float32_sha256']==old['actual_float32_sha256'] and new['shape']==old['shape'], 'fresh matrices differ from corresponding R28 ordinal'
        self.calls+=1;self.versions.append({'G_update_ordinal':self.calls,'matrices':rows,'same_draw_for_GENERATED_and_REAL':True,'same_R28_ordinal_matrices':True})
        self.last={n:p.copy() for n,p in zip(NAMES,arrays)}
        if self.first is None:
            self.first={n:p.copy() for n,p in zip(NAMES,arrays)};np.savez_compressed(self.root/'FRESH_PROJECTION_FIRST_PRIVATE.npz',**self.first)
        matrices=[torch.as_tensor(p,device=y.device,dtype=y.dtype) for p in arrays]
        a=self.fresh_projected(y,matrices);b=self.fresh_projected(real,matrices)
        values=[];records=[]
        for name,x,r in zip(NAMES,a,b):
            value,parts=energy_u(x,r);values.append(value)
            records.append({'group':name,'n':len(x),'projections':x.shape[1],**{k:float(v.detach()) for k,v in parts.items()},'U':float(value.detach()),'negative_U':bool(value.detach()<0)})
        mean=torch.stack(values).mean()
        self.energy_rows.append({'G_update_ordinal':self.calls,'groups':records,'equal_group_mean_U':float(mean.detach()),'negative_mean_U':bool(mean.detach()<0)})
        return mean,values

    def record_context(self,step,bins):
        assert step==self.calls
        values=np.asarray(bins).tolist();mixed=len(set(values))>1
        row={'G_update_ordinal':step,'mixed_remainder_batch':mixed,'bin_counts':{str(b+1):values.count(b) for b in sorted(set(values))}}
        self.context_rows.append(row)
        self.energy_rows[-1].update(row)
        if step%250==0:
            self.persist_energy()

    def summary(self):
        group={}
        for i,n in enumerate(NAMES):
            rows=[r['groups'][i] for r in self.energy_rows]
            group[n]={'updates':len(rows),'negative_U_count':sum(r['negative_U'] for r in rows),'negative_U_fraction':sum(r['negative_U'] for r in rows)/len(rows) if rows else None,'mean_negative_direction_fraction':float(np.mean([r['negative_direction_fraction'] for r in rows])) if rows else None,'mean_U':float(np.mean([r['U'] for r in rows])) if rows else None,'min_U':min((r['U'] for r in rows),default=None),'max_U':max((r['U'] for r in rows),default=None),'mean_twice_cross':float(np.mean([r['twice_cross'] for r in rows])) if rows else None,'mean_generated_self':float(np.mean([r['generated_self'] for r in rows])) if rows else None,'mean_REAL_self':float(np.mean([r['REAL_self'] for r in rows])) if rows else None}
        return {'groups':group,'all_three_diagonals_excluded':True,'two_generated_arguments_differentiable':True,'negative_U_preserved':True,'mean_negative_count':sum(r['negative_mean_U'] for r in self.energy_rows),'mean_negative_fraction':sum(r['negative_mean_U'] for r in self.energy_rows)/len(self.energy_rows) if self.energy_rows else None,'mixed_remainder_updates':[r for r in self.energy_rows if r.get('mixed_remainder_batch',False)],'actual_fresh_calls':self.calls,'same_R28_projection_versions_all_consumed_ordinals':True,'objective_scale_not_comparable_to_quantile_W1':True}

    def persist_energy(self):
        (self.root/'ENERGY_TERMS_PRIVATE.json').write_text(json.dumps({'formula':'mean_k sum_i!=j (2|Ai-Bj|-|Ai-Aj|-|Bi-Bj|)/(n(n-1))','updates':self.energy_rows,'summary':self.summary()},indent=2,allow_nan=False)+'\n','utf8')

    def persist(self):
        super().persist();self.persist_energy()
        if self.calls==self.expected['calls']:
            assert self.rng.bit_generator.state==self.expected['final_state']
