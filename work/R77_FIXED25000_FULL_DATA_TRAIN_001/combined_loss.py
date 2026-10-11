"""Same nine R72 laws: share the full-window and coordinate-RMS marginal graph."""
import math
import torch
from full_waveform_loss import FullWaveformLoss as R72FullWaveformLoss, TERM_NAMES, VIEWS, feature_vectors
from exact_uniform_w1 import UniformW1Cache


class FullWaveformLoss(R72FullWaveformLoss):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.w1=UniformW1Cache(self.device)

    def _shared_full_and_coordinate(self,context):
        values,bins,counts,directions,_,_=context
        # Reference projections are constants. SOURCE and REAL use the same
        # direction tensors and frozen references, so each is sorted only once.
        projected_reference={}
        domain_full=[];domain_coordinate=[]
        for v,b,c in zip(values[:2],bins,counts):
            full=v.new_zeros(());coordinate=v.new_zeros(())
            for name,features,direction in zip(VIEWS,feature_vectors(v),directions):
                feature_full=v.new_zeros(());feature_coordinate=v.new_zeros(())
                for slot,n in enumerate(c):
                    if not n:continue
                    a=(features[b==slot-1]-self.stats[name+'_center'][slot])/self.stats[name+'_scale'][slot]
                    marginal=self.w1.columns(a,self._reference_sorted[name][slot],b_sorted=True,check=False)
                    key=(name,slot)
                    if key not in projected_reference:
                        projected_reference[key]=torch.sort(self._reference_features[name][slot]@direction,dim=0,stable=True).values
                    sliced=self.w1.columns(a@direction,projected_reference[key],b_sorted=True,check=False).mean()
                    regional_full=.5*(marginal.mean()+sliced)
                    regional_coordinate=torch.linalg.vector_norm(marginal)/math.sqrt(features.shape[1])
                    feature_full=feature_full+(n/len(v))*regional_full
                    feature_coordinate=feature_coordinate+(n/len(v))*regional_coordinate
                full=full+feature_full;coordinate=coordinate+feature_coordinate
            domain_full.append(full/len(VIEWS));domain_coordinate.append(coordinate/len(VIEWS))
        return .5*sum(domain_full),.5*sum(domain_coordinate)

    def iter_gradient_groups(self,source_canonical,real_processed_canonical,source_preADC,REAL_preADC,source_raw,REAL_raw,directions4,temporal_directions):
        context=self._context(source_canonical,real_processed_canonical,source_preADC,REAL_preADC,source_raw,REAL_raw,directions4,temporal_directions)
        full,coordinate=self._shared_full_and_coordinate(context)
        pair={'full_window':full,'coordinate_RMS':coordinate}
        if not all(bool(torch.isfinite(v))for v in pair.values()):raise FloatingPointError('nonfinite shared full/coordinate law')
        yield tuple(pair),full+coordinate,pair,context[-1]
        for name in TERM_NAMES[2:]:
            value=self._compute_term(name,context)
            if not bool(torch.isfinite(value)):raise FloatingPointError('nonfinite '+name)
            yield (name,),value,{name:value},context[-1]

