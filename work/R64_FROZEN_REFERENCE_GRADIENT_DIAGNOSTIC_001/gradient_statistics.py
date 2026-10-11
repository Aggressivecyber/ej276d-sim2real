"""Finite-reference gradient descriptions using raw, unclipped vectors.

Pure NumPy; no RNG, fitting, file access, scientific execution or optimization.
The squared-error average is over 32 macros, NOT over vector coordinates.
The same caller-supplied 1000x32 paired bootstrap indices serve all five
components and their vector sum. No population bias/unbiasedness is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

D_NAMES = ('full_window', 'PSD', 'noise', 'coordinate_RMS', 'temporal_joint')
ALL_NAMES = D_NAMES + ('total_D',)
MACROS = 32
BOOTSTRAPS = 1000
DEFAULT_CHUNK = 65536
DECOMPOSITION_ATOL = 1e-10
DECOMPOSITION_RTOL = 5e-11


def _array(value, name, shape):
    if not isinstance(value, np.ndarray) or value.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise TypeError(name + ': a NumPy float32/float64 array or memmap is required')
    if value.shape != shape:
        raise ValueError(name + ': expected shape ' + str(shape) + ', got ' + str(value.shape))
    return value


def _finite_chunks(value, chunk, name):
    for k in range(0, value.shape[-1], chunk):
        if not np.isfinite(value[..., k:k+chunk]).all():
            raise ValueError(name + ': nonfinite coordinate; no update accepted')


def _bootstrap_indices(value):
    if not isinstance(value, np.ndarray) or value.shape != (BOOTSTRAPS, MACROS):
        raise ValueError('bootstrap_indices must be a fixed [1000,32] NumPy integer array')
    if value.dtype.kind not in 'iu' or (value < 0).any() or (value >= MACROS).any():
        raise ValueError('bootstrap indices must be integers in [0,31]')
    result = np.array(value, dtype='<i8', order='C', copy=True)
    result.flags.writeable = False
    return result


def _norm_and_cos(norm_sq, reference_sq, dot):
    # No clipping or replacement by a reference norm: all are raw coordinates.
    if norm_sq < 0 or reference_sq < 0:
        raise ArithmeticError('Negative squared norm')
    norm, refnorm = math.sqrt(norm_sq), math.sqrt(reference_sq)
    cosine = None if norm == 0 or refnorm == 0 else (dot / norm) / refnorm
    if cosine is not None and not math.isfinite(cosine):
        raise FloatingPointError('Cosine overflow')
    return norm, cosine


def _relative(numerator, denominator):
    if denominator == 0:
        return None
    result = numerator / denominator
    if not math.isfinite(result):
        raise FloatingPointError('Relative squared error overflow')
    return result


def _sum6(parts):
    result = np.array([math.fsum(float(p[j]) for p in parts) for j in range(6)], dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError('Float64 scalar reduction overflow')
    return result


def _dot_rows(a, b):
    with np.errstate(over='raise', invalid='raise'):
        return np.einsum('ij,ij->i', a, b, optimize=False)


def _six_rows(value, start, stop):
    # Only one coordinate chunk is copied; never materialize [32,5,P] scratch.
    result = np.empty((6, stop-start), dtype=np.float64)
    result[:5] = value[:, start:stop]
    result[5] = np.sum(result[:5], axis=0, dtype=np.float64)
    return result


def _metric_record(norm_sq, reference_sq, dot, squared_error):
    norm, cosine = _norm_and_cos(norm_sq, reference_sq, dot)
    return {
        'squared_norm': float(norm_sq), 'norm': norm,
        'dot_reference': float(dot), 'cosine_reference': cosine,
        'squared_l2_error': float(squared_error),
        'relative_squared_l2_error': _relative(squared_error, reference_sq),
    }


@dataclass(frozen=True)
class StatisticsResult:
    """summary is JSON-compatible; arrays are read-only views, without copies.

    mean_A/mean_B have shape [6,P] in ALL_NAMES order, including the vector sum.
    bootstrap_error_changes has shape [1000,6] using the same paired index rows.
    Caller owns persistence. The module never writes arrays or summary files.
    """
    summary: dict
    mean_A: np.ndarray
    mean_B: np.ndarray
    bootstrap_error_changes: np.ndarray


class GradientAccumulator:
    """Streaming 32 macro pairs; retain two mean vectors plus scalar records.

    reference: [5,P]. update(A_macro,B_macro): [5,P] each, in D_NAMES order.
    The five components' VECTOR SUM is independently measured as total_D.
    reference and means use [6,P] float64 internally. Persistent numerical
    storage is approximately 144*P bytes plus small scalar/bootstrap records;
    input arrays belong to the caller. This supports G_Y flattened to [5,N*480]
    without retaining 32 endpoint macro vectors. A context-specific label keeps
    endpoint and parameter statistics separate in the output.

    Welford's M2 uses denominator M=32, not M-1. Its separately accumulated
    dispersion checks the finite-sample squared-error decomposition. All
    thresholds below are fixed numerical checks, not scientific decision rules.
    """
    def __init__(self, reference, bootstrap_indices, *, coordinate_label='parameter',
                 feature_chunk=DEFAULT_CHUNK):
        if not isinstance(reference, np.ndarray) or reference.ndim != 2 or reference.shape[0] != 5 or reference.shape[1] < 1:
            raise ValueError('reference must be [5,P] with P>=1')
        if type(feature_chunk) is not int or feature_chunk < 1:
            raise ValueError('feature_chunk must be a positive integer')
        if not isinstance(coordinate_label, str) or not coordinate_label:
            raise ValueError('coordinate_label must be a nonempty string')
        self.dimension = int(reference.shape[1])
        self.feature_chunk = feature_chunk
        self.coordinate_label = coordinate_label
        self.bootstrap_indices = _bootstrap_indices(bootstrap_indices)
        reference = _array(reference, 'reference', (5,self.dimension))
        _finite_chunks(reference, feature_chunk, 'reference')
        self.reference = np.empty((6,self.dimension), dtype=np.float64)
        norm_parts = []
        for start, stop in self._chunks():
            block = _six_rows(reference,start,stop)
            self.reference[:,start:stop] = block
            norm_parts.append(_dot_rows(block,block))
        self.reference_sq = _sum6(norm_parts)
        for j in range(6):
            if self.reference_sq[j] == 0 and np.any(self.reference[j] != 0):
                raise FloatingPointError('Nonzero reference squared norm underflows float64')
        self.reference.flags.writeable = False
        self.means = np.zeros((2,6,self.dimension), dtype=np.float64)
        self.m2 = np.zeros((2,6), dtype=np.float64)
        self.records = []
        self._squared_errors = np.empty((MACROS,2,6), dtype=np.float64)
        self._count = 0
        self._failed = False
        self._finished = False
        self.dispersion_method = 'Welford online M2 / 32'
        self.dispersion_arithmetic = None

    def _chunks(self):
        return ((start,min(start+self.feature_chunk,self.dimension)) for start in range(0,self.dimension,self.feature_chunk))

    @property
    def count(self):
        return self._count

    def update(self, A_macro, B_macro):
        """Consume one pair in the externally frozen macro order.

        Returns a small JSON-compatible scalar record. Inputs are not retained.
        Invalid shapes/nonfinite inputs fail before mutation. An arithmetic
        failure after mutation marks this accumulator unusable (fail closed).
        """
        if self._failed or self._finished:
            raise RuntimeError('Accumulator is failed or finalized')
        if self._count >= MACROS:
            raise ValueError('Exactly 32 macro pairs are allowed')
        A_macro = _array(A_macro,'A_macro',(5,self.dimension))
        B_macro = _array(B_macro,'B_macro',(5,self.dimension))
        _finite_chunks(A_macro,self.feature_chunk,'A_macro')
        _finite_chunks(B_macro,self.feature_chunk,'B_macro')
        n = self._count + 1
        arm_norm, arm_dot, arm_error, arm_m2 = [[],[]], [[],[]], [[],[]], [[],[]]
        cross_dot, cross_error = [], []
        try:
            for start,stop in self._chunks():
                aa = _six_rows(A_macro,start,stop)
                bb = _six_rows(B_macro,start,stop)
                rr = self.reference[:,start:stop]
                for arm, gg in enumerate((aa,bb)):
                    arm_norm[arm].append(_dot_rows(gg,gg))
                    arm_dot[arm].append(_dot_rows(gg,rr))
                    difference = gg - rr
                    arm_error[arm].append(_dot_rows(difference,difference))
                    difference = gg - self.means[arm,:,start:stop]
                    # Welford M2 increment uses (n-1)/n * ||g-old_mean||^2.
                    arm_m2[arm].append(((n-1)/n) * _dot_rows(difference,difference))
                    self.means[arm,:,start:stop] += difference / n
                cross_dot.append(_dot_rows(aa,bb))
                difference = aa-bb
                cross_error.append(_dot_rows(difference,difference))
            norms = np.stack([_sum6(p) for p in arm_norm])
            dots = np.stack([_sum6(p) for p in arm_dot])
            errors = np.stack([_sum6(p) for p in arm_error])
            self.m2 += np.stack([_sum6(p) for p in arm_m2])
            abdot, aberror = _sum6(cross_dot), _sum6(cross_error)
            record = {'macro_index':self._count,'A':{},'B':{},'paired':{}}
            for j,name in enumerate(ALL_NAMES):
                for arm,label in enumerate(('A','B')):
                    record[label][name] = _metric_record(norms[arm,j],self.reference_sq[j],dots[arm,j],errors[arm,j])
                _,abcos = _norm_and_cos(norms[0,j],norms[1,j],abdot[j])
                record['paired'][name] = {
                    'B_minus_A_squared_l2_error':float(errors[1,j]-errors[0,j]),
                    'dot_A_B':float(abdot[j]),'cosine_A_B':abcos,
                    'squared_l2_A_minus_B':float(aberror[j]),
                }
            self._squared_errors[self._count] = errors
            self.records.append(record)
            self._count = n
            return record
        except BaseException:
            self._failed = True
            raise

    def finalize(self):
        """Return paired bootstrap, exact finite-sample check, means and records."""
        if self._failed or self._finished:
            raise RuntimeError('Accumulator is failed or already finalized')
        if self._count != MACROS:
            raise ValueError('Require exactly 32 macro pairs before finalizing')
        output = {
            'coordinate_label':self.coordinate_label,'component_order':list(D_NAMES),
            'reported_order':list(ALL_NAMES),'macros':MACROS,'vector_dimension':self.dimension,
            'squared_error_convention':'mean over macros of squared Euclidean vector error; no division by vector dimension',
            'raw_unclipped_gradients':True,'feature_chunk':self.feature_chunk,
            'dispersion_method':self.dispersion_method,'dispersion_arithmetic':self.dispersion_arithmetic,
            'reference':{},'arms':{'A':{},'B':{}},'paired':{},'macro_records':self.records,
        }
        mean_norm_parts, mean_dot_parts, mean_error_parts = [[],[]], [[],[]], [[],[]]
        for start,stop in self._chunks():
            rr = self.reference[:,start:stop]
            for arm in range(2):
                mean = self.means[arm,:,start:stop]
                mean_norm_parts[arm].append(_dot_rows(mean,mean))
                mean_dot_parts[arm].append(_dot_rows(mean,rr))
                delta = mean-rr
                mean_error_parts[arm].append(_dot_rows(delta,delta))
        mean_norm = np.stack([_sum6(p) for p in mean_norm_parts])
        mean_dot = np.stack([_sum6(p) for p in mean_dot_parts])
        mean_error = np.stack([_sum6(p) for p in mean_error_parts])
        checks = []
        for j,name in enumerate(ALL_NAMES):
            output['reference'][name] = {'norm':math.sqrt(self.reference_sq[j]),'squared_norm':float(self.reference_sq[j])}
            for arm,label in enumerate(('A','B')):
                items = [record[label][name] for record in self.records]
                mse = math.fsum(item['squared_l2_error'] for item in items)/MACROS
                dispersion = float(self.m2[arm,j]/MACROS)
                discrepancy = float(mean_error[arm,j])
                residual = mse-(discrepancy+dispersion)
                tolerance = DECOMPOSITION_ATOL + DECOMPOSITION_RTOL*max(abs(mse),abs(discrepancy+dispersion))
                passed = abs(residual) <= tolerance
                check = {'mean_squared_l2_error':mse,'mean_gradient_discrepancy_squared_l2':discrepancy,
                         'empirical_dispersion_mean_squared_l2':dispersion,'residual':residual,
                         'absolute_residual':abs(residual),'allowed_error':tolerance,'passed':passed}
                checks.append(passed)
                mean_metrics = _metric_record(mean_norm[arm,j],self.reference_sq[j],mean_dot[arm,j],discrepancy)
                cosines = [item['cosine_reference'] for item in items]
                output['arms'][label][name] = {
                    'mean_squared_l2_error':mse,'relative_mean_squared_l2_error':_relative(mse,self.reference_sq[j]),
                    'mean_norm':math.fsum(item['norm'] for item in items)/MACROS,
                    'mean_squared_norm':math.fsum(item['squared_norm'] for item in items)/MACROS,
                    'mean_dot_reference':math.fsum(item['dot_reference'] for item in items)/MACROS,
                    'mean_cosine_reference':None if any(c is None for c in cosines) else math.fsum(cosines)/MACROS,
                    'defined_cosine_count':sum(c is not None for c in cosines),
                    'mean_gradient':mean_metrics,'empirical_dispersion_mean_squared_l2':dispersion,
                    'finite_sample_decomposition':check,
                }
        if not all(checks):
            self._failed = True
            raise ArithmeticError('Finite-sample decomposition failed frozen numerical tolerance')
        changes = self._squared_errors[:,1,:]-self._squared_errors[:,0,:]
        # This is small: [1000,32,6] (1.54 MB), independent of parameter count.
        bootstrap = changes[self.bootstrap_indices].mean(axis=1,dtype=np.float64)
        intervals = np.quantile(bootstrap,[.025,.975],axis=0,method='linear')
        for j,name in enumerate(ALL_NAMES):
            output['paired'][name] = {
                'mean_B_minus_A_squared_l2_error':math.fsum(float(z) for z in changes[:,j])/MACROS,
                'bootstrap_percentile_95_interval':[float(intervals[0,j]),float(intervals[1,j])],
                'bootstrap_replicates':BOOTSTRAPS,'bootstrap_unit':'macro pair',
                'resampling':'same caller-supplied index row for A/B and all components/total',
                'primary':name=='total_D','multiple_testing_adjusted':False,
            }
        output['bootstrap_indices_sha256'] = hashlib.sha256(self.bootstrap_indices.tobytes()).hexdigest()
        output['bootstrap_quantile_method'] = 'linear'
        output['all_finite_sample_decompositions_passed'] = True
        output['numerical_decomposition_tolerance'] = {'atol':DECOMPOSITION_ATOL,'rtol':DECOMPOSITION_RTOL}
        output['interpretation'] = [
            'Finite empirical mean discrepancy and dispersion conditional on the supplied fixed reference; not population bias or an unbiased bias/variance estimate.',
            'A negative paired interval only describes smaller squared gradient error for this fixed diagnostic; not better training, nine-term descent, A-F performance or a selected next method.',
            'Zero-reference and zero-gradient cases retain absolute quantities; undefined relative errors and cosines are null. Mean cosine is null if any macro cosine is undefined; no macro is omitted.',
            'Total D is the vector sum before norms/errors; it is not the sum or mean of component squared errors.',
        ]
        self._finished = True
        means_a,means_b = self.means[0].view(),self.means[1].view()
        means_a.flags.writeable=False;means_b.flags.writeable=False;bootstrap.flags.writeable=False
        return StatisticsResult(output,means_a,means_b,bootstrap)


def summarize_gradients(reference,A,B,bootstrap_indices,*,coordinate_label='parameter',feature_chunk=DEFAULT_CHUNK):
    """Convenience wrapper for [5,P], [32,5,P], [32,5,P]; no giant copy.

    Array/memmap inputs are traversed one macro at a time, and float32-to-float64
    conversion occurs only in coordinate chunks. Use GradientAccumulator.update
    directly when endpoint macro vectors should be discarded immediately.
    """
    accumulator = GradientAccumulator(reference,bootstrap_indices,coordinate_label=coordinate_label,feature_chunk=feature_chunk)
    _array(A,'A',(MACROS,5,accumulator.dimension))
    _array(B,'B',(MACROS,5,accumulator.dimension))
    for macro in range(MACROS):
        accumulator.update(A[macro],B[macro])
    return accumulator.finalize()


def macro_gradient_record(reference,A_macro,B_macro,*,feature_chunk=DEFAULT_CHUNK):
    """Pure small sufficient-statistic record for one macro, retaining no vector.

    All three arrays are [5,P]. The returned nine lists have six values in
    ALL_NAMES order. Keeping these 32 records plus sum_A/sum_B [5,P] permits
    summarize_sufficient_statistics without storing any endpoint macro vectors.
    This routine neither updates sums nor draws bootstrap indices.
    """
    if not isinstance(reference,np.ndarray) or reference.ndim!=2 or reference.shape[0]!=5 or reference.shape[1]<1:
        raise ValueError('reference must be [5,P] with P>=1')
    if type(feature_chunk) is not int or feature_chunk<1:
        raise ValueError('feature_chunk must be a positive integer')
    dim=reference.shape[1]
    arrays=tuple(_array(v,name,(5,dim)) for name,v in (('reference',reference),('A_macro',A_macro),('B_macro',B_macro)))
    for name,value in zip(('reference','A_macro','B_macro'),arrays):
        _finite_chunks(value,feature_chunk,name)
    parts={key:[] for key in ('reference_squared_norm','A_squared_norm','B_squared_norm',
                              'A_dot_reference','B_dot_reference','A_squared_l2_error','B_squared_l2_error',
                              'dot_A_B','squared_l2_A_minus_B')}
    for start in range(0,dim,feature_chunk):
        stop=min(start+feature_chunk,dim)
        rr,aa,bb=(_six_rows(value,start,stop) for value in arrays)
        parts['reference_squared_norm'].append(_dot_rows(rr,rr))
        for label,gg in (('A',aa),('B',bb)):
            parts[label+'_squared_norm'].append(_dot_rows(gg,gg))
            parts[label+'_dot_reference'].append(_dot_rows(gg,rr))
            difference=gg-rr
            parts[label+'_squared_l2_error'].append(_dot_rows(difference,difference))
        parts['dot_A_B'].append(_dot_rows(aa,bb))
        difference=aa-bb
        parts['squared_l2_A_minus_B'].append(_dot_rows(difference,difference))
    result={key:_sum6(values).tolist() for key,values in parts.items()}
    result.update(schema='R64_MACRO_GRADIENT_SCALARS_V1',reported_order=list(ALL_NAMES))
    return result


def summarize_sufficient_statistics(reference,sum_A,sum_B,records,bootstrap_indices,*,
                                    coordinate_label='endpoint',feature_chunk=DEFAULT_CHUNK):
    """Summarize original reference + online sums + 32 small macro records.

    reference/sum_A/sum_B: [5,P] f64/f32 arrays; sums are the raw sums of 32
    corresponding macro component vectors. For scientific use accumulate sums
    in f64. records are exactly the dictionaries from macro_gradient_record.

    Dispersion = mean squared norm - squared norm of empirical mean, denominator
    32. This is the finite second-moment identity. Cancellation is exposed; a
    tiny negative within fixed arithmetic tolerance is reported and set to zero,
    never silently treated as negative dispersion. A larger negative or failed
    independent error-decomposition check raises. If numerical cancellation is
    unacceptable, the streaming GradientAccumulator computes Welford M2 using
    the same mean-vector storage order; no additional macro-vector archive is
    needed. No statistical definition or CI sampling unit changes.
    """
    accumulator=GradientAccumulator(reference,bootstrap_indices,coordinate_label=coordinate_label,feature_chunk=feature_chunk)
    sum_A=_array(sum_A,'sum_A',(5,accumulator.dimension))
    sum_B=_array(sum_B,'sum_B',(5,accumulator.dimension))
    _finite_chunks(sum_A,feature_chunk,'sum_A');_finite_chunks(sum_B,feature_chunk,'sum_B')
    if not isinstance(records,(list,tuple)) or len(records)!=MACROS:
        raise ValueError('records must contain exactly 32 macro_gradient_record dictionaries')
    keys=('reference_squared_norm','A_squared_norm','B_squared_norm','A_dot_reference','B_dot_reference',
          'A_squared_l2_error','B_squared_l2_error','dot_A_B','squared_l2_A_minus_B')
    for macro,record in enumerate(records):
        if not isinstance(record,dict) or record.get('schema')!='R64_MACRO_GRADIENT_SCALARS_V1' or tuple(record.get('reported_order',()))!=ALL_NAMES:
            raise ValueError('Invalid scalar record schema/order')
        data={}
        for key in keys:
            value=np.asarray(record.get(key),dtype=np.float64)
            if value.shape!=(6,) or not np.isfinite(value).all():
                raise ValueError('Invalid scalar record field '+key)
            if ('squared' in key) and (value<0).any():
                raise ValueError('Negative squared quantity in scalar record '+key)
            data[key]=value
        ref_error=np.abs(data['reference_squared_norm']-accumulator.reference_sq)
        ref_bound=DECOMPOSITION_ATOL+DECOMPOSITION_RTOL*np.maximum(np.abs(data['reference_squared_norm']),np.abs(accumulator.reference_sq))
        if not (ref_error<=ref_bound).all():
            raise ValueError('Scalar records do not match the supplied fixed reference norms')
        expanded={'macro_index':macro,'A':{},'B':{},'paired':{}}
        for j,name in enumerate(ALL_NAMES):
            for arm,label in enumerate(('A','B')):
                expanded[label][name]=_metric_record(data[label+'_squared_norm'][j],accumulator.reference_sq[j],
                    data[label+'_dot_reference'][j],data[label+'_squared_l2_error'][j])
                accumulator._squared_errors[macro,arm,j]=data[label+'_squared_l2_error'][j]
            _,cosine=_norm_and_cos(data['A_squared_norm'][j],data['B_squared_norm'][j],data['dot_A_B'][j])
            expanded['paired'][name]={'B_minus_A_squared_l2_error':float(data['B_squared_l2_error'][j]-data['A_squared_l2_error'][j]),
                'dot_A_B':float(data['dot_A_B'][j]),'cosine_A_B':cosine,'squared_l2_A_minus_B':float(data['squared_l2_A_minus_B'][j])}
        accumulator.records.append(expanded)
    for start,stop in accumulator._chunks():
        for arm,total in enumerate((sum_A,sum_B)):
            accumulator.means[arm,:,start:stop]=_six_rows(total,start,stop)/MACROS
    diagnostics=[]
    for arm,label in enumerate(('A','B')):
        norm_parts=[]
        for start,stop in accumulator._chunks():
            block=accumulator.means[arm,:,start:stop]
            norm_parts.append(_dot_rows(block,block))
        mean_norm=_sum6(norm_parts)
        for j,name in enumerate(ALL_NAMES):
            second=math.fsum(row[label][name]['squared_norm'] for row in accumulator.records)/MACROS
            raw=second-float(mean_norm[j])
            bound=DECOMPOSITION_ATOL+DECOMPOSITION_RTOL*max(abs(second),abs(float(mean_norm[j])))
            if raw < -bound:
                raise ArithmeticError('Negative sufficient-statistic dispersion beyond frozen tolerance')
            dispersion=max(0.,raw)
            accumulator.m2[arm,j]=dispersion*MACROS
            diagnostics.append({'arm':label,'term':name,'raw_second_moment_difference':raw,
                                'absolute_roundoff_bound':bound,'negative_roundoff_clamped_to_zero':raw<0})
    accumulator.dispersion_method='finite second moment from online sums and scalar records; denominator 32'
    accumulator.dispersion_arithmetic=diagnostics
    accumulator._count=MACROS
    return accumulator.finalize()
