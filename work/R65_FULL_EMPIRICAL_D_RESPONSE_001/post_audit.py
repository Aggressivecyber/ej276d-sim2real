"""R65 audit adapter; inherited constraint description is byte-preserved."""
import numpy as np
import evaluation as ev
from r65_training_audit import audit_saved_records

def constraint_description(d):
    n = len(d['constraint_projection_range_active'])
    fields = ['pre_constraint_out_of_range_sample_count','constraint_projection_RMS_V',
        'constraint_projection_maxabs_V','constraint_projection_rail_sample_count',
        'pre_constraint_charge_relative_error']
    def describe(use):
        return {'N':int(use.sum()),
            'range_active_events':int(d['constraint_projection_range_active'][use].sum()),
            'rail_sample_total':int(d['constraint_projection_rail_sample_count'][use].sum()),
            'fields':{k:ev.summary(d[k][use]) for k in fields}}
    result = describe(np.ones(n, bool))
    result.update(internal_model_operation='Euclidean box-charge projection after constant-field Heun',
        zero_ADC_clip_count_does_not_mean_zero_model_projection=True,
        equal_original_parent_weights=True, pass_claim=False,
        inside_TRAIN_support=describe(d['inside_TRAIN_support']),
        outside_TRAIN_support=describe(~d['inside_TRAIN_support']),
        by_original_charge_region=[dict(region=int(b),**describe(d['source_Qbin_actual_preADC']==b)) for b in range(-1,9)])
    return result


def training_audit(root, cfg, receipt):
    return audit_saved_records(root, cfg, receipt)
