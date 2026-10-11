"""Frozen common canonical representation; no model dependency."""
import numpy as np

def canonical(v):
    codes = np.clip(np.rint((np.asarray(v, dtype='f8') + .5) * 16383), 0, 16383)
    return -.5 + codes / 16383.
