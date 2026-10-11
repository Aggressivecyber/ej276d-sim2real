"""Import only byte-bound, definition-only R63 source copies, never its runner."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def _load(name, dependencies):
    path = ROOT / 'r63_original' / (name + '.py')
    bindings = json.loads((ROOT / 'SOURCE_BINDINGS.json').read_text('utf-8-sig'))
    expected = next(x['sha256'] for x in bindings if x['file'] == path.name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Original source binding mismatch: ' + name)
    unique = '_r64_preparation_original_' + name
    spec = importlib.util.spec_from_file_location(unique, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    old = {key: sys.modules.get(key) for key in dependencies}
    try:
        sys.modules.update(dependencies)
        spec.loader.exec_module(module)
    finally:
        for key, value in old.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


network = _load('network', {})
noise = _load('noise_core', {})
core = _load('source_flow_core', {'network': network, 'noise_core': noise})
precision = _load('precision_order', {'noise_core': noise})
old_loss = _load('R61_distribution_loss', {'precision_order': precision, 'source_flow_core': core})
loss = _load('distribution_loss', {'R61_distribution_loss': old_loss, 'source_flow_core': core})
R63 = SimpleNamespace(core=core, noise=noise, network=network, precision=precision,
                      old_loss=old_loss, loss=loss)
