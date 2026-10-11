"""Record root's completed visual inspection of all original and final figures."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import struct

R = Path(__file__).resolve().parents[1]
S = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << 2)
names = ['01_endpoint_geometry.png', '02_gradient_comparison.png', '03_paired_error_intervals.png', '04_charge_and_waveform.png']
comments = [
    'All three geometric comparisons, 32-pair ranges, same-treated REAL and five-view heatmap opened; labels, log units and legend readable, no clipping.',
    'Both gradient coordinate spaces, all five terms and total, both arms and cosine panels opened; labels and legends readable, no clipping.',
    'Original parameter-panel near-zero ticks overlapped. Separate display image reopened after explicit major ticks and axis limits; all original means/intervals and symlog linear thresholds retained, labels now clear.',
    'Both charge-bound bars, aggregate and ten-region PSD values, complete waveform mean/SD and four population labels opened; no clipping or label overlap.'
]
figures = []
for name, comment in zip(names, comments):
    final = name.replace('.png', '.display.png') if name.startswith('03_') else name
    original_path, final_path = S/'figures'/name, S/'figures'/final
    data = final_path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    width, height = struct.unpack('>II', data[16:24])
    figures.append(dict(original_path='figures/'+name, final_display_path='figures/'+final,
        original_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
        final_sha256=hashlib.sha256(data).hexdigest(), width=width, height=height,
        original_actually_opened=True, final_display_actually_opened=True, layout_pass=True,
        review_comment=comment))
receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), reviewer='/root', figure_count=4,
    original_images_actually_opened=4, additional_display_versions_actually_reopened=1,
    all_figures_opened=True, layout_pass=True, scientific_values_changed=False,
    original_images_and_frozen_renderer_preserved=True, figures=figures,
    display_only_receipt='DISPLAY_ONLY_RECEIPT.json', display_renderer_native_exit=0,
    no_new_scientific_computation=True)
(S/'FIGURE_QA.json').write_bytes((json.dumps(receipt, indent=2)+'\n').encode())
print(json.dumps(receipt, indent=2))
