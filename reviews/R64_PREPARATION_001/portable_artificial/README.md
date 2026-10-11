# Portable R64 artificial fixture

`portable_fixture.py` exports:

```python
run_check(device="cpu", *, adapter_root=None) -> dict
```

Launch the owning process with `python -B`. Pass the copied frozen adapter directory explicitly, for example:

```python
from portable_fixture import run_check

cpu_receipt = run_check("cpu", adapter_root=stage / "distribution_adapter")
cuda_receipt = run_check("cuda:0", adapter_root=stage / "distribution_adapter")
```

The supervisor owns resource limits, backend and thread configuration, logs and JSON writes. The portable function makes no file writes, applies no process limits, sets no torch backend/thread options, and consumes no torch random draws. It uses the original local NumPy PCG64 seed without touching the NumPy global RNG. The calling process must use the original float32 default dtype. Temporary adapter module bindings and the caller's gradient mode are restored. Before/after fingerprints verify thread, backend and RNG state preservation. Use sequential calls from the single owned supervisor thread.

If omitted, `adapter_root` is the adjacent `distribution_adapter` directory. Its three adapter modules, `SOURCE_BINDINGS.json`, `FROZEN_TOLERANCES.json`, six `r63_original/*.py` modules and `r63_original/config.json` are SHA-bound before loading. These are explicitly copied source/configuration files. No scientific arrays, checkpoint weights or DEV/EVAL/CAL inputs are read.

The preserved 69 checks include original R63 equal-32 value/gradient agreement; full unequal-count independent CDF oracle; full/equal-32/equal-128 fast versus generic branches; zero RMS; duplicate-parent scatter and exact global mass; region-major four-by-32 versus 128 layouts; wrong-mass/layout/term rejection; and original endpoint Box/STE single-graph versus two-pass endpoint and toy-parameter gradients. Toy coefficients, waveform construction, population counts, all checks and all mathematical tolerances are unchanged.

Both comparison arms run on the selected device. Features, standardization and float32 projection matrix products stay there. Necessary detached empirical-law columns for the generic W1 and independent CDF oracle transfer to CPU. Device-to-NumPy conversion uses `detach().cpu().numpy()`.

Successful JSON includes:

- `compact`: status, pass flag, check count, device, fixture/tolerance hashes, preserved-state flag and elapsed time.
- `tests`: all 69 named records with existing errors and tolerances.
- `original_nine_piece_values_32`, `D_F_values`, `D_A_four32_values`, `D_B128_values`.
- `D_F_parent_gradient_norms`, `D_F_parameter_gradients`, `D_A_B_endpoint_gradient_distance`, `same_schedule_replay_checks`.
- `fixture`, `fixture_arrays_sha256`, `tolerance_sha256_before_run`, `selected_device`, `fixture_hash_semantics`.
- `caller_state_before`, `caller_state_after`, `caller_state_preserved`, source bindings, implementation hashes and scope limitations.

Fixture/runtime check failures raise `ArtificialCheckFailure`; its `.receipt` is a JSON-compatible partial failure receipt. Device availability, launch requirements and source/hash binding failures can raise before fixture construction. A caller should record the exception and abort before reading science inputs when any check fails.

The fixture hash preserves the original ordered shape/dtype/raw-byte stream. It identifies the materialized fixture on the selected device. CUDA-native `sin` and reduction output may have different raw bytes than CPU; cross-device bitwise equivalence is not required. Every numerical gate remains within the selected device and retains the original fixed tolerance.

## Local receipt

Exactly one local CPU invocation completed with all 69 checks passing, native exit code 0, single-core affinity, 1 GiB process commit cap and 1 GiB RSS watchdog. Peak RSS was 403,587,072 bytes (384.890625 MiB); the process elapsed 33.2227296 seconds including startup/exit. All 12 compared key fields, including all check records, numerical summaries and the fixture hash, exactly equal the original artificial receipt.

The unchanged fixture SHA-256 is `8358df412f174e62fad3d848e38e50a6d39f906b6099b9c36696c5c56d2c3f1e`. Fixed-tolerance SHA-256 is `24c3c78dad18846fd56180fccfff19a1e0f835908f649ab2eecae50fe43c266c`.

The local resource owner is `run_cpu_once.py`; it is separate from the portable import and refuses a second invocation after a completed receipt. The frozen tolerance document's historical resource metadata remains unchanged; the separate local guard enforces the stricter 1 GiB bound. CUDA and the real ResponseNetwork were not executed here. This result is preparation evidence only.

`PORT_SOURCE_BINDING.json` records source provenance and static transformation checks. `PORTABLE_CPU_ARTIFICIAL.json` stores the returned full JSON. `LOCAL_CPU_RECEIPT.json` records exact parity and protected source hashes. `NATIVE_EXIT_RECEIPT.json` records observed process completion. `PREPARATION_RECEIPT.json` seals this directory's deliverables and confirms that the frozen source adapter and geometry files remain unchanged.
