# R65 thin objective adapter and artificial proofs

This directory contains a fixed API and synthetic-only proofs for one joint
five-term D endpoint gradient and the original four C terms. It does not load
scientific arrays or weights, register a scientific run, or train a model.

The actual local CPU run passed all 86 checks with native exit 0. PID 62016 was
then observed absent. Its affinity was one CPU and OS peak RSS was 340271104
bytes under the enforced 1073741824-byte limit. The artificial helper took
11.907 seconds; the full guarded process took 15.891 seconds. No scientific
arrays, scientific weights, DEV/EVAL/CAL, actual ResponseNetwork or optimizer
updates were used.

## Fixed API

Import objective_adapter.py. Importing only defines functions and imports
torch; it does not construct an objective, load data or execute tests.

- joint_total_D(objective, source_endpoint, reference, layout)
  accepts the caller's current FrozenDistributionObjective instance, projected
  SOURCE endpoint float64[Ns,480] before ADC, raw REAL float64[Nr,480], and
  explicit ParentLayout. It creates one detached endpoint leaf, calls all five
  D terms once, differentiates the existing total once, and returns detached
  total, pieces, GY, diagnostics and piece_order.
- The actual pieces and scalar-addition order remains full_window,
  coordinate_RMS, PSD, noise, temporal_joint. The GY already includes original
  masses, scales, weights, reductions and canonical STE. Inject it into the
  original projected output exactly once. Do not inject into raw or divide by
  parent count, chunk count, five terms or C microbatch count.
- original_C_microbatch(objective, output, raw, reference, identity_input,
  identity_bins, source_psd, source_h, directions, source_interval_low,
  source_interval_high, temporal_directions)
  calls the unchanged original nine-term DistributionLoss once. It directly
  adds source_order, source_valley, REAL_identity and box_correction in that
  order, preserving both live output/raw paths. Return total and pieces retain
  the graph. discarded_D_values and original_nine_total_value are detached
  logging fields. There is no total-minus-D cancellation or extracted C formula.

The caller must construct a fresh FrozenDistributionObjective at each update
with that update's full-window and temporal directions. Its constructor clones
and detaches these arrays. Reusing an old object would retain old directions.
The proof uses two different PCG64 direction sets, confirms both directional
terms and GY change, and confirms constructor tensors do not alias the input
arrays. The API caches no objective, REAL features, noise or directions.

The caller owns original init, source and REAL populations, noise, original
stream consumption, training mode, deterministic settings, replay, parameter
accumulation, clipping and Adam. The scientific design uses the existing
original C instance/formulas and one C occurrence batch, with fresh directions
passed explicitly to the original call. This adapter does not choose a
scientific endpoint or alter acceptance gates.

## Source bindings and portable execution

bound_r64 contains 13 byte-identical R64/R63 source/config/portable files.
BOUND_SOURCES.json binds their original paths and hashes. The original paths
are provenance only; runtime imports use this directory's relative copies.
fixture_factory.py copies lines 230-314 of the bound original portable fixture
without changing its body. Its local CPU array hash exactly matches the prior
R64 fixture hash 8358df412f174e62fad3d848e38e50a6d39f906b6099b9c36696c5c56d2c3f1e.

Copy all 22 files named in PREEXECUTION_BINDING.json, plus that manifest, to
the same relative layout on the server. Then call:

    result = portable_objective_checks.run_check(
        device="cuda:0", objective_adapter_root=explicit_adapter_directory
    )

CPU is also supported. The caller owns CPU/GPU/RSS admission and torch backend
settings. The helper reads only its bound Python/JSON files and returns a JSON
receipt. On failure it raises ArtificialCheckFailure with the partial receipt
in exception.receipt. It changes no caller RNG, thread, dtype or backend state;
the recorded state is compared before/after. This local receipt establishes
CPU only. CUDA remains for the separately supervised server call. The CUDA
suite has 85 checks because the exact inherited CPU-array-hash assertion is
CPU-specific; its selected-device array hashes are still recorded.

local_runner.py is a Windows-only optional runner. It sets one-CPU process
affinity and starts a 20-ms OS peak-RSS guard before importing torch, terminates
on a limit/probe failure, sets one intra/inter-op thread and the frozen numerical
settings, sends temp/cache writes under the new attempt on E/D, snapshots every
bound helper before execution, and writes complete/partial receipts. Every
attempt must use a fresh runs/<id> directory.

## Actual proof coverage

The 86 passing CPU checks include:

- Unequal 38-SOURCE/40-REAL complete parent laws, original ten regions,
  unequal per-domain masses, repeated sampled parents, tied ADC waveforms,
  exact zero coordinate-RMS value and selected gradient.
- Joint D values and GY against all five separate terms, direct joint graph,
  and original R63 equal32 occurrence graph with duplicate-parent scatter.
  Maximum joint-versus-five GY error was 4.3336734635e-10 for complete unequal
  parents and 6.1343644747e-10 for equal32 duplicates. Direct joint GY matched
  exactly. Tolerances were frozen before the first execution.
- Original float32 standardized full-window/temporal features and float64
  PSD/noise branches; 100 and 9 respective W1 calls in the joint objective.
  Deliberately using SOURCE masses for REAL changed total by 0.3711621 and
  GY by up to 0.1619772, so that error is detectable.
- Two fresh direction sets, constructor binding and no aliasing.
- Original endpoint/Box/STE replay on a nonzero four-parameter toy with
  7+7+7+7+7+3 rows. Projected output, raw output, ADC and strict free masks
  replayed bitwise. Float64 chunk accumulation versus the same-schedule joint
  graph had maximum parameter error 1.3113021851e-6 and passed the frozen
  absolute 1e-7 plus relative 1e-6 gate.
- Original C four-piece values and total, and independent out/raw partials,
  matched exactly. Every C piece and its output derivative was nonzero.
  Box partials were opposite; the other three C pieces had no independent raw
  path. The 320+64 C fixture used active rails and nonzero identity, without
  executing a 384-row actual neural network.
- Original all-nonzero-pair order denominator, unresolved interval masking,
  overlap/touch/equal-parent cases; 32-slot then population-h-fraction valley
  denominators and mean-before-square nonlinearity; raw original identity
  regions including below/above support; weighted SOURCE/mean identity box
  normalization. Wrong denominators, nonlinear ordering and clipped identity
  regions all produced detectable errors.

The artificial C values were source_order=2.7093012166,
source_valley=238.5599591047, REAL_identity=0.0263752366,
box_correction=0.0009616667. These deliberately synthetic values are not
training statistics, expected scientific losses or acceptance thresholds.

## Preserved failure and limits

Attempt 001 stopped at the extra assertion that five complete-unequal D terms
must all be exactly zero for identical canonical waveforms. full_window was
9.9127263590e-9. Its 34 preceding checks passed; native exit 1 and complete
input snapshot remain in runs/001. Different source/reference row shapes do
not guarantee identical final bits of the original float32 projected matmul.
This was an overstrong additional artificial assertion.

MECHANICAL_FIX_002.json documents the correction accepted by the root:
complete unequal-parent coordinate_RMS and its selected gradient still must
be exactly zero; the extra all-five-D zero test uses the existing equal32
layout with matching projected row shapes and still requires exact zero.
Attempt 002 reran the complete suite. The failed unequal full_window value is
still recorded in its receipt. No original source, production API or tolerance
was changed, and no failure was discarded.

The suite establishes algorithmic first-order STE/stable-tie/zero-RMS branches,
not hard-ADC finite-difference equivalence. It does not establish scientific
response gains, actual-network replay, combined D+C parameter clipping/Adam
equivalence, production RSS or runtime. The separately assigned integrated
ResponseNetwork CPU/CUDA proof and scientific gates remain required.

## Receipts

- PREEXECUTION_BINDING.json: prospective hash binding for passing attempt 002.
- runs/002/ARTIFICIAL_CPU.json: all 86 checks, fixture hashes, caller states.
- runs/002/SELF_PROCESS_RECEIPT.json: resource and timing receipt.
- runs/002/NATIVE_EXIT_RECEIPT.json: independently observed native exit/PID state.
- runs/001: preserved failure, old manifest and exact old input snapshot.
- DELIVERY_RECEIPT.json: final API/helper/receipt hashes and scope.

