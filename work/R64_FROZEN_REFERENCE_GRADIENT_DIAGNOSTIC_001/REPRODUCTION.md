# R64 reproduction

Use the frozen PROTOCOL.md, config.json, INPUT_BINDING.json, source binding
receipts and original private inputs. No model update or validation selection
occurs. All original legal TRAIN parents retain their original ordering and
domain masses. The input weight is the unique sealed R63 final state.

The original server environment is recorded in config.json. Run only through
`python -B linux_supervise.py diagnose_train.py` in a new registered R64
directory after admission and code/input binding checks. The root dispatcher
creates an exclusive launch marker. Do not repeat an uncertain launch: resolve
its actual worker identity and native receipt first. Original stage directories
are preserved. Local files, logs, source, caches and temporary files stay on E/D.

The integrated artificial fixture runs its frozen 69 checks on CPU and CUDA;
the actual ResponseNetwork runs nonzero artificial 4+1 CPU and 64+1 CUDA replay
checks. Both finish before the worker opens scientific arrays or checkpoint
files. Failures retain their partial receipt. Those artificial network rows and
VJPs are separate from scientific counters.

The worker uses original `source_flow_core.endpoint`, not the Heun solver.
Network parameters are unchanged but require gradients; model training mode is
retained and the optional MHA inference fast path is disabled in both collection
and replay. Deterministic algorithms are enabled and TF32 is disabled. Every
parent is replayed in the same complete original 64-row schedule even if its
sampled objective derivative is zero. SOURCE ends with one row and REAL with
55 rows. Every replay checks projected endpoint bits, canonical codes, and the
strict Box free mask before any VJP. Parameter VJPs inject at the projected
endpoint exactly once. Named parameter gradients are accumulated in float64
on CPU in original chunk order without another reduction divisor.

All random streams are independent NumPy PCG64 streams with seeds in config.
SOURCE and REAL full noise arrays draw standard normals in original full shape,
then cast to little-endian float32. Four full-window directions draw independent
matrices of shapes 480/480/479/478 by 64, normalize each column in float64, then
cast to float32. These three stream digests must match R60/R62. Temporal
directions similarly draw 31 by 64 and normalize before the cast. Each stream
digest starts with 32 zero bytes and updates as SHA256(previous_digest +
contiguous consumed-array bytes).

For macro index 0 through 31, and raw region -1 through 8 in order, each domain
draws 128 original global parent indices with `Generator.choice(pool, size=128,
replace=True)`. Each such draw casts to little-endian uint16, is consumed as one
digest block, and reshapes as [4,32]. Both arms use these same occurrences.
The resulting arrays have shape [32,10,4,32]; B flattens only the last two axes
within each region. A averages four complete D32 gradients in float64. Bootstrap
draws use `Generator.integers(0,32,size=(1000,32))` with its default integer dtype,
then cast to little-endian uint16 and consume one block. REAL floor scopes are
`all, region-1, region0, ..., region8`; each A/B stream draws a [32,N_SOURCE_scope]
array from that scope's complete original REAL pool, with replacement, casts to
uint16, and consumes one block. Saved indices, directions, epsilon, innovations,
and initial/final states permit independent reconstruction.

Only the five original distribution terms enter full F and sampled A/B. The
fixed REAL reference is canonical raw REAL without a network or innovation.
Canonical STE is applied once inside the objective, followed by the original
CUDA feature, frozen standardization, float32 cast, and projection chain.
Sorting is stable; tied distinct parents stay distinct; abs(0) and exactly zero
coordinate RMS select zero. Exact weighted quantile partitioning uses integer
CDF segments; the distance and reduction arithmetic is floating point. Regional
masses differ between SOURCE and REAL. NumPy endpoint geometry and CUDA loss
values are labeled separately; bitwise cross-device equivalence is not claimed.
For sampled global laws, the existing frozen adapter evaluates region_count /
domain_count / slots in binary64, converts those represented masses to exact
integer ratios, and normalizes by their exact integer total. Thus integer-CDF
exactness refers to these explicitly represented masses, rather than to an
arbitrary-precision rational evaluation of count/(domain_count*slots). Complete
uniform reference laws reduce to exact equal parent masses. This numeric detail
does not replace either domain's original population or regional proportions.

Files ending PRIVATE.npz and parameter_gradients_private/macro_XX.npz remain in
private storage. Each parameter file retains five float64 A and five float64 B
vectors in the manifest order. Total D is their vector sum. Full endpoint
gradients and both online mean vectors are retained; macro endpoint derivatives
are reconstructed from saved endpoints, indices and frozen code instead of
being duplicated on disk. Welford accumulators retain macro norms, errors,
cosines and independently checked finite-sample decompositions. The supplied
1,000 macro-level paired bootstrap rows are reused for all six reported vectors.
Relative metrics with zero reference are null; all absolute quantities remain.

The nominal charge-mean lower bounds and finite REAL resampling are descriptive
checks conditional on this endpoint. Neither constitutes original A-F
acceptance, a population uncertainty interval, an irreducibility certificate,
causal attribution or evidence that increasing a future training batch works.
The fixed-parent noise/directions differ from R63's original fresh-occurrence
noise and per-update directions. The four other training constraint terms are
excluded from this gradient comparison.

The fixed endpoint comprises full shared collection, geometry, full F and 32
paired macros, figures, final immutable input/weight bindings and native
resource/exit receipts. Planned scientific counters: 245209 neural rows, 3865
network calls, 33800 parameter VJPs, 161 five-term endpoint-objective calls, and
33 complete SOURCE replays. This endpoint is an experiment definition, not a
compute, time, method or output budget. Root must actually open the figures,
reconcile science/native receipts, and complete private Git/D delivery.
