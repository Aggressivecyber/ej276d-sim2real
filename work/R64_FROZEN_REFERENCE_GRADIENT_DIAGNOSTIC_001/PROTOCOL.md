# R64: frozen TRAIN reference geometry and distribution-gradient diagnostic

## Question and scientific endpoint

R63 is closed at its registered update 50000. Its original A/B/C failed; the
private Git and D-drive delivery is recorded in R63_DELIVERY_COMPLETION_RECEIPT.json.
This stage diagnoses which already-covered TRAIN statistics remain unfitted,
how much of the voltage coordinate discrepancy is associated with the retained
charge means, and how the distribution reduction over four batches of 32 differs
from one batch of 128 at the same frozen model and finite parent reference.
It does not train a candidate, choose a checkpoint, add features, or enlarge a
future training batch. R60's negative conflict markers remain unchanged.

The fixed endpoint is one complete shared parent endpoint collection; all
specified geometry summaries and 32 matched REAL resamples; one full-reference
gradient and 32 paired macro-batch gradient comparisons; registered figures,
weight/input immutability, actual native resource/exit receipts, and stage
delivery. These counts define the experiment, not a compute/time budget.

## Frozen inputs and population

Use the original legal TRAIN archive: all 6593 SOURCE parents and all 21047 REAL
parents in original archive order, their original IDs and ten raw charge regions
[-1,0,1,2,3,4,5,6,7,8]. No deletion, relabeling, regional renormalization, REAL
reweighting, or updated bin edge is allowed. Use R63's unique final weight,
original unit-charge/noise constants, frozen R61 distribution constants, and R63
31-feature temporal constants. The complete input paths and SHA256 bindings are
in config.json and INPUT_BINDING.json. Constants are loaded, never refitted.
Neither DEV, EVAL nor CAL arrays may be opened or decoded. No new HGB, density,
teacher, source-law fit, optimization, or neural update is part of this stage.

Supporting descriptions retain overall/all-ten-region PSD W1, original SOURCE
charge/PSD rank association, the nine fixed first/second-difference boundaries,
complete waveform means/SDs, projection activity and actual charge errors. These
reuse the shared endpoint collection without new network calls or density fits.

## Shared endpoint definition

Each original parent receives one PCG64 float32 epsilon vector. SOURCE and REAL
noise seeds reuse the already declared R60/R62 complete-parent streams; all three
old full noise/direction stream hashes must agree. The original innovation,
continuous charge condition, guide, network and BoxChargeProjection are used.
The shared endpoint is the original training function source_flow_core.endpoint:
initial plus displacement, then charge/box projection, before canonical ADC.
It is not the accumulated Heun inference endpoint. SOURCE and descriptive
same-treated REAL endpoints each receive one forward in fixed original-order
chunks of 64, including the original final partial chunk.

All inference weights are frozen; parameters retain requires_grad for the
diagnostic. Keep the original training mode (zero-dropout attention and no
BatchNorm running statistics) and disable the optional MHA inference fast path
throughout both passes. Enable deterministic algorithms; disable TF32 and cuDNN
benchmark; keep all other registered original dtypes and math. Direct endpoint,
canonical ADC codes, Box free mask, raw endpoint, and innovations are saved.
The fixed REAL distribution reference is canonical_REAL_torch(raw_REAL): it has
no network transformation and no added epsilon. The same-treated REAL network
output is a separate descriptive control and never replaces this reference.

The parent-fixed noise and fixed direction libraries define a conditional finite
empirical diagnostic. R63 training used independent noise for every occurrence
and fresh directions per update. This stage does not estimate that full
stochastic training expectation or a population-optimal gradient.

## Geometry and REAL resampling

Describe the frozen SOURCE and same-treated REAL endpoints against canonical
REAL using the original four view families V480/U480/DU479/D2U478 and the exact
R63 31 baseline features. Reuse frozen within-region centers/scales; standardize
in float64, cast to float32 as in the original loss, then use the fixed float32
direction libraries. Report every region's coordinate W1 vector, coordinate
mean, coordinate RMS, sliced mean, original SOURCE-mass weighted contributions,
and the five defined distribution terms. The global population has its own
original equal-parent empirical law; a ten-region weighted summary must be named
separately from a global statistic. No failure region is omitted.

For each of the ten regions and the entire original population, draw 32 pairs
of independent REAL parent bootstrap samples with replacement, each of size
equal to that scope's original SOURCE count. Preserve all duplicate masses and
record indices. Compare A against B for a matched-size REAL sampling description
and A/B against the complete REAL reference where reported. These are empirical
TRAIN reference distributions, not acceptance thresholds or independent
acquisition uncertainty. Sampled endpoint values are reused; no network call is
made for these resamples.

Geometry uses its bound NumPy implementation; CUDA is the numerical path of the
five-term gradient objective. Exact cross-device FFT/BLAS agreement is not
claimed. The descriptive NumPy projection results and CUDA loss values are
identified separately.

Report raw SOURCE, projected SOURCE, canonical SOURCE, raw REAL, and canonical
REAL charge means and 10/50/90 percentiles in the original q=sum(v[100:400])
minus 3*sum(v[:100]) code units. Charge source values must agree with the
original float32 waveforms promoted to float64. For every region define
a=(-3 on 0:100, +1 on 100:400, 0 elsewhere), c=abs(a)*frozen_V_scale,
Dout=abs(E q(Y_canonical)-E q(REAL_canonical)), and the actual standardization
representation discrepancy Erep. Report the nonnegative nominal lower bounds
[Dout-Erep]+/(480*max(c)) for coordinate mean and
[Dout-Erep]+/(sqrt(480)*norm(c)) for coordinate RMS. Also report the SOURCE-charge
form subtracting its actual postADC charge deviation. The combined V-only
contribution bound is sum_region mS*(LBmean/8 + LBRMS/4), with original unit outer
weights. All arithmetic is explicitly nominal float64, not a rigorous interval
certificate or a uniform irreducibility bound. Check each bound against the
corresponding measured coordinate contribution within a preregistered numerical
tolerance. Zero gives no compatibility proof; a positive value gives no original
A-F infeasibility result and explains neither the normalized-shape nor sliced
or temporal losses by itself.

## Five distribution terms and full empirical reference

Only D={full_window, PSD, noise, coordinate_RMS, temporal_joint} enters the
gradient comparison. All original outer weights are one. Four other original
terms C={source_order, source_valley, REAL_identity, box_correction} are excluded
from both arms and the full reference; no total-training conflict claim is made.

Within each region, both domains have their own unit-total empirical laws. Full
F includes every SOURCE and REAL parent once, with unequal regional counts.
No truncation, duplicating to equal counts, quantile grid or entropic approximation
is permitted. Scalar W1 is the exact finite weighted quantile integral with
integer CDF mass partitions; distances and reductions remain floating point.
The full reference keeps the original parent order. Sampled laws keep fixed
region/microbatch/slot order. Stable sorting chooses the tie branch;
abs(0) has gradient zero, and a zero coordinate-W1 RMS has gradient zero.
Tied distinct parents are never merged. Selected tie gradients are not unique
classical gradients. Equal-mass fast paths must agree with the generic integer
CDF implementation on artificial values and selected gradients before use.

* full_window: each of four families averages coordinate-W1 mean and 64-direction
  sliced-W1 mean, then sums regions with original SOURCE mass, then averages the
  four families.
* coordinate_RMS: per family and region take sqrt(mean_j W_j^2), sum regions with
  SOURCE mass, then average the four families. Do not interchange these reductions.
* temporal_joint: half the sum of the 31-coordinate marginal mean and 64-direction
  sliced mean, weighted by original SOURCE regional mass.
* PSD: 0.5*(global W1 + mean of seven Q1-Q7 conditional W1)/0.01. Global SOURCE and
  REAL masses are separate; the conditional part averages seven regions equally.
* noise: global W1 of baseline RMS and lag-1 ACF, each divided by its original
  frozen noise scale, then averaged. Each domain uses its own global masses.

At full F, every global SOURCE parent has mass 1/6593 and every global REAL
parent 1/21047. For a stratified sampled law, a SOURCE occurrence in region r
has mass mS_r/K_r and a REAL occurrence has mass mR_r/K_r. Both domains must not
be assigned the same regional masses. CUDA computes the original canonical,
features, standardization, casts and projected features; host integer-CDF W1
can supply values/selected derivatives without replacing the original chain.

## Paired 4-by-32 versus 128 construction

For each of 32 macro batches, independently sample with replacement 128 SOURCE
and 128 REAL parents per original region, then split each into four 32-slot
microbatches. SOURCE occurrences reuse that parent's frozen epsilon and endpoint;
REAL references reuse its canonical raw waveform. The two arms share all IDs,
multiplicities, endpoints, masses, constants, directions and frozen parameters.

A=(D32_1+D32_2+D32_3+D32_4)/4 and B=D128, with 128 constructed within every region
before stacking ten regions. Concatenating four complete region-major 320-row
batches and reshaping them as 10-by-128 is forbidden. Compute the selected
gradient of each of five terms and the sum in the same original parent endpoint
coordinates; gather/autograd scatter-add must retain duplicate occurrence mass.
There is no optimizer, gradient clipping, checkpoint selection or update.

## Two-pass VJP and verification

The first pass collects projected-before-ADC parent endpoints without retaining
network graphs. Build each D objective from a differentiable endpoint leaf,
apply original canonical STE exactly once, then obtain its parent G_Y.
Feature-coordinate blocking is allowed; sorting still spans all parents in the
relevant scalar law. Coordinate RMS coefficients require the complete W vector.

Replay every SOURCE parent in the same 64-row/final-partial schedule and same
device/mode/dtype/noise. Compare replayed projected outputs bitwise, canonical
codes exactly, and strict Box free masks exactly before VJP. A discrepancy stops
the scientific gradient comparison and is preserved, not silently tolerated.
Inject G_Y at the projected output, so the original Box backward and charge-null
network displacement derivative each occur once. Do not inject it at raw or
repeat Box backward. Reuse a chunk graph for five full-reference VJPs or ten
paired-arm VJPs. Accumulate returned parameter gradients in float64 CPU order,
with a recorded named-parameter manifest. Do not divide again by parents,
chunks or microbatches. Total D is the sum of its five component vectors.

The complete full reference plus 32 macro comparisons use 33 full-SOURCE replay
passes after the first complete SOURCE/REAL collection: planned neural parent
rows are 27640+33*6593=245209, apart from explicitly separate artificial tests.
An all-zero endpoint derivative may avoid its mathematical zero VJP only if the
receipt records that exception; it must not remove the parent from any law.
Weights and all input files must have unchanged final hashes.

## Reported gradient statistics

For every component and total D, save full gF and all gA/gB vectors privately,
plus loss values, norms, dot products, cosine where both norms are nonzero,
absolute squared errors ||gA-gF||^2 and ||gB-gF||^2, and errors relative to
||gF||^2 when it is nonzero. Report endpoint-gradient counterparts separately.
For the 32 macro samples report the empirical mean gradient, its discrepancy
from gF, and dispersion around that mean. Check the finite-sample identity
mean||g-gF||^2 = ||mean(g)-gF||^2 + mean||g-mean(g)||^2. These are finite empirical
descriptions, not an unbiased decomposition of population bias and variance.

Primary descriptive comparison: paired B-minus-A squared parameter-gradient
error for total D, with its mean and a fixed 1000-replicate macro-level paired
bootstrap percentile 95% interval (linear quantiles). The same resample indices
are used for all five components; component results are secondary and not
multiple-testing-adjusted. An interval below zero only supports smaller error
under this fixed finite-reference diagnostic. It does not establish improved
training, full-objective descent, original A-F performance, or a chosen next
method. Do not drop zero-reference cases; report undefined relative quantities
as null with the raw absolute results retained.

All parameter vectors are retained as float64; total D can be reconstructed from
its five saved component vectors. Save the full-reference endpoint gradients,
and macro endpoint norms/inner products/errors plus online mean endpoint vectors.
Do not duplicate all macro endpoint vectors on disk: their exact construction
remains reproducible from shared endpoints, frozen code and saved sampled IDs.
Check actual available disk capacity before dispatch and ordinary delivery;
this storage readiness check is not an output-volume or scientific budget.

## Artificial prerequisites and failure rules

Before reading scientific arrays/weights, check the integrated five-term
objective with fully artificial constants, real 480-coordinate waveforms,
nonzero artificial parameters, all ten regions, unequal counts and separate
domain masses, duplicate parents, ADC ties, and zero-RMS branch. Check selected
W1 derivatives against independent arithmetic, equal-size original R63 values
and algorithmic gradients within tolerances frozen before execution, correct
region-major construction, negative controls for wrong masses/layout, and
single-graph versus two-pass VJP. A small-network test does not establish actual
ResponseNetwork replay: also test the actual nonzero artificial network on CPU
and the server's CUDA runtime, including its 64-row and final-partial schedules.
Hard-ADC finite differences are not an oracle for its deliberately chosen STE.

Numerical tolerances and fixtures are immutable before scientific dispatch.
Routine code failures are recorded and repaired without reading outcomes to
change scientific definitions. If input scope, resource ownership, replay or
mathematical compatibility fails, stop the affected scientific stage and report
the failure. A diagnostic marker never replaces original A-F acceptance.

## Execution, resources and delivery

One scientific worker on rylan@100.82.231.102 in a new R64_ROOT directory under
the existing project. Server CPU<=12, RSS<=48GiB, GPU<=10GiB; original admission
checks and actual PID/exit/resource handshake remain. No science/time/method/
token/output-volume quota is introduced. Auxiliary agents share local CPU<=4
and RSS<=4GiB; each inherits the root model and reasoning effort without override.
The root owns registration, scientific direction, dispatch, synthesis and Git.
All new local source/data/log/cache/temp artifacts are on E/D. Prior originals
and results remain sealed; no migration consistency test is introduced.

Create a compact report, private reproducibility arrays and named seeds,
configuration/source/input bindings, figures actually opened by root, native
receipts and a private Git/D-drive delivery receipt. Project/model completion
remains false; the long-term goal remains active. Use the entire diagnostic
evidence to choose and preregister the next method or focused mechanism test,
without automatically combining a feature extension with a batch-size change.
