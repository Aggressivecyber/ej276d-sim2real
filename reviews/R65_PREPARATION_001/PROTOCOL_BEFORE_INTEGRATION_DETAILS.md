# R65: complete finite TRAIN distribution objective with original constraints

Status: prospective protocol prepared by the root; scientific registration and
execution require the separate registration receipt and frozen implementation
bindings. Original A-F acceptance and all closed historical results remain in force.

## Question and evidence

R64's fixed paired diagnostic found larger parameter total-D squared error for
joint128 than for the mean of four32 losses: B-A=206.36235254113632,
95% paired interval [99.05100094847836,343.48602557985976]. All finite mean
gradient discrepancies decreased but dispersion increased more. Both estimators
had substantial relative error against the complete finite-parent reference.
The conditional B128 training proposal is therefore not activated.

R65 tests whether evaluating that complete finite-parent distribution at every
update improves the original joint response requirements. It uses fresh noise
per parent and fresh directions, not R64's frozen endpoint library. The reference
is a finite empirical objective with an algorithmic surrogate derivative;
it is not a population oracle, exact conditional-noise integral, or proof of
success. R64 did not include C or identify a conflict in the full nine-term
training gradient. R60's negative conflict markers remain negative.

## Frozen population, architecture and endpoint

Use all original 6593 SOURCE and 21047 REAL TRAIN parents in archive order,
with the original IDs, labels, ten regions, per-domain population masses and
SOURCE weights. Keep all below/above-support parents. Load the frozen R61
distribution and R63 temporal constants without refitting. Preserve the
original noise calibration, unit-charge coordinates, valley definitions,
float32 PSD order intervals, architecture and original direct endpoint.
No labels enter the network. Do not read or decode DEV, EVAL or CAL during
training, preparation on scientific inputs, or training decisions.

Initialize the untrained original ResponseNetwork with model seed 20261010590
and verify tensor SHA
`4a51097e257d4b35976452982aa9a84f48e321d425e0ba3859e8afe8a7bd239b`.
Do not load the R63 final model. Retain original training mode, zero dropout,
deterministic algorithms, disabled TF32/cuDNN benchmark and disabled optional
MHA inference fast path. No model buffer may change between the two passes.

Keep the original Adam settings, clipping at 5, 500-update warmup from 1e-6
to 2e-4, and original cosine schedule ending at 2e-5. The sole scientific
endpoint is update 50000, matching the original optimizer-update schedule.
This fixed endpoint is a comparison design, not a compute budget. There is
no time, method-count, training-count or token budget. Intermediate losses
and recovery caches do not select a model, schedule extension or new endpoint.

## Five D terms on the complete parent law

At each update draw a fresh float32 epsilon[6593,480] in original SOURCE
order from a new independent PCG64 stream, seed 20261011650. Use the original
innovation with completion=True and store that step's innovations for replay.
Every parent has one occurrence in this D law. All raw REAL parents form the
reference after original canonical_REAL_torch, without a neural transformation
or added noise. Keep per-step full-window and temporal directions from the
original seven R63 streams, generated and consumed exactly once per update.

The D definitions and outer weights remain those of R63/R64:

- full_window: original SOURCE regional-mass average of the four view families'
  half coordinate-W1 mean plus half 64-direction sliced-W1 mean.
- coordinate_RMS: each family's root mean square of the complete vector of
  coordinate W1 values, then original SOURCE regional mass and four-family mean.
- temporal_joint: half 31-coordinate marginal mean plus half 64-direction
  sliced mean, then original SOURCE regional-mass average.
- PSD: half the global W1 plus half the seven Q1-Q7 equal-region W1 mean,
  divided by the original 0.01 scale.
- noise: global baseline RMS and lag1 ACF W1 divided by their original frozen
  scales, then averaged.

Global SOURCE and REAL parents have their own equal masses 1/6593 and
1/21047. Each region has its own complete uniform parent law. Use the bound
R64 integer-CDF finite W1 and uniform fast path, stable original-row sorting,
abs(0)=0 and zero-RMS selected gradient. No truncation, quantile-grid surrogate,
transport regularization, smoothing change or merging of distinct tied parents.

Use the existing joint `FrozenDistributionObjective.__call__` with all five
terms on one projected-before-ADC endpoint leaf. Its total scalar provides one
total GY. The unchanged canonical STE appears once on this route. No extra
REAL feature cache, extracted C implementation, or stochastic parameter-VJP
approximation is selected in this protocol.

## One original C microbatch

Keep the original seven R63 PCG64 streams and their exact per-step consumption:
source_rows, REAL_reference_rows, REAL_identity_rows, source_epsilon,
identity_epsilon, projection_directions, baseline_projection_directions.
Each update draws 32 SOURCE and 32 REAL reference occurrences per original
region, plus 64 uniform REAL identity occurrences. Each SOURCE/identity
occurrence receives fresh original noise, including duplicate parent draws.
Draw the old REAL reference rows even though full D now uses all REAL.

Run the unchanged original DistributionLoss on this 320+64 neural microbatch
and retain only source_order, source_valley, REAL_identity and box_correction.
Its five microbatch D values are computed and discarded. This avoids a new
implementation of original pair denominators and nonlinear valley reduction.
Keep both raw and projected output paths for box_correction, original source
order intervals, original valley denominators and identity normalization.
There is one C microbatch per optimizer update; no four-microbatch averaging.

The C marginal sampling law and formulas remain original. The joint D-C noise
coupling changes: original D and C shared the microbatch endpoints; full D
now uses a distinct fresh parent-noise stream. This may change covariance,
clipping and Adam trajectories. No claim of an identical joint stochastic law
or isolated effect solely from replacing a gradient estimator is made.

## Two passes and optimizer update

Use original-order SOURCE chunks of 64, with the final one-parent chunk.
First collect all projected endpoints without retaining neural graphs, at
unchanged theta. Build the complete five-term D law and GY, then replay the
same chunks at the same theta, mode, noise and numerical settings. Compare
projected endpoints bitwise, canonical ADC codes exactly and strict Box free
masks exactly before injecting each GY block into projected output.

No extra division by parent count, chunks or microbatch count is allowed.
Original charge-null displacement and Box backward appear exactly once.
Accumulate returned parameter gradients in float64 CPU named-parameter and
original chunk order. Add the one original C parameter gradient, convert once
to each parameter dtype, then clip the combined D+C gradient and take one
Adam step. Do not clip D or C separately and do not update theta between passes.

For an uninterrupted run, each update has 2*6593+384=13570 scientific neural
forward rows, 209 forward calls, 104 D parameter VJPs and one C reverse pass.
Across 50000 updates this is 678500000 neural rows and 10450000 forward calls,
excluding separately reported artificial checks and subsequent inference.
The D law sees 329650000 SOURCE and 1052350000 REAL parent occurrences;
C sees 16000000 SOURCE and 3200000 REAL identity occurrences. The old
16000000 reference occurrences remain recorded as computed/discarded-D cost.
This is not an equal-exposure or equal-compute comparison with R63.

## Artificial checks before scientific TRAIN access

Bind tolerances, synthetic inputs and helper hashes before execution. Reuse
the R64 artificial W1/unequal-count/duplicate-parent/tie/zero-RMS coverage and
actual ResponseNetwork CPU/CUDA replay checks. Add checks for the jointly
computed total D endpoint gradient against the sum of the five separately
computed gradients, and for retained C against the original four pieces.

Use nonzero synthetic network coefficients, nonzero C signals, active Box
coordinates and raw/output paths. Check source-order interval overlap and
original denominator, valley nonlinearity, REAL identity, and combined D+C
parameter accumulation. The single-graph reference must use the same chunk
schedule when comparing two-pass replay. CPU uses a small artificial schedule;
CUDA also tests the registered 64-row plus final-partial path. Purely artificial
preparation never opens scientific arrays or weights or advances a scientific
optimizer. Failure stops before TRAIN access; mechanical fixes preserve failed
receipts and must not loosen tolerances after seeing results.

## Recording, termination and original evaluation

Record all five D and four C losses, original source diagnostics, D and C
projection/charge summaries, finite gradient norms, clipping, complete parent
exposures, random stream chains/states, replay check counts and mismatch details,
actual resources, elapsed times, native exits and helper/input hashes. Avoid
storing every ephemeral endpoint; the seeds, source order, source code and
frozen inputs define reproducibility. Retain a single rolling same-run recovery
cache with exact model/optimizer/RNG/counters and input binding; never select
it as a scientific model endpoint. Any recovery must preserve original native
attempt receipts, resume that same registered run and record replayed compute.

Stop and preserve failures on nonfinite values, replay mismatch, violated
original charge/range constraints, mutated input bindings, hardware/resource
limits or failed scientific gates. Do not modify the active method, endpoint,
population, loss or learning schedule in response to training logs.

At the unique final weight run the original 128/256 TRAIN numerical gate,
then original complete SOURCE/same-treated REAL A-C evaluation and independent
4000-row replay using the frozen inherited evaluation implementation. Retain
all original regions and failed density slots. Passing permits the original
D/E continuation; failing closes this fixed candidate with its negative result.
No new acceptance threshold or independent-acquisition success is inferred.

Server limits remain CPU12, tree RSS48GiB and GPU allocated/reserved each10GiB;
one scientific job, fresh resource admission, real PID/native receipts.
Local auxiliary work shares CPU4/RSS4GiB. All new local files, logs, caches
and temporary files live on E/D. Preserve historical data and backups.
Deliver the unique weight, runnable inference, protocol, figures actually
opened, full evidence and native receipts via private Git and normal D backup.
R65 delivery alone cannot complete the long-term model goal; the same frozen
method must satisfy the original complete A-F requirements.
