# R66: capped local equal-cost benefit test

The user asked to establish improvement before paying the large R65 full-population cost. R65's original 50,000-update endpoint is incomplete and remains on cost hold. Its stopped process has been retired and archived; the immutable update500 checkpoint remains the common starting state. This separate pilot cannot satisfy original A-F or authorize the old long run.

## Frozen question and scope

From the same R65 update500 model, Adam moments/step, all eight PCG64 streams and torch CPU/CUDA RNG states, does full-population D plus original C improve common TRAIN measurements more than the original R63 mini-batch nine-term route at equal measured training time? Both arms use the SAME fixed learning rate equal to the original schedule at update501. This is a local fixed-LR continuation comparison, conditional on a starting state produced by full-D training. Different native Adam update counts, D/C coupling and accumulation policies remain part of the compared routes. No claim about original initialization, 50,000-update convergence, population truth or independent acquisition follows.

All original 6,593 SOURCE and 21,047 REAL TRAIN parents, original bins/masses, original frozen statistics, network, ADC/Box formulas, clip5, Adam, loss weights and occurrence noise policy are retained. No validation fields, DEV, EVAL or CAL are decoded. The original A-F thresholds and full source population are unchanged. The R65 driver and modules are read-only; the pilot writes only its own directory.

## Compute and endpoints

- One scientific worker, server CPU12/RSS48GiB/GPU10GiB. Local CPU4/RSS4GiB aggregate, all new local files on E/D.
- Hard maximum 600 seconds of worker wall time, including imports, binding checks, loading, all probes, training and output. External process-group watchdog plus internal deadline checks enforce the cap. Overshoot attributable to OS signal delivery is measured and reported; no result after a budget violation can pass.
- Full arm: exactly 30 completed updates, starting at original step501; 180 seconds maximum for this arm. The measured synchronized time T includes original draws, noise, D/C evaluation, VJPs, clipping, Adam and original per-step diagnostics/counters. Original periodic disk logging/checkpoint actions are excluded in both arms; pilot progress writes occur outside step timing only before/after each arm. Full phase hooks only check deadlines.
- Mini arm: restore the identical update500 start, run until the first complete update reaches T. Require abs(Tmini-T) <= max(0.5 seconds, 0.01*T). No intermediate quality evaluation, retry, endpoint selection, earlier endpoint substitution or result-based extension.
- First finish the full baseline evaluation described below and measure E0. Proceed to training only if elapsed + 360 + 2*1.10*E0 + 20 <= 600 seconds. Otherwise stop INCONCLUSIVE_BUDGET. The 20 seconds reserve is for synchronization and writing. No population, probe or metric reduction is allowed after observations.

## Common measurements

Exactly eight independent probe seeds 202610116601 through 202610116608. For every model, each seed provides one fresh full-SOURCE parent noise draw, original random 64 projection directions, and an independent original C microbatch (32 occurrences per original region and 64 REAL identity occurrences, with independent occurrence noise). Probe RNG is separate from training RNG. All three models share the same probe draw recipe. Compare full original five-D values plus four original-C values, all coefficients1, as one common nine-term score. Never compare training losses from incompatible estimators.

Each full SOURCE probe also reports canonical PSD W1 globally and in ALL ten original regions, original baseline-RMS and lag1-ACF W1 globally and by region, original-source PSD Spearman correlation globally and by region, Q3/Q4 h==1 hard valley fractions under the original bounds, range projection fraction and RMS globally and in original support-out regions -1/8, finite/charge/range checks. Undefined required correlations make the harm gate incomplete. Raw source valley fractions and original counts are retained.

Each model additionally processes all 21,047 REAL once, using common seed202610116699, original TRAIN normalization and original regions. Report the same five-D discrepancies against original REAL, PSD/noise/boundary metrics and normalized parentwise waveform identity error. In this REAL-vs-REAL control only, the source-side regional masses equal the actual REAL counts; this is a declared control, not a refit or change to SOURCE weights. No source-label valley or source-rank test is imposed on REAL control.

## Fixed decision

Let B/F/M be baseline/full/mini. Positive local evidence requires mean(scoreF) <= 0.90*mean(scoreM), mean(scoreF) <= 0.90*mean(scoreB), and at least6 of8 paired scores F<M. A paired 1,000-resample bootstrap of score differences with seed202610116680 is descriptive only, conditional on these fixed models, parent populations and noise/directions; it is not a population or training-replicate confidence interval.

For EACH five-D/four-C component, PSD/noise/projection nonnegative metric and single REAL-control discrepancy, require F <= min(B,M) + 0.05*max(B,M) + 1e-10. For each declared source PSD correlation require rhoF >= max(rhoB,rhoM)-0.02. For each Q3/Q4 h==1 valley fraction require pF <= min(pB,pM)+0.01. SOURCE metrics use the eight-probe mean; REAL uses the one fixed complete control. Exact original finite/voltage bounds and pre-ADC relative-charge errors <=1e-5 remain hard gates. All parents and failed regions remain in the report.

Positive requires complete binding, population, numerical, resource, budget and cost-match receipts plus every main/harm condition. It permits only a separately preregistered capped confirmation, never automatic expensive R65 continuation. Negative closes this local support test and supplies no reason to resume R65. Incomplete, timeout or invalid results are INCONCLUSIVE and do not authorize more compute or budget extension. Negative does not prove full-D generally ineffective. Preserve pilot weights as diagnostic artifacts only; no final inference candidate is selected here.

## Implementation and review

Main update bodies are AST-extracted verbatim from frozen R65/R63 drivers through original per-step diagnostics and resource assertions, before periodic progress/checkpoint blocks. SHA256/normalized-AST receipts bind them. The only scheduling substitution is the declared common fixed learning-rate function. Setup loads the same checkpoint separately per arm; exact recursive state hashes verify model, optimizer and RNG restoration. Baseline probes must not mutate model state or training streams. The previously passed eight artificial CPU/CUDA R65 checks are reused with their 115-file binding; this pilot adds no new loss or derivative formula.

The independent static design review is reviews/R66_EQUAL_COST_PILOT_001/design_review/DESIGN_REVIEW.md. Root adopts its equal-time, complete-budget and harm-gate cautions; root deliberately chooses common fixed LR instead of the review's original-step LR option. Root owns science, implementation review, job control and delivery. No additional agent work is currently active.
