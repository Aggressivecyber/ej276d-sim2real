# R69: current full-law objective with sampled parameter VJPs

Status: prospective protocol; no scientific execution is authorized by this file alone until the root registration and implementation review are sealed.

## Question and evidence

R67 did not support expensive R65: full20 had a worse common nine-term score than mini979 at equal time. R68 showed that D5 was slightly smaller but C4 larger; noise and temporal D were individually worse. This does not establish a D/C gradient conflict or the cause of the difference. R68 also bounded an invariant TRAIN charge mismatch. R47 already tested charge-only HGB on the original evaluation selections and found uncertain results. Neither proves universal failure of the original fixed two-dimensional HGB gate. Do not repeat that classifier or restore R65.

The sole new candidate reduces the cost of propagating the CURRENT full empirical distribution derivative to network parameters. It does not change any original event measure, D/C loss formula, input charge, network, or acceptance threshold. It is a new stochastic optimizer, not a bitwise implementation replacement for R65. A 1-full-plus-49-C-only proposal is rejected for lack of mechanism/benefit evidence; old D:C=1:1 is a binding of R65, not a universal restriction on future independently registered methods.

## Frozen estimator

At each candidate update, draw the original C microbatch and full SOURCE noise/directions exactly as in R65. At fixed theta, run the original 64-row schedule over all 6593 SOURCE parents, retain all current projected endpoints, and evaluate all five original D terms against all 21047 canonical REAL parents. Compute their joint endpoint derivative GY once, with all original population/region masses and straight-through ADC definition.

The schedule has M=104 blocks (103 of length64 and one of length1). Independently of theta, endpoints, noise, GY and C, select k=4 block indices uniformly without replacement using a new PCG64 stream, then replay them in ascending original schedule order. Check their endpoints, ADC codes and Box free masks exactly against the saved current endpoints. Let G_b=J_b^T GY_b. Use Ghat_D=(104/4)*sum_selected G_b, accumulated in CPU float64 and scaled once. The final one-parent block has the same inclusion probability4/104. GY already contains every scientific weight; do not add block-length weights or renormalize selected parents.

The conditional expectation of Ghat_D is the complete D gradient. Actual per-update backward exposure is sampled; all-parent forward/distribution exposure is retained. The original C microbatch gradient is then added, followed by one original cast, clip5 and Adam step. Clipping, Adam, floating arithmetic and the changing trajectory are not claimed unbiased or identical to R65. No cross-event BatchNorm/attention/state may be introduced. The original network uses per-event GroupNorm, attention and zero dropout.

## First endpoint: one no-update gradient admission

Before any response update or common response probes, restore the original durable R65 update500 model/Adam/RNG and calculate one complete current D+C gradient through all104 original blocks. This calculation performs no optimizer step. Record sum_b ||G_b||^2, ||sum_b G_b||^2 and ||G_D+G_C||^2. For SRSWOR k=4, the conditional mean squared norm of the D estimation error is

    V = M*(M-k)/(k*(M-1)) * (sum_b ||G_b||^2 - ||sum_b G_b||^2/M).

All quantities refer to this one frozen theta/noise/direction draw and the original mathematical first-order gradient before clipping. This is not training/physical uncertainty. Complete the finite artificial unequal-block expectation/variance check before reading scientific arrays.

Prospective cost admission requires finite computations, unchanged model/Adam states, and V <= ||G_D||^2 AND V <= ||G_D+G_C||^2. A zero norm passes only if V is also zero. This deliberately conservative engineering rule asks that conditional RMS approximation error not exceed either reference gradient magnitude. It is not an A-F criterion or a proof that every stochastic optimizer with larger variance is ineffective. If it fails, close this fixed k4 candidate as NEGATIVE_GRADIENT_ADMISSION, preserve the exact evidence, perform zero response updates and do not scan k or new seeds.

## Conditional equal-cost response pilot

Only if admission passes, restore the same update500 model, Adam, original PCG64 states and torch RNG separately for candidate and mini. Use the same fixed update501 learning rate from R67 (0.00019999999981874005). New block seed is202610116901. Candidate uses exactly100 updates, with a200-second arm cap. Mini uses the original R63 nine-term step from that same starting state until its measured update time first reaches candidate update time; require difference <=max(0.5s,1%). Original C sample size320 SOURCE+64 REAL identity and all original sampling/weights remain. This is a same-start, equal-time route comparison, not an isolated causal estimate of one source of randomness.

Both arms are rejected if completed-step elapsed time exceeds200 seconds, including mini's final complete-step overshoot; equal-time tolerance does not waive this condition. The600-second process watchdog remains the hard execution limit.

Baseline and both endpoints use eight common NEW full-SOURCE noise/direction probes, seeds202610116911..202610116918, plus one complete same-treated-REAL control seed202610116999. Every probe keeps6593 SOURCE and21047 REAL reference parents, all regions/support failures, all five D and four original C terms, overall/regional PSD and baseline RMS/ACF1 W1, source correlations/valleys, range projection and exact charge checks. C probe terms retain the R67 fixed original microbatch measurement scope. The complete common evaluation may not be shortened to fit the cap.

Retain R67's prospective local-benefit rule: candidate mean nine-term score must be <=90% of both mini and baseline, and beat mini in at least6/8 common probes. For every originally listed nonnegative component/distance, candidate mean <=min(baseline,mini)+0.05*max(baseline,mini)+1e-10; correlations >=max(baseline,mini)-0.02; valleys <=min(baseline,mini)+0.01. Apply the same nonnegative bound to full same-treated-REAL controls. Undefined values fail. Report the descriptive paired-noise bootstrap interval with seed202610116980 and1000 draws; it is not independent training/acquisition uncertainty. All local-benefit conditions must pass together. They do not replace original A-F.

## Execution, roles and stopping

One owned server process tree, exactCPU0..11, RSS48GiB, GPU allocated/reserved/process10GiB and fresh admission. Entire worker has a hard600-second cap, including loading, synthetic checks, gradient admission, possible training, probes and saving. Before training, measured baseline cost must reserve both200-second arms, all remaining complete evaluations and20 seconds saving. Machine safety or the fixed cap stops this attempt; no automatic retry, k/seed scan, intermediate checkpoint choice or long-run restart. Failure to execute is inconclusive, not a scientific negative.

Only existing legal TRAIN fields and frozen R65 dependencies are opened. No DEV, EVAL, CAL, response HGB, density fit, task utility or FULL/GAP execution. R65 stays retired and its original50000 endpoint incomplete. Old checkpoints/ledgers/results remain intact. Root owns registration, native launch/observation, scientific decisions and Git/D delivery; auxiliary agents provide scoped static/synthetic checks only. Shared localCPU4/RSS4GiB; each helper1CPU/RSS1GiB. All new local source/data/log/cache/temp remain on E/D.

Positive local evidence permits only a separately registered capped confirmation aimed at the same original A-F objective. Negative admission or negative joint pilot closes this fixed candidate with its evidence. Original A-F and full model goal stay uncompleted until one frozen method passes every original requirement and actual delivery.
