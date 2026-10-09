# R15 conditional residual flow

The residual flow improves held-out density and baseline discrimination but does not close the complete response discrepancy. Complete validation NLL changes from -2420.615 to -2436.294 nat/event. Full voltage BA drops0.970 to0.950 and baseline BA0.686 to0.604, while full shape remains0.902 and time-tail BA remains0.930. Charge-PSD double-band structure and narrow-bin support are essentially unchanged.

The experiment fixes R14 expanded21047 TRAIN-fit records, original599 internal validation and2000 DEV, plus the EXPOSURES head/conditional-Gaussian checkpoints selected at15700/15900. Representation/scales/m0 and empirical L still come from original2592. No original long waveform, CAL waveform, EVAL or new G4 is decoded. One acquisition and artificial groups remain the independence limit. These are target-density development results, not a verified G4-to-REAL conditional mapping.

## Fixed scientific comparison

The new eight-layer446-dimensional conditional affine RealNVP uses34 head coordinates,223/223 splits and257->128->128->446 SiLU networks, bounded logscale3*tanh(raw/3). All final matrices/biases start at zero; each layer restores original coordinate order. Only this network is fitted. Frozen m/ell remain unchanged, but the inverse flow need not have zero mean/unit variance: the contrast permits additional conditional mean and covariance changes as well as non-Gaussian structure, so it is not a pure higher-moment or variance-only mechanism test.

Complete density includes frozen head density, flow logdensity/logdet, -sum(ell), -logdetL, head standardization,33logQ and the FACTOR constant Jacobian. Tail coordinates span the full waveform; they are different from the time-tail220:400 classification window. The density gain cannot be assumed to repair that temporal region.

## Training and selection

Single seed20261009151, Adam3e-4/weightdecay1e-5/batch128/clip5, float32 cuda:0. Actual16300/16300 updates in100.504400s. All599 validation rows at step0/every100 select the strictly smallest corresponding conditional NLL; exact ties retain the earliest checkpoint. No restarts, schedule, seed or structure changes and no DEV selection.

| Checkpoint | Step | Conditional599 val NLL[nat/event] |
|---|---:|---:|
| Initial | 0 | -2306.726318 |
| Selected | 1400 | -2322.405273 |
| Final | 16300 | -2071.197998 |

Later validation worsens substantially and fluctuates; the full trace, selected/initial/final weights and final optimizer/random states are preserved. Extra updates are not hidden or claimed as improvement. CPUfloat64 final scores do not reselect the checkpoint.

| Model | Full599 val NLL | Full21047 TRAIN NLL | Val minus TRAIN |
|---|---:|---:|---:|
| GAUSSIAN_BASE | -2420.614998 | -2428.807348 | 8.192350 |
| RESIDUAL_FLOW | -2436.293911 | -2456.336744 | 20.042833 |

Both models use identical data/coordinates and complete density terms. Validation improves by15.678913nat/event but the gap increases from8.19235 to20.04283. This is descriptive evidence of improved selected density with a larger generalization gap; it does not establish response equivalence.

## Common grouped classification and paired effects

All2000 candidate and REAL rows pass inherited QC. The same original R7 common pools supply600 classifier-training and250 held-out rows per domain, six frozen views,18 HGB fits. REAL-REAL is same-N TRAIN-reference versus DEV sampling, not a physical equivalence threshold.

| View | Gaussian BA | Flow BA | REAL-REAL BA | Flow minus Gaussian BA | 500-group95%CI |
|---|---:|---:|---:|---:|---|
| full_historical_shape | 0.910 | 0.902 | 0.512 | -0.008 | [-0.03402687953056866, 0.015233788274809907] |
| full_voltage | 0.970 | 0.950 | 0.546 | -0.020 | [-0.03619726595035821, -0.0009328890444446421] |
| prompt_local | 0.578 | 0.562 | 0.490 | -0.016 | [-0.05594562872056017, 0.02455927978173747] |
| postpeak_local | 0.726 | 0.728 | 0.470 | 0.002 | [-0.031034067448382177, 0.04013935670113281] |
| tail_local | 0.930 | 0.930 | 0.490 | 0.000 | [-0.021925875701044227, 0.021593539384580793] |
| baseline_local | 0.686 | 0.604 | 0.522 | -0.082 | [-0.1250153939456982, -0.03889059429757812] |

Baseline and voltage BA intervals show improvement in this fixed contrast; shape/prompt/postpeak/tail intervals include zero. No familywise significance or equivalence claim is made. AUC/intervals and every view are retained in full/compact metrics. Intervals condition on these classifiers and artificial groups, not response retraining or independent acquisitions.

## PSD support, observable changes and tails

Both Gaussian and flow have qualified two-component fits only in bins1,2,6; REAL has all8. Pairwise common support is exactly[1,2,6] for peak means, raw widths and detrended widths. Bins3,4,5,7,8 fail the frozen convergence/support rules and remain undefined. Do not compare averages across different supports or promote fitted components into particle truth. The frozen q4 CAL-reference failure and q1 HC-neutron absence remain unmodified.

| PSD component | Field | Common bins | Gaussian MAE | Flow MAE |
|---|---|---|---:|---:|
| 0 | mean | [1, 2, 6] | 0.02540747 | 0.02527617 |
| 0 | sigma_PSD | [1, 2, 6] | 0.01192193 | 0.01188312 |
| 0 | detrended_sigma_PSD | [1, 2, 6] | 0.01259339 | 0.01254680 |
| 1 | mean | [1, 2, 6] | 0.03038692 | 0.03008722 |
| 1 | sigma_PSD | [1, 2, 6] | 0.01417721 | 0.01410426 |
| 1 | detrended_sigma_PSD | [1, 2, 6] | 0.01435471 | 0.01428365 |

These errors change only slightly; the five unsupported regions and displaced/blurred charge-PSD structure remain. All new2000 outputs are finite, unclipped by ADC and QC-valid; no redraw, truncation or CFD recrop. Three negative-PSD events in each candidate remain outside the shared0..0.6 body view and visible in the full-range plot; REAL has zero outside. All panels share the complete charge range.

Existing local waveform summaries (no added statistic calculation):

| Domain | Baseline RMS median/max[mV] | cross50 mean/SD[index] | peak mean/SD[index] | Prompt curvature median[mV] | Postpeak curvature median[mV] |
|---|---|---|---|---:|---:|
| REAL | 0.627988/1.592535 | 119.475478/0.281736 | 126.270500/2.109397 | 6.342334 | 3.974275 |
| GAUSSIAN_BASE | 0.749114/4.224051 | 119.492145/0.492408 | 126.297500/2.298616 | 6.206412 | 4.201934 |
| RESIDUAL_FLOW | 0.677808/5.303054 | 119.495543/0.482530 | 126.346500/2.322528 | 6.683735 | 4.037248 |

Baseline RMS median moves toward REAL, with a remaining extreme upper tail (flow max5.303mV versus REAL1.593mV). Baseline spectra decrease toward REAL but remain elevated, and ACF structure remains. Prompt curvature increases especially at q3/q4; prompt100:150 contains peak and early fall, not only the upstroke. Postpeak curvature is closer in point summaries without a clear postpeak classifier gain. Complete waveform quantile bands may look close while full/tail joint discrimination remains high.

REAL cross50 spans118.9995..120.0049; flow spans114.5043..127.0849, similar to Gaussian114.4951..127.1213. Its narrow observed alignment support is not restored. Full cross50/phase/peak quantiles and quartile curve/RMS/ACF/rise summaries remain in metrics. These residuals do not by themselves prove a processing-chain or physical cause.

Every new draw is queried in voltage and own-shape views against original2592 and expanded21047 references, with baseline/historic REAL summaries reused. Exact-copy counts are zero. Larger reference cardinality changes nearest distances; neither a shorter distance nor no exact copy proves or excludes memory. No retraining or reference selection is added.

## Mathematics, replay and execution

Step0 restores all599 complete validation scores and all2000 baseline pre-ADC waveforms with maximum difference0, post-ADC array_equal. Selected inverse error1.33e-15 and forward/inverse logdet error7.11e-15 at val0/199/399/598; the full446x446 fixed-head autodiff Jacobian logdet error at val0 is3.55e-15. These pass the predefined bounds without repeating the old representation audit.

Independent replay PID171999 reproduces2000 rows/960000 points: pre-ADC maxabs0 and bitwise true; post-ADC array_equal true. It loads actual saved head/innovations and selected weights, with no fit, new draws or complete evaluation rerun.

Remote numeric worker PID171461 and replay both exited0. Remote cumulative117.422093s; graphics-only label repair used native4c999a/PID53100/exit0,6.625092s including launcher overhead, so conservative cumulative124.047186s/2400. Peak supervised remote RSS2809225216B/4GiB, GPU allocated308759552B/4GiB. Local plot-fix RSS was not sampled; no science is rerun to fill this unknown.

One source-only read failure requested nonexistent old plots.py; correct plotting.py was then read. Normal return SCP succeeded, but safe extraction rejected pre-uploaded INPUT_BINDINGS.json before any extraction; the already transferred tar was unpacked while skipping this control, with no retransfer/science retry. Original failure/entry/tar and native codes are preserved. An inherited contrast legend in the original C2ST figure was corrected from saved aggregate JSON only; use c2st_paired_corrected.png. Original source/image are preserved. All eight key figures were actually viewed; none is claimed as distributional success from median curves.

## Implication and reproduction

Relaxing the conditional residual density helps baseline and selected complete likelihood, but this single architecture/budget leaves the joint time-tail and overall distributions far from REAL. The fixed head and alignment-support model remain candidate limits; this experiment does not isolate them as the sole cause. Do not respond by extending this fit or changing its selected point from DEV. ROOT will register any next controlled problem. The full project remains active, and target-density learning still precedes a validated G4 response mapping.

See REPRODUCTION.md, config.json/PROTOCOL.md/READINESS.json/INPUT_BINDINGS.json, RESIDUAL_INITIAL_STATE/RESIDUAL_FLOW/RESIDUAL_FINAL_STATE, saved permutations/random/optimizer states, ACTUAL_GENERATION_INPUTS.npz, generated.npz, density/C2ST/NN private payloads, complete metrics/trace, mathematical/replay/native/resource/QA receipts and the final inventory. Only aggregate compact metrics are intended for Git.
