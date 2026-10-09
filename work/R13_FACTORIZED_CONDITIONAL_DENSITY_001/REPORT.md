# R13 factorized conditional density

The conditional residual network improves held-out continuous likelihood substantially, but none of the four factorial arms reproduces the complete REAL waveform distribution. Increasing density flexibility at the head does not resolve the remaining domain difference. This is a same-acquisition TRAIN/DEV diagnostic of target density; it is not a validated G4-to-REAL event response.

## Fixed design and actual execution

The frozen FACTOR representation, PCA basis/scales, ridge conditional mean and R12 empirical residual Cholesky remain unchanged. There are 2592 TRAIN-fit records, 599 complete internal validation records, and the original 2000 DEV records. CAL boundaries/references are reused as control metadata; CAL waveform numeric decoding and EVAL reading are zero. No new PCA, GMM or ridge fitting occurred.

An eight-coupling RealNVP head and a head-conditioned residual mean/log-scale network each ran exactly 2000 Adam updates, batch128, lr0.0003, weight decay0.00001 and clip norm5, with fixed seeds 20261009131/20261009132. Initial states, random states, all 21 validation checkpoints and selected weights are saved. RealNVP is an established invertible density construction, not an algorithmic novelty claimed here ([original paper](https://arxiv.org/abs/1605.08803)).

The two networks were trained on RTX5070 cuda:0; final likelihood and generation use saved selected weights on CPU float64. Worker161998 ran from 04:37:00.204663 to 04:37:41.259889 UTC with native exit0. Independent replay162395 also exited0. Total numeric time is 42.658897 s, training counter20.894185 s; four CPU threads, peak RSS2590203904 B, GPU allocated peak80701952 B. No installation or migration content consistency test was performed.

## Training and complete validation

| Network | Initial val NLL | Best val NLL | Selected step | Final step2000 NLL |
|---|---:|---:|---:|---:|
| Head | 47.403652 | 34.034595 | 900 | 40.007610 |
| Conditional residual | −2260.954346 | −2287.472656 | 200 | −2276.621582 |

Both validation trajectories deteriorate after the selected early checkpoint. The head is particularly worse by the final checkpoint. These trajectories do not establish training insufficiency; the following larger-TRAIN experiment can test sample-size sensitivity without altering this baseline.

All599 validation events are legal. Complete480-dimensional NLL includes the invertible-representation Jacobian and was recomputed in CPU float64 from selected weights:

| Arm | Complete NLL, nat/event | DEV full-shape BA | DEV full-voltage BA |
|---|---:|---:|---:|
| GMM_GLOBAL | −2371.774569 | 0.924 | 0.976 |
| FLOW_GLOBAL | −2371.917438 | 0.900 | 0.980 |
| GMM_CONDITIONAL | −2398.293033 | 0.926 | 0.978 |
| FLOW_CONDITIONAL | −2398.435901 | 0.932 | 0.980 |
| Historical FACTOR_LW | −2373.852796 | 0.944 | 0.986 |
| Same-N REAL–REAL | — | 0.512 | 0.546 |

The selected flow head adds only0.142869 nat/event, while conditioning the residual adds26.518464 nat/event. Better likelihood does not translate into full waveform agreement. All six domains retain2000 QC-valid records. C2ST uses the same600/250 grouped training/test positions across all candidates. The REAL–REAL result is a sampling reference, not an equivalence threshold.

## Paired effects and remaining differences

For complete normalized shape, paired BA differences (new minus comparator) are: conditional residual on GMM +0.002, 95%CI[−0.015128,0.020191]; conditional residual on FLOW +0.032, CI[0.007963,0.057066]; FLOW replacing GMM with GLOBAL residual −0.024, CI[−0.052968,0.005124]; FLOW replacing GMM with CONDITIONAL residual +0.006, CI[−0.019173,0.030308]. The intervals use500 artificial-group bootstrap replicates conditional on fitted classifiers; they do not include network retraining uncertainty or independent acquisition uncertainty. No multiplicity-adjusted mechanism claim is made.

Conditional residual changes improve high-charge postpeak curvature relative to the GLOBAL residual, but the q4 prompt second-difference median rises to about8.5 mV versus REAL6.7 mV. The prompt100:150 window includes the peak and early fall, so this is not exclusively a rising-edge statement. Baseline RMS means are REAL0.6464 mV, GMM_GLOBAL0.7253, FLOW_GLOBAL0.7239, GMM_CONDITIONAL0.7138 and FLOW_CONDITIONAL0.7112; baseline spectral/ACF discrepancies remain. Similar conditional median curves are insufficient to establish distributional equality.

Two-component descriptions have no qualified bins for GMM_GLOBAL or GMM_CONDITIONAL; FLOW_GLOBAL and FLOW_CONDITIONAL qualify only bin7. Four-arm common support is empty, so four-way peak-position/width errors are undefined. Unsupported bins remain on the figure/table with failure reasons, including nonconvergence, nonseparation, support below20 and variance-floor contact. No fallback fit or cut change is used. Density components and high-confidence reference classes are not particle truth. The inherited q4 reference failure and q1 high-confidence neutron absence are preserved.

The shared innovation protocol keeps the actual R11 white draws/component positions. Residual contrasts share the same head by construction; FLOW versus GMM contrasts are paired by draw position, not the same physical head or an independent G4 parent. All2000 queries of each new arm were compared with2592 TRAIN-fit records; no exact nearest-waveform copy was found. This is not proof of no memorization.

## Mathematical checks and reproducibility

Flow inverse maximum error4.44e−16, forward/inverse logdet cancellation7.11e−15, and full34-dimensional autodiff logdet check0. Initial conditional residual identity matches the old Gaussian density within4.55e−13 nat; final full446-dimensional covariance-form comparison agrees within2.28e−12 nat. Frozen conditional covariance orientation is still L diag(exp(2ell)) Lᵀ; it cannot express arbitrary head-dependent covariance rotations.

The fresh replay process restored saved weights and actual innovations and reproduced6000 generated rows /2880000 points for all three new arms exactly, before and after ADC (maxabs0, bitwise pass). It did not refit or resample. Raw waveforms, private features/predictions/nearest indices and full traces remain local; METRICS_COMPACT.json contains aggregate publication results. All eight delivered figures were visually inspected, without additional fitting or metric evaluation.

The full invertible representation does not guarantee adequate head/tail density. This target-density experiment does not attribute the remaining discrepancy to G4, establish an event-level response, repair missing independent labels/runs, or create a new blind test. The next registered sample-size comparison keeps these results immutable.

## Artifacts

See METRICS_COMPACT.json and full metrics.json; selected FLOW.pt/TAIL.pt; three arm generated.npz files; complete HEAD_TRAINING.json/TAIL_TRAINING.json; MATHEMATICAL_CHECKS.json; REMOTE_NATIVE_RECEIPT.json/REPLAY_RECEIPT.json; results/figures; RESOURCE_RECEIPT.json; REPRODUCTION.md; and FINAL_MANIFEST.json. No further scientific computation was performed for closure.
