# R31 frozen normalization anchor diagnostic

Four saved domains: REAL DEV, REAL TRAIN reference, R28 FRESH_FINAL2950, R29 ENERGY_FINAL2950. Exact old classifier train600/holdout250, groups, QC, original Q bins, R30 domain moment parameters and its original six-domain support bins1–7 are reused. No moment estimation, response computation, EM, new REAL, EVAL or CAL reads.

Reconstruct the frozen full480 u from saved V, then apply the original R30 GLOBAL_CENTER/Q_CENTER/Q_CENTER_SCALE parameters with unchanged global fallback/floor. The sole new mask keeps0:100 and150:480 (430 columns). Never rebaseline, renormalize, rebin or change the mask. Full480 old scores remain visible. All source parents/events remain.

For every domain describe original peaks, peak-floor activation, max(u)-1 errors and argmax inside/outside100:150 for all/train/holdout rows. The no-fit decoder operates on full480 GLOBAL_CENTER holdout features w: e_A=abs(max(w+mu_A)-1), e_B=abs(max(w+mu_B)-1), predict the lower-error domain, abstain if abs(e_A-e_B)<=1e-12. True labels only score predictions. Keep floor-active events in main classification and separately report applicability; save error quantiles[0,.5,.9,.99,1], class coverage, nonabstention BA and all-event worst/best abstention bounds. No tuned threshold or fit.

Three comparisons x four430 views =12 HGB fits, old parameters/seed and train-only imputer/StandardScaler. Increment before fit; persist predictions before500 fixed-classifier artificial-group bootstrap replicates. Pair every430 view against its same-row old480 classifier, and three transformed430 views against RAW430. Any row/label/group mismatch makes a pair undefined. Original-Q regional BA at both domain N>=10 is descriptive; no regional fit or bootstrap.

Control rule: RAW430 REAL–REAL BA95CI must include.5. A transformed430 view is usable only if its paired-minus-RAW430 BA95CI is entirely inside[-.10,+.10]. This operational diagnostic tolerance is not physical equivalence. One mask only; if control still fails, close this domain-moment-C2ST causal attribution route. Masked success cannot promote complete waveforms. No unique covariance/electronics attribution.

CPU4 actual PID/RSS guard, science300s including failures and plots, RSS4GiB, output96MiB. Executor stop/seal17:11:45.842690UTC. E/D only, no SSH/install/Git/agents. All old stages remain immutable.
