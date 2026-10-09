# RNSF-001: frozen R3B plus a full-waveform neural spline transport

The user explicitly requested a new Git branch and server computation for the
previously proposed flow trial. Branch `codex/nsf-fullwave-20261009` starts at
`9da21785404db7783311fac261d8a67fe555ce24`; its E-drive worktree is isolated from
the active main checkout. This is one bounded trial, not completion of the
whole research project. Existing jobs and main-stage results are not modified.

Use `rylan@100.82.231.102`, the existing scientific Python interpreter and a
trial-local nflows 0.14 dependency directory. Shared environment packages are
not upgraded. CPU4, process RSS4GiB, Torch allocated GPU4GiB, one scientific
job, 512MiB scientific outputs, 7200s cumulative numeric budget including
engineering numerical checks, failed attempts, generation, evaluation and
saved-model replay. Before every launch check existing project workers; do not
resubmit unknown jobs. Transport is normal file copying, without cross-machine
file-content tests. Data/models remain E/D locally and private on the server.

Inputs: existing G4 TRAIN cache 8192 parents (6593 fit/1599 validation), frozen
R3B checkpoint, its 2000 DEV source conditions, REAL first5200 records
(DEV2000; TRAIN3200 with2592 fit/599 validation after existing QC), frozen CAL,
width intervals, and original R7 evaluator selections. Preserve unknown and
unsupported records. No new G4 and no EVAL. Source generation uses one fixed
draw per TRAIN parent; random repetitions never become new independent parents.

Model: all480 voltage coordinates, one REAL-fit per-time affine standardizer
with fixed 1/16383V scale floor, 8 rational-quadratic spline coupling blocks,
8 bins, linear tails at standardized +/-8, 32-channel convolutional parameter
networks with time coordinates and dilation[1,2,4,8,16,32,64]. Alternating
even/odd masks. No PCA truncation, classifier, cycle or auxiliary feature loss.
All three flows initialize at identity and have the same architecture.

Train source and REAL densities separately by negative log likelihood, freeze
their parameters, then train the forward/inverse transport with equal-weight
bidirectional likelihood (change-of-variables log determinants included).
Adam lr2e-4, batch64, clip gradient norm5, seed2026100909. Density stages each
at most4000 updates/1200s; transport at most8000 updates/3600s; 1200s reserved
for remaining numerical work. Validate initial state and every100 steps using
all allowed validation records, choose minimum validation objective, patience10
checks. No seed, architecture, learning-rate or loss-weight search. A plateau
or budget limit ends its stage. The three fits form one training pipeline.

A fixed common uniform jitter of +/-0.5/16383V regularizes the continuous
density proxy in both domains. It is a declared smoothing convention, not a
claim that actual REAL ADC bin probability is known. Fresh training jitter,
fixed validation jitter, and saved inference jitter use separate seeded RNGs.
Raw REAL evaluation remains unchanged. Final generated voltage is saved before
and after nominal14bit[-.5,.5] ADC. No CFD recrop, rejection or regeneration.

Main DEV uses a newly generated frozen R3B baseline and the same2000 inputs
through the learned transform; all source conditions, latent/white/jitter and
parents are saved. The current-server baseline is compared directly so
historic backend differences are not a treatment effect. Extra64 fixed parents
have (a) source conditions permuted with noise held fixed, (b) fresh noise with
source held fixed; each is run through the entire chain. These are sensitivity
checks, not proof of correct particle labels or physical mapping.

Evaluation: reuse six R7 full/local views and fixed group-separated600/250
positions; common QC intersection for R3B/NSF/REAL/RR, minimum150/75 per domain,
at most18 HGB fits including REAL-REAL, 500 conditional group bootstraps.
Compare total Q/PSD, eight fixed charge-window peak positions/widths and
coverage, baseline and signal fluctuations, full waveform shape, clipping/QC,
nearest TRAIN neighbours and source sensitivity. Width descriptions for
R3B/NSF/REAL use at most24 fixed slots; all failures remain visible. Main
populations and filtering counts are reported separately. A saved-model fresh
process replays the full2000 end-to-end inputs; compare pre-ADC within1e-7V and
post-ADC exact values. Do not refit metrics during replay.

Advance only on substantial complete-response improvement with simultaneous
width/noise/support evidence. Near0.99 classification, renewed contraction or
support loss closes this recipe. A single classifier near0.5 is insufficient.
The trial does not claim physical equivalence, true particle-label preservation
or independent-acquisition generalization. Report measured results either way.

References: https://arxiv.org/abs/2309.06472 ;
https://github.com/jraine/flows4flows ; https://arxiv.org/abs/1906.04032 ;
https://github.com/bayesiains/nflows . The CNN coupling conditioner is the
project adaptation; no pretrained external detector model is claimed.
