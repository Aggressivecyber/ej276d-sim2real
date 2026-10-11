# R65 full empirical TRAIN response: running

R65 is one registered response-training candidate. The eight frozen CPU/CUDA
artificial checks passed before scientific TRAIN decoding, and the owned server
worker then started the sole 50000-update run from the original untrained
initialization. No final scientific acceptance result exists yet.

The frozen method uses all 6593 SOURCE and 21047 REAL TRAIN parents for each
five-term distribution objective, with one fresh noise draw per SOURCE parent.
The endpoint gradient is replayed through the original 64-row network chunks,
summed over every parameter in CPU float64, and combined with one original
384-row constraint microbatch before a single cast, clip and Adam update.
The original acceptance thresholds, support-outside parents and source weights
remain in force. The D-C random coupling and compute exposure differ from R63;
this is not an equal-compute or noise-integrated population comparison.

The implementation review binds 112 files, including 59 parsed Python files.
The two new network/Adam checks cover all 1369889 parameters. Their 25 CPU and
21 CUDA assertions passed at the predeclared tolerances. Artificial synthetic
optimizer steps are recorded separately from scientific training updates.

## Evidence and current state

- [Registered protocol](tasks/R65_FULL_EMPIRICAL_D_RESPONSE_001.md)
- [Frozen implementation binding](work/R65_FULL_EMPIRICAL_D_RESPONSE_001/ROOT_IMPLEMENTATION_REVIEW.json)
- [Root review of eight actual artificial receipts and launch](work/R65_FULL_EMPIRICAL_D_RESPONSE_001/ROOT_LAUNCH_AND_ARTIFICIAL_REVIEW.json)
- [Latest actual process/progress observation](work/R65_FULL_EMPIRICAL_D_RESPONSE_001/LATEST_REMOTE_OBSERVATION.json)
- [Inherited full project acceptance](PROJECT_ACCEPTANCE.md)
- [Prior R64 fixed diagnostic and negative B128 result](R64_REPORT.md)

The server worker PID at launch is 291399; its native attempt is
`37c43e76bb064c44937ebe888d58d0da`. PID numbers alone are not permanent job
identities: the launch, handshake, working directory and current observation
must be used together. The server root is
`/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R65_ROOT_20261011_030107_UTC`.

## Completion and recovery

Only update 50000 is a scientific model endpoint. The original TRAIN 128/256
numerical gate and R65 record audit must pass before complete SOURCE and
same-treated REAL A-C evaluation and independent replay. Original D/E work is
conditional on the original gates. Intermediate losses cannot select a model
or alter this running method.

The trainer writes one atomic same-run recovery checkpoint every 500 committed
updates. Recovery needs proof that the previous actual process is gone,
preserved native-attempt evidence, the originally published checkpoint digest,
all frozen bindings and exact optimizer/RNG/counter restoration. Do not launch
the deployment script again to recover an existing run. A static recovery review
and synthetic record tests passed; native recovery has not been exercised.

Server limits remain CPU12, tree RSS48GiB and GPU10GiB, with one scientific job.
Local auxiliary work shares CPU4/RSS4GiB; all new local files are on E/D. The
fixed endpoint is not a time quota. The long-term model goal remains active
until the same frozen method passes original A-F and final delivery is verified.
