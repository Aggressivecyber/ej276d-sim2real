# R19 reproduction

Actual server: rylan@100.82.231.102. Executed root:
/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R19_REMOTE_20261009_080433
Interpreter /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python.
Entrypoint `python -B remote_supervise.py`: sequential numeric_worker.py then replay.py. CPU4/CUDA training; saved main/probe/repeat inference CPUfloat32. Host/environment/actual PIDs/native exit are in REMOTE_NATIVE_RECEIPT.json and LOCAL_NATIVE_RECEIPT.json.

Static/local package is this R19 directory. `deploy.py` creates only a fresh remote root and uses normal scp; `launch.py` launches one supervised job; `retrieve.py` fetches the return archive without migration equality tests. To run in a new directory, copy only this code/config/protocol/TRAIN_BAND_REFERENCE/REFERENCE_METRICS/VALLEY and corrected probe/repeat input packages; edit only the new copy's remote_root/execution.remote_root and deadlines. Preserve original R19, R18 and input caches. No environment installation needed. Validate scientific resource/deadline availability before another run; this stage itself is sealed and must not be rerun.

Required read-only remote inputs:
- R10_REMOTE_20261009_012338/inputs (G4_TRAIN_cache, old prefix REAL and metadata, group selection and fixed definitions). Only first5200 REAL rows viewed numerically; no CAL waveforms.
- R14_REMOTE_20261009_044129/TRAIN_ADDITION.npz and expanded_fit_original_index.bin.
- R18_REMOTE_20261009_071126/attempt_03: TRANSLATOR.pt (selectedstep3500), INITIAL_CRITIC.pt, VALIDATION_INPUTS.npz, ACTUAL_GENERATION_INPUTS.npz, generated.npz and old C2ST_PRIVATE. Local SOURCE_PROBES_CORRECTED.npz/REPEAT_NOISE_PRIVATE were copied to this new R19 package normally, not changed in R18.
- TRAIN_BAND_REFERENCE.json contains only R3B REAL_TRAIN_fit fixed Gaussian slices. No DEV parameters used to fit tilts; old REAL/SOURCE/R18 metric summaries and fits reused.
Input versions are historical bindings in INPUT_BINDINGS/config; no new source rehash or migration consistency test was performed.

Inference replay from saved outputs needs only config, model.py, replay.py, ACTUAL_GENERATION_INPUTS.npz and each arm's selected TRANSLATOR.pt/generated.npz. It loads no target/REAL/TRAIN array. Already executed independent worker PID206128, native0: 4000 rows/1,920,000 points, both pre/post ADC bitwise, maxabs0, source IDs identical. Evidence REPLAY_RECEIPT.json and REMOTE_NATIVE_RECEIPT.json; no need to replay again.

Training replication uses saved common seeds, same R18 initial model/critic, new Adam, TRAIN_SOURCE/REAL pools, TRAIN-frozen lambda. The private TRAINING_STREAMS contains actual source/target/bin/U/NCE selections; TRAINING_RANDOM_STATES holds initial/final noise RNG plus stream digest. Same source U and distinct CDF intentionally yield different source batches. Pairing receipt confirms common streams, equal4000 updates. Selected steps CONT2500/TILT3000 using the same weighted validation; finalstep4000 kept. Selected-vs-final trace differs and original R18 unweighted J is preserved. No DEV recipe/selection.

Full metrics, weights, source/target identities, private streams and waveforms remain local E/D and the dedicated server directory. Git publication should use only aggregate METRICS_COMPACT/REPORT/QA/config/code/figures, not private identities/checkpoints/payloads. No Git command was executed by this executor.
