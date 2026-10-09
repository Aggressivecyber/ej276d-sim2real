# R11 reproduction

Executed server: `rylan@100.82.231.102`, host `rylan-GTi15`.
Directory: `/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R11_REMOTE_20261009_020915`.
Python: `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`.
Main: `python -B remote_supervise.py`, invokes CPU4 `numeric_worker.py` then a separate `replay.py`.

Read-only inputs: sibling `R10_REMOTE_20261009_012338/inputs`, keyed by config input_files. Historical R10 `metrics.json` and `generated.npz` are read-only from that sibling root. Local originals are listed in config.json; REAL metadata/waveforms from P1K, fixed roles from R2, R7 position selections, R3B density definition/generated waves. REAL V file contains6800 rows but only5200 rows are numerically viewed; CAL1600 is not decoded. No EVAL files required.

For fresh reconstruction of saved science, restore source/config and the two model folders in a **new** E/D or Linux study directory. Keep readonly inputs separate and update only path locations in a new configuration, preserving science settings. `python -B replay.py` requires saved metrics.json, each representation.npz, selected GMM_K.npz, GENERATION_INPUTS.npz and generated.npz. It reconstructs all4000 draws and compares pre/post outputs without fitting or redrawing. The executed REPLAY_RECEIPT.json records PID135146, pass, pre/post bitwise and maxabs0. Restore all files before running; old scientific originals must not be overwritten.

For a from-scratch reproduction, create a new directory and run the saved numeric_worker/remote_supervise with readonly input path mapping. Six GMM seeds, initialization, solver, coordinate maps, ADC and chunk sizes are explicit. Current server Python3.12.3, NumPy1.26.4, sklearn1.8.0, SciPy1.11.4, CPU4. No environment install/upgrade was used. Floating point behavior across different environments is not claimed bitwise identical. The old R3B/R10 models and CPU/GPU histories are historical comparisons, not newly paired fits.

All saved random inputs are actual component/head_white/tail_white arrays. New draws have no G4 parent event correspondence. Normal source upload/result transfer exited0; no migration equality test or new original-input hash audit was executed. Full local metrics, model parameters, noise and identity selections stay E/D; Git publication uses aggregate summary only.
