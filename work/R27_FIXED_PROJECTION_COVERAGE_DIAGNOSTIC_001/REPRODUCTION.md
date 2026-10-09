# R27 reproduction

Actual local CPU interpreter: C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe. Versions: {"Python": "3.13.3", "NumPy": "2.2.6", "sklearn": "1.7.2"}. CPU threads4, no CUDA/Torch import required. Scientific formulas and hashes were frozen before the first HGB in PRECOMPUTE_FREEZE.json and config.json/PROTOCOL.md.

Original commands (cwd this stage, -B, all temporary/cache paths under E):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:TEMP='E:/GAN/research/sim2real_two_papers_20261008_0305/.runtime-cache/tmp'
$env:TMP=$env:TEMP
& 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe' -B supervise.py numeric_worker.py 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe'
& 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe' -B supervise.py render_figures.py 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe'
```

These commands were executed once in this stage; no extra science replay was performed or required because response inference/generation is0. Saved arrays, predictions, matrices and RNG states support detailed reproduction. The .used tokens deliberately prevent a duplicate submission here. A future authorized reproduction must use a new E/D directory and its own deadline/counters; never modify closed stages or remove tokens to force a rerun. Any numerical reproduction would count its15 HGB fits and runtime separately.

Reuse the original precomputed actual phase0/1 C2ST rows and cluster identities, TRAIN scaling and four new projection seeds, rather than resplitting or fitting scalers on holdout. Main output2000 and joint128×8 from the four historical endpoints are read-only. Historical493-column classifier results were not substituted for these pure480 views. PROJECT dependencies and original definitions are preserved through INPUT_BINDINGS.json; no new broad source hash or migration-content test was performed.

Inputs:

- generated/R23_D1_SELECTED: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R23_ONLINE_STANDARDIZED_CRITIC_001\STANDARDIZED\generated.npz`
- generated/HOMO_SELECTED: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\HOMO_SELECTED\generated.npz`
- generated/HOMO_FINAL: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\HOMO_FINAL\generated.npz`
- generated/R23_D1_FINAL2950: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\R23_D1_FINAL2950\generated.npz`
- joint/R23_D1_SELECTED: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R23_ONLINE_STANDARDIZED_CRITIC_001\results\STANDARDIZED_joint8_REPEAT_PRIVATE.npz`
- joint/HOMO_SELECTED: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\results\HOMO_SELECTED_joint8_REPEAT_PRIVATE.npz`
- joint/HOMO_FINAL: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\results\HOMO_FINAL_joint8_REPEAT_PRIVATE.npz`
- joint/R23_D1_FINAL2950: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\results\R23_D1_FINAL2950_joint8_REPEAT_PRIVATE.npz`
- real: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\REAL_DEV_AND_REFERENCE_PRIVATE.npz`
- source: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\ACTUAL_GENERATION_INPUTS.npz`
- source_metadata: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\DEV_SOURCE_METADATA.npz`
- fine_constants: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R23_ONLINE_STANDARDIZED_CRITIC_001\FINE_CONSTANTS.npz`
- pooled_constants: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R23_ONLINE_STANDARDIZED_CRITIC_001\VALIDATION_INPUTS.npz`
- classification_rows: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\results\C2ST_PRIVATE.npz`
- old_metrics_compact: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\METRICS_COMPACT.json`
- R26_manifest: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R26_CONDITIONAL_BATCH_TRANSLATION_001\FINAL_MANIFEST.json`
- R23_manifest: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R23_ONLINE_STANDARDIZED_CRITIC_001\FINAL_MANIFEST.json`

Evidence: numeric_worker.py.NATIVE_RECEIPT.json (actual52400, supervisor25396, native0), render_figures.py.NATIVE_RECEIPT.json (actual58504, supervisor60948, native0), RUN_HISTORY.json, NUMERIC_RECEIPT.json and PLOT_RECEIPT.json. Each actual worker PID was acknowledged before imports, with RSS monitoring failures0. TEMP/TMP remain E; C is used only to run the existing interpreter/system dependencies.

The dimensional span is not the full loss nullspace. New projection-block ranges are not confidence intervals. REAL TRAIN–DEV is one-acquisition descriptive sampling reference. DEV has repeated historical exposure; no EVAL/CAL waveform was opened. Shared total normalization and signal/electronic covariance prohibit a physical-independent-noise interpretation.
