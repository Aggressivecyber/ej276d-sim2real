R22 uses only local E/D. config.json defines all fixed input paths, seeds and ceilings. Old stages remain read-only.

CPU Python: C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe (Torch2.10.0+cpu/NumPy2.2.6). Existing GPU Python: E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe (Torch2.12.0+cu130/NumPy1.26.4). No installation. All child TEMP/TMP/MPL/JOBLIB/TORCH/CUDA cache variables point into this stage/cache.

In a fresh output directory copy the scripts/config and preserve input paths. Run CPU `supervise.py prepare_pool.py <CPU_python>`, then CPU `supervise.py train_attempt2.py <GPU_python>`, then CPU `supervise.py assess.py <CPU_python>`. One scientific process at a time. Do not run in the sealed original directory; used tokens prevent duplicates. No stage was scientifically replayed just for missingRSS.

FIXED_POOL_PRIVATE.npz preserves complete parent IDs/roles/random draws/ADC versions/standardizer and identical4000 sampling rows. RAW/ and STANDARDIZED/ store selected/final critic+optimizer. CRITIC_SCORES_PRIVATE/ALL_SCORES_PRIVATE retain all predictions; INPUT_GRADIENTS_PRIVATE includes64 actual input indices; BOOTSTRAP_PRIVATE stores paired replicates. These arrays/weights/models stay local. METRICS_COMPACT is aggregate-only. HGB pre/post are own newly fitted models.

This is a finite frozen TRAIN/internal-validation pool, not a new held-out acquisition or onlineGAN demonstration. CUDA supervision monitored a venv launcher, not actual scientificPID44800: actualRSSunknown; fix that before future CUDA use. Preserve failed_001 and native history. CPU/GPU version differences are explicit, not claimed bitwise equivalence.
