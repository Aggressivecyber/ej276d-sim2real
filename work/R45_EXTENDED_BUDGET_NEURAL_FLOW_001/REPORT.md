# R45: longer fixed neural-flow training, numerical prerequisite failed

The only registered 50,000-update endpoint was trained successfully, but its fixed 32-step Heun recipe failed the TRAIN numerical prerequisite. Close this recipe. All five distribution/scientific gates are **not evaluated**; numerical/replay gate 6 fails. No DEV arrays were decoded or generated, no HGB/EM was fitted, and no 4,000-row main replay was run. This provides no new claim about waveform or PSD distribution similarity.

R43 model and Adam at 2,950 were inherited exactly. New 47,050 updates consumed 1,505,600 rows per domain; including history, 1,600,000 exposures per domain. The unique sampled SOURCE/REAL populations are [3752, 18739]; loaded pools are [6593, 21047]. These repeated Monte Carlo exposures do not add independent parents. Original empirical coupling, constants, scalar-time architecture, loss and optimizer rules stayed fixed. Additional t continued R43's final PCG64 state; white used independent PCG64(20261010452). Statistical PSD-order coupling is not physical event pairing.

The fixed 4,096 TRAIN states use 512 occurrence positions per Q bin, one PCG64(20261010453) iterated bins 0–7, with independent probe t/white. They reuse TRAIN parents and are not unseen-parent validation or model selection. MSE fell 4.391% (final/initial 0.956090509); the registered 10% descriptive marker is false. PSD velocity RMSE improved overall but signed residual changed sign and the series fluctuates. None of these path-velocity quantities is generated-waveform PSD acceptance or proof of convergence.

| Total update | Fixed-probe MSE480 | PSD velocity RMSE | Signed PSD residual |
|---:|---:|---:|---:|
| 2950 | 0.875450319 | 0.029373776 | +0.009189923 |
| 5000 | 0.869177684 | 0.030071057 | +0.012485995 |
| 10000 | 0.857917774 | 0.026019746 | -0.007847388 |
| 15000 | 0.853879229 | 0.024520473 | -0.001225830 |
| 20000 | 0.848572971 | 0.023876418 | -0.001538003 |
| 25000 | 0.846398929 | 0.023190261 | +0.003536982 |
| 30000 | 0.845137971 | 0.023491878 | -0.006053565 |
| 35000 | 0.842185763 | 0.023443838 | +0.002939635 |
| 40000 | 0.837433780 | 0.022825448 | -0.002374759 |
| 45000 | 0.849223398 | 0.027958947 | +0.012318653 |
| 50000 | 0.837009741 | 0.023099292 | -0.007178327 |

On the unchanged 64 SOURCE + 64 REAL TRAIN inputs/white, CPU12/chunk64 preADC 32-vs-64 RMS median is **0.422870993 LSB > 0.25**, p95 **0.866297412 LSB <= 1**. Maximum relative q errors are 32-step f32 1.5287638e-08, 64-step f32 1.62241383e-08; f64 errors are about 2.6e-16. Charge preservation passes, median integration tolerance fails. The 64-step result is a numerical reference, not a rescue candidate. No solver/representation/floor was changed and no intermediate checkpoint was selected.

Training actual worker107155/super107150 ended 01:34:34.348301 UTC, native0; CPU check113870/super113865 ended 01:39:30.691453 UTC, native0; local plotting25148/super59088 ended 01:40:46.605670 UTC, native0. Total scientific charge **1463.158563411 s**, comprising GPU training 1443.364407056, CPU numerical check 16.143875255, plotting 3.650281100. Historical R43 worker training 76.152305712s is separately retained, not repeated here or treated as free.

Server worker RSS HWM 2,042,970,112 B; sampled tree peak 2,037,903,360 B. GPU process peak 629,145,600 B; allocator peak 313,275,392 B, reserved 343,932,928 B, with containment rather than addition. Two terminal RSS query gaps remain recorded with cause **unknown**; self HWM is available. CPU check and local plotting had zero gaps. There were three ordinary source/control query errors and no failed scientific process; their original tools and unknown PIDs remain in ENGINEERING_EVENTS/NATIVE_CHAIN. Normal transfer exits were successful; migration content consistency tests were not performed.

All five figures were actually opened and checked. Weights, Adam states, every-5,000 engineering checkpoints, dense loss/exposure records, t/white RNG states and per-batch hashes remain local/server research artifacts. Prepared generation/evaluation/replay code and config_cpu are **unexecuted** because the numerical stop fired. Only FINAL_FLOW/FINAL_OPTIMIZER at total50,000 represent this failed candidate. The project remains incomplete under PROJECT_ACCEPTANCE; this stage closes only its registered recipe and starts no new experiment.
