# R71 frozen charge component

This is the completed TRAIN charge component. Complete waveform A–F is unvalidated. The unique weights are `FINAL_HEAD.pt`; `HEAD_STATISTICS.npz` contains the input standardization and frozen continuous template/noise constants needed for inference. `TRAINING_RECEIPT.json` binds the weights by SHA256. All output rows retain their original parents; there is no domain flag.

Input NPZ contains `waveforms_V` with shape `(N,480)` in the original common window and volts. Use the original pre-response waveform, including its baseline. Supply either an NPZ with `epsilon` of the same shape or an explicit PCG64 seed. Keep the original row order. Any one-dimensional input metadata of length N is copied to output with prefix `input_`, without entering the model.

From this directory, using the recorded compatible PyTorch/NumPy environment:

```powershell
python -B infer.py --input E:/path/input.npz --epsilon E:/path/epsilon.npz --output E:/path/R71_output.npz --device cpu --threads 1
```

Alternatively replace `--epsilon ...` with `--seed 20261011711`. It draws the full `(N,480)` epsilon array once; changing the row count/order changes its row association. Output contains preADC, canonical ADC, Q0/Q*/actual Q, original and output conditions, support and projection metadata. Existing output files are never overwritten. New local data must remain on E/D. Apply the project resource guard for automated runs, and keep the one-scientific-job policy.

The delivered CLI was actually executed on the original first64 SOURCE plus first64 REAL numerical rows, using their saved epsilon. Its preADC outputs were bitwise identical to the CUDA endpoint. See `reviews/R71_REAL_CHARGE_FULL_TRAIN_001/inference_cli_check/CLI_EXECUTION_RECEIPT.json`.

Training is reproduced by the frozen `train_full.py`, `PROTOCOL.md`, `config.json` and `INPUT_BINDING.json` in the recorded server environment. It uses all6593/21047 parents each step, a separately counted50-update cost measurement followed by exact initialization/optimizer/RNG restoration, and the unique50000 endpoint. Do not relaunch a used entry or choose recovery checkpoints by quality. To relocate a reproduction, create a new owned directory and prospective execution receipt, resolve each input by its existing SHA, and preserve the original source/config bytes alongside the path mapping. The historical original TRAIN archive and frozen constants remain explicit dependencies. The D backup includes a copy of TRAIN and the original required constants in addition to this stage's complete artifacts.

`NUMERICAL_REFERENCE.json` uses a legacy field name `ADC_code_equal` for float64 voltage equality. Independent quantizer auditing found actual integer ADC codes identical; float64 voltages differ by at most1.11e-16V. The original field is preserved. `TRAIN_SERIALIZATION_REPLAY.json` is a TRAIN4000 component replay, not the final held-out independent4000 acceptance.

Figures and metrics use every original parent. Final source/REAL acceptance, task utility and REAL-only GAP verification have not been established by this component. See `R71_REPORT.md` in the project root and the actual `R71_DELIVERY_RECEIPT.json` for private Git/D backup delivery.
