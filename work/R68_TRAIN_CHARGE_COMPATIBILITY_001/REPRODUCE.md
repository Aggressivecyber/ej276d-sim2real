# Reproduce the finite TRAIN charge certificate

This is a charge compatibility diagnostic, not a trained response model or A-F acceptance. Original native execution completed once; replay was not repeated for delivery.

Use Windows Python with NumPy. All new output must be on E or D. The private Git revision supplies the scripts and compact receipts; the matching D backup supplies the private NPZ inputs. Restore their relative paths under one project root. Required dependencies are the files bound by `reviews/R68_TRAIN_CHARGE_COMPATIBILITY_001/REGISTRATION.json`, the original R64 `ENDPOINT_GEOMETRY_PRIVATE.npz`, and `reviews/R65_PREPARATION_001/training_audit_adapter/local_resource_guard.py`; the D snapshot includes these exact files. The archive SHA256 must equal `e5562444d91498a70412c28117936a059dcb084bf1999093b07c3434aca313da`.

From the project root, explicitly request a fresh output directory:

```powershell
python -B .\reviews\R68_TRAIN_CHARGE_COMPATIBILITY_001\reproduce_in_fresh_directory.py --output E:\GAN\reproductions\R68_new_copy
```

The entry refuses an existing output directory, keeps the sealed result intact, enforces one CPU/1GiB and a 60-second child limit, validates the frozen source/input hashes, and runs the same two prespecified scalar descriptions. It does not launch SSH, update the project job ledger or train a model. Concurrent scientific work must have ended before manually requesting this replay. Expected exact KS lower bound: `34468748/138762871`; expected threshold BA lower bound: `173231619/277525742`. Times, PIDs and local resource observations naturally differ. Five artificial endpoint cases are checked before scientific scalar decoding. The replay entry was statically checked, not run as a second scientific evaluation.

The PNG uses `CDF_ENVELOPE_PRIVATE.npz`, a saved derivative of the original completed run. Counts and bounds in RESULT.json are exact rational/integer evidence; plotting coordinates are float64 approximations.
