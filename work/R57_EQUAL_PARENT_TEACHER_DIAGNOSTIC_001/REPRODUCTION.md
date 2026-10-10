# R57 reconstruction

This is a TRAIN-only statistical teacher diagnostic. It is not a neural transformation or a passed A-F evaluation. The original SOURCE evaluation identities, order, composition, and weights remain unchanged.

`config.json` and `INPUT_BINDING.json` bind the original TRAIN archive, original logQ edges, frozen R54 pulse basis, and frozen REAL TRAIN valley definition. No source_val/target_val, DEV, EVAL, or CAL arrays are decoded. Raw input archives are retained at the bound original locations; the sparse coupling and every edge feature are saved privately, not in Git.

The CPU worker is `numeric_worker.py`, launched once via `linux_supervise_post.py numeric_worker.py` and `linux_owned_worker.py`. The supervisor enforces CPU12/RSS48GiB and a single project scientific process, with identity handshake before scientific imports. Scientific time, output volume, and method-count budget fields are null. Use the recorded environment and config paths on the authorized server. For a separately authorized replay, copy code and bindings to a new E/D or server directory and launch there, preserving the original `.used` marker and native receipts.

`coupling_core.py` constructs the exact integer coupling. The ten raw-Q regions include both support-exterior regions, and each has exactly 16 charge-mass groups. `COUPLING_PRIVATE.npz` stores source and REAL indices, charge group, positive integer edge mass, and global denominator. Each SOURCE has total mass exactly 1/6593; REAL parents are equal within each region, while their global region mass follows SOURCE. No latent class or composition label selects a teacher edge.

For edge j, reconstruct the full preADC teacher as `teacher_core.endpoint(real[REAL_index[j]], source_Q[source_index[j]], B)` with a batch dimension. The source Q uses `noise_core.q_np`. Then apply the unchanged `canonical.canonical` for the common 14-bit representation. Original REAL baseline samples 0:100 and tail samples 400:480 are copied exactly; the frozen basis changes only the pulse. Failed or overrange mass is retained.

`FEATURES_PRIVATE.npz` contains all source/REAL/teacher features, edge masses, fixed boundary descriptors, and all 480-sample waveform moments. `FIXED_EXAMPLES_PRIVATE.npz` retains the first edge in each raw-Q region. `RESULT.json` contains the predeclared teacher conditions and all negative findings. The six PNG figures use complete observed ranges. `NUMERIC_RECEIPT.json`, native receipts, resource observations, root review, and final delivery receipt distinguish scientific completion from model acceptance.
