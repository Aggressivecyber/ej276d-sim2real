# R65 equivalent implementation integration review

Prepared-code conclusion: **usable; no scientific-path code blocker found**.
This is a bounded local static review. It does not certify installation,
checkpoint contents, native recovery, or final scientific acceptance. No model
state, scientific arrays, SSH, Git, training, or CUDA execution was performed
for this integration review. Root's reported CUDA result is separate native
evidence; this review verifies that the prepared helper contains the tested
candidate implementation.

The reviewed prepared files are under
`reviews/R65_THROUGHPUT_OPTIMIZATION_001/prepared_upgrade/`. Original-source line
references below are to `work/R65_FULL_EMPIRICAL_D_RESPONSE_001/`; prepared line
references explicitly say `prepared`.

## Independently checked prepared diff

- Main `for step` loop AST is identical between original and prepared driver.
  The prepared loop starts at line 545 with the same
  `range(updates+1,C['updates']+1)`. Thus restoring committed update 500 continues
  logical updates 501 through 50,000 with the original chunk schedule, CPU f64
  sum order, objective calls, counters, clipping, Adam, and random draws.
- `parameter_vjp_cpu_f64`, `scalar_parameter_gradient_cpu_f64`, and
  `install_cpu_f64_clip_step` ASTs are unchanged. In the prepared helper these
  start at lines 68, 81, and 92. The candidate flatten body AST equals the tested
  `gradient_pack_candidate.py` body after mapping its function name to the
  original private API name. The additional legacy fallback is the old sequence.
- Prepared `network_update_check/SOURCE_BINDING.json:4` equals the actual
  prepared helper digest. Its tolerance, fixture, and objective-input bindings
  stay unchanged.
- All three before/after digests match the prepared receipt exactly:

| File | Original SHA256 | Prepared SHA256 |
| --- | --- | --- |
| `train_final.py` | `54834b0fd73ee08a420096e791be6038c2193acfb22c91c81563e059b64c67f3` | `d8cc8c3e9fd844bb6c375847fa8a25929c1d4a971931cc5813d17f8858c3a84b` |
| `network_update_check/network_update_check.py` | `db8976450aec139e61b7a39b5d6f2c8234028b676fb578eea1819cda8ee77945` | `cbd6c259cf9067b5ba376bc8f88a5e2f0fe40f5b55bf56b954672883c897dd69` |
| `network_update_check/SOURCE_BINDING.json` | `7e3a60f76968da95d96fd20953d0cd34459a8591406bad99aaa52ef137587afb` | `8c49cb9c5c528944cf5e44debdb0ac5874b93c8e9767107b8ca4e50e87fe66a4` |

## Minimum binding changes and compatibility

1. **Helper self binding.** Original helper `_verified_bundle` at lines 135–162,
   specifically 139, rejects a helper whose SHA differs from
   `SOURCE_BINDING.json:4`. The prepared self binding already matches. Preserve
   the old helper and old self-binding bytes under the old native-attempt archive.

2. **Current implementation review.** The original review binds the helper at
   lines 238–240, its self binding at 253–255, and the driver at 548–550; its
   top-level `driver_SHA256` is line 10. These need the three prepared digests and
   correct byte sizes. Driver startup at original lines 234–237 and finalization
   at 735–736 still require every current listed file to match. The original
   review digest is
   `e0a4b101886f5cb10b205364fe8414ca2db1fa211cbb058de7d4113f85ba24f5`.
   Preserve that original review byte-for-byte in the old native-attempt archive;
   issue a new current review containing the migration declaration and its
   CPU/CUDA evidence. Root's planned count 115 must be the actual new list count.
   Preserve original preparation receipts as historical evidence rather than
   editing them to claim they originally reviewed the new helper.

3. **Old checkpoint / new implementation bridge.** The original driver computes
   current `frozen_binding` at lines 250–255, then requires request, publication,
   and checkpoint all to equal it at 263, 268, and 275. Prepared lines 259–280
   correctly distinguish the checkpoint binding from current binding, accept
   only the fixed original review SHA, and require protocol/config/input binding
   and full environment equality. Prepared 298, 303, and 310 retain exact old
   request/publication/checkpoint equality. Checkpoint bytes and the original
   publication-receipt digest remain checked at prepared 301 and 307. Do not
   rewrite the original `.pt` or old publication JSON to contain the new review.

4. **Artificial suite provenance.** Original line 305 requires the suite's
   implementation review SHA to match current binding. Prepared 288–292 run
   the original eight-check suite with the current review before checkpoint
   deserialization at 308; prepared 340–341 then keep the strict current-suite
   assertion. This solves the compatibility issue without a weakened suite
   check. The old original suite bytes must be archived before line 292 publishes
   the new current suite receipt. Both suites have their own truthful binding.

5. **Recovery history and next checkpoint.** Original restore constraints at
   408–436 remain intact in prepared 444–472: strict model load, original Adam
   state, original initialization and parameter order, committed log lengths,
   counters, all eight PCG64 states/chains, exposures, direction counts, and CPU
   and CUDA RNG states. Prepared 326 records the engineering transition in
   recovery history; 473–482 write the restore receipt. New rolling checkpoints
   use current binding; final receipt at prepared 805–806 does too. The old
   checkpoint remains evidence of the old implementation epoch, not an edited
   checkpoint relabeled as new.

## Final audit can remain strict and unchanged

`r65_training_audit.py:201–216` compares the **final receipt's** frozen-binding
hashes with current canonical protocol/config/input/review files and all current
review file hashes. Writing the final receipt with the new current binding makes
these existing checks valid; carrying the old review SHA into the final receipt
would cause a correct rejection. `r65_training_audit.py:225–227` binds the
current artificial-suite file SHA, so the newly executed current suite works
without altering this audit code. `POSTPROCESS_FILE_BINDING.json:26` and the
review's audit entry at 508–510 therefore need no change on this minimal route.
If root actually modifies the audit, both digests would then need updating;
changing it is not necessary for the prepared route reviewed here.

The unchanged physical audit at lines 104–177 already permits recovery from
registered 500-step snapshots (120–121), requires unique attempt IDs, and checks
the history link, original checkpoint SHA, confirmed prior PID death, and
physical versus logical work (145–154). Preserve the actual pre-migration
attempt and any completed work beyond checkpoint 500. Such work is replayed
physical compute, not extra committed logical updates. Unknown intervals remain
unknown. A fresh `OWNED_SCIENCE_NONCE` is required for the new attempt; the driver
uses it at original line 165. Keep `config.remote_root` unchanged because it
participates in the logical run ID at line 166.

## Declaration still needs final sealing

At review time `prepared_upgrade/IMPLEMENTATION_UPGRADE.json` had not yet been
created; the prepared source and `PREPARATION_RECEIPT.json` existed. This is the
remaining deployment preparation, not a prepared-code defect. The finalized
declaration must contain exactly the three source names/digests in the table,
their verified old archive paths under `native_attempts`, the complete old
frozen binding, the selected original checkpoint SHA, logical run ID, and exact
CPU/CUDA evidence paths and digests. Prepared lines 271–279 verify every listed
source/archive/evidence entry but do not assert the changed-source name set is
exactly those three. Root's final declaration seal must therefore explicitly
check that the list is nonempty, has no duplicate, and equals this exact set.
No broader compatibility exception is needed.

The current new review must bind this declaration and evidence. Preserve the old
review, suite, checkpoint, publication receipt, source bytes, physical progress,
and native attempt records before replacing current files. Avoid a mutual hash
cycle: the declaration can contain the old review SHA and new source digests;
the new review then binds the declaration; the recovery history stores the new
review SHA after the new review is sealed.

## Post controllers

The inspected prepared `r65_controller_common.py:27` hard-codes the old review
SHA. Lines 187–193 require that SHA and count 112 in `CONTROLLER_BINDING.json`;
197–208 require the old local review and count 112; 215 sends that SHA remotely.
After migration the old controller is expected to refuse. Root's stated plan to
stop it and seal a new controller against the actual new review/count 115 is
correct. Also refresh the controller-source hashes in its seal and install
matching current canonical local source/review bytes before protected fetch:
common lines 269 and 283–284 refuse an existing protected local file differing
from remote bytes. Keep the strict checks.

`r65_remote_controller.py` does not exist at the requested prepared path.
Common line 216 actually loads `r65_remote_gate.py`, which exists. That gate was
not read in this bounded request; root should account for any review/count
binding it holds when sealing the new controller. This review did not invoke
either controller or any remote operation.
