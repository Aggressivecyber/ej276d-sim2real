# R65 conditional D/E continuation dependencies

This is a preparation review while R65 is training. It registers no scientific
job, reads no new waveform/weight/array, and changes neither the active recipe nor
original A-F. D/E remains conditional on the same final R65 weight passing the
original numerical, A-C and independent-replay requirements.

The governing acceptance is limited to a historically exposed single acquisition.
Independent-acquisition, true-particle-label and online hardware success are not
part of the evidence presently available, and must not be appended to a positive
development result. Conversely, those unclaimed extensions must not be introduced
as new completion requirements for the original A-F.

## D: fixed prefix task and current data cost

Reuse the original R5 52-feature prefix 0:150 construction, targets (signed PSD
and logQ), Ridge(alpha=0.001), fixed HGB, sample weights and bootstrap. The required
arms are REAL-only, REAL+original SIM and REAL+the single frozen candidate, for
nested REAL budgets 256/1024, plus full available REAL TRAIN. That is 28 task fits:
(2 budgets x 3 arms +1 full REAL arm) x 2 targets x 2 estimators. The former R5
44-fit count included additional historical response candidates and must not be
carried into this one-candidate study. Both estimators remain required.

The metadata for the current response data reports 21047 REAL fit events and 599
internal validation events. Old R5's 2592-event pool cannot be relabeled as current
full REAL. Before registration, bind actual source IDs and artificial 50-record
groups for every allowed TRAIN/validation/DEV/EVAL role; verify intersections and
the nested SHA ordering without using EVAL target values to make design choices.
The exact eligible full-REAL task count is a future identity/QC check, not proven
by this metadata review.

R65 already uses every one of 21047 REAL TRAIN parents in every full-D step, plus
an identity microbatch. It also inherits full-data statistics and older priors.
The R48 noise receipt records 18739 distinct matched REAL calibration events;
R63 temporal statistics record 21047 REAL fit rows. These are not disjoint counts:
the eventual cost ledger must use identity unions instead of adding them. A 1024
task-training budget is not a 1024-event acquisition cost claim.

Retain the original R5 fixed 2000 G4 parent selection from the bound 8192-parent
TRAIN cache, ordered by SHA256('R5-synth-v1|' + parent_id). Do not substitute the
DEV 2000 evaluation parents or R65's 6593-fit subset. Bind original no-noise inputs,
original SIM generation and explicit candidate noise before generation. Any
unsupported parent is a recorded protocol/data issue, not permission to silently
replace the fixed set. The current review has not decoded these inputs or tested
inference on them.

Save all task models, standards, weights, training identities, generated arrays
and input/random/code bindings before EVAL value decoding. The original acceptance
limits are still PSD MAE improvement of at least 5% for both estimators at 1024,
both required paired 95% comparison upper bounds below zero, no more than 5%
deterioration against full REAL PSD MAE, and no more than 5% logQ RMSE deterioration
against the budget-matched REAL-only arm. Do not select a winning estimator.

## E: the GAP recipe needs its own prospective implementation binding

Keep the original CAL q3 interval [3703.7782948464155,6711.804904974998)mV ns.
The old R6 FULL/GAP curriculum is historical evidence, not an implementation of
R65. R65 currently binds the original full-population counts, the eight random
streams and inherited full-data noise/coordinate/distribution/temporal constants.
A GAP worker cannot simply load the finished FULL weight or copy those constants
and claim exclusion of q3 from fitting and statistics.

The current trainer asserts all ten SOURCE and REAL regional pools are nonempty.
ParentLayout.validate also rejects an empty regional list, and original C draws
32 occurrences in each region. This is a concrete applicability dependency:
the excluded GAP population must be inspected under a separately registered
TRAIN-only preflight before claiming the unchanged recipe is defined. This
review establishes the code requirement, not which regions actually become empty.

For a defined GAP study, reconstruct each TRAIN-derived statistic with the same
bound construction algorithm using only allowed GAP parents, preserving evaluation
boundaries and the original method. Bind matched initial trainable tensors and
the prescribed update endpoint before running either paired arm. Reusing the
completed FULL is permitted only if its saved initialization, data and random
policy satisfy the prospective paired design; saved nominally equal seeds alone
do not prove this. Genuine data exclusions will change draw domains and need
explicit stream semantics, rather than fabricated equality to R63's seven final
stream hashes or fixed full-population counters.

If the original recipe is undefined after the exclusion, record that limitation
and its exact population/operation evidence. Do not silently delete empty regions,
borrow full-data moments, modify thresholds or introduce an unregistered smoothing
rule to obtain a GAP success. Any proposed method change is separate prospective
work. A GAP negative result can complete the scope investigation only with its
required evidence; this preparation document alone completes none of E.

Report all source-nominal and output-own regions, migration, q1/q2/q4 controls,
support failures, and original A-C results. Keep every evaluation parent. Only
actually passing regions can be named validated; the historical q4 reference
failure and single-acquisition limitation remain visible.

## Action at R65's fixed endpoint

If original numerical or A-C/replay gates fail, close and deliver R65's negative
result; do not spend D/E fits to retrospectively seek an alternative success.
If they pass, bind the actual final weight and register D first, with the current
identity/cost and pre-EVAL freeze checks above. Prepare and register the one E
study with its own explicit exclusion/statistic/initialization bindings. Final F
requires actual runnable inference, all necessary saved evidence, opened figures,
private Git push verification and D-drive final delivery; the running-source
snapshot 49d62f8 and its backup are not that final receipt.

Sources are project originals, listed with hashes in SOURCE_BINDINGS.json. No
literature claim or newly executed scientific result is introduced here.
