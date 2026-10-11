# R65 fixed implementation

The training entry runs the eight inherited/new CPU and CUDA artificial checks
before decoding TRAIN. The original ResponseNetwork starts from the R63 initial
tensor hash. Every update uses one new full-parent noise array, all 6593 SOURCE
and 21047 REAL parents, one joint five-D endpoint gradient, every original
64-row parameter VJP, and one unchanged 384-row C call. The shared checked
helper installs the CPU float64 sum once, clips once and takes one Adam step.

All original seven PCG64 chains and final states and C parent exposures are
checked against R63. Both frozen statistics archives and the original source
precision intervals are reused. No normalization, calibration or valley fit is
repeated. The eighth stream is separately declared and its actual byte chain
and states are saved; no second50000-update independent regeneration is claimed.

## Recovery transaction

The cache is saved after a complete optimizer/log/counter commit, every500
updates including50000. A same-directory temporary checkpoint is fsynced and
hashed. A small PENDING journal containing that SHA, committed update, creator
attempt and frozen bindings is atomically persisted before the rolling cache
is replaced. The cache directory is fsynced, then the COMMITTED receipt is
atomically persisted. Either receipt can identify valid cached bytes following
an interrupted publication. A receipt hash and the originally published cache
hash are required; a digest newly invented after interruption is insufficient.

Before a recovery launch, the root must confirm the actual prior PID is gone,
preserve its native/log/marker history and partial finalization bytes under
native_attempts, and prepare RECOVERY_REQUEST.json. The trainer validates all
bindings, the previously published journal SHA, cache SHA, and preserved copies.
The checkpoint creator may precede the immediately previous native attempt.
The real new worker nonce is used when appending the recovery history edge.
Update50000 is recoverable for final artifact publication with zero new updates.
The unique endpoint and learning-rate schedule are never extended or selected.

## Computation records

Completed logical updates, actual per-attempt forward/VJP/Adam operations, and
unknown crash intervals are different records. An Adam call may complete before
its subsequent log commit; such an operation remains in the physical record as
a known uncommitted optimizer step. Recovery does not subtract it. A normal
uninterrupted final run has exactly the preregistered count. Artificial network
and optimizer operations are separately recorded and are never scientific
response training updates.

The original full response acceptance remains unchanged. R65-specific audit
logic adapts record semantics, including D/C/discarded micro-D loss and exposure
fields, before the unchanged numerical and complete response evaluation gates.
